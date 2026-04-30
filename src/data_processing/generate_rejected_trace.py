"""
Generate rejected samples for DPO training using vLLM (in-process).

Input format (each line of input JSONL):
  {
    "user_prompt": "<the user message, e.g., the [PYTHON] block>",
    "assistant_prompt": "[MONOLOGUE]\\n...reasoning...\\n[/MONOLOGUE]\\n[ANSWER]\\n...\\n[/ANSWER]",
    "task_type": "output_pred" | "input_pred" | "main_solution",
    "idx": 0  (optional, will be auto-assigned)
  }

The script:
1. Parses each `assistant_prompt` into `reasoning_content` (from [MONOLOGUE]
   tags) and `content` (from [ANSWER] tags). Tags are stripped — the values
   stored are the raw text inside.
2. Builds a perturbation prompt and asks the LLM to produce a subtly wrong
   reasoning + answer.
3. Verifies the rejected `content` differs from the chosen `content`.
4. Writes DPO records with SEPARATED `reasoning_content` and `content` fields,
   matching how chat templates with native reasoning support consume them.

Output format (each line of output JSONL):
  {
    "prompt": "<user_prompt>",
    "chosen": {
      "reasoning_content": "<original reasoning, no wrapper tags>",
      "content": "<original answer, no wrapper tags>"
    },
    "rejected": {
      "reasoning_content": "<perturbed reasoning, no wrapper tags>",
      "content": "<perturbed answer, no wrapper tags>"
    },
    "metadata": { ... }
  }

Usage:
  python generate_dpo_rejected.py \\
      --input chosen.jsonl --output dpo_pairs.jsonl \\
      --model Qwen/Qwen2.5-32B-Instruct \\
      --tensor-parallel-size 2
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Wrapper-tag handling
# ---------------------------------------------------------------------------

MONOLOGUE_RE = re.compile(r"\[MONOLOGUE\](.*?)\[/MONOLOGUE\]", re.DOTALL)
ANSWER_RE = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", re.DOTALL)


def split_assistant_prompt(assistant_prompt: str) -> tuple[str, str] | None:
    """
    Split an assistant_prompt with [MONOLOGUE]...[/MONOLOGUE][ANSWER]...[/ANSWER]
    into (reasoning_content, content), with the wrapper tags stripped.

    Returns None if either tag block is missing — caller should skip such rows.
    """
    mono_match = MONOLOGUE_RE.search(assistant_prompt)
    ans_match = ANSWER_RE.search(assistant_prompt)
    if mono_match is None or ans_match is None:
        return None
    reasoning = mono_match.group(1).strip()
    content = ans_match.group(1).strip()
    if not reasoning or not content:
        return None
    return reasoning, content


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PERTURBATION_PROMPT = """You are helping create training data for preference learning. \
Given a problem, a correct reasoning trace, and the correct final answer, you must \
produce a SUBTLY INCORRECT version of the reasoning that leads to a DIFFERENT final answer.

Rules (follow strictly):
1. Change EXACTLY ONE step in the reasoning — a calculation, a logical inference, \
or an interpretation of the problem. Do not introduce multiple independent errors.
2. The error must be plausible — the kind a careful person might make — not absurd.
3. All reasoning AFTER the error must be internally consistent with the error \
(propagate the mistake faithfully through to the final answer).
4. The final answer MUST differ from the original answer because of the error.
5. Keep the same structure, tone, section headers, and overall formatting as the \
original reasoning.
6. Write with the same confidence as the original — do not hint that anything is wrong.
7. The final answer must follow EXACTLY the same syntactic shape as the original \
(same prefix like `assert`, same wrapper template, same syntax — only the value(s) change). \
Do NOT add commentary, do NOT add markdown, do NOT add tag wrappers like [ANSWER].

ORIGINAL PROBLEM:
{user_prompt}

ORIGINAL CORRECT REASONING:
{reasoning_content}

ORIGINAL CORRECT ANSWER:
{content}

Respond with ONLY a JSON object (no markdown fences, no commentary) with these \
exact keys:
{{
  "perturbation_description": "one sentence describing what single step you changed",
  "rejected_reasoning": "the full incorrect reasoning trace (no wrapper tags, just the reasoning text)",
  "rejected_answer": "the resulting wrong answer (no wrapper tags, same syntactic shape as the original)"
}}
"""


def build_user_message(sample: dict[str, Any]) -> str:
    return PERTURBATION_PROMPT.format(
        user_prompt=sample["user_prompt"],
        reasoning_content=sample["_chosen_reasoning"],
        content=sample["_chosen_content"],
    )


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

REQUIRED_OUTPUT_KEYS = {"perturbation_description", "rejected_reasoning", "rejected_answer"}


def parse_json_response(text: str) -> dict[str, Any] | None:
    """Strip markdown fences if present and parse JSON; tolerant of stray prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(obj, dict) or not REQUIRED_OUTPUT_KEYS.issubset(obj.keys()):
        return None
    return obj


# ---------------------------------------------------------------------------
# Defensive tag stripping (in case the model re-adds wrappers)
# ---------------------------------------------------------------------------

def _strip_answer_tags(s: str) -> str:
    """If text contains [ANSWER]...[/ANSWER], return the inside; else return as-is."""
    m = ANSWER_RE.search(s)
    return m.group(1).strip() if m else s.strip()


def _strip_monologue_tags(s: str) -> str:
    """If text contains [MONOLOGUE]...[/MONOLOGUE], return the inside; else as-is."""
    m = MONOLOGUE_RE.search(s)
    return m.group(1).strip() if m else s.strip()


# ---------------------------------------------------------------------------
# Lenient verification
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s.strip())


def verify_lenient(chosen_content: str, rejected_content: str) -> tuple[bool, str]:
    """Verify rejected content differs meaningfully from chosen content."""
    rejected = str(rejected_content).strip() if rejected_content is not None else ""
    chosen = str(chosen_content).strip() if chosen_content is not None else ""

    if not rejected:
        return False, "rejected_answer is empty"

    if _normalize(chosen) == _normalize(rejected):
        return False, "rejected_answer identical to chosen_answer (modulo whitespace)"

    return True, "ok"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

VALID_TASK_TYPES = {"input_pred", "output_pred", "main_solution"}
REQUIRED_FIELDS = {"user_prompt", "assistant_prompt", "task_type"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read input JSONL and split assistant_prompt into reasoning + content."""
    samples = []
    skipped_no_split = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] line {i} not valid JSON: {e}", file=sys.stderr)
                continue
            missing = REQUIRED_FIELDS - obj.keys()
            if missing:
                print(f"[warn] line {i} missing {missing}, skipping", file=sys.stderr)
                continue
            if obj["task_type"] not in VALID_TASK_TYPES:
                print(f"[warn] line {i} unknown task_type {obj['task_type']!r}, skipping",
                      file=sys.stderr)
                continue

            split = split_assistant_prompt(obj["assistant_prompt"])
            if split is None:
                skipped_no_split += 1
                if skipped_no_split <= 5:
                    print(f"[warn] line {i} could not split assistant_prompt "
                          f"(missing [MONOLOGUE] or [ANSWER] block), skipping",
                          file=sys.stderr)
                continue
            reasoning, content = split
            obj["_chosen_reasoning"] = reasoning
            obj["_chosen_content"] = content
            samples.append(obj)

    if skipped_no_split:
        print(f"[warn] total {skipped_no_split} rows skipped due to missing tag blocks",
              file=sys.stderr)
    return samples


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_trial_filename(output_path: Path, trial: int) -> Path:
    """Generate filename with trial suffix: output_trial_0.jsonl, etc."""
    stem = output_path.stem
    suffix = output_path.suffix
    parent = output_path.parent
    return parent / f"{stem}_trial_{trial}{suffix}"


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def try_make_record(sample: dict[str, Any], raw_text: str,
                    retry_attempt: int = 0) -> tuple[dict[str, Any] | None, str]:
    """Parse + verify model output. Returns (record, reason); record is None on failure."""
    parsed = parse_json_response(raw_text)
    if parsed is None:
        return None, "json_parse_failed"

    rejected_reasoning = str(parsed["rejected_reasoning"]).strip()
    rejected_answer = str(parsed["rejected_answer"]).strip()

    if not rejected_reasoning:
        return None, "rejected_reasoning is empty"

    # Defensive: strip wrapper tags the model might have added back even though we asked it not to.
    rejected_reasoning_clean = _strip_monologue_tags(rejected_reasoning)
    rejected_content_clean = _strip_answer_tags(rejected_answer)

    chosen_content = sample["_chosen_content"]
    chosen_reasoning = sample["_chosen_reasoning"]

    ok, reason = verify_lenient(chosen_content, rejected_content_clean)
    if not ok:
        return None, f"verify_failed:{reason}"

    if not rejected_reasoning_clean:
        return None, "rejected_reasoning empty after tag stripping"

    # Pass through any extra keys from the input row into metadata.
    # Skip:
    #   - keys we already use directly (user_prompt is the `prompt` field;
    #     assistant_prompt is split into chosen.reasoning_content/content)
    #   - internal working fields prefixed with "_"
    consumed_keys = {"user_prompt", "assistant_prompt"}
    extra = {
        k: v for k, v in sample.items()
        if k not in consumed_keys and not k.startswith("_")
    }

    metadata = {
        **extra,  # idx, task_type, and any user-provided extras come from here
        "perturbation": parsed["perturbation_description"],
    }
    if retry_attempt > 0:
        metadata["retry_attempt"] = retry_attempt

    record = {
        "prompt": sample["user_prompt"],
        "chosen": {
            "reasoning_content": chosen_reasoning,
            "content": chosen_content,
        },
        "rejected": {
            "reasoning_content": rejected_reasoning_clean,
            "content": rejected_content_clean,
        },
        "metadata": metadata,
    }
    return record, "ok"


def build_prompts(samples: list[dict[str, Any]], tokenizer) -> list[str]:
    prompts = []
    for s in samples:
        messages = [{"role": "user", "content": build_user_message(s)}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                    help="Input JSONL with chosen samples")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output JSONL with DPO triples")
    ap.add_argument("--failed-output", type=Path, default=None,
                    help="Optional JSONL for failed samples")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct",
                    help="HF model id or local path")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--dtype", default="auto", help="auto | float16 | bfloat16")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N samples (debugging)")
    ap.add_argument("--limit-per-type", type=int, default=None,
                    help="Limit examples per task type")
    ap.add_argument("--retries", type=int, default=1,
                    help="Number of retry attempts for failed samples")
    args = ap.parse_args()

    # ---- 1. Load data -----------------------------------------------------
    samples = read_jsonl(args.input)

    for i, sample in enumerate(samples):
        if "idx" not in sample:
            sample["idx"] = i

    if args.limit:
        samples = samples[:args.limit]
    elif args.limit_per_type:
        by_type: dict[str, list] = {}
        for s in samples:
            task_type = s["task_type"]
            if task_type not in by_type:
                by_type[task_type] = []
            if len(by_type[task_type]) < args.limit_per_type:
                by_type[task_type].append(s)
        samples = [s for task_list in by_type.values() for s in task_list]

    print(f"Loaded {len(samples)} samples from {args.input}")

    counts: dict[str, int] = {}
    for s in samples:
        counts[s["task_type"]] = counts.get(s["task_type"], 0) + 1
    print("Task type breakdown:", counts)

    if not samples:
        print("No samples to process. Exiting.")
        return

    # ---- 2. Init vLLM -----------------------------------------------------
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        seed=args.seed,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # ---- 3. Build prompts -------------------------------------------------
    print("Building prompts...")
    prompts = build_prompts(list(tqdm(samples, desc="Templating")), tokenizer)

    # ---- 4. Generate ------------------------------------------------------
    print(f"Generating {len(prompts)} responses with vLLM...")
    outputs = llm.generate(prompts, sampling_params)
    outputs = sorted(outputs, key=lambda o: int(o.request_id))

    # ---- 5. Parse + verify ------------------------------------------------
    ok_records: list[dict[str, Any]] = []
    retry_samples: list[tuple[int, dict[str, Any]]] = []
    failure_log: list[dict[str, Any]] = []

    for sample_idx, (sample, out) in enumerate(
        tqdm(zip(samples, outputs, strict=True), total=len(samples), desc="Parsing")
    ):
        text = out.outputs[0].text
        record, reason = try_make_record(sample, text, retry_attempt=0)
        if record is None:
            retry_samples.append((sample_idx, sample))
            failure_log.append({
                "idx": sample.get("idx"), "attempt": 0, "reason": reason,
            })
            continue
        ok_records.append(record)

    trial_path = get_trial_filename(args.output, trial=0)
    write_jsonl(trial_path, ok_records)
    print(f"Wrote {len(ok_records)} records to {trial_path} (initial pass)")

    # ---- 6. Retry failed samples ------------------------------------------
    for retry_attempt in range(1, args.retries + 1):
        if not retry_samples:
            break

        print(f"\nRetry attempt {retry_attempt}/{args.retries}: "
              f"{len(retry_samples)} samples")

        retry_prompts = build_prompts([s for _, s in retry_samples], tokenizer)

        retry_seed = args.seed + retry_attempt
        retry_sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=retry_seed,
        )
        retry_outputs = llm.generate(retry_prompts, retry_sampling_params)
        retry_outputs = sorted(retry_outputs, key=lambda o: int(o.request_id))

        new_retry_samples: list[tuple[int, dict[str, Any]]] = []
        for (sample_idx, sample), out in zip(retry_samples, retry_outputs, strict=True):
            text = out.outputs[0].text
            record, reason = try_make_record(sample, text, retry_attempt=retry_attempt)
            if record is None:
                new_retry_samples.append((sample_idx, sample))
                failure_log.append({
                    "idx": sample.get("idx"), "attempt": retry_attempt, "reason": reason,
                })
                continue
            ok_records.append(record)

        retry_samples = new_retry_samples

        trial_path = get_trial_filename(args.output, trial=retry_attempt)
        write_jsonl(trial_path, ok_records)
        print(f"Wrote {len(ok_records)} records to {trial_path} "
              f"(after retry {retry_attempt})")

    # ---- 7. Build failed_records list -------------------------------------
    failed_records: list[dict[str, Any]] = []
    for sample_idx, sample in retry_samples:
        reasons = [f["reason"] for f in failure_log if f["idx"] == sample.get("idx")]
        last_reason = reasons[-1] if reasons else "unknown"
        # Don't dump the internal _chosen_* fields into failure log
        clean_source = {k: v for k, v in sample.items() if not k.startswith("_")}
        failed_records.append({
            "_status": "all_retries_exhausted",
            "_last_reason": last_reason,
            "_all_reasons": reasons,
            "_source": clean_source,
            "_idx": sample.get("idx"),
        })

    # ---- 8. Report --------------------------------------------------------
    total = max(len(samples), 1)
    print(f"\nSuccess: {len(ok_records)} / {len(samples)} "
          f"({100 * len(ok_records) / total:.1f}%)")
    print(f"Failed:  {len(failed_records)}")

    per_task_ok: dict[str, int] = {}
    for r in ok_records:
        t = r["metadata"]["task_type"]
        per_task_ok[t] = per_task_ok.get(t, 0) + 1
    print("Success by task_type:", per_task_ok)

    fail_reasons: dict[str, int] = {}
    for r in failed_records:
        key = r.get("_last_reason", "unknown")
        fail_reasons[key] = fail_reasons.get(key, 0) + 1
    if fail_reasons:
        print("Failure reasons (last attempt):", fail_reasons)

    write_jsonl(args.output, ok_records)
    print(f"\nFinal cumulative DPO data saved → {args.output}")
    print("Trial-specific files:")
    print(f"  - {get_trial_filename(args.output, 0)}")
    for i in range(1, args.retries + 1):
        print(f"  - {get_trial_filename(args.output, i)}")

    if args.failed_output and failed_records:
        write_jsonl(args.failed_output, failed_records)
        print(f"Wrote failures → {args.failed_output}")


if __name__ == "__main__":
    main()
