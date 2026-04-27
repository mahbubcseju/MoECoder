"""
Generate rejected samples for DPO training using vLLM (in-process).

Supports multiple task types via a `task_type` field on each row:
  - "input_pred":    given output, predict input  (CRUXEval-style ??)
  - "output_pred":   given input, predict output  (CRUXEval-style ??)
  - "main_solution": given input dict, predict return value of main_solution

Pipeline:
1. Read JSONL of chosen samples.
2. Build a perturbation prompt for each, apply Qwen's chat template.
3. Run vLLM batched generation over all prompts.
4. Parse JSON output, lenient-verify (rejected_answer != chosen_answer).
5. Write a JSONL in DPO format: {"prompt", "chosen", "rejected", "metadata"}.

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
5. Keep the same structure, tone, section headers, and formatting as the original.
6. Write with the same confidence as the original — do not hint that anything is wrong.
7. The final answer must follow EXACTLY the same format as the original answer \
(same prefix, same wrapper, same syntax — only the value(s) change).

ORIGINAL PROBLEM:
{user_prompt}

ORIGINAL CORRECT REASONING:
{reasoning_content}

ORIGINAL CORRECT ANSWER:
{answer}

Now produce the subtly incorrect version. Respond with ONLY a JSON object \
(no markdown fences, no commentary) with these exact keys:
{{
  "perturbation_description": "one sentence describing what single step you changed",
  "rejected_reasoning": "the full incorrect reasoning trace",
  "rejected_answer": "the resulting wrong answer in the same format as the original"
}}
"""


def build_user_message(sample: dict[str, Any]) -> str:
    return PERTURBATION_PROMPT.format(
        user_prompt=sample["user_prompt"],
        reasoning_content=sample["reasoning_content"],
        answer=sample["answer"],
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
# Lenient verification
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s.strip())


def verify_lenient(chosen_answer: str, rejected_answer: str) -> tuple[bool, str]:
    # Convert to string in case model returned numeric values
    rejected_answer = str(rejected_answer).strip() if rejected_answer is not None else ""
    chosen_answer = str(chosen_answer).strip() if chosen_answer is not None else ""
    
    if not rejected_answer:
        return False, "rejected_answer is empty"
    if _normalize(chosen_answer) == _normalize(rejected_answer):
        return False, "rejected_answer is identical to chosen_answer (modulo whitespace)"
    return True, "ok"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

VALID_TASK_TYPES = {"input_pred", "output_pred", "main_solution"}
REQUIRED_FIELDS = {"user_prompt", "reasoning_content", "answer", "task_type"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    samples = []
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
            samples.append(obj)
    return samples


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_trial_filename(output_path: Path, trial: int) -> Path:
    """Generate filename with trial suffix: output_trial_0.jsonl, output_trial_1.jsonl, etc."""
    stem = output_path.stem  # e.g., "dpo_pairs"
    suffix = output_path.suffix  # e.g., ".jsonl"
    parent = output_path.parent
    return parent / f"{stem}_trial_{trial}{suffix}"


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
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="Number of GPUs for tensor parallelism")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="Max context length (prompt + generation)")
    ap.add_argument("--dtype", default="auto",
                    help="auto | float16 | bfloat16")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="Max generation tokens per sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N samples (debugging)")
    ap.add_argument("--limit-per-type", type=int, default=None,
                    help="Limit examples per task type (e.g., --limit-per-type 100)")
    ap.add_argument("--retries", type=int, default=1,
                    help="Number of retry attempts for failed samples")
    args = ap.parse_args()

    # ---- 1. Load data and add idx ------------------------------------------------
    samples = read_jsonl(args.input)
    
    # Add idx if not present
    for i, sample in enumerate(samples):
        if "idx" not in sample:
            sample["idx"] = i
    
    if args.limit:
        samples = samples[:args.limit]
    elif args.limit_per_type:
        # Limit samples per task type
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

    # ---- 3. Build prompts with the chat template --------------------------
    prompts: list[str] = []
    for s in tqdm(samples, desc="Templating"):
        messages = [{"role": "user", "content": build_user_message(s)}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # ---- 4. Generate with initial seed ----------------------------------------
    print(f"Generating {len(prompts)} responses with vLLM...")
    outputs = llm.generate(prompts, sampling_params)

    # vLLM returns outputs in the same order as inputs, but we sort by request_id
    # defensively in case that ever changes.
    outputs = sorted(outputs, key=lambda o: int(o.request_id))

    # ---- 5. Parse + verify with retries ----------------------------------------
    ok_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    retry_samples: list[tuple[int, dict[str, Any]]] = []  # (sample_idx, sample)

    for sample_idx, (sample, out) in enumerate(tqdm(zip(samples, outputs, strict=True),
                                                      total=len(samples), desc="Parsing")):
        text = out.outputs[0].text
        parsed = parse_json_response(text)
        if parsed is None:
            retry_samples.append((sample_idx, sample))
            continue

        rejected_answer = parsed["rejected_answer"]
        chosen_answer = sample["answer"]

        ok, reason = verify_lenient(chosen_answer, rejected_answer)
        if not ok:
            retry_samples.append((sample_idx, sample))
            continue

        # Convert to strings for concatenation
        chosen_text = sample["reasoning_content"].rstrip() + "\n\n" + str(chosen_answer)
        rejected_text = parsed["rejected_reasoning"].rstrip() + "\n\n" + str(rejected_answer)

        ok_records.append({
            "prompt": sample["user_prompt"],
            "chosen": chosen_text,
            "rejected": rejected_text,
            "metadata": {
                "idx": sample.get("idx"),
                "task_type": sample.get("task_type"),
                "perturbation": parsed["perturbation_description"],
                "chosen_answer": chosen_answer,
                "rejected_answer": rejected_answer,
            },
        })

    # Write after initial pass with trial suffix
    trial_path = get_trial_filename(args.output, trial=0)
    write_jsonl(trial_path, ok_records)
    print(f"Wrote {len(ok_records)} records to {trial_path} (initial pass)")

    # ---- 6. Retry failed samples -------------------------------------------------
    for retry_attempt in range(1, args.retries + 1):
        if not retry_samples:
            break
        
        print(f"\nRetry attempt {retry_attempt}/{args.retries}: {len(retry_samples)} samples")
        
        # Build prompts for retry samples
        retry_prompts = []
        for _, sample in retry_samples:
            messages = [{"role": "user", "content": build_user_message(sample)}]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            retry_prompts.append(prompt_text)
        
        # Generate with different seed for this retry
        retry_seed = args.seed + retry_attempt
        retry_sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=retry_seed,
        )
        retry_outputs = llm.generate(retry_prompts, retry_sampling_params)
        retry_outputs = sorted(retry_outputs, key=lambda o: int(o.request_id))
        
        # Parse retry results
        new_retry_samples = []
        for (sample_idx, sample), out in zip(retry_samples, retry_outputs, strict=True):
            text = out.outputs[0].text
            parsed = parse_json_response(text)
            if parsed is None:
                new_retry_samples.append((sample_idx, sample))
                continue

            rejected_answer = parsed["rejected_answer"]
            chosen_answer = sample["answer"]

            ok, reason = verify_lenient(chosen_answer, rejected_answer)
            if not ok:
                new_retry_samples.append((sample_idx, sample))
                continue

            # Success on retry
            chosen_text = sample["reasoning_content"].rstrip() + "\n\n" + str(chosen_answer)
            rejected_text = parsed["rejected_reasoning"].rstrip() + "\n\n" + str(rejected_answer)

            ok_records.append({
                "prompt": sample["user_prompt"],
                "chosen": chosen_text,
                "rejected": rejected_text,
                "metadata": {
                    "idx": sample.get("idx"),
                    "task_type": sample.get("task_type"),
                    "perturbation": parsed["perturbation_description"],
                    "chosen_answer": chosen_answer,
                    "rejected_answer": rejected_answer,
                    "retry_attempt": retry_attempt,
                },
            })
        
        retry_samples = new_retry_samples
        
        # Write after each retry with trial suffix
        trial_path = get_trial_filename(args.output, trial=retry_attempt)
        write_jsonl(trial_path, ok_records)
        print(f"Wrote {len(ok_records)} records to {trial_path} (retry {retry_attempt})")

    # Add remaining failed samples to failed_records
    for sample_idx, sample in retry_samples:
        failed_records.append({
            "_status": "all_retries_exhausted",
            "_source": sample,
            "_idx": sample.get("idx"),
        })

    # ---- 7. Report + write ------------------------------------------------
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
        key = r.get("_status", "unknown")
        if key == "verification_failed":
            key = f"verification_failed:{r.get('_reason', '?')}"
        fail_reasons[key] = fail_reasons.get(key, 0) + 1
    if fail_reasons:
        print("Failure reasons:", fail_reasons)

    # Write final cumulative file
    write_jsonl(args.output, ok_records)
    print(f"\nFinal cumulative DPO data saved → {args.output}")
    print(f"Trial-specific files:")
    print(f"  - {get_trial_filename(args.output, 0)}")
    for i in range(1, args.retries + 1):
        print(f"  - {get_trial_filename(args.output, i)}")

    if args.failed_output and failed_records:
        write_jsonl(args.failed_output, failed_records)
        print(f"Wrote failures → {args.failed_output}")


if __name__ == "__main__":
    main()
