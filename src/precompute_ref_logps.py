import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm

try:
    from model import create_model_and_tokenizer
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import create_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pre-compute reference model log probs for DPO training. "
            "Run this once on the SFT checkpoint before launching train.py --do_dpo."
        )
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Path to the SFT checkpoint (used as the frozen reference model).")
    parser.add_argument("--dpo_data_path", type=str, required=True,
                        help="JSONL file with prompt / chosen / rejected fields.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to write the augmented JSONL (adds ref_chosen_logp / ref_rejected_logp).")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Must match the --max_length used during DPO training.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    # Forwarded to create_model_and_tokenizer; leave empty to let the checkpoint
    # restore MoE config from its own moe_config.json metadata.
    parser.add_argument("--moe_layer_indices", type=int, nargs="*", default=[])
    parser.add_argument("--num_experts_temp", type=int, default=4)
    parser.add_argument("--moe_top_k", type=int, default=1)
    parser.add_argument("--router_aux_loss_weight", type=float, default=0.01)
    return parser.parse_args()


def render_chat_text(tokenizer, messages, add_generation_prompt, think=False):
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=think, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def tokenize_dpo_pair(example, tokenizer, max_length):
    """Tokenize one chosen/rejected pair and return labels with prompt positions masked."""
    prompt_text = render_chat_text(
        tokenizer,
        [{"role": "user", "content": example["prompt"]}],
        add_generation_prompt=True,
    )
    chosen_text = render_chat_text(
        tokenizer,
        [
            {"role": "user", "content": example["prompt"]},
            {
                "role": "assistant",
                "reasoning_content": example["chosen"]["reasoning_content"],
                "content": example["chosen"]["content"],
            },
        ],
        think=True,
        add_generation_prompt=False,
    )
    rejected_text = render_chat_text(
        tokenizer,
        [
            {"role": "user", "content": example["prompt"]},
            {
                "role": "assistant",
                "reasoning_content": example["rejected"]["reasoning_content"],
                "content": example["rejected"]["content"],
            },
        ],
        think=True,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    chosen_enc = tokenizer(
        chosen_text, truncation=True, max_length=max_length, add_special_tokens=False
    )
    rejected_enc = tokenizer(
        rejected_text, truncation=True, max_length=max_length, add_special_tokens=False
    )

    prompt_len_c = min(len(prompt_ids), len(chosen_enc["input_ids"]))
    prompt_len_r = min(len(prompt_ids), len(rejected_enc["input_ids"]))

    chosen_labels = chosen_enc["input_ids"].copy()
    chosen_labels[:prompt_len_c] = [-100] * prompt_len_c

    rejected_labels = rejected_enc["input_ids"].copy()
    rejected_labels[:prompt_len_r] = [-100] * prompt_len_r

    return (
        chosen_enc["input_ids"], chosen_enc["attention_mask"], chosen_labels,
        rejected_enc["input_ids"], rejected_enc["attention_mask"], rejected_labels,
    )


def pad_and_tensorize(sequences, pad_value, device):
    max_len = max(len(s) for s in sequences)
    padded = [s + [pad_value] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long, device=device)


@torch.no_grad()
def compute_sequence_log_probs(logits, labels):
    """
    logits: [B, T, V], labels: [B, T] with -100 for masked positions.
    Returns [B] — sum of log probs over non-masked response tokens.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)
    mask = (shift_labels != -100).float()
    return (token_log_probs * mask).sum(dim=-1)


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading reference model from: {args.model_name}")
    model, tokenizer = create_model_and_tokenizer(args)
    model = model.to(device)
    model.eval()

    print(f"Loading DPO data from: {args.dpo_data_path}")
    dataset = Dataset.from_json(args.dpo_data_path)
    print(f"  {len(dataset)} examples loaded.")

    pad_id = tokenizer.pad_token_id
    all_ref_chosen_logps: list[float] = []
    all_ref_rejected_logps: list[float] = []

    for start in tqdm(range(0, len(dataset), args.batch_size), desc="Pre-computing ref log probs"):
        batch_rows = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]

        tokenized = [tokenize_dpo_pair(row, tokenizer, args.max_length) for row in batch_rows]
        chosen_ids, chosen_mask, chosen_labels = zip(*[t[:3] for t in tokenized])
        rejected_ids, rejected_mask, rejected_labels = zip(*[t[3:] for t in tokenized])

        c_ids  = pad_and_tensorize(chosen_ids,    pad_id, device)
        c_mask = pad_and_tensorize(chosen_mask,   0,      device)
        c_labs = pad_and_tensorize(chosen_labels, -100,   device)
        r_ids  = pad_and_tensorize(rejected_ids,    pad_id, device)
        r_mask = pad_and_tensorize(rejected_mask,   0,      device)
        r_labs = pad_and_tensorize(rejected_labels, -100,   device)

        with torch.no_grad():
            chosen_logps   = compute_sequence_log_probs(model(input_ids=c_ids, attention_mask=c_mask).logits,   c_labs)
            rejected_logps = compute_sequence_log_probs(model(input_ids=r_ids, attention_mask=r_mask).logits, r_labs)

        all_ref_chosen_logps.extend(chosen_logps.cpu().tolist())
        all_ref_rejected_logps.extend(rejected_logps.cpu().tolist())

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(dataset):
            record = dict(row)
            record["ref_chosen_logp"]   = all_ref_chosen_logps[i]
            record["ref_rejected_logp"] = all_ref_rejected_logps[i]
            f.write(json.dumps(record) + "\n")

    print(f"Saved augmented dataset ({len(dataset)} rows) → {output_path}")


if __name__ == "__main__":
    main()
