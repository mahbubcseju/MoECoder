import argparse
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, Features, Value
from torch.utils.data import DataLoader
from transformers import get_scheduler

from model import create_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Custom multi-GPU LLM training with Accelerate + DeepSpeed")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--min_learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument(
        "--moe_layer_indices",
        type=int,
        nargs="*",
        default=[],
        help="Transformer block indices whose MLP will be replaced by MoE.",
    )
    parser.add_argument("--num_experts_temp", type=int, default=4)
    parser.add_argument("--moe_top_k", type=int, default=1)
    parser.add_argument("--router_aux_loss_weight", type=float, default=0.01)
    parser.add_argument(
        "--freeze_non_moe",
        action="store_true",
        help="Freeze all non-MoE parameters and train only MoE params (and optional output head).",
    )
    parser.add_argument(
        "--train_output_head",
        action="store_true",
        help="When --freeze_non_moe is set, also train output/classification head if present.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation to use. Options depend on the model and hardware, but may include 'flash_attention_2', 'triton', 'auto', etc.",
    )
    parser.add_argument(
        "--reasoning_weight",
        type=float,
        default=1.0,
        help="Loss weight for reasoning_content tokens. Default=1.0",
    )
    parser.add_argument(
        "--content_weight",
        type=float,
        default=1.0,
        help="Loss weight for content tokens. Default=1.0. Use higher values to emphasize content over reasoning.",
    )
    # DPO arguments
    parser.add_argument("--do_dpo", action="store_true",
                        help="Run DPO training after (or instead of) SFT.")
    parser.add_argument("--dpo_data_path", type=str, default="",
                        help="Path to the augmented DPO JSONL produced by precompute_ref_logps.py.")
    parser.add_argument("--dpo_beta", type=float, default=0.1,
                        help="DPO temperature β. Higher = tighter adherence to the reference policy.")
    parser.add_argument("--dpo_num_epochs", type=int, default=1)
    parser.add_argument("--dpo_learning_rate", type=float, default=1e-6)
    parser.add_argument("--dpo_output_dir", type=str, default="",
                        help="Where to save the DPO checkpoint. Defaults to <output_dir>/dpo.")
    parser.add_argument("--dpo_sample_size", type=int, default=0,
                        help="Use only the first N DPO examples. 0 = use all data (default). Useful for quick smoke tests.")
    return parser.parse_args()


def tokenize_fn(examples, tokenizer, max_length):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


def normalize_run_flags(args):
    if not args.do_train and not args.do_eval and not args.do_dpo:
        args.do_train = True
    if args.moe_layer_indices and args.num_experts_temp < 2:
        raise ValueError("--num_experts_temp must be >= 2 when using --moe_layer_indices.")
    if args.moe_top_k < 1 or args.moe_top_k > args.num_experts_temp:
        raise ValueError("--moe_top_k must be between 1 and --num_experts_temp.")
    return args


def render_chat_text(tokenizer, messages, add_generation_prompt, think=False):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=think, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)



concepts = [
    'for_statement',
    'while_statement',
    'comparison_operator',
    'boolean_operator',
    'not_operator',
    'call',
    'assignment',
    'list_comprehension',
    'augmented_assignment',
    'unary_operator',
    'binary_operator',
    'subscript',
    'pattern_list',
    'string',
]
concept_mapping = {concept: idx for idx, concept in enumerate(concepts)}

import numpy as np
trainer_flag_1 = False
def build_concept_matrix_tokencentric(example, input_ids,tokens, offsets, concept_mapping, start_code_token_id=None):
    """
    Builds (seq_len, num_concepts) matrix:
      - if token overlaps >=1 concept span => one-hot (or multi-hot if overlaps multiple)
      - if token overlaps none => all -1

    Requires concept spans in CHAR OFFSETS:
      example["concept"][name] = [(start_char, end_char), ...]
      or [[start_char, end_char], ...]
    """
    seq_len = len(input_ids)
    num_concepts = len(concept_mapping)

    # Default: tokens not covered by any concept => all -1
    mat = np.full((seq_len, num_concepts), -1, dtype=np.int8)

    # 1) Flatten all concept intervals
    intervals = []
    concept_dict = example.get("concepts", {})
    # print("concept_dict:", concept_dict)
    for cname, spans in concept_dict.items():
        if cname not in concept_mapping:
            continue
        c = concept_mapping[cname]
        for s in spans:
            # print(s)
            if not (isinstance(s, (list, tuple)) and len(s) == 2):
                continue
            a, b = int(s[0]), int(s[1])
            if a < b:
                intervals.append((a + start_code_token_id, b + start_code_token_id, c))

    # If no spans, return all -1
    if not intervals:
        return mat.tolist()

    # 2) Sort intervals by start
    intervals.sort(key=lambda x: x[0])
    # print(len(intervals))
    j = 0
    active = []  # list of (end_char, concept_id)

    global trainer_flag_1
    # 3) Single pass over tokens
    for tok_idx, (tok_s, tok_e) in enumerate(offsets):
        if tok_s == tok_e:
            continue  # skip empty offsets

        # Add all intervals that start before token ends
        while j < len(intervals) and intervals[j][0] < tok_e:
            _, end_c, c = intervals[j]
            active.append((end_c, c))
            j += 1

        # Remove intervals that ended before or at token start
        if active:
            active = [(end_c, c) for (end_c, c) in active if end_c > tok_s]

        # Determine which concepts overlap this token:
        # (Since we only keep end>tok_s and added with start<tok_e, overlap holds)
        # print("active:", len(active))
        if active:
            # switch from -1s to 0s (one-hot base)
            mat[tok_idx, :] = 0
            # If you truly want strictly one-hot and a token can belong to only ONE concept,
            # replace this loop by choosing one concept (e.g., first).
            for _, c in active:
                mat[tok_idx, c] = 1
            
            if not trainer_flag_1:
                print(tokens[tok_idx])
    trainer_flag_1 = True
        
    return mat.tolist()


trainer_flag = False
full_text_should_be_printed = True

def compute_loss_weights(full_text, offsets, prompt_len, reasoning_weight=1.0, content_weight=1.0):
    """
    Compute loss weights for each token to differentiate between reasoning and content.
    Returns: (seq_len,) array where reasoning tokens get reasoning_weight and content gets content_weight
    """
    seq_len = len(offsets)
    weights = [0.0] * seq_len
    
    # Tokens before prompt_len get 0 weight (ignored in loss)
    for i in range(prompt_len):
        weights[i] = 0.0
    
    # For remaining tokens, we need to find where reasoning ends and content begins
    # This requires parsing the full_text to identify these sections
    # For now, we'll use a heuristic: if reasoning markers exist in text, identify them
    
    # Look for <think> or similar markers that might separate reasoning from content
    think_end = full_text.find("</think>") if "</think>" in full_text else -1
    
    if think_end != -1:
        # Tokens within reasoning section get reasoning_weight
        for i in range(prompt_len, seq_len):
            token_start, token_end = offsets[i]
            if token_end <= think_end:
                weights[i] = reasoning_weight
            else:
                weights[i] = content_weight
    else:
        # If no reasoning markers, apply content_weight to all assistant tokens
        for i in range(prompt_len, seq_len):
            weights[i] = content_weight
    
    return weights


def build_tokenized_example(example, tokenizer, max_length, reasoning_weight=1.0, content_weight=1.0):
    full_text = example["text"]
    prompt_text = render_chat_text(
        tokenizer,
        [{"role": "user", "content": example["user_prompt"]}],
        add_generation_prompt=True,
    )
    global full_text_should_be_printed
    if full_text_should_be_printed:
        print("*" * 100)
        print("full_text:", full_text)
        print("*"* 100)
        # print("refcode:", example['refcode'])
        full_text_should_be_printed = False

    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    offsets = encoded["offset_mapping"]

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = min(len(prompt_ids), len(input_ids))

    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

    assistant_mask = [0] * len(input_ids)
    for i in range(prompt_len, len(input_ids)):
        assistant_mask[i] = 1
    
    # Compute loss weights for reasoning vs content
    loss_weights = compute_loss_weights(full_text, offsets, prompt_len, reasoning_weight, content_weight)

    # print("full_text:", full_text)
    # print("refcode:", example['refcode'])
    code_start = full_text.index(example['refcode'])
    code_end = code_start + len(example['refcode'])
    global trainer_flag
    if not trainer_flag:
        print(full_text[code_start:code_end])
        # print("Example refcode:", example['refcode'])
        # print("Code span in text:", (code_start, code_end))
        # print("Token offsets:", offsets)
        trainer_flag = True

    concept_mat = build_concept_matrix_tokencentric(
        example=example,
        input_ids=input_ids,
        tokens =tokenizer.convert_ids_to_tokens(input_ids),
        offsets=offsets,
        concept_mapping=concept_mapping,
        start_code_token_id=code_start,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "assistant_mask": assistant_mask,
        "concept_mat": concept_mat,
        "loss_weights": loss_weights,
        # "code_mask": code_mask,
    }


def build_supervised_collator(tokenizer):
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for batching.")

    pad_token_id = tokenizer.pad_token_id

    def collate_fn(features):
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "assistant_mask": [],
            "concept_mat": [],
            "loss_weights": [],
            # "code_mask": [],
        }

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
            batch["assistant_mask"].append(feature["assistant_mask"] + [0] * pad_len)
            batch["concept_mat"].append(feature["concept_mat"] + [[-1] * len(concept_mapping)] * pad_len)
            batch["loss_weights"].append(feature["loss_weights"] + [0.0] * pad_len)
            # batch["code_mask"].append(feature["code_mask"] + [0] * pad_len)

        batch_out = {"input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
                     "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
                     "labels": torch.tensor(batch["labels"], dtype=torch.long),
                     "assistant_mask": torch.tensor(batch["assistant_mask"], dtype=torch.long),
                     "concept_mat": torch.tensor(batch["concept_mat"], dtype=torch.long),
                     "loss_weights": torch.tensor(batch["loss_weights"], dtype=torch.float)}
        return batch_out

    return collate_fn


def split_batch_for_model(batch):
    aux = {
        "assistant_mask": batch.pop("assistant_mask", None),
        "loss_weights": batch.pop("loss_weights", None),
        "concept_mat": batch.pop("concept_mat", None),
    }
    return batch, aux


def load_tokenized_dataset(args, tokenizer):
    # dataset_path = Path(__file__).resolve().parents[1] / "data" / "ex_tr_data" / "final_traced_dataset_w_concepts.jsonl"
    dataset = Dataset.from_json(str(args.data_path))

    messages_list = []
    for row in dataset:
        messages_list.append(
            [
                {"role": "user", "content": row["user_prompt"]},
                {"role": "assistant",  "reasoning_content": row['reasoning_content'], "content": row['content']},
            ]
        )
        # if len(messages_list) >= 256:
        #     break

    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer '{args.model_name}' does not provide a chat template. "
            "Use a chat-model tokenizer (for example Qwen) or define a custom formatter."
        )

    inputs = render_chat_text(tokenizer, messages_list, think=True, add_generation_prompt=False)
    # print(inputs[0])

    org_count = len(inputs)
    # code_pattern = re.compile(r"```.*?```", flags=re.DOTALL)

    filtered_user_prompts = []
    filtered_messages_list = []
    filtered_texts = []
    filtered_refcodes = []
    filtered_concepts = []
    for row, messages, text in zip(dataset, messages_list, inputs):
        if len(text) <= args.max_length:

            filtered_user_prompts.append(row["user_prompt"])
            filtered_messages_list.append(messages)
            filtered_refcodes.append(row["refcode"])
            filtered_concepts.append(row["concepts"])
            filtered_texts.append(text)
            # print("full_text:", text)
            # print("refcode:", row["refcode"])
    messages_list = filtered_messages_list
    print(f"Loaded {org_count} rows. Kept {len(messages_list)} rows after chat-length filter.")

    print(len(filtered_user_prompts), len(filtered_texts), len(filtered_refcodes), len(filtered_concepts))
    formatted_dataset = Dataset.from_dict(
        {
            "user_prompt": filtered_user_prompts,
            "text": filtered_texts,
            "refcode": filtered_refcodes,
            "concepts": filtered_concepts,
        }
    )
    split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    split_dataset = DatasetDict({"train": split_dataset["train"], "validation": split_dataset["test"]})
    # split = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    # split_dataset = DatasetDict({
    #     "train": split["train"].select(range(min(64, len(split["train"])))),
    #     "validation": split["test"].select(range(min(64, len(split["test"])))),
    # })

    # Get weights from args (with defaults)
    reasoning_weight = getattr(args, 'reasoning_weight', 1.0)
    content_weight = getattr(args, 'content_weight', 1.0)
    
    tokenized = split_dataset.map(
        lambda example: build_tokenized_example(
            example,
            tokenizer=tokenizer,
            max_length=args.max_length,
            reasoning_weight=reasoning_weight,
            content_weight=content_weight,
        ),
        remove_columns=["user_prompt", "text", "refcode", "concepts"],
    )
    return tokenized.filter(lambda x: len(x["input_ids"]) > 0 and any(label != -100 for label in x["labels"]))


def configure_trainable_parameters(model, args):
    # if not args.freeze_non_moe:
    #     return
    
    def set_requires_grad(module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag


    set_requires_grad(model, True)

    # 2) Unfreeze MoE layers 34-35: router + experts + norms
    # for i in args.moe_layer_indices:
    #     layer = model.model.layers[i]

    #     # MoE parts
    #     set_requires_grad(layer.mlp.router, True)
    #     set_requires_grad(layer.mlp.experts, True)

    #     # Norms in those layers
    #     set_requires_grad(layer.input_layernorm, True)
    #     set_requires_grad(layer.post_attention_layernorm, True)

    # # 3) Final norm
    # set_requires_grad(model.model.norm, True)
    # for param in model.parameters():
    #     param.requires_grad = False

    # for moe_layer in getattr(model, "moe_layers", []):
    #     for param in moe_layer.parameters():
    #         param.requires_grad = True

    # if args.train_output_head:
    #     head_names = ("lm_head",)
    #     for head_name in head_names:
    #         if hasattr(model.base_model, head_name):
    #             head_module = getattr(model.base_model, head_name)
    #             for param in head_module.parameters():
    #                 param.requires_grad = True
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("✅", name)
        else:
            print("❌", name)


def build_train_components(args, model, tokenized, collator):
    if "train" not in tokenized:
        raise ValueError("Training requested but dataset has no 'train' split.")

    train_loader = DataLoader(
        tokenized["train"],
        shuffle=True,
        batch_size=args.per_device_batch_size,
        collate_fn=collator,
    )
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    # print(trainable_params)
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check freezing/MoE configuration.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.0,)
    update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * update_steps_per_epoch
    num_warmup_steps = int(0.03 * max_train_steps)
    # lr_scheduler = get_scheduler(
    #     "cosine",
    #     optimizer=optimizer,
    #     num_warmup_steps=args.num_warmup_steps,
    #     num_training_steps=max_train_steps,
    # )
    # lr_scheduler = get_scheduler(
    #     "cosine_with_min_lr",
    #     optimizer=optimizer,
    #     num_warmup_steps=args.num_warmup_steps,
    #     num_training_steps=max_train_steps,
    #     scheduler_specific_kwargs={
    #         "min_lr": args.min_learning_rate,
    #     },
    # )
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    return train_loader, optimizer, lr_scheduler


def get_eval_split_name(dataset):
    if "validation" in dataset:
        return "validation"
    if "test" in dataset:
        return "test"
    raise ValueError("No evaluation split found. Dataset must contain 'validation' or 'test'.")


def build_eval_loader(args, tokenized, collator):
    eval_split = get_eval_split_name(tokenized)
    return DataLoader(
        tokenized[eval_split],
        shuffle=False,
        batch_size=args.per_device_batch_size,
        collate_fn=collator,
    )


def build_mode_components(args, model, tokenized, collator):
    train_loader, optimizer, lr_scheduler = (None, None, None)
    eval_loader = None

    if args.do_train:
        train_loader, optimizer, lr_scheduler = build_train_components(args, model, tokenized, collator)
    if args.do_eval:
        eval_loader = build_eval_loader(args, tokenized, collator)

    return train_loader, optimizer, lr_scheduler, eval_loader


def prepare_distributed_components(
    accelerator, model, train_loader=None, optimizer=None, lr_scheduler=None, eval_loader=None
):
    to_prepare = [model]
    has_train = all(item is not None for item in (train_loader, optimizer, lr_scheduler))
    has_eval = eval_loader is not None

    if has_train:
        to_prepare.extend([optimizer, train_loader, lr_scheduler])
    if has_eval:
        to_prepare.append(eval_loader)

    prepared = accelerator.prepare(*to_prepare)
    if not isinstance(prepared, tuple):
        prepared = (prepared,)

    idx = 0
    model = prepared[idx]
    idx += 1

    if has_train:
        optimizer = prepared[idx]
        idx += 1
        train_loader = prepared[idx]
        idx += 1
        lr_scheduler = prepared[idx]
        idx += 1

    if has_eval:
        eval_loader = prepared[idx]

    return model, train_loader, optimizer, lr_scheduler, eval_loader


def evaluate(model, eval_loader, accelerator):
    was_training = model.training
    model.eval()
    losses = []
    for batch in eval_loader:
        batch, _ = split_batch_for_model(batch)
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss.detach()
        losses.append(accelerator.gather_for_metrics(loss.unsqueeze(0)))

    if not losses:
        raise ValueError("Evaluation dataloader is empty after preprocessing.")

    eval_loss = torch.cat(losses).mean().item()
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    if accelerator.is_main_process:
        print(f"eval_loss={eval_loss:.4f} perplexity={perplexity:.4f}")

    if was_training:
        model.train()
    return eval_loss, perplexity


def train(model, train_loader, optimizer, lr_scheduler, accelerator, args, eval_loader=None):
    model.train()
    global_step = 0

    if accelerator.is_main_process:
        print(len(train_loader))
        print("********************")
    for epoch in range(args.num_epochs):
        for batch in train_loader:
            batch, aux = split_batch_for_model(batch)
            with accelerator.accumulate(model):
                outputs = model(
                    **batch,
                    loss_weights=aux["loss_weights"],
                    concept_mat=aux["concept_mat"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                    print(f"epoch={epoch} step={global_step} loss={loss.item():.4f}")
                    # break
        # break

        # if eval_loader is not None:
        #     if accelerator.is_main_process:
        #         print(f"running evaluation at end of epoch {epoch}")
        #     evaluate(model, eval_loader, accelerator)

def save_checkpoint(model, tokenizer, accelerator, output_dir):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=True,
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        tokenizer.save_pretrained(output_dir)


# ============================================================
# DPO data pipeline
# ============================================================

def _tokenize_dpo_pair(example, tokenizer, max_length):
    """Tokenize one chosen/rejected pair. Returns a 6-tuple of lists (no ref log probs)."""
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
                "content": f"\n\n[ANSWER]\n{example['chosen']['content']}\n[/ANSWER]",
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
                "content": f"\n\n[ANSWER]\n{example['rejected']['content']}\n[/ANSWER]",
            },
        ],
        think=True,
        add_generation_prompt=False,
    )

    prompt_ids   = tokenizer(prompt_text,   add_special_tokens=False)["input_ids"]
    chosen_enc   = tokenizer(chosen_text,   truncation=True, max_length=max_length, add_special_tokens=False)
    rejected_enc = tokenizer(rejected_text, truncation=True, max_length=max_length, add_special_tokens=False)

    prompt_len_c = min(len(prompt_ids), len(chosen_enc["input_ids"]))
    prompt_len_r = min(len(prompt_ids), len(rejected_enc["input_ids"]))

    chosen_labels   = chosen_enc["input_ids"].copy()
    chosen_labels[:prompt_len_c] = [-100] * prompt_len_c

    rejected_labels = rejected_enc["input_ids"].copy()
    rejected_labels[:prompt_len_r] = [-100] * prompt_len_r

    return (
        chosen_enc["input_ids"],   chosen_enc["attention_mask"],   chosen_labels,
        rejected_enc["input_ids"], rejected_enc["attention_mask"], rejected_labels,
    )


def build_dpo_tokenized_example(example, tokenizer, max_length):
    """Tokenize one DPO example. Prompt positions are masked (-100) in labels."""
    c_ids, c_mask, c_labs, r_ids, r_mask, r_labs = _tokenize_dpo_pair(example, tokenizer, max_length)
    return {
        "chosen_input_ids":        c_ids,
        "chosen_attention_mask":   c_mask,
        "chosen_labels":           c_labs,
        "rejected_input_ids":      r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_labels":         r_labs,
        "ref_chosen_logp":         example["ref_chosen_logp"],
        "ref_rejected_logp":       example["ref_rejected_logp"],
    }


def precompute_dpo_ref_logps_inline(raw_dataset, model, tokenizer, args, accelerator):
    """
    Compute ref log probs from the current (SFT) model weights in a single no-grad pass.
    Used when --do_train and --do_dpo are combined, so no separate precompute script is needed.
    Returns an augmented Dataset with ref_chosen_logp and ref_rejected_logp added.
    """
    pad_id    = tokenizer.pad_token_id
    ref_model = accelerator.unwrap_model(model)
    device    = next(ref_model.parameters()).device

    was_training = ref_model.training
    ref_model.eval()

    all_chosen_logps, all_rejected_logps = [], []

    def _pad(seqs, val):
        ml = max(len(s) for s in seqs)
        return torch.tensor([s + [val] * (ml - len(s)) for s in seqs], dtype=torch.long, device=device)

    for start in range(0, len(raw_dataset), args.per_device_batch_size):
        rows = [raw_dataset[i] for i in range(start, min(start + args.per_device_batch_size, len(raw_dataset)))]
        tok  = [_tokenize_dpo_pair(r, tokenizer, args.max_length) for r in rows]

        c_ids  = [t[0] for t in tok];  c_mask = [t[1] for t in tok];  c_labs = [t[2] for t in tok]
        r_ids  = [t[3] for t in tok];  r_mask = [t[4] for t in tok];  r_labs = [t[5] for t in tok]

        with torch.no_grad():
            chosen_logps = compute_sequence_log_probs(
                ref_model(input_ids=_pad(c_ids, pad_id), attention_mask=_pad(c_mask, 0), skip_moe_losses=True).logits,
                _pad(c_labs, -100),
            )
            rejected_logps = compute_sequence_log_probs(
                ref_model(input_ids=_pad(r_ids, pad_id), attention_mask=_pad(r_mask, 0), skip_moe_losses=True).logits,
                _pad(r_labs, -100),
            )

        all_chosen_logps.extend(chosen_logps.cpu().tolist())
        all_rejected_logps.extend(rejected_logps.cpu().tolist())

    if was_training:
        ref_model.train()

    augmented = []
    for i, row in enumerate(raw_dataset):
        record = dict(row)
        record["ref_chosen_logp"]   = all_chosen_logps[i]
        record["ref_rejected_logp"] = all_rejected_logps[i]
        augmented.append(record)

    return Dataset.from_list(augmented)


def _load_dpo_json(path):
    import json
    needed = {"reasoning_content", "content"}
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            clean = {
                "prompt": row["prompt"],
                "chosen": {k: row["chosen"][k] for k in needed},
                "rejected": {k: row["rejected"][k] for k in needed},
            }
            if "ref_chosen_logp" in row:
                clean["ref_chosen_logp"] = row["ref_chosen_logp"]
            if "ref_rejected_logp" in row:
                clean["ref_rejected_logp"] = row["ref_rejected_logp"]
            rows.append(clean)
    return Dataset.from_list(rows)


def load_dpo_dataset(args, tokenizer, raw_dataset=None):
    if raw_dataset is None:
        raw_dataset = _load_dpo_json(str(args.dpo_data_path))
    print(f"Loaded {len(raw_dataset)} DPO examples from {args.dpo_data_path}")

    tokenized = raw_dataset.map(
        lambda ex: build_dpo_tokenized_example(ex, tokenizer, args.max_length),
        remove_columns=raw_dataset.column_names,
    )
    tokenized = tokenized.filter(
        lambda x: (
            any(l != -100 for l in x["chosen_labels"])
            and any(l != -100 for l in x["rejected_labels"])
        )
    )
    split = tokenized.train_test_split(test_size=0.1, seed=42)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def build_dpo_collator(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate_fn(features):
        max_c = max(len(f["chosen_input_ids"])   for f in features)
        max_r = max(len(f["rejected_input_ids"]) for f in features)
        batch = {k: [] for k in (
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
            "ref_chosen_logp", "ref_rejected_logp",
        )}
        for f in features:
            cp = max_c - len(f["chosen_input_ids"])
            batch["chosen_input_ids"].append(f["chosen_input_ids"]       + [pad_id] * cp)
            batch["chosen_attention_mask"].append(f["chosen_attention_mask"] + [0]      * cp)
            batch["chosen_labels"].append(f["chosen_labels"]             + [-100]  * cp)

            rp = max_r - len(f["rejected_input_ids"])
            batch["rejected_input_ids"].append(f["rejected_input_ids"]       + [pad_id] * rp)
            batch["rejected_attention_mask"].append(f["rejected_attention_mask"] + [0]      * rp)
            batch["rejected_labels"].append(f["rejected_labels"]             + [-100]  * rp)

            batch["ref_chosen_logp"].append(f["ref_chosen_logp"])
            batch["ref_rejected_logp"].append(f["ref_rejected_logp"])

        return {
            "chosen_input_ids":        torch.tensor(batch["chosen_input_ids"],        dtype=torch.long),
            "chosen_attention_mask":   torch.tensor(batch["chosen_attention_mask"],   dtype=torch.long),
            "chosen_labels":           torch.tensor(batch["chosen_labels"],           dtype=torch.long),
            "rejected_input_ids":      torch.tensor(batch["rejected_input_ids"],      dtype=torch.long),
            "rejected_attention_mask": torch.tensor(batch["rejected_attention_mask"], dtype=torch.long),
            "rejected_labels":         torch.tensor(batch["rejected_labels"],         dtype=torch.long),
            "ref_chosen_logp":         torch.tensor(batch["ref_chosen_logp"],         dtype=torch.float),
            "ref_rejected_logp":       torch.tensor(batch["ref_rejected_logp"],       dtype=torch.float),
        }

    return collate_fn


# ============================================================
# DPO loss and training
# ============================================================

def compute_sequence_log_probs(logits, labels):
    """Returns [B] — sum of log probs over non-masked response tokens."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)
    mask = (shift_labels != -100).float()
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    chosen_rewards   = beta * (policy_chosen_logps   - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    losses   = -F.logsigmoid(chosen_rewards - rejected_rewards)
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    return losses.mean(), chosen_rewards.detach().mean(), rejected_rewards.detach().mean(), accuracy


def build_dpo_train_components(args, model, dpo_tokenized, collator):
    dpo_train_loader = DataLoader(
        dpo_tokenized["train"],
        shuffle=True,
        batch_size=args.per_device_batch_size,
        collate_fn=collator,
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.dpo_learning_rate, weight_decay=0.0)
    update_steps_per_epoch = math.ceil(len(dpo_train_loader) / args.gradient_accumulation_steps)
    max_train_steps = args.dpo_num_epochs * update_steps_per_epoch
    num_warmup = int(0.03 * max_train_steps)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=max_train_steps,
    )
    return dpo_train_loader, optimizer, lr_scheduler


def train_dpo(model, dpo_train_loader, optimizer, lr_scheduler, accelerator, args):
    model.train()
    global_step = 0

    if accelerator.is_main_process:
        print(f"DPO: {args.dpo_num_epochs} epoch(s), {len(dpo_train_loader)} steps/epoch, β={args.dpo_beta}")

    for epoch in range(args.dpo_num_epochs):
        for batch in dpo_train_loader:
            with accelerator.accumulate(model):
                chosen_out = model(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"],
                    skip_moe_losses=True,
                )
                policy_chosen_logps = compute_sequence_log_probs(
                    chosen_out.logits, batch["chosen_labels"]
                )

                rejected_out = model(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"],
                    skip_moe_losses=True,
                )
                policy_rejected_logps = compute_sequence_log_probs(
                    rejected_out.logits, batch["rejected_labels"]
                )

                loss, chosen_rew, rejected_rew, acc = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    batch["ref_chosen_logp"],
                    batch["ref_rejected_logp"],
                    args.dpo_beta,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                    print(
                        f"[DPO] epoch={epoch} step={global_step} "
                        f"loss={loss.item():.4f} "
                        f"chosen_rew={chosen_rew.item():.4f} "
                        f"rejected_rew={rejected_rew.item():.4f} "
                        f"acc={acc.item():.4f}"
                    )


def run_requested_tasks(args, model, tokenizer, accelerator, train_loader, optimizer, lr_scheduler, eval_loader):
    if args.do_train:
        train(model, train_loader, optimizer, lr_scheduler, accelerator, args, eval_loader=eval_loader)
        save_checkpoint(model, tokenizer, accelerator, args.output_dir)
        return
    if args.do_eval:
        evaluate(model, eval_loader, accelerator)


def main():
    args = normalize_run_flags(parse_args())
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    # DeepSpeed strips train_micro_batch_size_per_gpu from the YAML and may not
    # re-inject it into deepspeed_config, causing _prepare_deepspeed to raise when
    # no dataloader is passed (e.g. DPO-only mode). Set it explicitly here.
    if (hasattr(accelerator.state, "deepspeed_plugin")
            and accelerator.state.deepspeed_plugin is not None):
        ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
        if not isinstance(ds_cfg.get("train_micro_batch_size_per_gpu"), int):
            ds_cfg["train_micro_batch_size_per_gpu"] = args.per_device_batch_size

    model, tokenizer = create_model_and_tokenizer(args)

    configure_trainable_parameters(model, args)

    # print(model)
    if accelerator.is_main_process:
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in model.parameters())
        print(
            f"Trainable parameters: {trainable_params:,}/{total_params:,} "
            f"({100.0 * trainable_params / total_params:.2f}%)"
        )
        if model.converted_layer_indices:
            print(
                "Converted MLP->MoE layers: "
                f"{model.converted_layer_indices} | num_experts_temp={model.num_experts_temp} | top_k={model.moe_top_k}"
            )
        else:
            print("No MoE conversion requested. Using the original dense model.")
    # === SFT data & components ===
    train_loader = optimizer = lr_scheduler = eval_loader = None
    if args.do_train or args.do_eval:
        tokenized = load_tokenized_dataset(args, tokenizer)
        collator = build_supervised_collator(tokenizer)
        print("before build mode components")
        train_loader, optimizer, lr_scheduler, eval_loader = build_mode_components(
            args, model, tokenized, collator
        )

    # === DPO data & components (built BEFORE prepare so DeepSpeed sees the optimizer) ===
    dpo_train_loader = dpo_optimizer = dpo_scheduler = None
    dpo_output_dir = None
    if args.do_dpo:
        if args.do_train:
            raise ValueError(
                "Combined --do_train --do_dpo is not supported with DeepSpeed "
                "(cannot re-initialize the engine mid-script). "
                "Run SFT first, then run --do_dpo as a separate job."
            )
        if not args.dpo_data_path:
            raise ValueError("--dpo_data_path is required when --do_dpo is set.")
        dpo_output_dir = args.dpo_output_dir or str(Path(args.output_dir) / "dpo")

        if accelerator.is_main_process:
            print(f"\n{'='*60}\nPreparing DPO  →  {dpo_output_dir}\n{'='*60}")

        raw_dpo = _load_dpo_json(str(args.dpo_data_path))
        if args.dpo_sample_size > 0:
            raw_dpo = raw_dpo.select(range(min(args.dpo_sample_size, len(raw_dpo))))
            if accelerator.is_main_process:
                print(f"DPO sample size: using {len(raw_dpo)} examples.")

        if "ref_chosen_logp" not in raw_dpo.column_names:
            if accelerator.is_main_process:
                print("ref_chosen_logp not found — computing inline from current model weights.")
            # Move to GPU for fast precompute; DeepSpeed will manage placement during prepare.
            model.to(accelerator.device)
            raw_dpo = precompute_dpo_ref_logps_inline(raw_dpo, model, tokenizer, args, accelerator)

        dpo_tokenized = load_dpo_dataset(args, tokenizer, raw_dataset=raw_dpo)
        dpo_collator  = build_dpo_collator(tokenizer)
        dpo_train_loader, dpo_optimizer, dpo_scheduler = build_dpo_train_components(
            args, model, dpo_tokenized, dpo_collator
        )
        # Route into the prepare call so DeepSpeed gets model + optimizer together.
        train_loader  = dpo_train_loader
        optimizer     = dpo_optimizer
        lr_scheduler  = dpo_scheduler

    # === Single prepare call — DeepSpeed requires model + optimizer together ===
    print("before prepare distribution")
    model, train_loader, optimizer, lr_scheduler, eval_loader = prepare_distributed_components(
        accelerator,
        model,
        train_loader=train_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        eval_loader=eval_loader,
    )

    # Route prepared components back to their respective phases.
    if args.do_dpo:
        dpo_train_loader = train_loader
        dpo_optimizer    = optimizer
        dpo_scheduler    = lr_scheduler

    # === SFT ===
    print("run task")
    run_requested_tasks(args, model, tokenizer, accelerator, train_loader, optimizer, lr_scheduler, eval_loader)

    # === DPO ===
    if args.do_dpo:
        if accelerator.is_main_process:
            print(f"\n{'='*60}\nStarting DPO training\n{'='*60}")
        train_dpo(model, dpo_train_loader, dpo_optimizer, dpo_scheduler, accelerator, args)
        save_checkpoint(model, tokenizer, accelerator, dpo_output_dir)


if __name__ == "__main__":
    main()
