import argparse
import math
import re
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import get_scheduler

from model import create_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Custom multi-GPU LLM training with Accelerate + DeepSpeed")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
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
    parser.add_argument("--num_experts", type=int, default=4)
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
    return parser.parse_args()


def tokenize_fn(examples, tokenizer, max_length):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


def normalize_run_flags(args):
    if not args.do_train and not args.do_eval:
        args.do_train = True
    if args.moe_layer_indices and args.num_experts < 2:
        raise ValueError("--num_experts must be >= 2 when using --moe_layer_indices.")
    if args.moe_top_k < 1 or args.moe_top_k > args.num_experts:
        raise ValueError("--moe_top_k must be between 1 and --num_experts.")
    return args


def render_chat_text(tokenizer, messages, add_generation_prompt):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_tokenized_example(example, tokenizer, max_length, code_pattern):
    full_text = example["text"]
    prompt_text = render_chat_text(
        tokenizer,
        [{"role": "user", "content": example["user_prompt"]}],
        add_generation_prompt=True,
    )

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

    code_spans = [match.span() for match in code_pattern.finditer(full_text)]
    code_mask = [0] * len(input_ids)
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        for code_start, code_end in code_spans:
            if start < code_end and end > code_start:
                code_mask[idx] = 1
                break

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "assistant_mask": assistant_mask,
        "code_mask": code_mask,
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
            "code_mask": [],
        }

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
            batch["assistant_mask"].append(feature["assistant_mask"] + [0] * pad_len)
            batch["code_mask"].append(feature["code_mask"] + [0] * pad_len)

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    return collate_fn


def split_batch_for_model(batch):
    aux = {
        "assistant_mask": batch.pop("assistant_mask", None),
        "code_mask": batch.pop("code_mask", None),
    }
    return batch, aux


def load_tokenized_dataset(args, tokenizer):
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "execution_grounded_reasoning" / "all.jsonl"
    dataset = Dataset.from_json(str(dataset_path))

    messages_list = []
    for row in dataset:
        messages_list.append(
            [
                {"role": "user", "content": row["user_prompt"]},
                {"role": "assistant", "content": row["assistant_prompt"]},
            ]
        )

    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer '{args.model_name}' does not provide a chat template. "
            "Use a chat-model tokenizer (for example Qwen) or define a custom formatter."
        )

    inputs = render_chat_text(tokenizer, messages_list, add_generation_prompt=False)
    org_count = len(inputs)
    code_pattern = re.compile(r"```.*?```", flags=re.DOTALL)

    filtered_user_prompts = []
    filtered_assistant_prompts = []
    filtered_messages_list = []
    filtered_texts = []
    for row, messages, text in zip(dataset, messages_list, inputs):
        if len(text) <= args.max_length:
            filtered_user_prompts.append(row["user_prompt"])
            filtered_assistant_prompts.append(row["assistant_prompt"])
            filtered_messages_list.append(messages)
            filtered_texts.append(text)
    messages_list = filtered_messages_list
    print(f"Loaded {org_count} rows. Kept {len(messages_list)} rows after chat-length filter.")

    formatted_dataset = Dataset.from_dict(
        {
            "user_prompt": filtered_user_prompts,
            "assistant_prompt": filtered_assistant_prompts,
            "text": filtered_texts,
        }
    )
    # split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    # split_dataset = DatasetDict({"train": split_dataset["train"], "validation": split_dataset["test"]})
    split = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    split_dataset = DatasetDict({
        "train": split["train"].select(range(min(64, len(split["train"])))),
        "validation": split["test"].select(range(min(64, len(split["test"])))),
    })

    tokenized = split_dataset.map(
        lambda example: build_tokenized_example(
            example,
            tokenizer=tokenizer,
            max_length=args.max_length,
            code_pattern=code_pattern,
        ),
        remove_columns=["user_prompt", "assistant_prompt", "text"],
    )
    return tokenized.filter(lambda x: len(x["input_ids"]) > 0 and any(label != -100 for label in x["labels"]))


def configure_trainable_parameters(model, args):
    if not args.freeze_non_moe:
        return
    
    def set_requires_grad(module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag


    set_requires_grad(model, False)

    # 2) Unfreeze MoE layers 34-35: router + experts + norms
    for i in [34, 35]:
        layer = model.model.layers[i]

        # MoE parts
        set_requires_grad(layer.mlp.router, True)
        set_requires_grad(layer.mlp.experts, True)

        # Norms in those layers
        set_requires_grad(layer.input_layernorm, True)
        set_requires_grad(layer.post_attention_layernorm, True)

    # 3) Final norm
    set_requires_grad(model.model.norm, True)
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
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * update_steps_per_epoch
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
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
            batch, _ = split_batch_for_model(batch)
            with accelerator.accumulate(model):
                # print(batch)
                outputs = model(**batch)
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

    # args.gradient_accumulation_steps = accelerator.gradient_accumulation_steps
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
                f"{model.converted_layer_indices} | num_experts={args.num_experts} | top_k={args.moe_top_k}"
            )
        else:
            print("No MoE conversion requested. Using the original dense model.")
    tokenized = load_tokenized_dataset(args, tokenizer)
    collator = build_supervised_collator(tokenizer)
    print("before build mode components")
    train_loader, optimizer, lr_scheduler, eval_loader = build_mode_components(args, model, tokenized, collator)
    print("before prepare distribution")
    model, train_loader, optimizer, lr_scheduler, eval_loader = prepare_distributed_components(
        accelerator,
        model,
        train_loader=train_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        eval_loader=eval_loader,
    )
    print("run task")
    run_requested_tasks(args, model, tokenizer, accelerator, train_loader, optimizer, lr_scheduler, eval_loader)


if __name__ == "__main__":
    main()
