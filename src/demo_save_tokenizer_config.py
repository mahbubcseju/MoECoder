import argparse
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a tokenizer from Hugging Face and save tokenizer_config.json."
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "saved_models" / "tokenizer_demo"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.save_pretrained(str(output_dir))
    config_path = output_dir / "tokenizer_config.json"

    print(f"Saved tokenizer config to: {config_path}")


if __name__ == "__main__":
    main()
