import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=str)
    parser.add_argument("--cot", action="store_true")
    args = parser.parse_args()

    results_root = Path(__file__).resolve().parents[3] / "data" / "results"
    project_dir = results_root / args.project

    if args.cot:
        input_path = project_dir / "generation_cot.json"
        output_path = project_dir / "generation_cot_processed.json"

    else:
        input_path = project_dir / "generation.json"
        output_path = project_dir / "generation_processed.json"
    
    with open(input_path, "r") as f:
        generations = json.load(f)

    final_result = {}
    for key in generations.keys():
        # print(key, generations[key])
        final_result[f"sample_{key}"] = generations[key]

    with open(output_path, "w") as f:
        json.dump(final_result, f, indent=2)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
