#!/bin/bash
set -euo pipefail

project="qwen3-4B-combined-all_tokens-wo-expert"
cot=true
model_dir="../../../data/saved_models/qwen3-4B-combined-all_tokens-wo-expert"


bash generate.sh ${project} ${cot} ${model_dir}
bash process.sh ${project} ${cot}
bash eval.sh ${project} ${cot}
