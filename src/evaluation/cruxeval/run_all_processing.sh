#!/bin/bash
set -euo pipefail

# project="qwen3-4B-combined-e_24-layer_33_34_-k_2-all_tokens_lr_5e-5_6_batch_512"
# project="qwen3-4B-combined-e_24-layer_33_34_-k_2-all_tokens"
# project="qwen3-4B-combined-e_16-layer_31_32_33_34_-all_layers-all_tokens-k_2"

# project="qwen3-4B-wo_exprt"
# project="qwen3-4B-wo_expert-assistant-only"
project="qwen3-4B-combined-e_16-layer_31_32_33_34_-all_layers-assistant_only-k_1"
project="qwen3-4B-combined-thinking_mode-e_16-layer_31_32_33_34_-all_layers-assistant_only-k_1"


cot=true
model_dir="../../../data/saved_models/${project}/"


bash generate.sh ${project} ${cot} ${model_dir}
bash process.sh ${project} ${cot}
bash eval.sh ${project} ${cot}
