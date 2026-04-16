#!/bin/bash
set -euo pipefail

# project="qwen3-4B-combined-e_24-layer_33_34_-k_2-all_tokens_lr_5e-5_6_batch_512"
# project="qwen3-4B-combined-e_24-layer_33_34_-k_2-all_tokens"
# project="qwen3-4B-code-separate-combined_io-e_16-layer_33_34_-k_2-all_tokens_lr_5e-5_6_batch_512"
# project="qwen3-4B-code-separate-combined_io-all_tokens-layers_5-moe_32_34-lr_5e-5_6_batch_512"
# project="qwen3-4B-code-separate-combined_io-all_tokens-layers_10-moe_28_30_32_34-k_1"
# project="qwen3-4B-code-separate-combined_io-all_tokens-layers_all-moe_28_30_32_34-k_1"
project="qwen3-4B-code-separate-combined_io-all_tokens-layers_all-moe_20_22_24_26_28_30_32_34-E_8-k_1"

cot=true
model_dir="../../../data/saved_models/${project}/"


bash generate.sh ${project} ${cot} ${model_dir}
bash process.sh ${project} ${cot}
bash eval.sh ${project} ${cot}


project="qwen3-4B-code-separate-combined_io-all_tokens-layers_all-moe_32_33_34_35-k_1"
cot=true
model_dir="../../../data/saved_models/${project}/"


bash generate.sh ${project} ${cot} ${model_dir}
bash process.sh ${project} ${cot}
bash eval.sh ${project} ${cot}
