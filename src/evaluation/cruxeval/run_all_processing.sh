#!/bin/bash
set -euo pipefail

project="qwen3-4B-combined-e_24-layer_33_34_-k_2-all_tokens_lr_5e-5_6_batch_512"

cot=true
model_dir="../../../data/saved_models/${project}/"


bash generate.sh ${project} ${cot} ${model_dir}
bash process.sh ${project} ${cot}
bash eval.sh ${project} ${cot}
