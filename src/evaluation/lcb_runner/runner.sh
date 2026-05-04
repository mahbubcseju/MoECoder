#!/bin/bash

project="qwen3-4B-semcoder-thinking_mode_on-e_16-layer_33_34_-all_layers-assistant_only-k_1-reasoning_0.5_content_1.0-dpo-v2"

# project="qwen3-4B-semcoder-thinking_mode_on-e_16-layer_33_34_-all_layers-assistant_only-k_1-reasoning_0.5_content_1.0-dpo"
model_dir="../../../data/saved_models/${project}/"
# project="qwen3-4B-test-e_24-layer_33_34_-k_1-all_tokens_test"
# model_dir="../../../data/saved_models/${project}/"
# project="Qwen3-4B"
# model_dir="Qwen/Qwen3-4B"
task="codeexecution"

log_dir="../../../data/logs/${project}/"
result_dir="../../../data/results/"

mkdir -p ${log_dir}/
mkdir -p ${result_dir}/

python main.py \
    --model ${model_dir} \
    --model_repr ${project} \
    --scenario codeexecution \
    --evaluate \
    --trust_remote_code \
    --tensor_parallel_size 1 \
    --custom_output_dir ${result_dir} \
    --cot_code_execution \
    --stop "[/ANSWER]" \
    --n 1 \
    --release_version release_v6 > ${log_dir}/livecodebench_${task}.log  2>&1
