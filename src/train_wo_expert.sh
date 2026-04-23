#!/usr/bin/env bash
set -euo pipefail

project="qwen3-4B-wo_expert-assistant-only"

saved_model_dir="../data/saved_models/${project}/"
log_dir="../data/logs/${project}"

mkdir -p ${saved_model_dir}/
mkdir -p ${log_dir}/

accelerate launch --config_file accelerate_config.yaml train.py \
  --do_train \
  --do_eval \
  --model_name Qwen/Qwen3-4B \
  --output_dir ${saved_model_dir} \
  --data_path "../data/ex_tr_data/final_traced_dataset_w_concepts.jsonl" \
  --max_length 8192 \
  --per_device_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --num_epochs 1 > ${log_dir}/train.log 2>&1
