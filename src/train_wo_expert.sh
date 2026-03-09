#!/usr/bin/env bash
set -euo pipefail

project="qwen3-4B-combined-all_tokens-wo-expert"

saved_model_dir="../data/saved_models/${project}/"
log_dir="../data/logs/${project}"

mkdir -p ${saved_model_dir}/
mkdir -p ${log_dir}/

accelerate launch --config_file accelerate_config.yaml train.py \
  --do_train \
  --do_eval \
  --model_name Qwen/Qwen3-4B \
  --output_dir ${saved_model_dir} \
  --data_path "../data/combined/final_dataset_w_concepts.jsonl" \
  --max_length 8192 \
  --per_device_batch_size 4 \
  --moe_top_k 2 \
  --gradient_accumulation_steps 8 \
  --router_aux_loss_weight 0.01 \
  --freeze_non_moe \
  --learning_rate 5e-6 \
  --num_epochs 2 \
  --train_output_head > ${log_dir}/train.log 2>&1
