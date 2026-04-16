#!/usr/bin/env bash
set -euo pipefail

project="qwen3-4B-code-separate-combined_io-all_tokens-layers_all-moe_32_33_34_35-k_1"

saved_model_dir="../data/saved_models/${project}/"
log_dir="../data/logs/${project}"

mkdir -p ${saved_model_dir}/
mkdir -p ${log_dir}/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch --config_file accelerate_config.yaml train.py \
  --do_train \
  --do_eval \
  --model_name Qwen/Qwen3-4B \
  --output_dir ${saved_model_dir} \
  --data_path "../data/combined/final_dataset_w_io.jsonl" \
  --max_length 8192 \
  --per_device_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --aux_loss_weight 0.01 \
  --statement_loss_weight 0.01 \
  --freeze_non_moe \
  --learning_rate 5e-5 \
  --min_learning_rate 5e-6 \
  --num_epochs 3 \
  --moe_layer_indices 32 33 34 35  \
  --num_experts_temp 16 \
  --num_code_experts 8 \
  --moe_top_k 1 \
  --train_output_head > ${log_dir}/train.log 2>&1
