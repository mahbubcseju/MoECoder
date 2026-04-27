#!/usr/bin/env bash
set -euo pipefail

k=(1 2)

export PYTORCH_ALLOC_CONF=expandable_segments:True 
for item in "${k[@]}"; do
  echo "Running experiment with k=${item}"
  project="qwen3-4B-code-separate-combined_io-E_8-assistant_only-layers_all-moe_32_34-k_${item}"

  saved_model_dir="../data/saved_models/${project}/"
  log_dir="../data/logs/${project}"

  mkdir -p ${saved_model_dir}/
  mkdir -p ${log_dir}/

  accelerate launch --config_file accelerate_config.yaml train.py \
    --do_train \
    --do_eval \
    --model_name Qwen/Qwen3-4B \
    --output_dir ${saved_model_dir} \
    --data_path "../data/combined/final_dataset_w_io.jsonl" \
    --max_length 8192 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --aux_loss_weight 0.01 \
    --statement_loss_weight 0.01 \
    --freeze_non_moe \
    --learning_rate 5e-5 \
    --min_learning_rate 5e-6 \
    --num_epochs 1 \
    --moe_layer_indices 32 34  \
    --num_experts_temp 8 \
    --num_code_experts 4 \
    --moe_top_k ${item} \
    --train_output_head > ${log_dir}/train.log 2>&1
done

for item in "${k[@]}"; do
  echo "Running experiment with k=${item}"
  project="qwen3-4B-code-separate-combined_io-E_8-assistant_only-layers_all-moe_33_34-k_${item}"
  saved_model_dir="../data/saved_models/${project}/"
  log_dir="../data/logs/${project}"

  mkdir -p ${saved_model_dir}/
  mkdir -p ${log_dir}/

  # newer format
  accelerate launch --config_file accelerate_config.yaml train.py \
    --do_train \
    --do_eval \
    --model_name Qwen/Qwen3-4B \
    --output_dir ${saved_model_dir} \
    --data_path "../data/combined/final_dataset_w_io.jsonl" \
    --max_length 8192 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --aux_loss_weight 0.01 \
    --statement_loss_weight 0.01 \
    --freeze_non_moe \
    --learning_rate 5e-5 \
    --min_learning_rate 5e-6 \
    --num_epochs 1 \
    --moe_layer_indices  33 34  \
    --num_experts_temp 8 \
    --num_code_experts 4 \
    --moe_top_k ${item} \
    --train_output_head > ${log_dir}/train.log 2>&1
done
