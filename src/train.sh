#!/usr/bin/env bash
set -euo pipefail

project="qwen3-4B-code-separate-combined_io-all_tokens-layers_all-moe_33_34-lr_5e-5_6_batch_512"

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
  --per_device_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --aux_loss_weight 0.01 \
  --statement_loss_weight 0.01 \
  --freeze_non_moe \
  --learning_rate 5e-5 \
  --min_learning_rate 5e-6 \
  --num_epochs 2 \
  --moe_layer_indices 33 34 \
  --num_experts_temp 16 \
  --num_code_experts 8 \
  --moe_top_k 2 \
  --train_output_head > ${log_dir}/train.log 2>&1

# accelerate launch --config_file accelerate_config.yaml train.py \
#   --do_train \
#   --do_eval \
#   --model_name Qwen/Qwen3-4B \
#   --output_dir ${saved_model_dir} \
#   --data_path "../data/combined/final_dataset_w_io.jsonl" \
#   --max_length 8192 \
#   --moe_layer_indices 33 34 \
#   --num_experts_temp 16 \
#   --num_code_experts 8 \
#   --per_device_batch_size 4 \
#   --moe_top_k 2 \
#   --gradient_accumulation_steps 32 \
#   --aux_loss_weight 0.01 \
#   --statement_loss_weight 0.01 \
#   --freeze_non_moe \
#   --learning_rate 5e-5 \
#   --min_learning_rate 5e-6 \
#   --num_epochs 2 \
#   --train_output_head > ${log_dir}/train.log 2>&1

# python train.py \
#   --do_eval \
#   --model_name ../data/saved_models/ckpt-qwen \
#   --output_dir ../data/saved_models/ckpt-qwen-v2 \
#   --max_length 8192 \
#   --moe_layer_indices 34 35 \
#   --num_experts 16 \
#   --per_device_batch_size 4 \
#   --moe_top_k 2 \
#   --gradient_accumulation_steps 8 \
#   --router_aux_loss_weight 0.01 \
#   --freeze_non_moe \
#   --train_output_head > ../data/logs/test.log 2>&1