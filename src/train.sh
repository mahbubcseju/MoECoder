#!/usr/bin/env bash
set -euo pipefail

project="qwen3-4B-combined-e_24-layer_33_34_-k_2-all_tokens_lr_5e-5_6_batch_512"

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
  --moe_layer_indices 33 34 \
  --num_experts_temp 24 \
  --per_device_batch_size 4 \
  --moe_top_k 2 \
  --gradient_accumulation_steps 32 \
  --router_aux_loss_weight 0.01 \
  --freeze_non_moe \
  --learning_rate 5e-5 \
  --learning_rate 5e-6 \
  --num_epochs 2 \
  --train_output_head > ${log_dir}/train.log 2>&1

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