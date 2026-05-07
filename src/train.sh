#!/usr/bin/env bash
set -euo pipefail

# Architecture: 15 code experts (14 concept types + 1 other, fixed) + num_nl_experts NL experts
project="qwen3-4B-semcoder-thinking_mode_on-code_e_15-nl_e_4-layer_33_34_-k_nl_1"

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
  --data_path "../data/semcoder/final_dataset_w_concepts_and_thoughts.jsonl" \
  --max_length 8192 \
  --moe_layer_indices 33 34 \
  --num_nl_experts 4 \
  --moe_top_k 1 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 32 \
  --router_aux_loss_weight 0.01 \
  --freeze_non_moe \
  --learning_rate 5e-6 \
  --num_epochs 1 \
  --reasoning_weight 0.50 \
  --content_weight 1.0 \
  --train_output_head > ${log_dir}/train.log 2>&1

# project="qwen3-4B-combined-thinking_mode_off-code_e_15-nl_e_4-layer_33_34_-k_nl_2"

# saved_model_dir="../data/saved_models/${project}/"
# log_dir="../data/logs/${project}"

# mkdir -p ${saved_model_dir}/
# mkdir -p ${log_dir}/

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# accelerate launch --config_file accelerate_config.yaml train.py \
#   --do_train \
#   --do_eval \
#   --model_name Qwen/Qwen3-4B \
#   --output_dir ${saved_model_dir} \
#   --data_path "../data/combined/final_dataset_w_thinking_mode_and_concepts.jsonl" \
#   --max_length 8192 \
#   --moe_layer_indices 33 34 \
#   --num_nl_experts 4 \
#   --moe_top_k 2 \
#   --per_device_batch_size 4 \
#   --gradient_accumulation_steps 16 \
#   --router_aux_loss_weight 0.01 \
#   --freeze_non_moe \
#   --learning_rate 5e-6 \
#   --num_epochs 1 \
#   --train_output_head > ${log_dir}/train.log 2>&1


# project="qwen3-4B-combined-thinking_mode_off-code_e_15-nl_e_4-layer_32_33_34_-k_nl_1"

# saved_model_dir="../data/saved_models/${project}/"
# log_dir="../data/logs/${project}"

# mkdir -p ${saved_model_dir}/
# mkdir -p ${log_dir}/

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# accelerate launch --config_file accelerate_config.yaml train.py \
#   --do_train \
#   --do_eval \
#   --model_name Qwen/Qwen3-4B \
#   --output_dir ${saved_model_dir} \
#   --data_path "../data/combined/final_dataset_w_thinking_mode_and_concepts.jsonl" \
#   --max_length 8192 \
#   --moe_layer_indices 32 33 34 \
#   --num_nl_experts 4 \
#   --moe_top_k 1 \
#   --per_device_batch_size 4 \
#   --gradient_accumulation_steps 16 \
#   --router_aux_loss_weight 0.01 \
#   --freeze_non_moe \
#   --learning_rate 5e-6 \
#   --num_epochs 1 \
#   --train_output_head > ${log_dir}/train.log 2>&1