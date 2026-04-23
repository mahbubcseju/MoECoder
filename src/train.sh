#!/usr/bin/env bash
set -euo pipefail

project="qwen3-4B-moe-lora-r16-e16-k2-q_k_v_o_proj-lr_2e-4-assistant-only"

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
  --data_path "../data/ex_tr_data/final_traced_dataset_w_concepts.jsonl" \
  --max_length 8192 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_num_experts 16 \
  --lora_top_k 2 \
  --lora_target_modules q_proj k_proj  v_proj o_proj \
  --router_aux_loss_weight 0.01 \
  --per_device_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --min_learning_rate 2e-5 \
  --num_warmup_steps 100 \
  --num_epochs 2 \
  --attn_implementation flash_attention_2 > ${log_dir}/train.log 2>&1

# ---- MoEMLP (full block replacement) training ----
# project_moe="qwen3-4B-moe-e24-layer_33_34-k2-lr_5e-5"
# accelerate launch --config_file accelerate_config.yaml train.py \
#   --do_train \
#   --do_eval \
#   --model_name Qwen/Qwen3-4B \
#   --output_dir "../data/saved_models/${project_moe}/" \
#   --data_path "../data/combined/final_dataset_w_concepts.jsonl" \
#   --max_length 8192 \
#   --moe_layer_indices 33 34 \
#   --num_experts_temp 24 \
#   --moe_top_k 2 \
#   --router_aux_loss_weight 0.01 \
#   --freeze_non_moe \
#   --per_device_batch_size 4 \
#   --gradient_accumulation_steps 32 \
#   --learning_rate 5e-5 \
#   --min_learning_rate 5e-6 \
#   --num_warmup_steps 100 \
#   --num_epochs 2 \
#   --train_output_head \
#   --attn_implementation flash_attention_2 > "../data/logs/${project_moe}/train.log" 2>&1
