#!/usr/bin/env bash
set -euo pipefail

# accelerate launch --config_file accelerate_config.yaml train.py \
#   --do_eval \
#   --model_name Qwen/Qwen3-4B \
#   --output_dir ../data/saved_models/ckpt-qwen \
#   --max_length 8192 \
#   --moe_layer_indices 34 35 \
#   --num_experts 16 \
#   --per_device_batch_size 4 \
#   --moe_top_k 2 \
#   --gradient_accumulation_steps 8 \
#   --router_aux_loss_weight 0.01 \
#   --freeze_non_moe \
#   --train_output_head > ../data/logs/test.log 2>&1

python train.py \
  --do_eval \
  --model_name ../data/saved_models/ckpt-qwen \
  --output_dir ../data/saved_models/ckpt-qwen-v2 \
  --max_length 8192 \
  --moe_layer_indices 34 35 \
  --num_experts 16 \
  --per_device_batch_size 4 \
  --moe_top_k 2 \
  --gradient_accumulation_steps 8 \
  --router_aux_loss_weight 0.01 \
  --freeze_non_moe \
  --train_output_head > ../data/logs/test.log 2>&1