#!/usr/bin/env bash
set -euo pipefail

sft_project="qwen3-4B-semcoder-thinking_mode_on-e_16-layer_33_34_-all_layers-assistant_only-k_1"
dpo_project="${sft_project}-dpo-v2"

sft_model_dir="../data/saved_models/${sft_project}/"
dpo_model_dir="../data/saved_models/${dpo_project}/"
log_dir="../data/logs/${dpo_project}"

mkdir -p ${dpo_model_dir}/
mkdir -p ${log_dir}/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch --config_file accelerate_config.yaml train.py \
  --do_dpo \
  --model_name ${sft_model_dir} \
  --dpo_data_path "../data/semcoder/final_dataset_w_concepts_and_thoughts_rejected_trial_8.jsonl" \
  --dpo_output_dir ${dpo_model_dir} \
  --max_length 8192 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 32 \
  --dpo_beta 0.1 \
  --dpo_num_epochs 1 \
  --dpo_learning_rate 1e-7 \
  --freeze_non_moe \
  --train_output_head > ${log_dir}/train_dpo.log 2>&1
