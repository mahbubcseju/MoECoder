python generate_rejected_trace.py \
    --input ../../data/combined/final_dataset_w_thinking_mode_and_concepts_task_type.jsonl \
    --output ../../data/combined/final_dataset_w_thinking_mode_concepts_task_type_and_rejected.jsonl \
    --failed-output ../../data/combined/failures_dpo.jsonl \
    --model Qwen/Qwen3-32B \
    --retries 20 \
    --max-model-len 16384 > ../../data/logs/generate_dpo_rejected.log 2>&1