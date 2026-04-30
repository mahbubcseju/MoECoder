python generate_rejected_trace.py \
    --input ../../data/semcoder/final_dataset_w_concepts_and_thoughts.jsonl \
    --output ../../data/semcoder/final_dataset_w_concepts_and_thoughts_rejected.jsonl \
    --failed-output ../../data/semcoder/failures_dpo.jsonl \
    --model Qwen/Qwen3-32B \
    --retries 20 \
    --max-model-len 16384 > ../../data/logs/generate_semcoder_rejected.log 2>&1