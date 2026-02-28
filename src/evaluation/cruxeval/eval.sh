
result_dir="../../../data/results/qwen3-4B-test-v1/"

python evaluate_generations.py \
    --generations_path ${result_dir}/generation_cot_processed.json \
    --scored_results_path ${result_dir}/scored_cot_results.json \
    --mode output \
    2>&1 | tee ../../../data/logs/train_qwen3-4B-test-v1/eval_cot.log