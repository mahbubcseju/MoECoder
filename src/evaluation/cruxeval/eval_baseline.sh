
result_dir="../../data/results/qwen3-4b/"

python evaluate_generations.py \
    --generations_path ${result_dir}/generation_processed.json \
    --scored_results_path ${result_dir}/scored_results.json \
    --mode output \
    2>&1 | tee ../../data/logs/qwen3-4b/eval.log