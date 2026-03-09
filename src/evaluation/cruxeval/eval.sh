
project="qwen3-4B-test-e_24-layer_33_34_-k_1-all_tokens_test"
result_dir="../../../data/results/${project}/"

# python evaluate_generations.py \
#     --generations_path ${result_dir}/generation_processed.json \
#     --scored_results_path ${result_dir}/scored_results.json \
#     --mode output \
#     2>&1 | tee ../../../data/logs/${project}/eval.log

python evaluate_generations.py \
    --generations_path ${result_dir}/generation_cot_processed.json \
    --scored_results_path ${result_dir}/scored_cot_results.json \
    --mode output \
    2>&1 | tee ../../../data/logs/${project}/eval_cot.log

