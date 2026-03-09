
# project="qwen3-4B-combined-e_24-layer_33_34_-k_2-all_tokens"
project="$1"
cot="$2"

result_dir="../../../data/results/${project}/"

suffix=""
if [ "$cot" = true ]; then
    suffix="_cot"
fi


# python evaluate_generations.py \
    # --generations_path ${result_dir}/generation_processed.json \
    # --scored_results_path ${result_dir}/scored_results.json \
    # --mode output \
    # 2>&1 | tee ../../../data/logs/${project}/eval.log

python evaluate_generations.py \
    --generations_path ${result_dir}/generation${suffix}_processed.json \
    --scored_results_path ${result_dir}/scored${suffix}_results.json \
    --mode output \
    2>&1 | tee ../../../data/logs/${project}/eval${suffix}.log

