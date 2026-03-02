
result_dir="../../../data/results/qwen3-4B/"
cot=false

suffix=""
if [ "$cot" = true ]; then
    suffix="_cot"
fi

python evaluate_generations.py \
    --generations_path ${result_dir}/generation${suffix}_processed.json \
    --scored_results_path ${result_dir}/scored${suffix}_results.json \
    --mode output \
    2>&1 | tee ../../../data/logs/qwen3-4B/eval${suffix}.log
