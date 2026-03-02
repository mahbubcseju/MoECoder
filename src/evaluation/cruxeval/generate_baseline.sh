model_dir="Qwen/Qwen3-4B"
output_dir="../../../data/results/qwen3-4B"
log_dir="../../../data/logs/qwen3-4B"

mkdir -p ${output_dir}/
mkdir -p ${log_dir}/
python run_cruxeval.py \
    --model  ${model_dir} \
    --use_auth_token \
    --trust_remote_code \
    --tasks output_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 2048 \
    --precision fp16 \
    --limit 800 \
    --temperature 0.2 \
    --save_generations \
    --save_generations_path ${output_dir}/generation.json \
    --start 0 \
    --end 800 \
    --shuffle \
    --tensor_parallel_size 1 \
    2>&1 | tee ${log_dir}/eval.log