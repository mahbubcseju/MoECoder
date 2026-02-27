output_dir="../../../data/results/qwen3-4B-test-v1/"
model_dir="../../../data/saved_models/qwen3-4B-test-v1/"
log_dir="../../../data/logs/train_qwen3-4B-test-v1/"

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
    --save_generations_path ${output_dir}/generation_cot.json \
    --start 0 \
    --end 800 \
    --shuffle \
    --tensor_parallel_size 1 \
    --cot \
    2>&1 | tee ${log_dir}/eval_cot.log