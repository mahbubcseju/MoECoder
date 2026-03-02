
project="qwen3-4B-test-e32-layer_33_35_all_tokens"
cot=false


output_dir="../../../data/results/${project}/"
model_dir="../../../data/saved_models/${project}/"
log_dir="../../../data/logs/${project}/"

suffix=""
extra_args=""
if [ "$cot" = true ]; then
    suffix="_cot"
    extra_args="--cot"
fi


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
    --save_generations_path ${output_dir}/generation${suffix}.json \
    --start 0 \
    --end 800 \
    --shuffle \
    --tensor_parallel_size 1 \
    ${extra_args} \
    2>&1 | tee ${log_dir}/eval${suffix}.log