
# project="qwen3-4B-combined-e_24-layer_33_34_-k_2-all_tokens"
# cot=true

project="$1"
cot="$2"
model_dir="$3"

echo "Project: ${project}"
echo "CoT: ${cot}"
echo "Model Dir: ${model_dir}"

output_dir="../../../data/results/${project}/"
# model_dir="../../../data/saved_models/${project}/"
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
    --tensor_parallel_size 2 \
    ${extra_args} \
    2>&1 | tee ${log_dir}/generate${suffix}.log