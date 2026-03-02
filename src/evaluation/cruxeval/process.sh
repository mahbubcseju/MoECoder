model_dir="Qwen/Qwen3-4B"
cot=false
extra_args=""
if [ "$cot" = true ]; then
    extra_args="--cot"
fi

python process_generations.py \
    --project ${model_dir} \
    ${extra_args} \
    2>&1 | tee ../../../data/logs/${model_dir}/process.log