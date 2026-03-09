# project="qwen3-4B-combined-e_24-layer_33_34_-k_2-all_tokens"
# cot=true
project="$1"
cot="$2"

extra_args=""
suffix=""
if [ "$cot" = true ]; then
    extra_args="--cot"
    suffix="_cot"
fi

echo ${pwd}
python process_generations.py \
    --project ${project} \
    ${extra_args} \
    2>&1 | tee ../../../data/logs/${project}/process${suffix}.log