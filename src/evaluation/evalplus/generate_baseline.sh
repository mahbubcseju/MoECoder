export HUMANEVAL_OVERRIDE_PATH=../../../data/evalplus/humanevalplus.jsonl
export MBPP_OVERRIDE_PATH=../../../data/evalplus/mbppplus.jsonl

MODEL_ID="qwen3-4B"
chat_mode=true

extra_args=""
if [ "$chat_mode" = true ]; then
    extra_args="--chat_mode"
fi



INPUT_MODEL="Qwen/Qwen3-4B"
RESULTS_DIR="../../../data/results/evalplus/${MODEL_ID}/"
log_dir="../../../data/logs/${MODEL_ID}/"


echo "Running EvalPlus::[HumanEval]"
mkdir -p ${RESULTS_DIR}/humaneval
python generate.py \
    --model ${INPUT_MODEL} \
    --tp $TP \
    --bs 1 \
    --temperature 0 \
    --n_samples 1 \
    --greedy  \
    --save_folder ${RESULTS_DIR}/humaneval \
    --dataset humaneval \
    ${extra_args} \
    2>&1 | tee ${log_dir}/eval_humaneval_chat_mode_${chat_mode}.log

echo "Running EvalPlus::[MBPP]"
mkdir -p ${RESULTS_DIR}/mbpp
python generate.py \
    --model ${INPUT_MODEL} \
    --tp $TP \
    --bs 1 \
    --temperature 0 \
    --n_samples 1 \
    --greedy  \
    --save_folder ${RESULTS_DIR}/mbpp \
    --dataset mbpp \
    ${extra_args} \
    2>&1 | tee ${log_dir}/eval_mbpp_chat_mode_${chat_mode}.log