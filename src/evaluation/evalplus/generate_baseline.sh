export HUMANEVAL_OVERRIDE_PATH=../../../data/evalplus/humanevalplus.jsonl
# export MBPP_OVERRIDE_PATH=../../../data/evalplus/MbppPlus-v0.1.0.jsonl


INPUT_MODEL="Qwen/Qwen3-4B"
MODEL_ID="qwen3-4B"
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
    --chat_mode \
    2>&1 | tee ${log_dir}/eval_humaneval.log

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
    --chat_mode \
    2>&1 | tee ${log_dir}/eval_mbpp.log