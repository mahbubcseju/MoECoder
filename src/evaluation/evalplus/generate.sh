export HUMANEVAL_OVERRIDE_PATH=../../../data/evalplus/humanevalplus.jsonl
export MBPP_OVERRIDE_PATH=../../../data/evalplus/mbppplus.jsonl

# project="qwen3-4B-combined-e_24-layer_33_34_-k_2-all_tokens"
# INPUT_MODEL="../../../data/saved_models/${project}/"
# project="Qwen/Qwen3-4B"
project="qwen3-4B-combined-all_tokens-wo-expert"
INPUT_MODEL="../../../data/saved_models/${project}/"
# INPUT_MODEL="../../../data/saved_models/${model_dir}/"
TP=2

RESULTS_DIR="../../../data/results/${project}/"
log_dir="../../../data/logs/${project}/"


echo "Running EvalPlus::[HumanEval]"
mkdir -p ${RESULTS_DIR}/humaneval
mkdir -p ${log_dir}
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