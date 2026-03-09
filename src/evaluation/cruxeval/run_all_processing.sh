project="Qwen/Qwen3-4B"
cot=true
model_dir="Qwen/Qwen3-4B"


bash generate.sh ${project} ${cot} ${model_dir}
bash process.sh ${project} ${cot}
bash eval.sh ${project} ${cot}
