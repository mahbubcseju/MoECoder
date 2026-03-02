import json
import re
from vllm import LLM, SamplingParams
from datasets import load_dataset

dataset = load_dataset("dongwonj/Execution-Grounded-Reasoning")
data_to_process = dataset['train'] 


llm = LLM(model="Qwen/Qwen2.5-32B-Instruct", tensor_parallel_size=1) 
sampling_params = SamplingParams(temperature=0.2, max_tokens=1024, stop=["</ans>"])

def build_prompt(row):
    return f"""You are a Python expert. Fix the indentation of the provided 'refcode'. 
Use the 'input' and 'answer' provided to understand the logic.
Return ONLY the properly formatted code inside <ans> and </ans> tags.

### Input:
{row['input']}

### Reference Answer:
{row['answer']}

### Unformatted Refcode:
{row['refcode']}

### Formatted Code:"""


prompts = [build_prompt(row) for row in data_to_process]


outputs = llm.generate(prompts, sampling_params)


output_file = "formatted_dataset.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for i, output in enumerate(outputs):
        original_row = data_to_process[i]
        
        raw_text = output.outputs[0].text
        
        match = re.search(r'<ans>(.*?)</ans>', raw_text, re.DOTALL)
        if match:
            clean_code = match.group(1).strip()
        else:
            clean_code = raw_text.replace("<ans>", "").replace("</ans>", "").strip()

        new_entry = {
            "user_prompt": original_row.get("user_prompt"),
            "assistant_prompt": original_row.get("assistant_prompt"),
            "input": original_row.get("input"),
            "answer": original_row.get("answer"),
            "execution_trace": original_row.get("execution_trace"),
            "refcode": original_row.get("refcode"),
            "code": clean_code 
        }
        f.write(json.dumps(new_entry) + "\n")

print(f"Success! Saved to {output_file}")