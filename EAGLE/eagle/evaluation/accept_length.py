import json
from transformers import AutoTokenizer
import numpy as np

tokenizer=AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--jsonl_file", type=str, default="llama-2-chat-70b-fp16-ea-in-temperature-0.0.jsonl")
jsonl_file = parser.parse_args().jsonl_file

data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)




total_tokens = 0
total_steps = 0
total_times = 0
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    total_tokens += sum(datapoint["choices"][0]['new_tokens'])
    total_times += sum(datapoint["choices"][0]['wall_time'])
    total_steps += sum(datapoint["choices"][0]['idxs'])

print ('average accept length:', total_tokens/total_steps)
print ('average throughput:', total_tokens/total_times)



