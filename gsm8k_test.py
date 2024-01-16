#prevent the local variables from being imported into the remote environment as they can cuase crashes
import os
if 'LIBRARY_ROOTS' in os.environ:
    del os.environ['LIBRARY_ROOTS']


# this code assumes that you cloned the GSM8K repo into the same directory as this repo: git clone https://github.com/openai/grade-school-math/tree/master
from dataset import get_examples, GSMDataset
from calculator import sample as gsm_sample, use_calculator


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split



base_model_id = "microsoft/phi-2"


#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
#Load the model with fp16
model =  AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map={"": 3})



prompt = ("INPUT: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"+ \
         "SOLUTION: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72 ~~\n"+ \
         "INPUT: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n"+ \
         "SOLUTION: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10 ~~\n"+ \
         "INPUT: %s\nSOLUTION: ")


def sample(model, qn, tokenizer, device, sample_len):
    # Inefficient version of calculator sampling -- no batches, doesn't
    # cache activations from previous tokens
    END_OF_TEXT_TOKEN = tokenizer.eos_token_id
    EQUALS_TOKENS = set([28, 796, 47505])

    for _ in range(sample_len):
        with torch.no_grad():
            toks = tokenizer([qn], padding=False, return_tensors="pt").to(device)
            orig_len = toks["input_ids"].shape[1]

            out = model.generate(
                **toks, max_length=orig_len + 1, pad_token_id=model.config.eos_token_id
            )
            text = tokenizer.batch_decode(out)[0]
            if out[0, -1].item() == END_OF_TEXT_TOKEN:
                break
            if out[0, -1].item() in EQUALS_TOKENS:
                answer = use_calculator(text)
                if answer is not None:
                    print("Triggered calculator, answer", answer)
                    text = text + str(answer) + ">>"

            qn = text
    return qn


def solve(problem, prompt=prompt):
    input = prompt % problem
    model_input = tokenizer(input, return_tensors="pt").to("cuda:3")
    start_time = time.time()
    output = model.generate(**model_input, max_length=500)[0]
    duration = float(time.time() - start_time)
    total_length = len(output)
    tok_sec_prompt = round(len(output)/float(time.time() - start_time),3)
    print("Prompt --- %s tokens/seconds ---" % (tok_sec_prompt))
    solution = tokenizer.decode(output, skip_special_tokens=True)
    #exclude the input string from the solution
    solution = solution.split("INPUT: ")[2]
    #answer should be between #### and \n (exclusive)
    answer = solution.split("####")[1].split("\n")[0].strip()
    return solution, answer

if __name__ == '__main__':
    test_examples = get_examples("test")
    qn = test_examples[2]["question"]
    sample_len = 200
    print(qn.strip())
    print(sample(model, prompt % qn, tokenizer, "cuda:3", sample_len))


