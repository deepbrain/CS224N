#prevent the local variables from being imported into the remote environment as they can cuase crashes
import os
if 'LIBRARY_ROOTS' in os.environ:
    del os.environ['LIBRARY_ROOTS']


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

duration = 0.0
total_length = 0
prompt = []
prompt.append("INPUT: Think step by step and solve a math problem: X^2 -10x + 24 = 0. Use the discriminant method. Elaborate your solution.")

for i in range(len(prompt)):
  model_inputs = tokenizer(prompt[i], return_tensors="pt").to("cuda:3")
  start_time = time.time()
  output = model.generate(**model_inputs, max_length=500)[0]
  duration += float(time.time() - start_time)
  total_length += len(output)
  tok_sec_prompt = round(len(output)/float(time.time() - start_time),3)
  print("Prompt --- %s tokens/seconds ---" % (tok_sec_prompt))
  print(tokenizer.decode(output, skip_special_tokens=True))