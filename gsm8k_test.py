#prevent the local variables from being imported into the remote environment as they can cuase crashes
import os
if 'LIBRARY_ROOTS' in os.environ:
    del os.environ['LIBRARY_ROOTS']

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

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
from loguru import logger
import logging
from contextlib import contextmanager
import signal
from vllm import LLM, SamplingParams


base_model_id = "microsoft/phi-2"
#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
#Load the model with fp16

USE_VLLM = True

if USE_VLLM:
    model = LLM(model=base_model_id)
else:
    model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, torch_dtype=torch.float16,
                                                 device_map={"": 0})

function_name = "solve_math_problem"
prompt = f"def {function_name}() -> int:\n    \"\"\"%s\n    \"\"\"\n"
#         "       End with a \"return result\" line.\n" + \
#         "       You must elaborate your thinking in code via comments below\n"

def sample(model, qn, tokenizer, device, sample_len, temperature = 0., top_p = 0.01):
    # Inefficient version of calculator sampling -- no batches, doesn't
    # cache activations from previous tokens
    if USE_VLLM:
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=len(qn) + sample_len)
        out = model.generate(qn, sampling_params=sampling_params, use_tqdm=False)
        output = []
        for qn, solution in zip(qn, out):
            output.append(qn+solution.outputs[0].text)
        return output
    

    END_OF_TEXT_TOKEN = tokenizer.eos_token_id

    for _ in range(sample_len):
        with torch.no_grad():
            toks = tokenizer([qn], padding=False, return_tensors="pt").to(device)
            orig_len = toks["input_ids"].shape[1]

            out = model.generate(**toks, max_length=orig_len + 1, pad_token_id=model.config.eos_token_id, temperature=temperature, top_p=top_p,)
            text = tokenizer.batch_decode(out)[0]
            if out[0, -1].item() == END_OF_TEXT_TOKEN:
                break
            qn = text
            split = qn.split(' ')
            if split[-1] == 'result' and split[-2] == 'return':
                break
    return qn


# taken from
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, program):
    def timeout_handler(signum, frame):
        raise Exception(f"'{program}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def compute_result(input_code_string):
    try:
        # Create a new, isolated namespace for each invocation
        local_namespace = {}

        # Execute the code in the string within the isolated namespace
        exec('import math\n' +input_code_string, local_namespace)

        # Assuming the function name is known and consistent
        func_name = function_name  # Adjust if the function name varies
        max_time = 3

        if func_name in local_namespace:
            # Call the function and return the result
            with timeout(max_time, input_code_string):
                try:
                    return local_namespace[func_name]()
                except Exception as e:
                    logger.error(f"An error occurred: {e}")
                    logger.error(f"Code that caused the error: {input_code_string}")
                    return -99999
        else:
            # Function name not found
            return -99999
    except Exception as e:
        # Handle any exception that occurs
        logger.error(f"An error occurred: {e}")
        logger.error(f"Code that caused the error: {input_code_string}")
        return -99999

if __name__ == '__main__':
    logger.add("gsm8k_test_set_e5.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    examples = get_examples("test")
    logger.info("Loaded %d examples" % len(examples))
    logger.info("Using prompt: %s" % prompt)
    correct = 0
    total = 0
    device = "cuda:0"
    # remember the start time:
    start_time = time.time()
    batch = []
    answers = []
    for i in range(len(examples)):
        total += 1
        example = examples[i]
        batch.append(prompt % example["question"])
        answers.append(example["answer"])
        if len(batch) == 64:
            solutions = sample(model, batch, tokenizer, device, 500) #solve(qn)
            j = 0
            for qn, solution, ds_answer in zip(batch, solutions, answers):
                answer = compute_result(solution)
                try:
                    correct_answer = int(ds_answer.split("####")[1].split("<|endoftext|>")[0].strip())
                except ValueError:
                    correct_answer = -88888
            #remove the commas from the answer:
                if answer == correct_answer:
                    correct += 1
                logger.info("Total processed: %d, percent correct: %.3f" % (total, 100.0*correct / total))
                if answer != correct_answer:
                    logger.info(f"Question number: {i+j}, Question: {qn}")
                    logger.info(f"Dataset answer: {correct_answer}")
                    logger.info("Dataset solution: %s" % solution)
                    logger.info(f"LLM answer: {answer}")
                    logger.info(f"LLM solution: {solution}")
                j += 1
            batch = []
            answers = []
    logger.info("Accuracy: %.3f%%" % (100*correct/total))
    logger.info("Total time: %.1f minutes" % ((time.time() - start_time)/60))
    logger.info("Solutions generated per minute: %.1f" % (total / (time.time() - start_time) * 60))
    logger.info("Used prompt: %s" % prompt)



