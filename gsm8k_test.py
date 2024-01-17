#prevent the local variables from being imported into the remote environment as they can cuase crashes
import os
if 'LIBRARY_ROOTS' in os.environ:
    del os.environ['LIBRARY_ROOTS']


# this code assumes that you cloned the GSM8K repo into the same directory as this repo: git clone https://github.com/openai/grade-school-math/tree/master
from dataset import get_examples, GSMDataset
from calculator import sample as gsm_sample, use_calculator

import warnings
warnings.filterwarnings("ignore")
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




base_model_id = "microsoft/phi-2"


#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
#Load the model with fp16
model =  AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map={"": 3})


function_name = "solve_math_problem"
prompt = f"def {function_name}() -> int:\n    \"\"\"%s Your code must end with a \"return result\" line.\"\"\"\n"

def sample(model, qn, tokenizer, device, sample_len):
    # Inefficient version of calculator sampling -- no batches, doesn't
    # cache activations from previous tokens
    END_OF_TEXT_TOKEN = tokenizer.eos_token_id

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
        exec(input_code_string, local_namespace)

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
    logger.add("gsm8k_test_set_e2.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    test_examples = get_examples("test")
    logger.info("Loaded %d test examples" % len(test_examples))
    logger.info("Using prompt: %s" % prompt)
    correct = 0
    total = 0
    device = "cuda:3"
    # remember the start time:
    start_time = time.time()
    for num, example in enumerate(test_examples):
        total += 1
        qn = example["question"]
        solution = sample(model, prompt % qn, tokenizer, device, 500) #solve(qn)
        answer = compute_result(solution)
        try:
            correct_answer = int(example["answer"].split("####")[1].split("<|endoftext|>")[0].strip())
        except ValueError:
            correct_answer = -88888
        #remove the commas from the answer:
        if answer == correct_answer:
            correct += 1
            logger.info("---Total processed: %d, fraction correct %.3f" % (total, correct / total))
        else:
            logger.info("---Total processed: %d, fraction correct %.3f" % (total, correct / total))
            logger.info(f"Question number: {num}, Question: {qn}")
            logger.info(f"Correct answer: {correct_answer}")
            logger.info(f"Your answer: {answer}")
            logger.info(f"Solution: {solution}")
            logger.info("Answer: %s" % example["answer"])
    logger.info("Accuracy: %.3f %" % (100*correct/total))
    logger.info("Total time: %.3f seconds" % (time.time() - start_time))
    logger.info("Solutions generated per minute: %.3f" % (total / (time.time() - start_time) * 60))
    logger.info("Used prompt: %s" % prompt)



