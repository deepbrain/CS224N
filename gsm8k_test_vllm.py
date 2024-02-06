#prevent the local variables from being imported into the remote environment as they can cuase crashes
import os
if 'LIBRARY_ROOTS' in os.environ:
    del os.environ['LIBRARY_ROOTS']


# adjust this to the GPU you want to use:
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = 3

# this code assumes that you cloned the GSM8K repo into the same directory as this repo: git clone https://github.com/openai/grade-school-math/tree/master
from dataset import get_examples, GSMDataset
import warnings
warnings.filterwarnings("ignore")
import time
import torch
from loguru import logger
import logging
from contextlib import contextmanager
import signal
from transformers import (AutoModelForCausalLM,  AutoTokenizer)
import sympy
import numpy as np
import sympy
import math



base_model_id = "microsoft/phi-2"
base_model_revision = "accfee56d8988cae60915486310362db5831b1bd"
#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

USE_VLLM = True #TODO: debug code to work without vllm

if USE_VLLM:
    from vllm import LLM, SamplingParams
    model = LLM(model=base_model_id, revision=base_model_revision)
else:
    model = AutoModelForCausalLM.from_pretrained(base_model_id, revision=base_model_revision, trust_remote_code=True, torch_dtype=torch.float16, device_map={"": device})

function_name = "problem"
prompt = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       Elaborate your thinking step by step in comments before each code line below.\n"

prompt2 = f"def {function_name}() -> int:\n    \"\"\"%s" + \
         "       Add comments before each line.\n"

prompt3 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       Be accurate and think step by step in comments before each code line below.\n"

prompt4 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       Find unusual solution and comment before each of your line of code.\n"

prompt5 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       In your comments write an algebraic formula based on the problem, solve it algebraically, then write code to calculate the result.\n"

prompt6 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       Find the most elegant and correct solution.\n"

prompt7 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       Think step by step in comments before each code line below."

prompt8 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       You must elaborate your thinking in comments below."

prompt9 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
"""      Is this a simple math or algebra problem? For algebra problems, you must elaborate and solve it algebraically in the comments first, then write code to calculate the result. For simple math problems, you can write code to calculate the result directly.\n"""

prompt10 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
"""    First, let's solve this problem using pure symbolic expressions. Elaborate with your algebraic skills below. Use x,y,z...to denote unknown variables. Use a,b,c... to denote given constants. Then write a pure python code to compute and return the result.\n    Let x be"""


def sample(model, qn, tokenizer, device, sample_len, temperature = 0.05, top_p = 0.1): #temps higher than 2 do not work well
    if USE_VLLM:
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, presence_penalty=0.1, frequency_penalty=0, max_tokens=len(qn) + sample_len)
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

check = """def problem() -> int:
    \"\"\"Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

       Elaborate your thinking step by step in comments before each code line below.
    \"\"\"
    house_price = 80000
    repair_cost = 50000
    increase_percentage = 150
    
    # Calculate the increase in value of the house
    increase_in_value = (house_price + repair_cost) * increase_percentage / 100
    
    # Calculate the total value of the house after repairs
    total_value = house_price + repair_cost + increase_in_value
    
    # Calculate the profit made by Josh
    profit = total_value - (house_price + repair_cost)
    
    result = profit
    return result
    
    #Task: Identify logic issues and write code to fix: 
"""
batch = [check]
sol = sample(model, batch, tokenizer, device, 1000, temperature=0.05, top_p=0.1)

if __name__ == '__main__':
    logger.add("gsm8k_test_set_e5.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    examples = get_examples("test")
    logger.info("Loaded %d examples" % len(examples))
    logger.info("Using prompt: %s" % prompt)
    correct = 0
    total = 0
    device = "cuda:0"
    start_time = time.time()
    batch = []
    answers = []
    for i in range(64):
        batch.append(prompt % examples[i]["question"])
        answers.append(examples[i]["answer"])
        if len(batch) == 64:
            solutions = sample(model, batch, tokenizer, device, 500) #solve(qn)
            for qn, solution, ds_answer in zip(batch, solutions, answers):
                answer = compute_result(solution)
                try:
                    correct_answer = int(ds_answer.split("####")[1].split("<|endoftext|>")[0].strip())
                except ValueError:
                    correct_answer = -88888
                if answer == correct_answer:
                    correct += 1
                total += 1
                logger.info("Total processed: %d, percent correct: %.3f" % (total, 100.0*correct / total))
                if answer != correct_answer:
                    logger.info(f"---Dataset answer: {correct_answer} != LLM answer: {answer}\n" + \
                                "---Dataset solution: %s \n" % ds_answer + \
                                f"---LLM solution: {solution}")
            batch = []
            answers = []
    logger.info("Accuracy: %.3f%%" % (100*correct/total))
    logger.info("Total time: %.1f minutes" % ((time.time() - start_time)/60))
    logger.info("Solutions generated per minute: %.1f" % (total / (time.time() - start_time) * 60))
    logger.info("Used prompt: %s" % prompt)



