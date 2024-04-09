import os
if 'LIBRARY_ROOTS' in os.environ:
    del os.environ['LIBRARY_ROOTS']
import time
from math_llm import MathLLM
from tokenized_dataset import TokenizedQADataset
import pandas as pd
from multiprocessing import Process, Queue
import multiprocessing as mp
from loguru import logger
import random
import json

def get_prompt_response(problem, sol1, sol2, analysis_sol1, analysis_sol2, comp, better_id):
    primer = "You are a math professor. You will be comparing two responses to a math problem, and determining how they are different, and why one is better.\n"
    problem = f"The problem is:\n{problem.rstrip()}\n"
    sol1 = f"Solution 1:\n{sol1.rstrip()}\n"
    sol2 = f"Solution 1:\n{sol2.rstrip()}\n"
    instruction = f"Find the reason why the solutions results differ. Which one solution is correct? Structure your respose as follows:\n"
    structure = '{"Analysis of solution 1": "Analyze solution 1 here.", "Analysis of solution 2": "Analyze solution 2 here.", "Comparison of solutions": "Compare both solutions, determine which solution is better, and why.", "Better solution ID": This will be 1 or 2.}'
    prompt = primer + problem + sol1 + sol2 + instruction + structure

    response = f'{{"Analysis of solution 1": "{analysis_sol1}", "Analysis of solution 2": "{analysis_sol2}", "Comparison of solutions": "{comp}", "Better solution ID": {better_id}}}'
    return prompt, response

def parse_input_file(filename):
    res = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            sample = json.loads(line.strip())
            res.append(sample)
    return res

def get_comparator_dataset():
    parsed_lines = parse_input_file("../finetune-test.json")
    dataset = []
    for sample in parsed_lines:
        dataset.append(get_prompt_response(sample["Question"], sample["Solution 1"], sample["Solution 2"], sample["Analysis of solution 1"], sample["Analysis of solution 2"], sample["Comparison of solutions"], sample["Better solution ID"]))

    perc_train = 0.9
    train_samples = dataset[:int(len(dataset)*perc_train)]
    eval_samples = dataset[int(len(dataset)*perc_train):]

    random.seed(123)
    random.shuffle(train_samples)
    random.shuffle(eval_samples)
    return train_samples, eval_samples

def train(model_id, epoch, MPqueue):
    logger.add("learning.log", rotation = "100 MB")
    rephrase_llm = MathLLM(
        model_id=model_id,
        use_vllm=True,
        load=False,
        dataset_class=TokenizedQADataset
    )
    iteration = time.strftime("%Y%m%d-%H%M%S")
    train_samples, eval_samples = get_comparator_dataset()
    train_eos = epoch <= 1 # train for the first 2 epochs
    logger.info(f"Train eos: {train_eos}")
    rephrase_llm.train(train_samples, eval_samples, 'comparator-phi-'+iteration, lr = 1e-4, merge = True, train_eos=train_eos)
    MPqueue.put(rephrase_llm.model_id)

def multiprocessing_training(model_id, epoch, GPU=-1):
    MPqueue = Queue()
    if GPU != -1:
        bkp = os.environ["CUDA_VISIBLE_DEVICES"]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    p = Process(target=train, args=(model_id, epoch, MPqueue))
    p.start()
    p.join()
    if GPU != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = bkp
    return MPqueue.get()


def train_multiple_epochs(epochs):
    model_id = "microsoft/phi-2"
    for epoch in range(epochs):
        logger.info(f"Doing epoch {epoch}, based on {model_id}")
        model_id = multiprocessing_training(model_id, epoch)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    logger.add("learning.log", rotation = "100 MB")
    model_id1 = train_multiple_epochs(6)