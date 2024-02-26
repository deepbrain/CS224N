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


def get_assistant_dataset():
    prompts_dataset = list(pd.read_pickle("prompt_dataset.pkl").items())
    problem_dataset = list(pd.read_pickle("problem_1k_dataset.pkl").items())

    def process_dataset(ds, prompt):
        questions_answers = []
        for problem, rephrases in ds:
            problem = f"{prompt} {problem} "
            for rephrase in rephrases:
                questions_answers.append((problem, rephrase))
        return questions_answers

    perc_train = 0.9
    train_prompts_dataset = prompts_dataset[:int(len(prompts_dataset)*perc_train)]
    eval_prompts_dataset = prompts_dataset[int(len(prompts_dataset)*perc_train):]
    train_prompts_dataset = process_dataset(train_prompts_dataset, "Rephrase the following prompt:")
    eval_prompts_dataset = process_dataset(eval_prompts_dataset, "Rephrase the following prompt:")

    perc_train = 0.9
    train_problem_dataset = problem_dataset[:int(len(problem_dataset)*perc_train)]
    eval_problem_dataset = problem_dataset[int(len(problem_dataset)*perc_train):]
    train_problem_dataset = process_dataset(train_problem_dataset, "Rephrase the following problem:")
    eval_problem_dataset = process_dataset(eval_problem_dataset, "Rephrase the following problem:")

    train_samples = train_prompts_dataset + train_problem_dataset
    eval_samples = eval_prompts_dataset + eval_problem_dataset

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
    train_samples, eval_samples = get_assistant_dataset()
    train_eos = epoch <= 1 # train for the first 2 epochs
    rephrase_llm.train(train_samples, eval_samples, 'rephrase-phi-'+iteration, lr = 1e-4, merge = True, train_eos=model_id == train_eos)
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