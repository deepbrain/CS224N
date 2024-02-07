#prevent the local variables from being imported into the remote environment as they can cuase crashes
import os
if 'LIBRARY_ROOTS' in os.environ:
    del os.environ['LIBRARY_ROOTS']
# adjust this to the GPU you want to use:
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from loguru import logger
import logging
from dataset import get_examples, GSMDataset
from math_problem import Problem, Solution
from math_llm import MathLLM
import numpy as np
import asyncio
from collections import deque
import time
from prompt import get_old_prompts, Prompt, compute_oracle_accuracy, compute_majority_answer_accuracy, compute_mean_accuracy, find_unsolvable_problems
import json
import random


class ModelManager:
    def __init__(self, model_id, batch_size=64, problem_batch_size=32):
        self.batch_size = batch_size #inference batch size
        self.test_problems = get_examples("test")
        self.train_problems = get_examples("train")
        self.MathLLM = MathLLM(model_id, use_vllm=True, load=True)
        self.queue = deque()
        self.shuffle_and_batch(problem_batch_size)
        self.prompts = get_old_prompts()
        self.problems = []

    def shuffle_and_batch(self, batch_size):
        # Shuffle the list of problems to ensure randomness
        random.shuffle(self.train_problems)
        # Split the list into batches
        self.batches = [self.train_problems[i:i + batch_size] for i in range(0, len(self.train_problems), batch_size)]
        self.batch_index = 0
        return self.batches

    async def create_problems(self):
        selected_problems = self.batches[self.batch_index]
        self.batch_index += 1
        problems = []
        for problem in selected_problems:
            question = problem['question']
            ground_answer = problem['answer']
            try:
                ground_numeric_answer = int(ground_answer.split("####")[1].split("<|endoftext|>")[0].strip())
            except ValueError:
                ground_numeric_answer = -88888
            p = Problem(self, question, ground_answer, ground_numeric_answer)
#            rephrased_problems = await p.rephrase()
            problems.extend([p])
        self.problems.extend(problems)
        return problems


    async def generate_solutions(self, solution_class, problems): #generates multiple solutions for each problem by varying prompts
        tasks = [asyncio.create_task(problem.solve(self.prompts, solution_class)) for problem in problems]
        results = await asyncio.gather(*tasks) # produces a list of lists of correctness values like [True, False, True]
        mean_accuracy = compute_mean_accuracy(self.problems, self.prompts)
        majority_accuracy = compute_majority_answer_accuracy(self.problems, self.prompts)
        oracle = compute_oracle_accuracy(self.problems, self.prompts)
        unsolvable_problems = find_unsolvable_problems(self.problems, self.prompts)
        logger.info(f"Overall mean accuracy: {mean_accuracy:.2f}, Majority accuracy: {majority_accuracy:.2f}, Oracle accuracy: {oracle:.2f}, problems solved: {len(self.problems)}")
#        for rephrasing in range(0, 5):
#            mean_accuracy = compute_mean_accuracy(self.problems, self.prompts, rephrasing)
#            majority_accuracy = compute_majority_answer_accuracy(self.problems, self.prompts, rephrasing)
#            oracle = compute_oracle_accuracy(self.problems, self.prompts, rephrasing)
#            logger.info(f"Rephrasing{rephrasing} mean accuracy: {mean_accuracy:.2f}, Majority accuracy: {majority_accuracy:.2f}, Oracle accuracy: {oracle:.2f}, problems solved: {len(self.problems)}")

        prompt_accuracies = ''
        for i, prompt in enumerate(self.prompts):
            prompt_accuracies += f"prompt{i}: {prompt.compute_accuracy():.2f}, "
        logger.info(prompt_accuracies)
        train_samples = [problem.get_train_sample() for problem in self.problems]
        return train_samples


    async def get_completion(self, prompt, max_tokens = 1000, completion_only=False):
        result_queue = asyncio.Queue(1)
        params = {"prompt": prompt, "max_tokens": max_tokens, "completion_only": completion_only, "result_queue": result_queue}
        self.queue.append(params)
        return await result_queue.get()

    async def run(self):
        asyncio.create_task(model_manager.process_queue())
        while True:
            problems = await self.create_problems()
            train_samples = await self.generate_solutions(Solution, problems)
            self.upload_solutions(train_samples)

    def upload_solutions(self, train_samples, filename = "train_samples.txt"):
        with open(filename, 'a', encoding='utf-8') as file:
            for sample in train_samples:
                serialized_sample = json.dumps(sample, ensure_ascii=False)
                file.write(serialized_sample + "\n")

    def load_solutions(self, filename="train_samples.txt"):
        train_samples = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                sample = json.loads(line.strip())
                train_samples.append(sample)
        return train_samples

    async def process_queue(self):
        while True:
            if len(self.queue) > 0:
                batch_prompts = []
                batch_max_tokens = []
                batch_completion_only = []
                batch_result_queue = []
                count = 0
                while len(self.queue) > 0:
                    params = self.queue.popleft()
                    batch_prompts.append(params["prompt"])
                    batch_max_tokens.append(params["max_tokens"])
                    batch_completion_only.append(params["completion_only"])
                    batch_result_queue.append(params["result_queue"])
                    count += 1
                    if count == self.batch_size:
                        break
                solutions = await self.MathLLM.process_batch_async(batch_prompts, max(batch_max_tokens))
                for i in range(count):
                    sol = solutions[i]
                    if batch_completion_only[i]:
                        sol = sol[len(batch_prompts[i]):]
                    await batch_result_queue[i].put(sol)
            else:
                await asyncio.sleep(0.1)


if __name__ == '__main__':
    logger.add("learning.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    model_manager = ModelManager("microsoft/phi-2")
    #run the process_queue method in the background
    asyncio.run(model_manager.run())
