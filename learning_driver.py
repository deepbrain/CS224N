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


class ModelManager:
    #maintains a list of local problems and solutions
    #generates solutions for each problem and uploads them to github
    def __init__(self, model_id):
        self.batch_size = 64 #inference batch size
        self.test_problems = get_examples("test")
        self.train_problems = get_examples("train")
        self.MathLLM = MathLLM(model_id, use_vllm=True, load=True)
        self.queue = deque()


    def create_problems(self):
        self.problems = []
        # select 1000 random problems from the training set
        selected_problems = np.random.choice(self.train_problems, 1000, replace=False)
        self.problems = []
        for problem in selected_problems:
            question = problem['question']
            ground_answer = problem['answer']
            try:
                ground_numeric_answer = int(ground_answer.split("####")[1].split("<|endoftext|>")[0].strip())
            except ValueError:
                ground_numeric_answer = -88888
            p = Problem(self, question, ground_answer, ground_numeric_answer)
            self.problems.append(p)

    async def generate_solutions(self, solution_class): #generates multiple solutions for each problem by varying prompts
        tasks = [asyncio.create_task(problem.solve(solution_class)) for problem in self.problems]
        results = await asyncio.gather(*tasks) # produces a list of lists of correctness values like [True, False, True]
        # convert to numpy array
        correctness = np.array(results)
        # calculate the accuracy
        accuracy = np.mean(correctness, axis=0)
        logger.info(f"Accuracy first prompt: {accuracy[0]:.2f}, majority prompt: {accuracy[1]:.2f}, best prompt: {accuracy[2]:.2f}")
        train_samples = [problem.get_train_sample() for problem in self.problems]
        return train_samples


    async def solve(self, prompt, max_tokens = 500, completion_only=False):
        result_queue = asyncio.Queue(1)
        params = {"prompt": prompt, "max_tokens": max_tokens, "completion_only": completion_only, "result_queue": result_queue}
        self.queue.append(params)
        return await result_queue.get()

    async def run(self):
        asyncio.create_task(model_manager.process_queue())
        while True:
            self.create_problems()
            train_samples = await self.generate_solutions(Solution)
            self.upload_solutions(train_samples)

    def upload_solutions(self, train_samples):
        #append train file with new samples
        with open("train.csv", "a") as f:
            for sample in train_samples:
                f.write(sample + "\n")

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
