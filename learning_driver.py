#prevent the local variables from being imported into the remote environment as they can cuase crashes
from multiprocessing import Process, Queue
import multiprocessing as mp
import os
if 'LIBRARY_ROOTS' in os.environ:
    del os.environ['LIBRARY_ROOTS']
# adjust this to the GPU you want to use:
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from loguru import logger
import logging
from dataset import get_examples, GSMDataset
from tokenized_dataset import TokenizedQADataset
from math_problem import Problem, Solution
from math_llm import MathLLM, BASE_PHI_REVISION
import numpy as np
import asyncio
from collections import deque
import time
from prompt import get_old_prompts, Prompt, compute_oracle_accuracy, compute_majority_answer_accuracy, compute_mean_accuracy, find_unsolvable_problems
import json
import random


class ModelManager:
    def __init__(self, model_id, start_from = 0, num_samples = -1, inference_batch_size=32, problem_batch_size=32):
        self.inference_batch_size = inference_batch_size #inference batch size
        self.batch_size = problem_batch_size
        self.test_problems = get_examples("test")
        self.train_problems = get_examples("train")
        if num_samples == -1:
            num_samples = len(self.train_problems) // 2
        self.num_samples = num_samples
        self.model_id = model_id
        self.MathLLM = MathLLM(model_id, use_vllm=True, load=False, dataset_class=TokenizedQADataset) #revision = BASE_PHI_REVISION
        self.queue = deque()
        self.shuffle_and_batch(start_from, num_samples)
        self.prompts = get_old_prompts()
        self.problems = []
        self.test = False
        self.learning_iteration = 0
        self.code_interpreter_lock = asyncio.Lock()


    def shuffle_and_batch(self, start_from, num_samples):
        logger.info(f"Batching {num_samples} samples starting from {start_from}")
        train_problems = self.train_problems[start_from:start_from+num_samples]
        self.batches = [train_problems[i:i + self.batch_size] for i in range(0, len(train_problems), self.batch_size)]
        self.batch_index = 0
        self.test_batches = [self.test_problems[i:i + self.batch_size] for i in range(0, len(self.test_problems), self.batch_size)]
        self.test_batch_index = 0
        return self.batches

    async def create_problems(self):
        if self.test:
            if self.test_batch_index >= len(self.test_batches):
                return []
            selected_problems = self.test_batches[self.test_batch_index]
            self.test_batch_index += 1
        else:
            if self.batch_index >= len(self.batches):
                return []
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
        tasks = [asyncio.create_task(problem.solve(self.prompts, solution_class, self.test)) for problem in problems]
        results = await asyncio.gather(*tasks) # produces a list of lists of correctness values like [True, False, True]
        mean_accuracy = compute_mean_accuracy(self.problems, self.prompts)
        majority_accuracy = compute_majority_answer_accuracy(self.problems, self.prompts)
        oracle = compute_oracle_accuracy(self.problems, self.prompts)
        unsolvable_problems = find_unsolvable_problems(self.problems, self.prompts)
        if self.test:
            test = "test " + str(self.learning_iteration)
        else:
            test = "train " + str(self.learning_iteration)
        logger.info(f"{test} all prompts mean accuracy: {mean_accuracy:.2f}, Majority accuracy: {majority_accuracy:.2f}, Oracle accuracy: {oracle:.2f}, problems solved: {len(self.problems)}")
#        for rephrasing in range(0, 5):
#            mean_accuracy = compute_mean_accuracy(self.problems, self.prompts, rephrasing)
#            majority_accuracy = compute_majority_answer_accuracy(self.problems, self.prompts, rephrasing)
#            oracle = compute_oracle_accuracy(self.problems, self.prompts, rephrasing)
#            logger.info(f"Rephrasing{rephrasing} mean accuracy: {mean_accuracy:.2f}, Majority accuracy: {majority_accuracy:.2f}, Oracle accuracy: {oracle:.2f}, problems solved: {len(self.problems)}")

        prompt_accuracies = test
        for i, prompt in enumerate(self.prompts):
            prompt_accuracies += f" prompt{i}: {prompt.compute_accuracy():.2f},"
        logger.info(prompt_accuracies)
        if self.test:
            return [], []
        train_samples = []
        for problem in problems:
            train_sample = problem.get_train_samples()
            if train_sample is not None:
                train_samples.extend(train_sample)
        all_samples = []
        for problem in problems:
            all_samples.extend(problem.get_all_samples())
        return train_samples, all_samples


    async def get_completion(self, prompt, max_tokens = 1000, temperature = 0.3, completion_only=False):
        result_queue = asyncio.Queue(1)
        params = {"prompt": prompt, "max_tokens": max_tokens, "completion_only": completion_only, "result_queue": result_queue, "temperature": temperature}
        self.queue.append(params)
        return await result_queue.get()

    async def run_test(self):
        self.lock = asyncio.Lock()
        for prompt in self.prompts:
            prompt.reset_stats()
        self.problems = []
        logger.info(f"Testing model {self.MathLLM.model_id}")
        #self.MathLLM.evaluate()
        self.test = True
        self.test_batch_index = 0
        while True:
            problems = await self.create_problems()
            if len(problems) == 0:
                break
            await self.generate_solutions(Solution, problems)
        logger.info(f"Testing complete for model{self.MathLLM.model_id}")
        for prompt in self.prompts:
            prompt.reset_stats()
        self.problems = []
        self.test = False


    async def run_inference(self, iteration, do_test): #this code will execute in separate process
        self.lock = asyncio.Lock()
        logger.info(f"Starting inference loop for model {self.MathLLM.model_id} revision {str(self.MathLLM.revision)}")
        asyncio.create_task(self.process_queue())
        self.learning_iteration = iteration
        if do_test:
            await self.run_test()
        while True:
            problems = await self.create_problems()
            if len(problems) == 0:
                break
            train_samples, all_samples = await self.generate_solutions(Solution, problems)
            self.upload_solutions(train_samples, all_samples)

    def spawn_inference(self, start_from, num_samples, iteration, do_test, GPU=-1):
        MPqueue = Queue()
        if GPU != -1:
            bkp = os.environ["CUDA_VISIBLE_DEVICES"]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
        p = Process(target=run_inference, args=(self.model_id, start_from, num_samples, iteration, do_test, MPqueue))
        p.start()
        p.join()
        if GPU != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = bkp
        return MPqueue.get()

    def spawn_training(self, GPU=-1):
        MPqueue = Queue()
        if GPU != -1:
            bkp = os.environ["CUDA_VISIBLE_DEVICES"]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
        p = Process(target=run_training, args=(self.model_id, MPqueue))
        p.start()
        p.join()
        if GPU != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = bkp
        return MPqueue.get()


    def run(self):
        self.lock = asyncio.Lock()
        start_from = 0
        inference_batch_size = self.num_samples
        for i in range(100):
            do_test = True#i != 0
            self.spawn_inference(start_from, num_samples=inference_batch_size, iteration=i, do_test=do_test, GPU=-1)
            start_from += inference_batch_size
            if start_from > (len(self.train_problems) - inference_batch_size):
                start_from = 0
            self.model_id = self.spawn_training()

    def upload_solutions(self, train_samples, all_samples, filename = "train_samples.txt", all_filename = "all_samples.txt"):
        with open(filename, 'a', encoding='utf-8') as file:
            for sample in train_samples:
                serialized_sample = json.dumps(sample, ensure_ascii=False)
                file.write(serialized_sample + "\n")
        with open(all_filename, 'a', encoding='utf-8') as file:
            for sample in all_samples:
                serialized_sample = json.dumps(sample, ensure_ascii=False)
                file.write(serialized_sample + "\n")

    def load_solutions(self, filename="train_samples.txt", max_samples=2000):
        train_samples = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                sample = json.loads(line.strip())
                train_samples.append(sample)
                if len(train_samples) >= max_samples:
                    break
        return train_samples
    def archive_solutions(self, filename="train_samples.txt", all_filename="all_samples.txt"):
        if os.path.exists(filename):
            os.rename(filename, filename + '-' + self.iteration)
        if os.path.exists(all_filename):
            os.rename(all_filename, all_filename + '-' + self.iteration)

    def train(self):
        train_samples = self.load_solutions(max_samples = 1000000)
        eval_samples = train_samples[len(train_samples)-self.batch_size*2:]
        train_samples = train_samples[:len(train_samples)-self.batch_size*2]
        self.iteration = time.strftime("%Y%m%d-%H%M%S")
        model_id = self.MathLLM.model_id
        self.MathLLM.unload_model()
        del self.MathLLM
        logger.info(f"Training model {model_id} from iteration {self.iteration}")
        self.MathLLM = MathLLM(model_id, use_vllm=True, load=False, dataset_class=TokenizedQADataset)
        self.MathLLM.train(train_samples, eval_samples, 'trained_iter_' + self.iteration, lr = 2e-5, merge = True, train_eos=model_id == "microsoft/phi-2", use_dpo = True)
        self.archive_solutions()

    async def process_queue(self):
        while True:
            if len(self.queue) > 0:
                batch_prompts = []
                batch_max_tokens = []
                batch_completion_only = []
                batch_result_queue = []
                batch_temperature = None
                count = 0
                while len(self.queue) > 0:
                    params = self.queue.popleft()
                    T = params["temperature"]
                    if batch_temperature is None:
                        batch_temperature = T
                    elif T != batch_temperature:
                        self.queue.appendleft(params)
                        break
                    batch_prompts.append(params["prompt"])
                    batch_max_tokens.append(params["max_tokens"])
                    batch_completion_only.append(params["completion_only"])
                    batch_result_queue.append(params["result_queue"])
                    count += 1
                    if count == self.inference_batch_size:
                        break
                solutions = await self.MathLLM.process_batch_async(batch_prompts, max(batch_max_tokens), temperature = batch_temperature, top_p=0.6, presence_penalty=0.2, frequency_penalty = 0.2)
                for i in range(count):
                    sol = solutions[i]
                    if batch_completion_only[i]:
                        sol = sol[len(batch_prompts[i]):]
                    await batch_result_queue[i].put(sol)
            else:
                await asyncio.sleep(0.1)

def run_inference(model_name, start_from, num_samples, iteration, do_test, MPqueue):
    logger.add("learning.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    model_manager = ModelManager(model_name, start_from, num_samples)
    asyncio.run(model_manager.run_inference(iteration, do_test))
    MPqueue.put("done")

def run_training(model_name, MPqueue):
    logger.add("learning.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    model_manager = ModelManager(model_name)
    logger.info(f"Training model {model_manager.MathLLM.model_id}")
    model_manager.train()
    MPqueue.put(model_manager.MathLLM.model_id)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    logger.add("learning.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    model_manager = ModelManager("trained_iter_20240222-154927", num_samples = 1024) #("trained_iter_20240220-235255", num_samples=1024)
    #run the process_queue method in the background
    asyncio.run(model_manager.run())
