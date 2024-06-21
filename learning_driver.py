# (C) 2024 Stanford CS224N Group Custom Project by Artyom Shaposhnikov, Shubhra Mishra, Roberto Garcia

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
from code_interpreter import INVALID_ANSWER

class ModelManager:
    def __init__(self, model_id, model_type, start_from = 0, num_samples = -1, inference_batch_size=32, problem_batch_size=32, method = 'cross'):
        self.method = method
        self.inference_batch_size = inference_batch_size #inference batch size
        self.batch_size = problem_batch_size
        self.test_problems = get_examples("test")
        self.train_problems = get_examples("train")
        if num_samples == -1:
            num_samples = len(self.train_problems) // 2
        self.num_samples = num_samples
        self.model_id = model_id
        self.model_type = model_type
        self.MathLLM = MathLLM(model_id, model_type, use_vllm=False, load=False, dataset_class=TokenizedQADataset) #revision = BASE_PHI_REVISION
        self.queue = deque()
        self.shuffle_and_batch(start_from, num_samples)
        self.prompts = get_old_prompts()
        self.problems = []
        self.learning_iteration = 0
        self.code_interpreter_lock = asyncio.Lock()

    def add_train_sample(self, dict):
        self.train_samples.append(dict)

    def add_all_sample(self, dict):
        self.all_samples.append(dict)

    def add_problem_sample(self, problem):
        self.problem_samples.append(problem)

    def compute_train_sample_accuracy(self, problems):
        total = 0
        correct = 0
        for p in problems:
            if p.is_learning_sample:
                total += 1
                if p.learning_sample_correct:
                    correct += 1
        return correct, total

    def load_rephrased_problems(self):
        self.rephrased_problems = []
        problems_dict = {}
        q1 = ""
        for p in self.train_problems:
            if p['question'] not in problems_dict:
                problems_dict[p['question'].strip()] = p
        self.rephrased_problems = []
        if self.model_type == 'mistral':
            LT = "mistral_problem_samples_rephrases_low_temp.txt"
            HT = "mistral_problem_samples_rephrases_high_temp.txt"
        else:
            LT = "problem_samples_rephrases_low_temp.txt"
            HT = "problem_samples_rephrases_high_temp.txt"
        lines = []
        with open(LT, 'r', encoding='utf-8') as file:
            for line in file:
                lines.append(line)
        with open(HT, 'r', encoding='utf-8') as file:
            for line in file:
                lines.append(line)
        for line in lines:
            js = json.loads(line)
            question = js['problem']
            problem = problems_dict[question.strip()]

            # Collect existing rephrases to ensure uniqueness
            existing_rephrases = {problem[key] for key in problem if key.startswith('rephrase')}

            # Identify the highest existing rephrase key number in the problem dictionary
            highest_number = -1
            for key in problem.keys():
                if key.startswith('rephrase'):
                    num = int(key.replace('rephrase', ''))
                    highest_number = max(highest_number, num)

            # Extend the problem dict with the rephrased problems in js
            for key, value in js.items():
                if key.startswith('rephrase') and value not in existing_rephrases:
                    # Since this is a new unique rephrase, add it to existing_rephrases set
                    existing_rephrases.add(value)

                    # Check if this rephrase key is already present, if so, find a new unique key
                    if key in problem:
                        highest_number += 1
                        new_key = f'rephrase{highest_number}'
                        problem[new_key] = value
                    else:
                        # This ensures even if the rephrase key from js isn't directly used, it's unique
                        highest_number += 1
                        new_key = f'rephrase{highest_number}'
                        problem[new_key] = value

        for problem in problems_dict:
            self.rephrased_problems.append(problems_dict[problem])


    def shuffle_and_batch(self, start_from, num_samples):
        if self.method == 'rephrase' or self.method == 'rephrase_test':
            self.load_rephrased_problems()
            self.batches = [self.rephrased_problems[i:i + self.batch_size] for i in range(0, len(self.rephrased_problems), self.batch_size)]
            self.test_batches = self.batches
            self.test_batch_index = 0
            self.batch_index = 0
            return self.batches
        else:
            logger.info(f"Batching {num_samples} samples starting from {start_from}")
            train_problems = self.train_problems[start_from:start_from+num_samples]
            self.batches = [train_problems[i:i + self.batch_size] for i in range(0, len(train_problems), self.batch_size)]
        self.batch_index = 0
        self.test_batches = [self.test_problems[i:i + self.batch_size] for i in range(0, len(self.test_problems), self.batch_size)]
        self.test_batch_index = 0
        return self.batches

    async def create_problems(self):
        if self.method == 'test':
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
            p = Problem(self, question, ground_answer, ground_numeric_answer, use_dpo = False, rephasings=problem)
#            rephrased_problems = await p.rephrase()
            problems.extend([p])
        self.problems.extend(problems)
        return problems


    async def generate_solutions(self, solution_class, problems, dataset_problems): #generates multiple solutions for each problem by varying prompts
        tasks = [asyncio.create_task(problem.solve(self.prompts, solution_class, self.method)) for problem in problems]
        results = await asyncio.gather(*tasks) # produces a list of lists of correctness values like [True, False, True]
        if dataset_problems:
            mean_accuracy = compute_mean_accuracy(self.problems, self.prompts)
            majority_accuracy = compute_majority_answer_accuracy(self.problems, self.prompts)
            oracle = compute_oracle_accuracy(self.problems, self.prompts)
            unsolvable_problems = find_unsolvable_problems(self.problems, self.prompts)
            if self.method == 'test':
                test = self.method + str(self.learning_iteration)
            else:
                test = self.method + str(self.learning_iteration)
            logger.info(f"{test} all prompts mean accuracy: {mean_accuracy:.4f}, Majority accuracy: {majority_accuracy:.4f}, Oracle accuracy: {oracle:.4f}, problems solved: {len(self.problems)}")
#        for rephrasing in range(0, 5):
#            mean_accuracy = compute_mean_accuracy(self.problems, self.prompts, rephrasing)
#            majority_accuracy = compute_majority_answer_accuracy(self.problems, self.prompts, rephrasing)
#            oracle = compute_oracle_accuracy(self.problems, self.prompts, rephrasing)
#            logger.info(f"Rephrasing{rephrasing} mean accuracy: {mean_accuracy:.2f}, Majority accuracy: {majority_accuracy:.2f}, Oracle accuracy: {oracle:.2f}, problems solved: {len(self.problems)}")

            prompt_accuracies = test
            for i, prompt in enumerate(self.prompts):
                prompt_accuracies += f" prompt{i}: {prompt.compute_accuracy():.4f},"
            logger.info(prompt_accuracies)
        else:
            logger.info(f"Generated {len(self.train_samples)} train samples")
            for sample in self.train_samples:
                logger.info(f"Prompt: {sample['prompt']}, Solution: {sample['solution']}")


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
#        self.MathLLM.evaluate()
        self.test = True
        method = self.method
        self.method = 'test'
        self.test_batch_index = 0
        while True:
            problems = await self.create_problems()
            if len(problems) == 0:
                break
            await self.generate_solutions(Solution, problems, True)
        logger.info(f"Testing complete for model{self.MathLLM.model_id}")
        for prompt in self.prompts:
            prompt.reset_stats()
        self.method = method
        self.problems = []
        self.test = False

    def generate_problem_prompt(self):
        max_problems = len(self.problem_samples)
        #choose 3 random problems between 0 and max_problems
        problems = []
        cnt = 3
        while len(problems) < cnt:
            p = self.problem_samples[random.randint(0, max_problems-1)]
            if p not in problems:
                problems.append(p)
        prompt = ''
        for p in problems:
            # remove newlines from p
            p = p.replace('\n', ' ')
            prompt += 'Problem: ' + p + '\n'
        prompt += 'Problem: '
        return prompt

    async def generate_problems(self, problem_filename = "problem_samples.txt"):
        self.problem_samples = []
        with open(problem_filename, 'r', encoding='utf-8') as file:
            for line in file:
                problem = line
                self.problem_samples.append(problem)
        tasks = []
        prompts = []
        for i in range(48):
            prompt = self.generate_problem_prompt()
            prompts.append(prompt)
            task = asyncio.create_task(self.get_completion(prompt, max_tokens=200, temperature=0.3, completion_only=True))
            tasks.append(task)
        completions = await asyncio.gather(*tasks)
        problems = []
        problem_set = []
        for completion in completions:
            problem_str = completion.split('\n')[0]
            #check if 'Solition' is in the completion
            if (len(problem_str) > 0) and (not ('Solution' in problem_str)):
                problem = Problem(self, problem_str, "INVALID_ANSWER", -88888)
                problems.append(problem)
                problem_set.append(problem_str)
        print(problem_set)
        self.problem_samples = []
        return problems

    async def run_inference(self, iteration, do_test): #this code will execute in separate process
        #randomize the random seed
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        #randomize the random seed
        random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        self.lock = asyncio.Lock()
        self.train_samples = []
        self.all_samples = []
        self.problems = []
        self.problem_samples = []

        logger.info(f"Starting inference loop for model {self.MathLLM.model_id} revision {str(self.MathLLM.revision)}")
        asyncio.create_task(self.process_queue())
        self.learning_iteration = iteration
        if do_test:
            await self.run_test()
        correct = 0
        total = 0
        dataset_problems = True
        total_len = 0
#        problems = await self.generate_problems()
        todo_problem_batches = len(self.test_batches) if self.method == "test" else len(self.batches)
        processed_batches = 0
        logger.info(f"Running inference on {todo_problem_batches} problem batches")
        while True:
            s_time = time.time()
            if dataset_problems:
                problems = await self.create_problems()
                if len(problems) == 0:
                    dataset_problems = False
                    self.upload_solutions()
                    break
            await self.generate_solutions(Solution, problems, dataset_problems)
            if dataset_problems:
                c, t = self.compute_train_sample_accuracy(problems)
                total_len += len(self.train_samples)
                correct += c
                total += t
            self.upload_solutions()
            processed_batches += 1 
            e_time = time.time() 
            logger.info(f"Done inference on  {processed_batches}/{todo_problem_batches} problem batches, took {e_time-s_time} seconds")
            if total > 0:
                logger.info(f"Chosen train samples accuracy: {correct}/{total} = {correct/total:.4f}")


    def spawn_inference(self, start_from, num_samples, iteration, do_test, GPU=-1):
        MPqueue = Queue()
        if GPU != -1:
            bkp = os.environ["CUDA_VISIBLE_DEVICES"]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
        p = Process(target=run_inference, args=(self.model_id, self.model_type, start_from, num_samples, iteration, do_test, self.method, MPqueue))
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
        p = Process(target=run_training, args=(self.model_id, self.model_type, self.method, MPqueue))
        p.start()
        p.join()
        if GPU != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = bkp
        self.model_id = MPqueue.get()
        if not self.model_id.endswith("-lora"):
            self.model_id += "-lora"
        self.model_id = self.spawn_merging()
        return self.model_id

    def spawn_merging(self, GPU=-1):
        #sleep for 5 seconds
        if not self.model_id.endswith("-lora"):
            return
        time.sleep(5)
        MPqueue = Queue()
        if GPU != -1:
            bkp = os.environ["CUDA_VISIBLE_DEVICES"]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
        p = Process(target=run_merging, args=(self.model_id, MPqueue))
        p.start()
        p.join()
        if GPU != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = bkp
        self.model_id = MPqueue.get()
        return self.model_id


    def run(self):
        self.lock = asyncio.Lock()
        start_from = 0
        inference_batch_size = self.num_samples
        for i in range(3):
            do_test = i != 0
            self.spawn_merging()
            self.spawn_inference(start_from, num_samples=inference_batch_size, iteration=i, do_test=do_test, GPU=-1)
            if self.method == 'test':
                return
            start_from += inference_batch_size
            if start_from > (len(self.train_problems) - inference_batch_size):
                start_from = 0
            self.model_id = self.spawn_training()

    def upload_solutions(self, filename = "train_samples.txt", problem_filename = "problem_samples.txt"):
        with open(self.method + '_' + self.model_id + '_' + filename, 'a', encoding='utf-8') as file:
            for sample in self.train_samples:
#                js = {'problem' : sample['problem'], 'prompt': sample['prompt'],  'solution' : sample['solution']}
                serialized_sample = json.dumps(sample, ensure_ascii=False)
                file.write(serialized_sample + "\n")
        with open(self.method + '_' + self.model_id + '_' +problem_filename, 'a', encoding='utf-8') as file:
            for problem in self.problem_samples:
                problem = problem.replace('\n', ' ')
                file.write(problem + "\n")
        with open(self.method + '_' + self.model_id + '_' + "_all_samples.txt", 'a', encoding='utf-8') as file:
            for sample in self.all_samples:
#                js = {'problem' : sample['problem'], 'prompt': sample['prompt'], 'solution' : sample['solution'], 'answer' : str(sample['answer']), 'ground_answer': sample['ground_numeric_answer']}
                try:
                    serialized_sample = json.dumps(sample, ensure_ascii=False)
                    file.write(serialized_sample + "\n")
                #handle all exceptions:
                except Exception as e:
                    logger.info(f"Error serializing sample: {sample}, exception: {e}")
        self.train_samples = []
        self.all_samples = []
        self.problem_samples = []

    def load_solutions(self, filename="train_samples.txt", problem_filename = "problem_samples.txt"):
        train_samples = []
        problems = []
        solutions = []
        prev_problem = None
        def process_line(line):
            nonlocal prev_problem, solutions, train_samples
            sample = json.loads(line.strip())
            problem = sample['problem']
            if prev_problem is not None and problem != prev_problem:
                # select random solution:
                idx = random.randint(0, len(solutions) - 1)
                train_samples.append(solutions[idx])
                solutions = []
                prev_problem = problem
            if prev_problem is None:
                prev_problem = problem
            solutions.append({'solution': sample['solution'], 'prompt': sample['prompt']})

        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                process_line(line)
        with open("train_samples0.txt", 'r', encoding='utf-8') as file:
            for line in file:
                process_line(line)

        with open(problem_filename, 'r', encoding='utf-8') as file:
           for line in file:
              problem = line
              problems.append(problem)

        return train_samples #############################################

        problem_samples = []
        for i in range(3000):
            cnt = 0
            problem_batch = []
            while cnt < 4:
                problem = problems[random.randint(0, len(problems)-1)]
                if problem not in problem_batch:
                    problem_batch.append(problem)
                    cnt += 1
            prompt = ''
            for p in problem_batch[:-1]:
                prompt += 'Problem: ' + p + '\n'
            prompt += 'Problem: '
            problem_samples.append({'prompt': prompt, 'solution': problem_batch[-1]})
        #mix problem_samples and train_samples randomly
        train_samples.extend(problem_samples)
        random.shuffle(train_samples)
        return train_samples


    def load_all_solutions(self, all = False):
        train_samples = []
        problems = {}
        total_samples = 0
        total_problems = 0
        def process_line(line, embedded_prompt = False):
            nonlocal problems, total_samples, total_problems
            sample = json.loads(line.strip())
            if embedded_prompt:
                solution = {'solution': sample['solution'], 'prompt': sample['prompt']}
            else:
                solution = {'solution': sample['solution'], 'prompt': sample['prompt'] % sample['problem']}

            problem = sample['problem'].strip()
            if problem in problems:
                if solution in problems[problem]:
                    return
                else:
                    problems[problem].append(solution)
                    total_samples += 1
            else:
                problems[problem] = [solution]
                total_problems += 1
                total_samples += 1

        with open(self.method  + '_' + self.model_id + '_train_samples.txt', 'r', encoding='utf-8') as file:
            for line in file:
                process_line(line)

        logger.info(f"Loaded {total_samples} samples, {total_problems} problems")

        train_samples = []
        for problem in problems:
            prompts = {}
            for solution in problems[problem]:
                prompt = solution['prompt']
                if prompt in prompts:
                    prompts[prompt].append(solution)
                else:
                    prompts[prompt] = [solution]
            for prompt in prompts:
                solutions = prompts[prompt]
                if all:
                    for solution in solutions:
                        train_samples.append(solution)
                else:
                    if len(solutions) > 1:
                        idx = random.randint(0, len(solutions) - 1)
                        train_samples.append(solutions[idx])
                    else:
                        train_samples.append(solutions[0])
        #randomly shuffle the train samples
        if all:
            #sort the train samples by prompt and solution:
            train_samples = sorted(train_samples, key = lambda x: (x['prompt'], x['solution']))
        else:
            random.shuffle(train_samples)
        return train_samples


    def load_and_filter_all_solutions(self, all = False):
        problems = {}
        total = 0
        with open(self.method  + '_' + self.model_id + '_all_samples.txt', 'r', encoding='utf-8') as file:
            for line in file:
                total += 1
                sample = json.loads(line.strip())
                problem = sample['problem'].strip()
                if problem in problems:
                    problems[problem].append(sample)
                else:
                    problems[problem] = [sample]
        logger.info(f"Loaded {total} samples")
        train_samples = []
        for problem in problems:
            answer_counts = {}
            for sample in problems[problem]:
                a = sample['answer']
                if a != INVALID_ANSWER:
                    if a in answer_counts:
                        answer_counts[a] += 1
                    else:
                        answer_counts[a] = 1
            if len(answer_counts) > 0:
                best_answer = max(answer_counts, key=answer_counts.get)
                # compute percent of best answer in answer_counts:
                percent = answer_counts[best_answer] / sum(answer_counts.values())
                if answer_counts[best_answer] > 25:
                    if self.method == 'cross':
                        prompts = []
                        for sample in problems[problem]:
                            if sample['answer'] != best_answer:
                                if sample['prompt'] not in prompts:
                                    prompts.append(sample['prompt'])
                        if len(prompts) > 0:
                            for sample in problems[problem]:
                                if sample['answer'] == best_answer:
                                    prompt = prompts[random.randint(0, len(prompts)-1)]
                                    train_samples.append({'prompt': prompt % sample['problem'], 'solution': sample['solution']})
                                    break
                    elif self.method == 'temperature':
                        prompts = {}
                        for sample in problems[problem]:
                            if sample['answer'] == best_answer:
                                if sample['prompt'] in prompts:
                                    prompts[sample['prompt']].append(sample)
                                else:
                                    prompts[sample['prompt']] = [sample]
                        if len(prompts) > 0:
                            for prompt in prompts:
                                if all:
                                    for solution in prompts[prompt]:
                                        train_samples.append({'prompt': prompt % solution['problem'], 'solution': solution['solution']})
                                else:
                                    solutions = prompts[prompt]
                                    if len(solutions) > 1:
                                        idx = random.randint(0, len(solutions) - 1)
                                        train_samples.append({'prompt': prompt % solutions[idx]['problem'], 'solution': solutions[idx]['solution']})
                                    elif len(solutions) == 1:
                                        train_samples.append({'prompt': prompt % solutions[0]['problem'], 'solution': solutions[0]['solution']})

        #shuffle the train samples
        if all:
            #sort the train samples by prompt and solution:
            train_samples = sorted(train_samples, key = lambda x: (x['prompt'], x['solution']))
        else:
            random.shuffle(train_samples)
        logger.info(f"Loaded {len(train_samples)} train samples, using method: {self.method}")
        return train_samples


    def archive_solutions(self, filename="train_samples.txt"):
#        return
        if os.path.exists(filename):
            os.rename(filename, filename + '-' + self.iteration)

    def train(self):
        train_samples = self.load_all_solutions()
        eval_samples = train_samples[len(train_samples)-self.batch_size*2:]
#        train_samples = train_samples[:len(train_samples)-self.batch_size*2]
        self.iteration = time.strftime("%Y%m%d-%H%M%S")
        model_id = self.MathLLM.model_id
        self.MathLLM.unload_model()
        del self.MathLLM
        logger.info(f"Training model {model_id} from iteration {self.iteration}")
        self.MathLLM = MathLLM(model_id, self.model_type, use_vllm=False, load=False, dataset_class=TokenizedQADataset)
        train_eos = False
        if self.MathLLM.model_id == "microsoft/phi-2":
            the_samples = train_samples[:len(train_samples)//32]
            train_eos = True
        else:
            the_samples = train_samples
        self.MathLLM.train(the_samples, eval_samples, 'trained_iter_' + self.iteration, lr = 5e-5, merge = False, train_eos=train_eos)
        if train_eos:
            self.iteration = time.strftime("%Y%m%d-%H%M%S")
            self.MathLLM.train(train_samples, eval_samples, 'trained_iter_' + self.iteration, lr = 1e-4, merge = True, train_eos=False)
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

def run_inference(model_name, model_type, start_from, num_samples, iteration, do_test, method, MPqueue):
    logger.add("learning.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    model_manager = ModelManager(model_name, model_type, start_from, num_samples, method = method)
    asyncio.run(model_manager.run_inference(iteration, do_test))
    MPqueue.put("done")

def run_training(model_name, model_type, method, MPqueue):
    logger.add("learning.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    model_manager = ModelManager(model_name, model_type, method = method)
    logger.info(f"Training model {model_manager.MathLLM.model_id}")
    model_manager.train()
    MPqueue.put(model_manager.MathLLM.model_id)

def run_merging(model_name, MPqueue):
    logger.add("learning.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    M = MathLLM(model_name, "mistral", use_vllm=False, load=False, dataset_class=TokenizedQADataset)  # revision = BASE_PHI_REVISION
    M.merge_lora()
    MPqueue.put(M.model_id)


def load_solutions(filename="all_samples1.txt"):
    problems = {}

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            sample = json.loads(line.strip())
            problem = sample['problem']
            if problem not in problems:
                problems[problem] = [sample]
            else:
                problems[problem].append(sample)
    hesitant_problems = []
    total_correct = 0
    total = 0
    for problem, samples in problems.items():
        correct = 0
        ts = 0
        for sample in samples:
            if int(float(sample['answer'])) != INVALID_ANSWER:
                if int(float(sample['answer'])) == int(float(sample['ground_answer'])):
                    correct += 1
                    total_correct += 1
                    total += 1
                else:
                    total += 1
                ts += 1
        if (ts == 0) or (correct/ts < 0.5):
            hesitant_problems.append(problem)

    logger.info(f"Loaded {len(problems)} problems, {total_correct}/{total} = {total_correct/total:.4f} correct samples, {len(hesitant_problems)/len(problems):.4f} hesitant problems")
    return problems, hesitant_problems




if __name__ == '__main__':
#    p, hp = load_solutions()
    mp.set_start_method('spawn')
    logger.add("learning.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')

#    M = MathLLM("trained_iter_20240401-101540-lora", "mistral", use_vllm=False, load=False, dataset_class=TokenizedQADataset)  # revision = BASE_PHI_REVISION
#    M.merge_lora()
#    model_manager = ModelManager("microsoft/phi-2", start_from=0, num_samples = 7473) #("trained_iter_20240220-235255", num_samples=1024)
#    model_manager = ModelManager("trained_iter_20240214-181649", start_from=0, num_samples = 7473) #("trained_iter_20240220-235255", num_samples=1024)
#    model_manager = ModelManager("trained_iter_20240215-134533", start_from=0, num_samples = 7473) #("trained_iter_20240220-235255", num_samples=1024)
#    model_manager = ModelManager("trained_iter_20240309-070712", start_from=0, num_samples = 7473) #("trained_iter_20240220-235255", num_samples=1024)
#    model_manager = ModelManager("trained_iter_20240318-090716", start_from=0, num_samples=7473, method = 'test')  # ("trained_iter_20240220-235255", num_samples=1024)
#    model_manager = ModelManager("trained_iter_20240328-194728", "mistral", start_from=0, num_samples=7473, method='temperature')  # ("trained_iter_20240220-235255", num_samples=1024)
#    model_manager = ModelManager("trained_iter_20240401-101540", "mistral", start_from=0, num_samples=7473, method='temperature')

#    model_manager = ModelManager("trained_iter_20240404-015311", 'mistral', start_from=5700, num_samples=7473, method='rephrase')
    # model_manager = ModelManager("trained_iter_20240420-084455", 'mistral', start_from=0, num_samples=7473, method='rephrase')
    model_manager = ModelManager("mistralai/Mistral-7B-Instruct-v0.1", "mistral", start_from=0, num_samples=7473, method='temperature') # ("trained_iter_20240220-235255", num_samples=1024)
#    model_manager = ModelManager("mistralai/Mistral-7B-Instruct-v0.1", "mistral", start_from=0, num_samples=7473, method='temperature')  # ("trained_iter_20240220-235255", num_samples=1024)
#    samples1 = model_manager.load_all_solutions(all=True)
#    samples2 = model_manager.load_and_filter_all_solutions(all=True)
#    NF = 0
#    for sample in samples1:
#        if sample not in samples2:
#            NF += 1
#    logger.info(f"Found {NF} samples in samples1 that are not in samples2")
#    for sample in samples2:
#        if sample not in samples1:
#            NF += 1
#    logger.info(f"Found {NF} samples in samples2 that are not in samples1")
    #    model_manager = ModelManager("trained_iter_20240322-192937", start_from=0, num_samples=1024,
#                             method='cross')  # ("trained_iter_20240220-235255", num_samples=1024)
#    model_manager = ModelManager("trained_iter_20240323-072154", start_from=0, num_samples=7473, method='cross')
#model_manager.load_all_solutions()
    #run the process_queue method in the background
    asyncio.run(model_manager.run())
