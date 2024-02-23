import numpy as np
from loguru import logger
from code_interpreter import compute_result, INVALID_ANSWER, crop_solution
import asyncio
import random
import re

class Solution:
    def __init__(self, problem, model_manager, prompt, is_test):
        self.problem = problem
        self.model_manager = model_manager
        self.prompt = prompt
        self.solutions = []
        self.answers = []
        self.is_test = is_test

    async def solve(self, temperature):
        self.initial_prompt = self.prompt.get_prompt() % self.problem.question
        self.solution = await self.model_manager.get_completion(self.initial_prompt, max_tokens=1000, temperature = temperature, completion_only=True)
#        logger.info(f"Solving problem: " + self.initial_prompt+self.get_train_solution())
        async with self.model_manager.code_interpreter_lock:
            self.answer, error, lls = compute_result(
                self.initial_prompt,
                self.get_train_solution(self.solution),
                self.prompt.get_function_name(),
                should_crop_solution=True,
            )
        if error != "":
            self.answer = INVALID_ANSWER

        self.solutions.append(self.solution)
        self.answers.append(self.answer)

        self.prompt.add_solution(self.problem, self.problem.ground_numeric_answer, self.answer)
        return self.answer

    def get_train_prompt(self):
        return self.prompt.get_train_prompt() % self.problem.question

    def get_train_solution(self, solution_text):
        s = solution_text
        # Split the string into lines
        lines = s.split('\n')
        # Remove lines with print statements, and empty comment lines
        cleaned_lines = [line for line in lines if not (line.strip().startswith('print(') or line.strip() == '#' or line.strip() == '' or line.strip().startswith("if __name__") or line.strip().startswith("if __name__ =="))]
        # Join the cleaned lines back into a single string
        cleaned_code = '\n'.join(cleaned_lines)
        cleaned_code = crop_solution(cleaned_code)
        return cleaned_code

    def __str__(self):
        return self.solution

    def __repr__(self):
        return self.__str__()



class Problem:
    def __init__(self, model_manager, question, ground_answer, ground_numeric_answer, rephasings=0):
        self.model_manager = model_manager
        self.ground_answer = ground_answer
        self.ground_numeric_answer = ground_numeric_answer
        self.solutions = []
        self.question = question
        self.rephasings = rephasings

    def get_question(self):
        return self.question

    async def rephrase(self):
        #creates a rephrased version of the problem
        p1 = "Two problems are the same:\nProblem1:%sProblem2:" % self.question
        problems = [p1]
        tasks = [asyncio.create_task(self.model_manager.get_completion(p, max_tokens=1000, completion_only=True)) for p in problems]
        rephrased_problems = await asyncio.gather(*tasks)
        result = [Problem(self.model_manager, p, self.ground_answer, self.ground_numeric_answer, 1+i) for i, p in enumerate(rephrased_problems)]

    async def solve(self, prompts, solution_class=Solution, is_test = False):
        for p in prompts:
            self.solutions.append(solution_class(self, self.model_manager, p, is_test))

        Temp = 0.3
        for i in range(5):
            tasks = [asyncio.create_task(solution.solve(temperature = Temp)) for solution in self.solutions]
            answers = await asyncio.gather(*tasks)
            if is_test:
                return
            Temp += 0.1
        answer_counts = {}
        for s in self.solutions:
            for a in s.answers:
                if a != INVALID_ANSWER:
                    if a in answer_counts:
                        answer_counts[a] += 1
                    else:
                        answer_counts[a] = 1

        self.all_samples = []
        for solution in self.solutions:
            for s,a in zip(solution.solutions, solution.answers):
                self.all_samples.append({'prompt' : solution.get_train_prompt(), 'solution' : solution.get_train_solution(s), 'answer' : a, 'ground_numeric_answer' : self.ground_numeric_answer})

        self.samples = []
        if len(answer_counts) == 0:
            self.best_answer = INVALID_ANSWER
            self.best_solution = random.choice(self.solutions)
            logger.error(f"No valid answers for problem: {self.question}")
            self.train_json = None
            return

        self.best_answer = max(answer_counts, key=answer_counts.get)
        best_len = 100000
        for solution in self.solutions: #TODO group solutions by answers and then by logic similarity. Include numeric answer into solution before comparing. Rank resulting groups by self evaluation. Select the correct answer by overall highest rank. Choose negative sampples from similar solutions!
            negative_sample = None
            positive_sample = None
            for s,a in zip(solution.solutions, solution.answers):
                if a == self.best_answer:
                    if len(s) < best_len:
                        positive_sample = solution.get_train_solution(s)
                        best_len = len(s)
                elif a == INVALID_ANSWER:
                    negative_sample = s
                else:
                    if (negative_sample is None) and (answer_counts[a] <= 2):
                        negative_sample = s
            if positive_sample is not None and negative_sample is not None:
                self.samples.append({'prompt' : solution.get_train_prompt(), 'solution' : positive_sample, 'negative' : negative_sample})

        return # training on the same prompt

    def get_train_samples(self):
        return self.samples

    def get_all_samples(self):
        return self.all_samples

    def get_comparative_samples(self):
        return self.comparative_solutions


    def __str__(self):
        return self.question

    def __repr__(self):
        return self.__str__()


