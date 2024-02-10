import numpy as np
from loguru import logger
from code_interpreter import compute_result, INVALID_ANSWER
import asyncio
import random
#from prompts.prompt_generators import get_all_prompts
import re

class Solution:
    def __init__(self, problem, model_manager, prompt):
        self.problem = problem
        self.model_manager = model_manager
        self.prompt = prompt

    async def solve(self):
        self.initial_prompt = self.prompt.get_prompt() % self.problem.question
        self.solution = await self.model_manager.get_completion(self.initial_prompt, max_tokens=1000, completion_only=True)
#        logger.info(f"Solving problem: " + self.initial_prompt+self.get_train_solution())
        self.answer, error = await compute_result(self.initial_prompt+self.get_train_solution(), self.prompt.get_function_name(), self.model_manager.lock)
        if error != "":
            self.answer = INVALID_ANSWER
        self.prompt.add_solution(self.problem, self.problem.ground_numeric_answer, self.answer)
        return self.answer
    def get_train_prompt(self):
        return self.initial_prompt

    def get_train_solution(self):
        s = self.solution
        # Split the string into lines
        lines = s.split('\n')
        # Remove lines with print statements, and empty comment lines
        cleaned_lines = [line for line in lines if not (line.strip().startswith('print(') or line.strip() == '#' or line.strip() == '' or line.strip().startswith("if __name__") or line.strip().startswith("if __name__ =="))]
        # Join the cleaned lines back into a single string
        cleaned_code = '\n'.join(cleaned_lines)
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

    async def solve(self, prompts, solution_class=Solution):
        for p in prompts:
            self.solutions.append(solution_class(self, self.model_manager, p))
        tasks = [asyncio.create_task(solution.solve()) for solution in self.solutions]
        answers = await asyncio.gather(*tasks)
        answer_counts = {}
        for a in answers:
            if a != INVALID_ANSWER:
                if a in answer_counts:
                    answer_counts[a] += 1
                else:
                    answer_counts[a] = 1
        if len(answer_counts) == 0:
            self.best_answer = INVALID_ANSWER
            self.best_solution = random.choice(self.solutions)
            logger.error(f"No valid answers for problem: {self.question}")
            return
        self.best_answer = max(answer_counts, key=answer_counts.get)
        best_solutions = []
        for solution in self.solutions:
            if solution.answer == self.best_answer:
                best_solutions.append(solution)
        self.best_solution = random.choice(best_solutions)


    def get_train_sample(self):
        return {'prompt' : self.best_solution.get_train_prompt(), 'solution' : self.best_solution.get_train_solution()}

    def __str__(self):
        return self.question

    def __repr__(self):
        return self.__str__()


