import numpy as np
from loguru import logger
from code_interpreter import compute_result, INVALID_ANSWER
import asyncio
import random
from prompts.prompt_generators import get_all_prompts

function_name = "problem"
prompt = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       Elaborate your thinking step by step in comments before each code line below. End with return result\n"
prompt2 = f"def {function_name}() -> int:\n    \"\"\"%s" + \
         "       Add comments before each line. End with return result\n"
prompt3 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       Be accurate and think step by step in comments before each code line below. End with return result\n"
prompt4 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       Find unusual solution and comment before each of your line of code. End with return result\n"
prompt5 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       In your comments write an algebraic formula based on the problem, solve it algebraically, then write code to calculate the result. End with return result\n"
prompt6 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       Find the most elegant and correct solution. End with return result\n"
prompt7 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       Think step by step in comments before each code line below. End with return result"
prompt8 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
         "       You must elaborate your thinking in comments below. End with return result"
prompt9 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
"""      Is this a simple math or algebra problem? For algebra problems, you must elaborate and solve it algebraically in the comments first, then write code to calculate the result. For simple math problems, you can write code to calculate the result directly. End with return result\n"""
prompt10 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
"""    First, let's solve this problem using pure symbolic expressions. Elaborate with your algebraic skills below. Use x,y,z...to denote unknown variables. Use a,b,c... to denote given constants. Then write a pure python code to compute and return the result. End with return result\n    Let x be"""
prompts = [prompt, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10]

compare1 = f"Task: Compare two solutions for accuracy, Solution 1:\n%s \n Solution 2:\n%s\nWhich solution is better, Solution1 or Solution2: Solution"
compare2 = f"Task: Compare two solutions for accuracy, Solution 1:\n%s \n Solution 2:\n%s\nWhich solution is better, answer in one word Solution2 or Solution1: Solution"
compare3 = f"Solution 1: \n%s \n Solution 2: \n%s\nTask: Evaluate the Solutions and choose one Solution1 or Solution2: Solution"
compare4 = f"Solution 1: \n%s \n Solution 2: \n%s\nTask: Evaluate the Solutions and choose one Solution2 or Solution1: Solution"
compare5 = f"Analyze the two solutions and choose the better one: Solution 1:\n%s \n Solution 2:\n%s\nWhich solution is better Solution1 or Solution2: Solution"
compare6 = f"Analyze the two solutions and choose the better one: Solution 1:\n%s \n Solution 2:\n%s\nWhich solution is better Solution2 or Solution1: Solution"
comparisons = [compare1, compare2, compare3, compare4, compare5, compare6]


class Solution:
    def __init__(self, problem, model_manager, prompt, log=False):
        self.problem = problem
        self.model_manager = model_manager
        self.prompt = prompt
        self.function_name = "problem"
        self.log = log


    async def solve(self):
        # start from the initial prompt
        self.solutions = []
        self.answers = []
        solution = await self.model_manager.solve(self.prompt)
        self.best_answer, error = compute_result(solution, self.function_name)
        #check if best_answer is a number:
        self.best_solution = solution
        if self.best_answer != INVALID_ANSWER:
            question = solution + f"#the result is {self.best_answer} is it correct? Answer Yes or No:"
            g = await self.model_manager.solve(question, max_tokens=2, completion_only=True)
            if error != "":
                logger.info(f"Error: {error}")
            if "No" in g:
                self.best_solution = solution + "#Review: this solution may have errors."


        if self.log:
            self.problem.first_answer = self.best_answer
            self.problem.first_solution = self.best_solution
        self.solutions.append(self.best_solution)
        self.answers.append(self.best_answer)
        """        if self.best_answer == INVALID_ANSWER:
            issues = "Yes"
        else:
            task = "\n\nAre there any logic issues in solution above, answer in one word Yes or No:\n"
            issues = await self.model_manager.solve(solution + task, max_tokens = 2, completion_only=True)
        if "Yes" in issues:
            task = f"\n\n#Task: Identify logic issues and write code to fix: \n"
            response = await self.model_manager.solve(solution + task)
            # parse the closest to the end of string self.function_name and extract the code starting from the def + function_name to the end:
            next_solution = response.split(self.function_name)[-1]
            # add def:
            next_solution = "def " + self.function_name + next_solution
            next_answer = compute_result(next_solution, self.function_name)
            if next_answer != INVALID_ANSWER:
                self.solutions.append(next_solution)
                self.answers.append(next_answer)
                self.best_solution, idx = await self.problem.compare_solutions((solution, 0), (next_solution, 1))
                if idx == 1:
                    self.best_answer = next_answer
        else:
            self.best_solution = solution
        self.best_answer = compute_result(self.best_solution, self.function_name)
        """

    def __str__(self):
        return self.best_solution

    def __repr__(self):
        return self.__str__()


class Problem:
    def __init__(self, model_manager, question, ground_answer, ground_numeric_answer):
        self.model_manager = model_manager
        self.ground_answer = ground_answer
        self.ground_numeric_answer = ground_numeric_answer
        self.solutions = []
        self.question = question


    def get_question(self):
        return self.question

    async def solve(self, solution_class=Solution):
        #prompts = get_all_prompts(self.question)
        log = True
        for p in prompts:
            prompt = p % self.question
            self.solutions.append(solution_class(self, self.model_manager, prompt, log))
            log = False
        # await concurrently for all solutions:
        tasks = [asyncio.create_task(solution.solve()) for solution in self.solutions]
        await asyncio.gather(*tasks)
        answer_counts = {}
        for solution in self.solutions:
            for answer in solution.answers:
                if answer != INVALID_ANSWER:
                    if answer in answer_counts:
                        answer_counts[answer] += 1
                    else:
                        answer_counts[answer] = 1
        self.majority_answer = max(answer_counts, key=answer_counts.get)

        sols = [[solution.best_solution, i, 0] for i, solution in enumerate(self.solutions) if solution.best_answer != INVALID_ANSWER]
#        for i in range(len(sols)):
#            if self.solutions[i].best_answer == self.majority_answer:
#                sols[i][2] += 1 #score for majority answer

        for i in range(5*len(comparisons)):
            random.shuffle(sols)
            index = i % len(comparisons)
            await self.score_solutions(sols, comparisons[index])
        self.majority_solution_index = -1
        max_score = -1
        for i in range(len(sols)):
            if sols[i][2] > max_score:
                max_score = sols[i][2]
                self.best_solution = sols[i][0]
                self.best_solution_index = sols[i][1]

                if self.solutions[sols[i][1]].best_answer == self.majority_answer:
                    self.majority_solution_index = sols[i][1]
                    self.majority_solution = sols[i][0]

        best_answer = self.solutions[self.best_solution_index].best_answer
        self.correctness = [self.first_answer == self.ground_numeric_answer, self.majority_answer == self.ground_numeric_answer, best_answer == self.ground_numeric_answer]
        return self.correctness

    async def score_solutions(self, sols, prompt):
        tasks = []
        i = 0
        while i < len(sols) - 1:
            sol1 = sols[i]
            sol2 = sols[i + 1]
            tasks.append(asyncio.create_task(self.compare_solutions(sol1, sol2, prompt)))
            i += 2
        await asyncio.gather(*tasks)



    async def compare_solutions(self, sol1, sol2, prompt = compare1):
        task = prompt % (sol1[0], sol2[0])
        response = await self.model_manager.solve(task, max_tokens = 2, completion_only=True)
        if "1" in response:
            sol1[2] += 1
            return sol1
        else:
            sol2[2] += 1
            return sol2

    def get_train_sample(self):
        return self.best_solution

    def __str__(self):
        return self.question + " = " + str(self.train_solution)

    def __repr__(self):
        return self.__str__()



