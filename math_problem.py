from loguru import logger
from code_interpreter import compute_result, INVALID_ANSWER
import asyncio
from prompts.prompt_generators import get_all_prompts


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
        self.solutions.append(solution)
        self.best_answer = compute_result(solution, self.function_name)
        if self.log:
            self.problem.first_answer = self.best_answer
        self.answers.append(self.best_answer)
        if self.best_answer == INVALID_ANSWER:
            issues = "Yes"
        else:
            task = """\n\nAre there any logic issues in solution above, answer in one word Yes or No:\n"""
            issues = await self.model_manager.solve(solution + task, max_tokens = 2, completion_only=True)
        if "Yes" in issues:
            task = f"""\n\n#Task: Identify logic issues and write code to fix: \n"""
            response = await self.model_manager.solve(solution + task)
            # parse the closest to the end of string self.function_name and extract the code starting from the def + function_name to the end:
            next_solution = response.split(self.function_name)[-1]
            # add def:
            next_solution = "def " + next_solution
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
        prompts = get_all_prompts(self.question)
        log = True
        for prompt in prompts:
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

        sols = [(solution.best_solution, i) for i, solution in enumerate(self.solutions) if solution.best_answer != INVALID_ANSWER]

        self.majority_solution_index = -1
        while len(sols) > 1:
            i = 0
            while i < len(sols) - 1:
                sol1 = sols[i]
                sol2 = sols[i + 1]
                asyncio.create_task(self.compare_solutions(sol1, sol2))
                i += 2
            results = await asyncio.gather(*tasks)
            for result in results:
                if self.solutions[result[1]].best_answer == self.majority_answer:
                    self.majority_solution_index = result[1]
            if len(sols) % 2 == 1:
                results.append(sols[-1])
            sols = results
        self.majority_solution = self.solutions[self.majority_solution_index]
        self.best_solution = sols[0][0]
        self.best_solution_index = sols[0][1]
        best_answer = self.solutions[self.best_solution_index].best_answer
        self.correctness = [self.first_answer == self.ground_numeric_answer, self.majority_answer == self.ground_numeric_answer, best_answer == self.ground_numeric_answer]
        return self.correctness


    async def compare_solutions(self, sol1, sol2):
        task = f"\n\n#Task: Compare two solutions for accuracy, Solution 1: \n{sol1[0]} \n Solution 2: \n{sol2[0]}\n\nWhich solution is better, answer in one word Solution1 or Solution2"
        response = await self.model_manager.solve(task, max_tokens = 2, completion_only=True)
        if "1" in response:
            return sol1
        else:
            return sol2


    def get_best_solution(self):
        return self.best_solution

    def __str__(self):
        return self.question + " = " + str(self.train_solution)

    def __repr__(self):
        return self.__str__()



