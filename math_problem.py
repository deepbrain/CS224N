import numpy as np
from loguru import logger
from code_interpreter import compute_result, INVALID_ANSWER, crop_solution, valStr, trace_valid
import asyncio
import random
import re
import traceback

class Solution:
    def __init__(self, problem, model_manager, prompt, full_prompt, is_test):
        self.problem = problem
        self.model_manager = model_manager
        self.prompt = prompt
        self.full_prompt = full_prompt
        self.solutions = []
        self.answers = []
        self.traces = []
        self.is_test = is_test

    async def solve(self, temperature):
        self.solution = await self.model_manager.get_completion(self.full_prompt, max_tokens=1000, temperature = temperature, completion_only=True)
#        logger.info(f"Solving problem: " + self.initial_prompt+self.get_train_solution())
        async with self.model_manager.code_interpreter_lock:
            self.answer, error, lls = compute_result(
                self.full_prompt,
                self.get_train_solution(self.solution),
                self.prompt.get_function_name(),
                should_crop_solution=True,
            )
        if error != "":
            self.answer = INVALID_ANSWER

        self.solutions.append(self.solution)
        self.answers.append(self.answer)
        self.traces.append(lls)

        self.prompt.add_solution(self.problem, self.problem.ground_numeric_answer, self.answer)
        return self.answer

    def get_train_prompt(self):
        return self.prompt.get_train_prompt()

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
    def __init__(self, model_manager, question, ground_answer, ground_numeric_answer, use_dpo, rephasings = {}):
        self.model_manager = model_manager
        self.ground_answer = ground_answer
        self.ground_numeric_answer = ground_numeric_answer
        self.solutions = []
        self.question = question
        self.rephasings = rephasings
        self.use_dpo = use_dpo

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
        self.is_learning_sample = False
        for p in prompts:
            if len(self.rephasings) > 0:
                for i in range(10):
                    if f'rephrase{i}' in self.rephasings:
                        rephrasing = self.rephasings[f'rephrase{i}']
                        self.solutions.append(solution_class(self, self.model_manager, p, p.get_prompt() % rephrasing, is_test))
            self.solutions.append(solution_class(self, self.model_manager, p, p.get_prompt() % self.question, is_test))
        tasks = [asyncio.create_task(solution.solve(temperature = 0.5)) for solution in self.solutions]
        answers = await asyncio.gather(*tasks)

#        Temp = 0.3
#        for i in range(5):
#            tasks = [asyncio.create_task(solution.solve(temperature = Temp)) for solution in self.solutions]
#            answers = await asyncio.gather(*tasks)
#            if is_test:
#                return
#            Temp += 0.1
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
                self.model_manager.add_all_sample({'problem' : self.question, 'prompt' : solution.get_train_prompt(), 'solution' : solution.get_train_solution(s), 'answer' : str(a), 'ground_numeric_answer' : self.ground_numeric_answer})

        self.samples = []
        if len(answer_counts) == 0:
            self.best_answer = INVALID_ANSWER
            self.best_solution = random.choice(self.solutions)
            logger.error(f"No valid answers for problem: {self.question}")
            self.train_json = None
            return

        self.best_answer = max(answer_counts, key=answer_counts.get)

        if self.use_dpo:
            self.create_dpo_samples(answer_counts)
            return

        self.evaluate_solutions(answer_counts)


        return # training on the same prompt

    def evaluate_solutions(self, answer_counts):
        try:
            traces = {}
            last_traces = {}
            total_valid = 0
            total = 0
            for solution in self.solutions:
                st = []
                for s,a,t in zip(solution.solutions, solution.answers, solution.traces):
                    total += 1
                    if a != INVALID_ANSWER:
                        total_valid += 1
                        while (len(t) >= 1) and (valStr(t[-1]) == a):
                            t.pop()
                        if len(t) == 0:
                            last = '-9999'
                        else:
                            last = t[-1]
                        t.extend([a])
                        if trace_valid(t):
                            if a in traces:
                                if not t in traces[a]:
                                    traces[a].append(t)
                            else:
                                traces[a] = [t]
                            if a in last_traces:
                                if not last in last_traces[a]:
                                    last_traces[a].append(last)
                            else:
                                last_traces[a] = [last]
                        st.extend([t])
                solution.traces = st
            ac = []
            for a in answer_counts:
                if a in traces:
                    l = len(traces[a])
                else:
                    l = 0
                if a in last_traces:
                    ll = len(last_traces[a])
                else:
                    ll = 0
                ac.append((answer_counts[a], l, ll, a))
            #sort ac by decending answer counts key
            ac.sort(reverse=True)
            self.is_learning_sample = False
            self.learning_sample_correct = True
            train_samples = []
            self.best_answer = ac[0][3]
            if (self.best_answer >= 0) and ((self.best_answer - round(self.best_answer)) == 0) and (len(ac) < 30):
                if len(ac) >= 2: #select problems with some indeterminancy in answers
                    if ac[0][0] >= 2*ac[1][0]: #one solution must dominate the others
                        if (ac[0][1] >= 2*ac[1][1]):# and (ac[0][2] >= 2*ac[1][2]): #number of traces and the last trace numbers of that solution must also dominate
                            #generate learning samples
                            self.samples = []
                            for solution in self.solutions:
                                for s,a in zip(solution.solutions, solution.answers):
                                    if a == self.best_answer:
                                        train_samples.append({'problem' : self.question, 'prompt' : solution.get_train_prompt(), 'solution' : solution.get_train_solution(s), 'answer' : a, 'ground_numeric_answer' : self.ground_numeric_answer})
                                        if self.learning_sample_correct and (a != self.ground_numeric_answer) and (self.ground_numeric_answer != -88888):
                                            logger.info(f"Learning sample incorrect: {self.question} {a} != {self.ground_numeric_answer}")
                                            self.learning_sample_correct = False
                                        self.is_learning_sample = self.ground_numeric_answer != INVALID_ANSWER
            if (len(ac) == 1) and (self.best_answer >= 0) and ((self.best_answer - round(self.best_answer)) == 0):
                self.samples = []
                for solution in self.solutions:
                    for s, a in zip(solution.solutions, solution.answers):
                        if a == self.best_answer:
                            train_samples.append({'problem': self.question, 'prompt': solution.get_train_prompt(),
                                                  'solution': solution.get_train_solution(s), 'answer': a,
                                                  'ground_numeric_answer': self.ground_numeric_answer})
                            if self.learning_sample_correct and (a != self.ground_numeric_answer) and (self.ground_numeric_answer != -88888):
                                logger.info(f"Learning sample incorrect: {self.question} {a} != {self.ground_numeric_answer}")
                                self.learning_sample_correct = False
                            self.is_learning_sample = self.ground_numeric_answer != INVALID_ANSWER

            #            if len(train_samples) > 0: #choose one random sample
#                index = random.randint(0, len(train_samples)-1) #TODO instead of random, choose the one that looks better to LLM
            for train_sample in train_samples:
                self.model_manager.add_train_sample(train_sample)

            #save hard problems for later analysis
            self.problem_samples = []
            do_sample = (self.best_answer < 0) or ((self.best_answer - round(self.best_answer)) != 0)
            if do_sample or (total_valid < 0.5 * total) or (len(ac) >= 2 and ac[0][0] < 2*ac[1][0]) or (len(ac) >= 2 and ac[0][1] < 2*ac[1][1]) or (len(ac) >= 7):
                self.model_manager.add_problem_sample(self.question)
        except Exception as e:
            logger.error(f"Error in problem evaluation: {e}, stack: {traceback.format_exc()}")



    def create_dpo_samples(self, answer_counts):
        best_len = 100000
        for solution in self.solutions: #TODO group solutions by answers and then by logic similarity. Include numeric answer into solution before comparing. Rank resulting groups by self evaluation. Select the correct answer by overall highest rank. Choose negative sampples from similar solutions!
            for s,a,t in zip(solution.solutions, solution.answers):
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


