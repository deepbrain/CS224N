import numpy as np
from loguru import logger
class Prompt:
    def __init__(self, prompt, function_name=None):  # prompt is assumed to have the % for the problem definition
        self.prompt = prompt
        if function_name is None:
            self.function_name = "problem"
        self.solution_answers = {}
        self.ground_answers = {}
        self.steps = {}

    def get_function_name(self):
        return self.function_name

    def get_prompt(self):
        return self.prompt

    def add_solution(self, problem, ground_numeric_answer, solution_answer):
        if problem in self.problems:
            return
        self.problems.append(problem)
        self.solution_answers[problem] = solution_answer
        self.ground_answers[problem] = ground_numeric_answer

    def add_step_solution(self, problem, step_result):
        if problem in self.steps[problem]:
            self.steps[problem].append(step_result)
        else:
            self.steps[problem] = [step_result]

    def compute_accuracy(self):
        correct = 0
        total = 0
        for problem in self.problems:
            if self.solutions_answers[problem] == self.ground_answers[problem]:
                correct += 1
            total += 1
        return correct / total


def compute_mean_accuracy(problems, prompts):
# computes the mean accuracy of all prompts
    correct = 0
    total = 0
    for problem in problems:
        for prompt in prompts:
            if problem in prompt.problems:
                if prompt.solution_answers[problem] == prompt.ground_answers[problem]:
                    correct += 1
                total += 1
    return correct / total


def compute_majority_step_accuracy(problems, prompts):
# computes the accuracy of the prompt that has the most repeating results in steps for each problem
    correct = 0
    for problem in problems:
        answers = {}
        for prompt in prompts:
            if problem in prompt.problems:
                steps = prompt.steps[problem]
                for step in steps:
                    if step in answers:
                        answers[step] += 1
                    else:
                        answers[step] = 1
        ranks = np.zeros(len(prompts))
        for prompt in prompts:
            for step in prompt.steps[problem]:
                ranks[prompts.index(prompt)] += answers[step]  # todo add higher weights for later steps
        top_rank_index = np.argmax(ranks)
        if prompts[top_rank_index].solution_answers[problem] == prompts[top_rank_index].ground_answers[problem]:
            correct += 1
    return correct / len(problems)


def compute_majority_answer_accuracy(problems, prompts):
    correct = 0
    for problem in problems:
        answers = {}
        for prompt in prompts:
            if problem in prompt.problems:
                answer = prompt.solution_answers[problem]
                ground_answer = prompt.ground_answers[problem]
                if answer in answers:
                    answers[answer] += 1
                else:
                    answers[answer] = 1
        majority_answer = max(answers, key=answers.get)
        if majority_answer == ground_answer:
            correct += 1
    return correct / len(problems)


def find_unsolvable_problems(problems, prompts):
    unsolvable = []
    for problem in problems:
        solved = False
        for prompt in prompts:
            if problem in prompt.problems:
                if prompt.solution_answers[problem] == prompt.ground_answers[problem]:
                    solved = True
                    break
        if not solved:
            unsolvable.append(problem)
    return unsolvable


def compute_oracle_accuracy(problems, prompts):
    return 1 - len(find_unsolvable_problems(problems, prompts)) / len(problems)
