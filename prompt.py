import numpy as np
from loguru import logger
#from prompts.prompt_generators import get_all_generated_prompts
from code_interpreter import INVALID_ANSWER

class Prompt:
    def __init__(self, prompt, train_prompt, function_name=None):  # prompt is assumed to have the % for the problem definition
        self.prompt = prompt
        self.train_prompt = train_prompt
        if function_name is None:
            self.function_name = "problem"
        else:
            self.function_name = function_name
        self.solution_answers = {}
        self.ground_answers = {}
        self.steps = {}
        self.problems = []

    def get_function_name(self):
        return self.function_name

    def get_prompt(self):
        return self.prompt

    def get_train_prompt(self):
        return self.train_prompt

    def add_solution(self, problem, ground_numeric_answer, solution_answer):
        if problem in self.problems:
            return
        self.problems.append(problem)
        self.solution_answers[problem] = solution_answer
        self.ground_answers[problem] = ground_numeric_answer

    def reset_stats(self):
        self.solution_answers = {}
        self.ground_answers = {}
        self.steps = {}
        self.problems = []

    def add_step_solution(self, problem, step_result):
        if problem in self.steps[problem]:
            self.steps[problem].append(step_result)
        else:
            self.steps[problem] = [step_result]

    def compute_accuracy(self, rephrasing=-1):
        correct = 0
        total = 0
        for problem in self.problems:
            if rephrasing != -1:
                if problem.rephasings != rephrasing:
                    continue
            if self.solution_answers[problem] == self.ground_answers[problem]:
                correct += 1
            total += 1
        return correct / total


def compute_mean_accuracy(problems, prompts, rephrasing=-1):
# computes the mean accuracy of all prompts
    correct = 0
    total = 0
    for problem in problems:
        if rephrasing != -1:
            if problem.rephasings != rephrasing:
                continue
        for prompt in prompts:
            if problem in prompt.problems:
                if prompt.solution_answers[problem] == prompt.ground_answers[problem]:
                    correct += 1
                total += 1
    return correct / total


def compute_majority_step_accuracy(problems, prompts, rephrasing = -1):
# computes the accuracy of the prompt that has the most repeating results in steps for each problem
    correct = 0
    for problem in problems:
        if rephrasing != -1:
            if problem.rephasings != rephrasing:
                continue
        answers = {}
        for prompt in prompts:
            if problem in prompt.problems:
                steps = prompt.steps[problem]
                for step in steps:
                    if step != INVALID_ANSWER:
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

def compute_majority_answer_accuracy(problems, prompts, rephrasing=-1):
    correct = 0
    for problem in problems:
        if rephrasing != -1:
            if problem.rephasings != rephrasing:
                continue
        answers = {}
        for prompt in prompts:
            if problem in prompt.problems:
                answer = prompt.solution_answers[problem]
                ground_answer = prompt.ground_answers[problem]
                if answer != INVALID_ANSWER:
                    if answer in answers:
                        answers[answer] += 1
                    else:
                        answers[answer] = 1
        if len(answers) == 0:
            continue
        majority_answer = max(answers, key=answers.get)
        if majority_answer == ground_answer:
            correct += 1
    return correct / len(problems)

def find_unsolvable_problems(problems, prompts, rephrasing=-1):
    unsolvable = []
    total = 0
    for problem in problems:
        if rephrasing != -1:
            if problem.rephasings != rephrasing:
                continue
        total += 1
        solved = False
        for prompt in prompts:
            if problem in prompt.problems:
                if prompt.solution_answers[problem] == prompt.ground_answers[problem]:
                    solved = True
                    break
        if not solved:
            unsolvable.append(problem)
    return unsolvable, total

def compute_oracle_accuracy(problems, prompts, rephrasing=-1):
    unsolvable, total = find_unsolvable_problems(problems, prompts, rephrasing)
    return 1 - len(unsolvable) / total

def get_old_prompts(): #returns instances of Prompt for each of the 10 old prompts
    function_name = "problem"
    prompt = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
             "        Elaborate your thinking step by step in comments before each code line below\n    \"\"\"\n"
    train_prompt = prompt
    prompt2 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
              "       Add comments before each line\n    \"\"\"\n"
    train_prompt2 = prompt2
    prompt3 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
              "       Be accurate and think step by step in comments before each code line below\n    \"\"\"\n"
    train_prompt3 = prompt3
    prompt4 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
              "       Find unusual solution and comment before each of your line of code\n    \"\"\"\n"
    train_prompt4 = prompt4
    prompt5 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
              "       In your comments write an algebraic formula based on the problem, solve it algebraically, then write code to calculate the result\n    \"\"\"\n"
    train_prompt5 = prompt5
    prompt6 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
              "       Find the most elegant and correct solution\n    \"\"\"\n"
    train_prompt6 = prompt6
    prompt7 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
              "       Think step by step in comments before each code line below\n    \"\"\"\n"
    train_prompt7 = prompt7
    prompt8 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
              "       You must elaborate your thinking in comments below\n    \"\"\"\n"
    train_prompt8 = prompt8

    prompt9 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
              "       Is this a simple math or algebra problem? For algebra problems, you must elaborate and solve it algebraically in the comments first, then write code to calculate the result. For simple math problems, you can write code to calculate the result directly\n    \"\"\"\n"
    train_prompt9 = prompt9
    prompt10 = f"def {function_name}() -> int:\n    \"\"\"%s\n" + \
              "       First, let's solve this problem using pure symbolic expressions. Elaborate with your algebraic skills below. Use x,y,z...to denote unknown variables. Use a,b,c... to denote given constants. Then write a python code to compute the result\n    \"\"\"\n"
    train_prompt10 = prompt10
    prompt_texts = [prompt, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10]
    train_prompts = [train_prompt, train_prompt2, train_prompt3, train_prompt4, train_prompt5, train_prompt6, train_prompt7, train_prompt8, train_prompt9, train_prompt10]
    return [Prompt(prompt_text, train_prompt, function_name) for prompt_text,train_prompt in zip(prompt_texts, train_prompts)]

def get_all_prompts():
    function_name = "problem"
    old_prompts = get_old_prompts()
    new_prompt_strings = get_all_generated_prompts("%s\n")
    new_prompts = [Prompt(prompt_text, function_name) for prompt_text in new_prompt_strings]
    return old_prompts + new_prompts