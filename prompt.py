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



def get_old_prompts(): #returns instances of Prompt for each of the 10 old prompts
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
    prompt_texts = [prompt, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10]
    return [Prompt(prompt_text, function_name) for prompt_text in prompt_texts]


