from loguru import logger
from code_interpreter import compute_result
from enum import Enum


class State(Enum):
    START = 'start'
    PROCESSING = 'processing'
    FINISHED = 'finished'

class Solution:
    def __init__(self, solution, model_id, prompt, function_name):
        self.solution = solution
        self.model_id = model_id
        self.prompt = prompt
        self.function_name = function_name
        self.answer = None
        self.state = State.START
        self.score = None

    def compute_answer(self):
        if self.answer is None:
            self.answer = compute_result(self.solution, self.function_name)
        if self.answer == -99999:
            self.score = -99999
        return self.answer

    def continue_prompt(self):
        raise NotImplementedError #TODO subclass this class and implement this method for different solution approaches


    def get_prompt(self):
        if self.state == State.START:
            self.state = State.PROCESSING
            return self.prompt
        elif self.state == State.PROCESSING: # the model is processing solution - we might need to sample multiple steps from the model
            return self.continue_prompt()
        elif self.state == State.FINISHED: #no more prompts to generate
            return None

    def set_score(self, score):
        self.score = score

    def self_evaluate(self): # evaluate the solution based on the answer, log_likelihood, self-evaluating prompts
        raise NotImplementedError

    def get_confidence(self): # are we sure we want to train on this solution? compute this via prompts?
        raise NotImplementedError


    def __str__(self):
        return self.solution

    def __repr__(self):
        return self.__str__()


class MathProblem:
    def __init__(self, question, answer, train_solution=None):
        self.question = question
        self.answer = answer
        self.train_solution = train_solution
        self.solutions = []

    def add_solution(self, solution, model_id, prompt, function_name):
        self.solutions.append(Solution(solution, model_id, prompt, function_name))

    def compute_answers(self):
        for solution in self.solutions:
            solution.compute_answer()

    def compare_solutions(self): #TODO implement this method
        raise NotImplementedError

    def get_best_solution(self, method='majority'):
        raise NotImplementedError #TODO implement this method


    def __str__(self):
        return self.question + " = " + str(self.train_solution)

    def __repr__(self):
        return self.__str__()


class MathLLM:
    def __init__(self, model_id):
        self.model_id = model_id

    def process_batch(self, batch): #processes a batch of prompts and returns a batch of solutions one per prompt
        raise NotImplementedError

    def train(self, problems, new_model_name): # trains with a lora model on a batch of problems and saves the model to a new_model_name
        raise NotImplementedError

    def get_embeddings(self, problems): #returns embeddings for a batch of problems
        raise NotImplementedError

    def update(self, model_id): #load/update a model from model_id in the cloud
        raise NotImplementedError

class LearningDriver:
    #maintains a list of local problems and solutions
    #generates solutions for each problem and uploads them to github
    def __init__(self, problems):
        raise NotImplementedError #TODO implement this class

    def generate_solutions(self): #generates multiple solutions for each problem by varying prompts
        raise NotImplementedError

    def find_best_solutions(self): # selects the best solutions from the generated solutions by evaluating them
        raise NotImplementedError

    def compute_insample_accuracy(self): # computes accuracy of the best selected solutions on the training set
        raise NotImplementedError

    def upload_best_solutions(self): # uploads locally best solutions to github once a day
        raise NotImplementedError

    def mege_best_solutions(self): #downloads best solutions from github and merges them with the local best solutions
        raise NotImplementedError

    def train_models(self): #trains lora models on the merged best solutions
        raise NotImplementedError

    def submit_new_models(self): #uploads the trained models to huggingface repository
        raise NotImplementedError

    def update_best_models(self): #downloads the best models from huggingface repository and uses them to generate new solutions
        raise NotImplementedError

    def run(self): #implements the main loop of the learning driver
        raise NotImplementedError

