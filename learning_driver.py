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

