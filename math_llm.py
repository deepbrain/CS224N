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
