from openai import OpenAI
import copy
import concurrent.futures
from loguru import logger
import logging

class GPT:
    def __init__(
        self,
        model="gpt-3.5-turbo-1106",
        temperature = 0.0,
        top_p = 0.0,
        freq_penalty = 0.0,
        presence_penalty = 0.0,
        prompt=None,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = freq_penalty
        self.presence_penalty = presence_penalty
        self.client = OpenAI()
        if prompt is None:
            self.prompt = [
                {
                    "role":"user",
                    "content":"", # this will be replaced with your message
                }
            ]
        else:
            self.prompt = prompt

    def ask_openai2(self, message, model=None, temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None):
        """
        * message: is basically what's sent to GPT as a prompt for it to give us a response back.
        * the other args are just overrides to the query parameters of the gpt model
        """
        model = self.model if model is None else model
        temperature = self.temperature if temperature is None else temperature
        top_p = self.top_p if top_p is None else top_p
        frequency_penalty = self.freq_penalty if frequency_penalty is None else frequency_penalty
        presence_penalty = self.presence_penalty if presence_penalty is None else presence_penalty

        prompt = copy.deepcopy(self.prompt)
        prompt[0]["content"] = message

        try:
            if 'instruct' in model:
                prompt = "\n\n".join(part["content"] for part in prompt)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.client.completions.create, model=model, prompt=prompt,
                                             temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty,
                                             presence_penalty=presence_penalty, max_tokens=1000)
                    response = future.result(timeout=60)  # Timeout set for 60 seconds
                    return response.choices[0].text
            else:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.client.chat.completions.create, model=model, messages=prompt,
                                             temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty,
                                             presence_penalty=presence_penalty, max_tokens=1000)
                response = future.result(timeout=60)  # Timeout set for 60 seconds
                return response.choices[0].message.content
        except concurrent.futures.TimeoutError:
            logger.error(f"Timeout Error")
            return None
        except Exception as e:
            logger.error(f"Error: {e}")
            return None

class MathGPT(GPT):
    def __init__(
        self,
        model="gpt-3.5-turbo-1106",
        temperature = 0.0,
        top_p = 0.0,
        freq_penalty = 0.0,
        presence_penalty = 0.0,
        prompt=None,
    ):
        super().__init__(model, temperature, top_p, freq_penalty, presence_penalty, prompt)
        if prompt is None:
            self.prompt = [
                {"role": "user",
                "content": "Pretend you are an expert Math teacher and python coder. \nCompare the following two solutions to a math problem: \"{problem}\"\n" +
                "The Solution1 is: {sol1}\nSolution1 ends here. \n\n\nThe Solution2 is: {sol2}\nSolution2 ends here.\n" +
                "Write a JSON string comparing the solutions. Think step by step and identify what is incorrect. Example:\n" +
                            "{{\n" +
                            "\"discussion\": \"Let's think step by step. First ... Therefore SolutionX seems to be correct, while SolutionY is likely incorrect\",\n" +
                            "\"better_solution_is\": \"1\"\n" +
                            "}}"
                }
            ]

    def ask_openai2(self, problem, sol1, sol2, model=None, temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None):
        message = "Pretend you are an expert Math teacher and python coder. \nCompare the following two solutions to a math problem: \"{problem}\"\n" +\
                "The Solution1 is: {sol1}\nSolution1 ends here. \n\n\nThe Solution2 is: {sol2}\nSolution2 ends here.\n" +\
                "Write a JSON string comparing the solutions. Think step by step and identify what is incorrect. Example:\n" +\
                            "{{\n" +\
                            "\"discussion\": \"Let's think step by step. First ... Therefore SolutionX seems to be correct, while SolutionY is likely incorrect\",\n" +\
                            "\"better_solution_is\": \"1\"\n" +\
                            "}}"
        message = message.format(problem=problem, sol1=sol1, sol2=sol2)
        return super().ask_openai2(message, model, temperature, top_p, frequency_penalty, presence_penalty)

if __name__ == '__main__':

    logger.add("learning.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    math_gpt = MathGPT()
    problem = "Solve the equation 2x + 3 = 7"
    sol1 = "x = 2"
    sol2 = "x = 4"
    result = math_gpt.ask_openai2(problem, sol1, sol2)
    print(result)
    logger.info(f"Result: {result}")
