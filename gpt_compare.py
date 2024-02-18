from openai import OpenAI
import copy
import concurrent.futures
from loguru import logger
import logging

class MathGPT:
    def __init__(self, args):
        self.args = args
        self.client = OpenAI()
        self.gpt4 = "gpt-4-1106-preview"
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

    def ask_openai2(self, problem, sol1, sol2, model="gpt-3.5-turbo-1106", prompt=None, temperature=0.0, top_p=0.0,
                    frequency_penalty=0.0, presence_penalty=0.0):
        if prompt is None:
            prompt = self.prompt
        messages = copy.deepcopy(prompt)
        try:
            for message in messages:
                message['content'] = message['content'].format(problem=problem, sol1=sol1, sol2=sol2)
            if 'instruct' in model:
                prompt = "\n\n".join(part["content"] for part in messages)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.client.completions.create, model=model, prompt=prompt,
                                             temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty,
                                             presence_penalty=presence_penalty, max_tokens=1000)
                    response = future.result(timeout=60)  # Timeout set for 60 seconds
                    return response.choices[0].text
            else:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.client.chat.completions.create, model=model, messages=messages,
                                             temperature=temperature)
                response = future.result(timeout=60)  # Timeout set for 60 seconds
                return response.choices[0].message.content

        except concurrent.futures.TimeoutError:
            logger.error(f"Timeout Error")
            return None
        except Exception as e:
            logger.error(f"Error: {e}")
            return None



if __name__ == '__main__':

    logger.add("learning.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    args = None
    math_gpt = MathGPT(args)
    problem = "Solve the equation 2x + 3 = 7"
    sol1 = "x = 2"
    sol2 = "x = 4"
    result = math_gpt.ask_openai2(problem, sol1, sol2)
    print(result)
    logger.info(f"Result: {result}")
