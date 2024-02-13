from openai import OpenAI
import copy
import concurrent.futures
from loguru import logger
from logging

class MathGPT:
    def __init__(self, args):
        self.args = args
        self.client = OpenAI()
        self.gpt4 = "gpt-4-1106-preview"
        self.prompt = [
            {"role": "system",
             "content": "Pretend you are an expert Math teacher and python coder. "},
            {"role": "user",
             "content": "Compare the following two solutions to a math problem:\n {problem}" +
            {"role": "user", "content": "Solution1: {sol1} \n Solution2: {sol2} \n"},
            {"role": "system", "content": "Write a JSON string comparing the solutions and identifying better solution. Example:\n" +
                                          "{{\n" +
                                          "\"discussion\": \"Lets think step by step and compare the solutions ...\",\n" +
                                          "\"better_solution_is\": \"1\"\n" +
                                          "}}"
             }]


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
        logger.error(f"Timeout Error: Ticker: {ticker}, News: {news}")
        return None
    except Exception as e:
        logger.error(f"Error: {e}, Ticker: {ticker}, News: {news}")
        return None

