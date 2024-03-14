# (C) 2024 Stanford CS224N Group Custom Project by Artyom Shaposhnikov, Shubhra Mishra, Roberto Garcia

from openai import OpenAI
from openai._types import NOT_GIVEN
import copy
import concurrent.futures
from loguru import logger
import logging
import tiktoken
import numpy as np
import time

class GPT:
    def __init__(
        self,
        model="gpt-3.5-turbo-1106",
        prompt=None,

        # OPENAI QUERY PARAMS, NOTGIVEN is the default
        temperature = NOT_GIVEN,
        top_p = NOT_GIVEN,
        freq_penalty = NOT_GIVEN,
        presence_penalty = NOT_GIVEN,
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

    def ask_openai2(self, message, model=None, temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None, timeout=60, **query_kwargs):
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
        prompt[-1]["content"] = message

        try:
            if 'instruct' in model:
                prompt = "\n\n".join(part["content"] for part in prompt)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.client.completions.create, model=model, prompt=prompt,
                                             temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty,
                                             presence_penalty=presence_penalty, max_tokens=1000, **query_kwargs)
                    response = future.result(timeout=timeout)  # Timeout set for 60 seconds
                    return response.choices[0].text
            else:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.client.chat.completions.create, model=model, messages=prompt,
                                             temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty,
                                             presence_penalty=presence_penalty, max_tokens=1000, **query_kwargs)
                response = future.result(timeout=timeout)  # Timeout set for 60 seconds
                return response.choices[0].message.content
        except concurrent.futures.TimeoutError:
            logger.error(f"Timeout Error")
            return None
        except Exception as e:
            logger.error(f"Error: {e}")
            return None

class RephraseGPT(GPT):
    def __init__(
        self,
        model="gpt-3.5-turbo-1106",
        prompt=None,

        # OPENAI QUERY PARAMS, NOTGIVEN is the default
        temperature = NOT_GIVEN,
        top_p = NOT_GIVEN,
        freq_penalty = NOT_GIVEN,
        presence_penalty = NOT_GIVEN,
    ):
        super().__init__(model, temperature, top_p, freq_penalty, presence_penalty, prompt)
        if prompt is None:
            self.prompt = [
                {
                    "role": "system",
                    "content": "You are a rephrasing assistant. "\
                               "You are brillinat in creatively rephrasing whatever the user indicates you. "\
                               "When rephrasing, you can cleverly vary the grammatical structure of sentences without changing their overall meaning. "\
                               "Additionally, you can change words too, as long as the meaning of the sentence is preserved. "\
                               "Also, you do not miss any detail when rephrasing sentences."
                },
                {
                    "role":"user",
                    "content":"", # this will be replaced with your message
                }
            ]
        else:
            self.prompt = prompt

    def get_repetition_penalty_dict(self, texts, penalty_multiplier=None):
        penalty_multiplier = 1 if penalty_multiplier is None else penalty_multiplier
        tokenizer_id = None
        if "gpt-3" in self.model: tokenizer_id = "gpt-3.5"
        elif "gpt-4" in self.model: tokenizer_id = "gpt-4"
        enc = tiktoken.encoding_for_model(tokenizer_id)
        occurrences = {}
        for i, text in enumerate(texts):
            text_enc = enc.encode(text)
            for tok in set(text_enc):
                if tok not in occurrences:
                    occurrences[tok]=0
                occurrences[tok] += (1 if i > 0 else 3)
        penalty = {tok:-np.log(occ+1)*penalty_multiplier for tok, occ in occurrences.items()}
        sorted_penalties = sorted(penalty.items(), key=lambda item: item[1])
        sorted_penalties = sorted_penalties[:300] # API only allows 300
        penalty = {k:v for k,v in sorted_penalties}
        return penalty

    def ask_openai2(self, prompt=None, problem=None, model=None, temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None, **query_kwargs):
        if prompt is None:
            assert problem is not None
            message = f"Rephrase the following problem: {problem} "
        elif problem is None:
            assert prompt is not None
            message = f"Rephrase the following prompt: {prompt}"
        return super().ask_openai2(message, model, temperature, top_p, frequency_penalty, presence_penalty, **query_kwargs)
    
    def ask_openai2_multiple(self, prompt=None, problem=None, num_samples=1, repetition_penalty_multiplier=None, **kwargs):
        results = []
        if prompt is None:
            assert problem is not None
            new_kwargs = dict(problem=problem)
            orig_text = problem
        elif problem is None:
            assert prompt is not None
            new_kwargs = dict(prompt=prompt)
            orig_text = prompt
 
        new_kwargs.update(kwargs)
        for i in range(num_samples):
            repetition_penalty_dict = self.get_repetition_penalty_dict([orig_text] + results, repetition_penalty_multiplier)
            for j in range(3):
                res = self.ask_openai2(**new_kwargs, timeout=15, logit_bias=repetition_penalty_dict)
                if res is None:
                    time.sleep(5) # sleep a bit and try again
                    res = self.ask_openai2(**new_kwargs, timeout=15, logit_bias=repetition_penalty_dict)

                if len(res) < len(orig_text) * 2.5:
                    break
            if len(res) >= len(orig_text) * 2.5:
                return results
            results.append(res)
        return results

class MathGPT(GPT):
    def __init__(
        self,
        model="gpt-3.5-turbo-1106",
        prompt=None,

        # OPENAI QUERY PARAMS, NOTGIVEN is the default
        temperature = NOT_GIVEN,
        top_p = NOT_GIVEN,
        freq_penalty = NOT_GIVEN,
        presence_penalty = NOT_GIVEN,
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

    def ask_openai2(self, problem, sol1, sol2, model=None, temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None, **query_kwargs):
        message = "Pretend you are an expert Math teacher and python coder. \nCompare the following two solutions to a math problem: \"{problem}\"\n" +\
                "The Solution1 is: {sol1}\nSolution1 ends here. \n\n\nThe Solution2 is: {sol2}\nSolution2 ends here.\n" +\
                "Write a JSON string comparing the solutions. Think step by step and identify what is incorrect. Example:\n" +\
                            "{{\n" +\
                            "\"discussion\": \"Let's think step by step. First ... Therefore SolutionX seems to be correct, while SolutionY is likely incorrect\",\n" +\
                            "\"better_solution_is\": \"1\"\n" +\
                            "}}"
        message = message.format(problem=problem, sol1=sol1, sol2=sol2)
        return super().ask_openai2(message, model, temperature, top_p, frequency_penalty, presence_penalty, **query_kwargs)

if __name__ == '__main__':

    logger.add("learning.log", rotation = "100 MB")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    
    print("========= TESTING MATH GPT =========")
    math_gpt = MathGPT()
    problem = "Solve the equation 2x + 3 = 7"
    sol1 = "x = 2"
    sol2 = "x = 4"
    result = math_gpt.ask_openai2(problem, sol1, sol2)
    print(result)

    print("========= TESTING REPHRASE GPT =========")
    rephrase_gpt = RephraseGPT()
    problem = "If 3 boys each make 12 muffins for a bake sale, and 2 other girls are making 20 muffins each, how many total muffins will be for sale?"
    rephrase_gpt.ask_openai2_multiple(problem=problem, num_samples=3, repetition_penalty_multiplier=5)
    print(result)
    logger.info(f"Result: {result}")
