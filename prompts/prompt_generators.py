from typing import Optional, Iterable, List
from cot_prompt_examples import COT_3_shot_promtps

PRIMER = ("You carefully provide accurate answers, and are brilliant at math and reasoning.\n"
          +"Think thoroughly of your code, explain the reasoning of each line of code in a comment.")
FUNCTION_NAME = "problem"

class PromptBase():
    def __init__(self, prompt:Optional[str] = None):
        self.prompt = prompt
    
    def __repr__(self):
        return self.prompt
    
    def get_prompt(self) -> str:
        return self.prompt

class Prompt(PromptBase):
    def __init__(
        self,
        question:str,
        primer:Optional[str] = PRIMER,
        answer:Optional[str] = None,
        question_pfix:str = "Q",
        answer_pfix:str = "A",
    ):
        prompt = (
            # Optionally add primer
            +(f"{primer}\n" if primer else "")
            +f"{question_pfix}: {question}"
            # Optionally add answer
            +(f"{answer_pfix}: {answer}" if answer else "")
        )
        super().__init__(prompt)

class CodePrompt(PromptBase):
    def __init__(
        self,
        question:str,
        function_name:str = FUNCTION_NAME,
        primer:Optional[str] = PRIMER,
        suffix:Optional[str] = None,
        answer:Optional[str] = None,
    ):
        prompt = (
            f"def {function_name}() -> int:\n"
            +f"    \"\"\""
            # Optionally add primer
            +(f"{primer}\n" if primer else "")
            +f"Solve the following problem. {question}"
            +(f"{suffix}\n" if suffix else "")
            +f"Your code must end with a \"return result\" line.\"\"\"\n"
            # Optionally add answer
            +(answer if answer else "")
        )
        super().__init__(prompt)

class ConcatPrompt(PromptBase):
    def __init__(self, prompts:Iterable[PromptBase]):
        prompt = ""
        for p in prompts[:-1]:
            prompt += f"{self._format_prompt(p)}\n\n"
        prompt += f"{self._format_prompt(prompts[-1])}\n"
        super().__init__(prompt)

    def _format_prompt(self, prompt:PromptBase) -> PromptBase:
        if prompt.get_prompt()[-1] == "\n":
            return PromptBase(prompt.get_prompt()[:-1])
        return prompt


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# Prompt library

def _get_regular_prompts(qn: str) -> List[str]:
    prompt1_fn = lambda qn: CodePrompt(qn).get_prompt()

    prompt2_primer = "You carefully provide accurate answers, and are brilliant at math and reasoning."
    prompt2_suffix = ("First, think thoroughly of the problem, and write a comment with a detailed step by step solution.\n"
                    "Then, cautiously write the code that solves the problem.")
    prompt2_fn = lambda qn: CodePrompt(qn, primer=prompt2_primer, suffix=prompt2_suffix).get_prompt()

    prompt3_primer = ("You carefully provide accurate answers, and are brilliant at math and reasoning.\n"
                    "Think thoroughly of your code, explain your reasoning using comments."
                    )
    prompt3_fn = lambda qn: CodePrompt(qn, primer=prompt3_primer).get_prompt()

    return [prompt_fn(qn) for prompt_fn in [prompt1_fn, prompt2_fn, prompt3_fn]]

def _get_cot_prompts(qn: str) -> List[str]:
    prompt1_fn = lambda qn: CodePrompt(qn).get_prompt()
    kshot_prompts = []
    for cot_prompt_group in COT_3_shot_promtps:
        prompt1_3shot_fn = lambda qn: ConcatPrompt(list(map(PromptBase, cot_prompt_group)) + [PromptBase(prompt1_fn(qn))]).get_prompt()
        kshot_prompts.append(prompt1_3shot_fn(qn))
    return kshot_prompts

def get_all_prompts(question: str) -> List[str]:
    """ Main entrance API, given a question as a string, returns multiple
    useful prompts to generate answers.
    """
    return _get_regular_prompts(question) + _get_cot_prompts(question)
    