from typing import Optional, Iterable, List
from prompts.cot_prompt_examples import COT_3_shot_promtps
from itertools import chain

PRIMER = ("You carefully provide accurate answers, and are brilliant at math and reasoning.\n"
          + "Think thoroughly of your code, explain the reasoning of each line of code in a comment.")
FUNCTION_NAME = "problem"


class PromptBase():
    def __init__(self, prompt: Optional[str] = None):
        self.prompt = prompt

    def __repr__(self):
        return self.prompt

    def get_prompt(self) -> str:
        return self.prompt


class Prompt(PromptBase):
    def __init__(
            self,
            question: str,
            primer: Optional[str] = PRIMER,
            answer: Optional[str] = None,
            question_pfix: str = "Q",
            answer_pfix: str = "A",
    ):
        prompt = (
            # Optionally add primer
                +(f"{primer}\n" if primer else "")
                + f"{question_pfix}: {question}"
                # Optionally add answer
                + (f"{answer_pfix}: {answer}" if answer else "")
        )
        super().__init__(prompt)


class CodePrompt(PromptBase):
    def __init__(
            self,
            question: str,
            function_name: str = FUNCTION_NAME,
            primer: Optional[str] = PRIMER,
            suffix: Optional[str] = None,
            answer: Optional[str] = None,
    ):
        prompt = (
                f"def {function_name}() -> int:\n"
                + f"    \"\"\""
                # Optionally add primer
                + (f"{primer}\n" if primer else "")
                + f"Solve the following problem. {question}"
                + (f"{suffix}\n" if suffix else "")
                + f"\"\"\"\n"
                # Optionally add answer
                + (answer if answer else "")
        )
        super().__init__(prompt)


class ConcatPrompt(PromptBase):
    def __init__(self, prompts: Iterable[PromptBase]):
        prompt = ""
        for p in prompts[:-1]:
            prompt += f"{self._format_prompt(p)}\n\n"
        prompt += f"{self._format_prompt(prompts[-1])}\n"
        super().__init__(prompt)

    def _format_prompt(self, prompt: PromptBase) -> PromptBase:
        if prompt.get_prompt()[-1] == "\n":
            return PromptBase(prompt.get_prompt()[:-1])
        return prompt


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# Prompt library

def _get_regular_prompts(qn: str) -> List[str]:
    prompt1_fn = lambda qn: CodePrompt(qn).get_prompt()

    prompt2_primer = "You carefully provide accurate answers, and are brilliant at math and reasoning."
    prompt2_suffix = (
        "First, think thoroughly of the problem, and write a comment with a detailed step by step solution.\n"
        "Then, cautiously write the code that solves the problem.")
    prompt2_fn = lambda qn: CodePrompt(qn, primer=prompt2_primer, suffix=prompt2_suffix).get_prompt()

    prompt3_primer = ("You carefully provide accurate answers, and are brilliant at math and reasoning.\n"
                      "Think thoroughly of your code, explain your reasoning using comments."
                      )
    prompt3_fn = lambda qn: CodePrompt(qn, primer=prompt3_primer).get_prompt()

    prompt4_suffix = "Elaborate your thinking step by step in comments before each code line below"
    prompt4_fn = lambda qn: CodePrompt(qn, primer=prompt3_primer, suffix=prompt4_suffix).get_prompt()

    prompt5_suffix = "Add comments before each line."
    prompt5_fn = lambda qn: CodePrompt(qn, primer=prompt3_primer, suffix=prompt5_suffix).get_prompt()

    prompt6_suffix = "Be accurate and think step by step in comments before each code line below."
    prompt6_fn = lambda qn: CodePrompt(qn, primer=prompt3_primer, suffix=prompt6_suffix).get_prompt()

    prompt7_suffix = "Find unusual solution and comment before each of your line of code."
    prompt7_fn = lambda qn: CodePrompt(qn, primer=prompt3_primer, suffix=prompt7_suffix).get_prompt()

    prompt8_suffix = "In your comments write an algebraic formula based on the problem, solve it algebraically, then write code to calculate the result."
    prompt8_fn = lambda qn: CodePrompt(qn, primer=prompt3_primer, suffix=prompt8_suffix).get_prompt()

    prompt9_suffix = "Find the most elegant and correct solution."
    prompt9_fn = lambda qn: CodePrompt(qn, primer=prompt3_primer, suffix=prompt9_suffix).get_prompt()

    prompt10_suffix = "Think step by step in comments before each code line below."
    prompt10_fn = lambda qn: CodePrompt(qn, primer=prompt3_primer, suffix=prompt10_suffix).get_prompt()

    prompt11_suffix = "You must elaborate your thinking in comments below."
    prompt11_fn = lambda qn: CodePrompt(qn, primer=prompt3_primer, suffix=prompt11_suffix).get_prompt()

    prompt12_suffix = (
        "Is this a simple math or algebra problem? For algebra problems, you must elaborate and solve it algebraically in the comments first, then write code to calculate the result.\n"
        "For simple math problems, you can write code to calculate the result directly.")
    prompt12_fn = lambda qn: CodePrompt(qn, primer=prompt3_primer, suffix=prompt12_suffix).get_prompt()

    prompt13_suffix = "First, let's solve this problem using pure symbolic expressions.\n"
    "Elaborate with your algebraic skills below. Use x,y,z...to denote unknown variables.\n"
    "Use a,b,c... to denote given constants. Then write a pure python code to compute and return the result."
    prompt13_fn = lambda qn: CodePrompt(qn, primer=prompt3_primer, suffix=prompt13_suffix).get_prompt()

    # print()

    prompt_fn_list = [prompt1_fn, prompt2_fn, prompt3_fn, prompt4_fn, prompt5_fn,
                      prompt6_fn, prompt7_fn, prompt8_fn, prompt9_fn, prompt10_fn,
                      prompt11_fn, prompt12_fn, prompt13_fn]

    return [prompt_fn(qn) for prompt_fn in prompt_fn_list]


def _get_few_shot_prompts(qn: str) -> List[str]:
    prompt1_fn = lambda qn: CodePrompt(qn).get_prompt()
    kshot_prompts = []

    # We currently have to do this conversion since we currently only effectively support
    # a context window of 1024 tokens and a 3shot prompt results in ~900 tokens already.
    # Hence we split up 5 3-shot prompts into 15 1-shot prompts.
    COT_1_shot_promtps = chain.from_iterable(COT_3_shot_promtps)
    COT_1_shot_promtps = [[one_shot_prompt] for one_shot_prompt in COT_1_shot_promtps]

    for cot_prompt_group in COT_1_shot_promtps:
        prompt1_1shot_fn = lambda qn: ConcatPrompt(
            list(map(PromptBase, cot_prompt_group)) + [PromptBase(prompt1_fn(qn))]).get_prompt()
        kshot_prompts.append(prompt1_1shot_fn(qn))
    return kshot_prompts


def get_all_prompts(question: str) -> List[str]:
    """ Main entrance API, given a question as a string, returns multiple
    useful prompts to generate answers.
    """
    return _get_regular_prompts(question) + _get_few_shot_prompts(question)
