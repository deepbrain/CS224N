import re
from math_problem import Problem, Solution

def split_and_delete_comments(s):
    # remove all python style comments from the text s
    # remove all # style comments:
    s = re.sub(r'#.*', '', s)
    # remove all """ style comments:
    s = re.sub(r'""".*?"""', '', s, flags=re.DOTALL)
    # remove 'return result' lines from s:
    s = re.sub(r'return result', '', s)
    #remove all empty lines:
    s = re.sub(r'\n\s*\n', '\n', s)
    #remove def function_name lines:
    s = re.sub(r'def \w*\(.*\):', '', s)
    #split the text into lines:
    s = s.split("\n")
    return s

class StepwiseSolution(Solution): #todo: implement this
    def __init__(self, problem, model_manager, prompt):
        super().__init__(problem, model_manager, prompt)
        self.steps = []
    def solve(self):
        return

    async def eval_solution(self, solution):
        shot_commenter = ""
        prompt = shot_commenter + self.question + "\nSolution:\n"
        s = split_and_delete_comments(solution)
        errors = 0
        warnings = 0
        good = 0
        for line in s:
            prompt += line + "\n#"
            response = await self.model_manager.get_completion(prompt, max_tokens=100, completion_only=True, stop_string = "\n")
            if "Error" in response:
               errors += 1
            elif "Potential" in response:
               warnings += 1
            elif "Good" in response:
               good += 1
            prompt += response
        return errors, warnings, good

