from contextlib import contextmanager
import signal
from loguru import logger
import sympy
import math

INVALID_ANSWER = -99999

# taken from
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, program):
    def timeout_handler(signum, frame):
        raise Exception(f"'{program}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def crop_solution(out):
    # Crops solution such that the last function in the output code is the one containing the solution.
    # This is particularly useful for few-shot prompts, where the model just keeps generating
    # mode problems and their solutions afterwards.
    out = out + "\n" 
    s = out.find("\n    return")
    e = out.find("\n", s+1)
    if e == -1:
        return out
    return out[:e]

def compute_result(prompt_string, output_string, function_name, should_crop_solution=True):
    try:
        # fix recursion
        modified_name = function_name + "_modified5765765" #new name to prevent recursion
        lines = prompt_string.split("\n") # prompt string contains the function
        for i in range(len(lines)):
            if function_name in lines[i]:
                lines[i] = lines[i].replace(function_name, modified_name)
                # no break needed since we are changing the function name on the prompt
        prompt_string = "\n".join(lines)

        # potentially crop solution
        if should_crop_solution:
            output_string = crop_solution(output_string)

        # concatenate prompt and code solution
        input_code_string = prompt_string + output_string

        # Create a new, isolated namespace for each invocation
        local_namespace = {}
        # Execute the code in the string within the isolated namespace
        exec("import math\nimport sympy\n" +input_code_string, local_namespace)
        # Assuming the function name is known and consistent
        func_name = function_name  # Adjust if the function name varies
        max_time = 3
        error = ""

        if modified_name in local_namespace:
            # Call the function and return the result
            with timeout(max_time, input_code_string):
                try:
                    res = local_namespace[modified_name]()
                    try:
                        res = int(res)
                    except:
                        res = INVALID_ANSWER
                    return res, error
                except Exception as e:
                    #logger.error(f"An error occurred: {e}")
                    error = str(e)
                    #logger.error(f"Code that caused the error: {input_code_string}")
                    return INVALID_ANSWER, error
        else:
            # Function name not found
            return INVALID_ANSWER, f"Function name '{func_name}' not found in code"
    except Exception as e:
        # Handle any exception that occurs
        #logger.error(f"An error occurred: {e}")
        error = str(e)
        #logger.error(f"Code that caused the error: {input_code_string}")
        return INVALID_ANSWER, error
