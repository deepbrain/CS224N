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


def compute_result(input_code_string, function_name):
    try:
        modified_name = function_name + "_modified5765765" #new name to prevent recursion
        lines = input_code_string.split("\n")
        for i in range(len(lines)):
            if function_name in lines[i]:
                lines[i] = lines[i].replace(function_name, modified_name)
            if "while True" in lines[i]:
                return INVALID_ANSWER, "Infinite loop detected"
        input_code_string = "\n".join(lines)
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
