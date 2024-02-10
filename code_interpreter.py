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
        lines = prompt_string.split("\n")
        for i in range(len(lines)):
            if function_name in lines[i]:
                lines[i] = lines[i].replace(function_name, modified_name)
        prompt_string = "\n".join(lines)

        lines = output_string.split("\n") # prompt string contains the function
        for i in range(len(lines)):
            if "while True" in lines[i]:
                lines[i] = lines[i].replace("while True", "for i in range(1)")
        output_string = "\n".join(lines)

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


test_code = """def problem() -> int:
    # Solution 1
    # We can use a while loop to keep track of the time and the state of the bulbs.
    # We initialize the time to 0 and the states to 0, 0, 0.
    # We then use a while loop to increment the time by 1 and check if all three bulbs are on.
    # If they are, we return the time.
    # If not, we update the states of the bulbs based on their intervals and continue the loop.
    # This solution has a time complexity of O(n) where n is the maximum interval of the bulbs.
    time = 0
    states = [0, 0, 0]
    while True:
        time += 1
        for i in range(3):
            if states[i] == 0:
                states[i] = time % (2, 3, 4)[i]
        if all(state == 0 for state in states):
            return time
    # Solution 2
    # We can use a list comprehension to generate all possible combinations of states for the bulbs.
    # We then use another list comprehension to filter out the combinations where all three bulbs are off.
    # Finally, we return the minimum time from the remaining combinations.
    # This solution has a time complexity of O(n^2) where n is the maximum interval of the bulbs.
    intervals = [2, 3, 4]
    states = [[0, 0, 0]]
    for interval in intervals:
        new_states = []
        for state in states:
            for i in range(3):
                new_state = state[:]
                new_state[i] = (new_state[i] + interval) % interval
                new_states.append(new_state)
        states = new_states
    return min(time for state in states if all(state) for time in range(1, max(intervals) + 1))"""

if __name__ == "__main__":
    # test case
    print(compute_result("", test_code, "problem"))
