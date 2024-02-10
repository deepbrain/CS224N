from loguru import logger
import sympy
import math

import asyncio
import multiprocessing
import time

import concurrent.futures
INVALID_ANSWER = -99999

def compute_result_impl(input_code_string, function_name):
    modified_name = function_name + "_modified5765765"  # new name to prevent recursion
    lines = input_code_string.split("\n")
    for i in range(len(lines)):
        if function_name in lines[i]:
            lines[i] = lines[i].replace(function_name, modified_name)
            break
    input_code_string = "\n".join(lines)
    local_namespace = {}
    exec(input_code_string, local_namespace)
    func_name = function_name  # Adjust if the function name varies
    error = ""
    if modified_name in local_namespace:
        # Call the function and return the result
        try:
            res = local_namespace[modified_name]()
            try:
                res = int(res)
            except:
                res = INVALID_ANSWER
            return res, error
        except Exception as e:
            # logger.error(f"An error occurred: {e}")
            error = str(e)
            # logger.error(f"Code that caused the error: {input_code_string}")
            return INVALID_ANSWER, error
    else:
        # Function name not found
        return INVALID_ANSWER, f"Function name '{func_name}' not found in code"

def target(queue, code, name):
    # Wrap the original function to store its result in a queue
    try:
        # Your logic here, ensure compute_result_impl is accessible
        result = compute_result_impl(code, name)
        queue.put(result)
    except Exception as error:
        queue.put((INVALID_ANSWER, str(error)))



async def run_subprocess_with_timeout(input_code_string, function_name, timeout):
    #serialize calls here via critical section
        ctx = multiprocessing.get_context('spawn')
        result_queue = ctx.Queue()
        process = ctx.Process(target=target, args=(result_queue, input_code_string, function_name))
        process.start()

            # Set up a future for the result queue to get an item
        start_time = time.time()
        while True:
            await asyncio.sleep(0.1)
            if result_queue.qsize() > 0:
                return result_queue.get()
            if (time.time()-start_time) > timeout:
                if process.is_alive():
                    process.kill()
                return (INVALID_ANSWER, "Timeout")

async def compute_result(input_code_string, function_name, lock, timeout = 10):
    async with lock:
        result = await run_subprocess_with_timeout(input_code_string, function_name, timeout)
        return result

# test case

test1 = """def problem() -> int:
    # Marion's score is 6 more than half of Ella's score
    # Ella got 4 incorrect answers
    # Let's assume Ella's score is x
    # Marion's score is (x/2) + 6
    # Marion's score is (x/2) + 6 = 40 - 4
    # (x/2) + 6 = 36
    # (x/2) = 36 - 6
    # (x/2) = 30
    # x = 30 * 2
    # x = 60
    # Marion's score is 60
    result = 60
    return result
"""

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
    lock = asyncio.Lock()
    # Test the compute_result function using async loop:
    res = asyncio.run(compute_result(test1, "problem", lock, 1))
    print(res)
    res = asyncio.run(compute_result(test_code, "problem", lock, 5))
    print(res)
    res = asyncio.run(compute_result(test_code, "problem", lock, 5))
    print(res)
    res = asyncio.run(compute_result(test_code, "problem", lock, 1))
    print(res)
