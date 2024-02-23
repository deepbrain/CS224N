from multiprocessing import Pool
import multiprocessing
from loguru import logger
import sympy
from sympy import Symbol
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError

INVALID_ANSWER = -99999


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


log_lines = []
def log_trace(s):
    log_lines.append(str(s))


def add_log_traces_to_code(code_string):
    lines = code_string.split('\n')  # Split the code into lines.
    modified_lines = []

    for line in lines:
        leading_spaces = len(line) - len(line.lstrip())
        indent = ' ' * leading_spaces
        stripped_line = line.strip()

        # Only process non-comment lines to decide if a log_trace should be added.
        if not stripped_line.startswith('#'):
            modified_lines.append(line)  # Add the original line to the output.

            # Check for an assignment operation in the line.
            if '=' in stripped_line and not any(
                    kw in stripped_line for kw in ('if', 'for', 'while', 'elif', 'else', '==')):
                variable_name = stripped_line.split('=')[0].strip()
                # Handle only simple assignments, ignoring compound assignments or comparisons.
                if '+' in variable_name or '-' in variable_name or '*' in variable_name or '/' in variable_name:
                    variable_name = variable_name[0:-2]
                modified_lines.append(f"{indent}log_trace({variable_name})")
        else:
            # For comment lines, just add them as they are.
            modified_lines.append(line)

    return '\n'.join(modified_lines)


def child_task(input_code_string, function_name):
    original_globals = set(globals().keys())
    exec(input_code_string, globals())
    error = ''
    global log_lines
    log_lines = []
    try:
        res = globals()[function_name]()
        try:
            res = int(res)
        except Exception as e:
            res = INVALID_ANSWER
        return res, error, log_lines
    except Exception as e:
        error = str(e)
        return INVALID_ANSWER, error, log_lines
    finally:
        # Clean up the global namespace
        for name in set(globals().keys()) - original_globals:
            del globals()[name]


total_calls = 0
timeouts = 0
max_time = 30
global_pool = None
def compute_result(prompt_string, output_string, function_name, should_crop_solution=True):
    global total_calls
    global timeouts
    global max_time
    total_calls += 1
    global global_pool
    # disallow recursion
    modified_name = function_name + "_modified5765765" #new name to prevent recursion
    lines = prompt_string.split("\n")
    for i in range(len(lines)):
        if function_name in lines[i]:
            lines[i] = lines[i].replace(function_name, modified_name)
            break
    prompt_string = "\n".join(lines)

    # potentially crop solution
    if should_crop_solution:
        output_string = crop_solution(output_string)

    # concatenate prompt and code solution
    input_code_string = prompt_string + output_string
    input_code_string = add_log_traces_to_code(input_code_string)
    if global_pool is None:
        global_pool = Pool(processes=1)
    result_async = global_pool.apply_async(child_task, (input_code_string, modified_name))
    try:
        res, error, lls = result_async.get(timeout=max_time)
        max_time = 3
        return res, error, lls
    except Exception as e:
        error = str(e)
        if isinstance(e, multiprocessing.context.TimeoutError):
            error = f"'{input_code_string}': timed out after {max_time} seconds"
            global_pool.terminate()
            global_pool = None
            max_time = 30
            timeouts += 1
        return INVALID_ANSWER, error, []


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

test_code2 = """def problem() -> int:
    # number of guests at the first venue
    guests_1 = 0
    # number of guests at the second venue
    guests_2 = 0
    # cost of the first venue
    cost_1 = 200
    # cost of the second venue
    cost_2 = 0
    # cost of food for each guest at the first venue
    food_cost_1 = 5
    # calculate the cost of the first venue for a given number of guests
    def cost_1_for_guests(guests):
        return cost_1 + (guests * food_cost_1)
    # calculate the cost of the second venue for a given number of guests
    def cost_2_for_guests(guests):
        return cost_2 + (guests * 25)
    # find the number of guests at which the costs of the two venues are equal
    while cost_1_for_guests(guests_1) != cost_2_for_guests(guests_2):
        guests_1 += 1
    return guests_1': timed out after 3 seconds
"""

ps3 = """def problem() -> int:\n"""
test_code3 = """
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = sympy.Symbol('c')
    d = sympy.Symbol('d')
    e = sympy.Symbol('e')
    f = sympy.Symbol('f')
    g = sympy.Symbol('g')
    h = sympy.Symbol('h')
    i = sympy.Symbol('i')
    j = sympy.Symbol('j')
    eq1 = x + y + z + a + b + c + d + e + f + g + h + i + j - 76000
    eq2 = x - 0.3 * y - 0.3 * z - a - b - c - d - e - f - g - h - i - j
    eq3 = x * y * z * a * b * c * d * e * f * g * h * i * j - (76000) ** 2
    sol = sympy.solve((eq1, eq2, eq3), (x, y, z, a, b, c, d, e, f, g, h, i, j))
    return sol[x] + sol[y] + sol[z]
"""

test_code4 = """
    cream_price = cheddar_price / 2
    cold_cut_price = cheddar_price * 2
    total_cost = cheddar_price + cream_price + cold_cut_price
    result = total_cost
    return result
"""

test_code5 = """
    # Initialize variables
    eggs_per_day = 16
    eggs_eaten = 3
    eggs_baked = 4
    price_per_egg = 2
    # Calculate the number of eggs remaining after Janet eats breakfast and bakes muffins
    eggs_remaining = eggs_per_day - eggs_eaten - eggs_baked
    # Calculate the amount of money Janet makes every day at the farmers' market
    money_made = eggs_remaining * price_per_egg
    return money_made
"""

test_code6 = """
    # The shadows start at zero at noon
    shadow_length = 0
    # Every hour past noon shadows stretch an extra 5 feet
    for hour in range(1, 7):
        shadow_length += 5
    # Convert the shadow length from feet to inches
    shadow_length *= 12
    # Return the shadow length in inches
    return shadow_length
"""

if __name__ == "__main__":
    # test case
    print(compute_result(ps3, test_code6, "problem"))
    print(compute_result(ps3, test_code6, "problem"))
    print(compute_result(ps3, test_code5, "problem"))
    print(compute_result(ps3, test_code5, "problem"))
    print(compute_result(ps3, test_code5, "problem"))
