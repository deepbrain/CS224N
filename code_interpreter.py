from contextlib import contextmanager
import signal

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
        # Create a new, isolated namespace for each invocation
        local_namespace = {}
        # Execute the code in the string within the isolated namespace
        exec('import math\n' +input_code_string, local_namespace)
        # Assuming the function name is known and consistent
        func_name = function_name  # Adjust if the function name varies
        max_time = 3

        if func_name in local_namespace:
            # Call the function and return the result
            with timeout(max_time, input_code_string):
                try:
                    return local_namespace[func_name]()
                except Exception as e:
                    logger.error(f"An error occurred: {e}")
                    logger.error(f"Code that caused the error: {input_code_string}")
                    return -99999
        else:
            # Function name not found
            return -99999
    except Exception as e:
        # Handle any exception that occurs
        logger.error(f"An error occurred: {e}")
        logger.error(f"Code that caused the error: {input_code_string}")
        return -99999
