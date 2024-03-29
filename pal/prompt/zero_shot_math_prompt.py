ZERO_SHOT_MATH_PROMPT = (
    """
\"\"\"Write a function in python code that solves the following math problem. The function should take no arguments and use only local variables. The function should have the name 'solution' and should return the result of the computation not print it."""
    + "\n\n"
)


# Write a function in python code that solves the following math problem.
# The function should take no arguments and only use local variables.
# The function should have the name "solution" and should return only the result of the computation.
# Please use legible variable names and use minimal, comments only where they're needed.
# Do not write anything other than the function.

EXCEPTION_PROMPT = """\n\nRunning the code above fails and results in the stack trace below the next sentence. Take a 
look at the stacktrace below and try to fix the problem in the code.\n\n"""
