import traceback

from typing import List, Optional, Tuple

import pandas as pd

import black.parsing
from black import format_str, FileMode

from pal.core.backend import call_gpt
import pal.prompt.zero_shot_math_prompt

SOLVE_REQUEST = (
    """Could you rewrite the code above and fix the problem the stack trace mentions?"""
)


def read_json(path):
    import json

    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))

    task_df = pd.DataFrame(rows)
    return task_df


def format_response(solution: str) -> Optional[List[str]]:
    # getting the part that's got the solution in it:
    parts = [
        part
        for part in solution.split("\n\n")
        if len(part) > 0 and "def solution" in part
    ]

    return parts


def fix_function(solution: str) -> Optional[str]:
    # read the solution string from the function definition to
    # and just rely on black to format it correctly. Remove
    # any print lines, and just return the solution

    try:
        formatted = format_str(solution[: solution.rfind("\n\n")], mode=FileMode())
        # formatted = format_str(solution, mode=FileMode())
    except black.parsing.InvalidInput as invalid:
        # logger.exception(invalid)
        # logger.error(f"Could not format:\n{solution}")
        return None

    # assign return function to a variable that we'll access after execution
    formatted += "\n" + "prompt_result = solution()"
    return formatted


def exec_code(original_prompt: str, snippet: str) -> str | Exception:
    """ """
    # try to execute the code
    local = {}
    try:
        exec(snippet, globals(), local)
        return local["solution"]()
    except Exception as e:
        # logger.error(f"Error executing sample code:\n\n {sample}\n\n {e}")
        # logger.info("Trying again...")
        raise e  # just bump it up


def solve_question(
    question_prompt: str, target: int, depth: int = 1
) -> Tuple[bool, int]:
    # get response from the model
    result = call_gpt(question_prompt, max_tokens=256)
    if not result:
        return False, depth  # fixme: return something more useful

    possible_snippets = format_response(result[0])
    if not possible_snippets:
        return False, depth

    for snippet in possible_snippets:
        try:
            solution = exec_code(question_prompt, snippet)
        except Exception:
            if depth >= 2:  # don't try more than twice
                return False, depth

            # get stack trace
            tb = traceback.format_exc()

            # add stack trace to prompt and try to solve again
            second_attempt = (
                question_prompt
                + "\n\n"
                + snippet
                + "\n\n"
                + pal.prompt.zero_shot_math_prompt.EXCEPTION_PROMPT
                + "\n\n"
                + str(tb)  # str(e)
                + "\n\n"
                + SOLVE_REQUEST
            )

            return solve_question(second_attempt, target, depth + 1)

        if int(solution) == target:
            return True, depth
        return False, depth
