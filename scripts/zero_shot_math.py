import argparse
import logging
import time


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename='../logs/run.log')  # pass explicit filename here
logger = logging.getLogger()  # get the root logger

logger.info(f"Starting run at: {time.time()}")


import json
import signal

import black.parsing
from black import format_str, FileMode

import pandas as pd

from contextlib import contextmanager
from tqdm.auto import tqdm
from typing import Optional

from pal.core.backend import call_gpt
import pal.prompt.zero_shot_math_prompt


# logging.basicConfig(filename="run.log", filemode="w+", level=logging.DEBUG)
# logger = logging.getLogger(__name__)
# # import logging

# borrowed from PAL codebase
@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"block timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def read_json(path):
    import json
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))

    task_df = pd.DataFrame(rows)
    return task_df


def fix_function(solution: str) -> Optional[str]:
    # read the solution string from the function definition to
    # and just rely on black to format it correctly. Remove
    # any print lines, and just return the solution

    try:
        formatted = format_str(solution[:solution.rfind('\n\n')], mode=FileMode())
        # formatted = format_str(solution, mode=FileMode())
    except black.parsing.InvalidInput as invalid:
        logger.exception(invalid)
        logger.error(f"Could not format:\n{solution}")
        return None

    # assign return function to a variable that we'll access after execution
    formatted += '\n' + "prompt_result = solution()"
    return formatted


def evaluate_prompts(path: str, out_file_path: str):
    # open file to save results of successful prompts
    out_file = open(out_file_path, "w")
    assert out_file

    gsm_data = read_json(path)

    correct_results = 0
    count = 0
    failed = 0
    wrong_results = 0

    for idx, row in tqdm(gsm_data.iterrows()):
        # did we get the expected answer correctly?
        expected = False

        prompt = pal.prompt.zero_shot_math_prompt.ZERO_SHOT_MATH_PROMPT + row['input'] + '"""'
        # assert prompt

        # Call GPT-3
        response = call_gpt(prompt, max_tokens=256)

        # record the response anyway, so we can analyze the code
        # for future improvement purposes, whether it runs or not

        # format returned prompt into executable code
        formatted_response = fix_function(response[0])
        if formatted_response is None:
            logger.error("Code formatting failed.")
            failed += 1
            count += 1
            continue

        # execute code for prompt if it's valid
        try:
            local = {}
            # some code is causing problems
            with timeout(2):
                exec(formatted_response, globals(), local)  # output will be stored in variable called prompt_result
        except Exception as e:
            print(f"Failed to exec:\n {formatted_response}")
            logger.exception(e)
            # need to log e for further examination of failure
            failed += 1
            count += 1
            continue

        # evaluate return value by comparing to target from input file
        expected_target = float(row['target'])

        try:
            if int(local['prompt_result']) == int(expected_target):
                expected = True
                correct_results += 1
            else:
                expected = False
                wrong_results += 1

            count += 1
        except Exception as e:
            logger.exception(e)
            failed += 1
            continue

        # save
        res = {
            "input": prompt,
            "response": formatted_response,  # fixme: perhaps we should save the actual unformatted response?
            "target": row["target"],
            "result": local['prompt_result'],
            "is_correct": "yes" if expected else "no"
        }
        out_file.write(json.dumps(res))

    out_file.close()  # we're done here

    return count, correct_results, wrong_results, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default="../datasets/gsm.jsonl")
    parser.add_argument("--output", type=str, default="../datasets/outputs/one_shot_gsm.jsonl")

    args = parser.parse_args()

    count, correct, wrong, failed = evaluate_prompts(args.path, args.output)

    print(f"Count: {count}\nCorrect: {correct}\nWrong: {wrong}\nFailed: {failed}\n")

    # save to file
    with open("../logs/results.txt", "w+") as ff:
        metrics = {
            "count": count,
            "correct": correct,
            "wrong": wrong,
            "failed": failed
        }
        ff.write(json.dumps(metrics))
    ff.close()
