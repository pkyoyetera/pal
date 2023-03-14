import argparse
import logging
import time

import json

from tqdm.auto import tqdm

import pal.prompt.zero_shot_math_prompt
from util import solve_question, read_json


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    filename="logs/run.log",
)  # pass explicit filename here
logger = logging.getLogger()  # get the root logger

logger.info(f"Starting run at: {time.time()}")


def evaluate_prompts(path: str, out_file_path: str, result_loc: str):
    # open file to save results of successful prompts
    out_file = open(out_file_path, "w")
    assert out_file

    # file to save raw results from model
    # result_file = open(result_loc, "w")
    # assert result_file

    gsm_data = read_json(path)

    # some variables
    counts, successes = 0, 0
    attempts = {}

    for idx, row in tqdm(gsm_data.iterrows()):
        counts += 1

        prompt = (
            pal.prompt.zero_shot_math_prompt.ZERO_SHOT_MATH_PROMPT
            + row["input"]
            + '"""'
        )
        target = int(row["target"])

        solved_correctly, tries = solve_question(prompt, target)

        if solved_correctly:
            successes += 1

        # records number of attempts
        if tries in attempts:
            attempts[tries] += 1
        else:
            attempts[tries] = 1

        # records results
        record = {
            "Input": row["input"],
            "Target": row["target"],
            "Solved correctly": solved_correctly,
            "Tries": tries,
        }
        out_file.write(json.dumps(record) + "\n")

    out_file.close()  # we're done here


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default="datasets/gsm.jsonl")
    parser.add_argument(
        "--output", type=str, default="datasets/outputs/gsm_multiple_attempts.jsonl"
    )
    # parser.add_argument("--save_returned_prompts", type=bool, default=True)
    parser.add_argument("--loc", type=str, default="logs/multiple_tries.txt")
    args = parser.parse_args()

    evaluate_prompts(args.path, args.output, args.loc)
