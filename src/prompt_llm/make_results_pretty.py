import sys
import os
import re
import json
from tqdm import tqdm

if len(sys.argv) > 1:
    datapath = sys.argv[1]
else:
    raise ValueError("Please provide a datapath")

destpath = f"llm_prompt_sessions/prettygpt/{os.path.basename(datapath)}"
if not os.path.exists(destpath):
    os.makedirs(destpath)
else:
    print("Destination path already exists. Overwriting files.")

goal_re = re.compile(r"(?<=\n\n\n)Goal: (?:(?!COMMANDER response:).)+", re.MULTILINE + re.DOTALL)

for file in tqdm(os.listdir(datapath)):
    if "result" not in file:
        continue
    # game = sorted(json.load(open(os.path.join(datapath, file), "r")), key=lambda x: len(goal_re.search(x["prompt"]).group()))
    game = json.load(open(os.path.join(datapath, file), "r"))

    turns = [goal_re.search(game[0]["prompt"]).group().strip()]
    responses = [game[0]["response"]]
    answers = [game[0]["answer"]]

    heights = [max(len(turns[0].split("\n")), len(responses[0].split("\n")), len(answers[0].split("\n")))]

    for turn in game[1:]:
        t = goal_re.search(turn["prompt"]).group().strip().removeprefix("\n".join(turns)).strip()
        responses.append(turn["response"])
        answers.append(turn["answer"])
        turns.append(t)
        heights.append(max(len(t.split("\n")), len(turn["response"].split("\n")), len(turn["answer"].split("\n"))))

    # Width for each column
    width1 = 0
    width2 = 0
    width3 = 0

    for turn, response, answer in zip(turns, responses, answers):
        for line in turn.splitlines():
            width1 = max(width1, len(line))
        for line in response.splitlines():
            width2 = max(width2, len(line))
        for line in answer.splitlines():
            width3 = max(width3, len(line))

    # Print the pretty table

    table = "+=====+=" + "=" * width1 + "=+=" + "=" * width2 + "=+=" + "=" * width3 + "=+"
    table += f"\n|  #  | {'Turn'.center(width1)} | {'Response'.center(width2)} | {'Answer'.center(width3)} |"
    table += "\n" + "+=====+=" + "=" * width1 + "=+=" + "=" * width2 + "=+=" + "=" * width3 + "=+"

    for i, (height, turn, response, answer) in enumerate(zip(heights, turns, responses, answers), start=0):
        turn_lines = [""] * ((height - len(turn.splitlines())) / 2).__floor__() + turn.splitlines() + [""] * ((height - len(turn.splitlines())) / 2).__ceil__()
        response_lines = [""] * ((height - len(response.splitlines())) / 2).__floor__() + response.splitlines() + [""] * ((height - len(response.splitlines())) / 2).__ceil__()
        answer_lines = [""] * ((height - len(answer.splitlines())) / 2).__floor__() + answer.splitlines() + [""] * ((height - len(answer.splitlines())) / 2).__ceil__()
        for t, r, a in zip(turn_lines, response_lines, answer_lines):
            table += f"\n| {f'{i: ^3}'} | {t.ljust(width1)} | {r.center(width2)} | {a.center(width3)} |"

        table += "\n+-----+-" + "-" * width1 + "-+-" + "-" * width2 + "-+-" + "-" * width3 + "-+"

    open(os.path.join(destpath, file.replace("_result.json", "_pretty.txt")), "w").write(table)


