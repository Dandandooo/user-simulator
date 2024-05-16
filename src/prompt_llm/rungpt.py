import numpy as np
import os
import sys

from src.prompt_llm.gpt4_entire import run_gpt
from src.prompt_llm.gpt4_entire_eval import conf_matrix, metric_string

if not os.path.exists("llm_prompt_sessions/gpt4_valid/temp"):
    os.mkdir("llm_prompt_sessions/gpt4_valid/temp")

if len(sys.argv) > 1:
    datapath = sys.argv[1]
else:
    datapath = "llm_prompts_data/turns/valid/"
if len(sys.argv) > 2:
    destpath = sys.argv[2]
else:
    destpath = "llm_prompt_sessions/gpt4_valid/temp/"

for i, folder in enumerate(sorted(os.listdir(datapath))):
    # Todo: run with 0, 2, and 5 examples
    if os.path.exists(destpath + folder + "_result.json"):
        print("Skipping", folder, f"({i}/{len(os.listdir(datapath))})")
        continue
    print("Running", folder, f"({i}/{len(os.listdir(datapath))})")
    run_gpt(os.path.join(datapath, folder))

    with open(f"{destpath}{folder}_score.txt", "w") as f:
        f.write(metric_string())
    os.rename("gpt4_result.json", f"{destpath}{folder}_result.json")

overall = np.zeros((2, 2))
for folder in sorted(os.listdir(datapath)):
    overall += conf_matrix(f"{destpath}{folder}_result.json")
print(overall)
