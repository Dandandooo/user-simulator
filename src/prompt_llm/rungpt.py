import numpy as np
import os
import sys

# from src.prompt_llm.gpt4_entire import run_gpt
from src.prompt_llm.gpt4o_entire import run_gpt
from src.prompt_llm.gpt4_entire_eval import conf_matrix, metric_string, all_matrix, das_confusion, das_stats

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

if not os.path.exists(destpath):
    os.mkdir(destpath)

for i, folder in enumerate(sorted(os.listdir(datapath)), start=1):
    # Todo: run with 0, 2, and 5 examples
    if os.path.exists(os.path.join(destpath, folder + "_result.json")):
        print("Skipping", folder, f"({i}/{len(os.listdir(datapath))})")
        continue
    print("Running", folder, f"({i}/{len(os.listdir(datapath))})")
    run_gpt(os.path.join(datapath, folder))

    with open(os.path.join(destpath, f"{folder}_score.txt"), "w") as f:
        f.write(metric_string())
    os.rename("gpt4_result.json", os.path.join(destpath, f"{folder}_result.json"))

overall = np.zeros((2, 2))
alls = np.zeros((3, 2))
speaks = np.zeros((3, 2))
observes = np.zeros((3, 2))
actions = np.zeros((3, 2))
das = np.zeros((20, 20))
for folder in sorted(os.listdir(datapath)):
    try:
        overall += conf_matrix(os.path.join(destpath, f"{folder}_result.json"))
        alls += all_matrix(os.path.join(destpath, f"{folder}_result.json"))
        speaks += all_matrix(os.path.join(destpath, f"{folder}_result.json"), prev="speak")
        observes += all_matrix(os.path.join(destpath, f"{folder}_result.json"), prev="observe")
        actions += all_matrix(os.path.join(destpath, f"{folder}_result.json"), prev="action")
        das += das_confusion(os.path.join(destpath, f"{folder}_result.json"))
    except FileNotFoundError:
        # print("Skipping", folder)
        continue
print("Overall Confusion Matrix:", overall, sep="\n")
print("Entire Confusion Matrix:", alls, sep="\n")
print("Speak Confusion Matrix:", speaks, sep="\n")
print("Observe Confusion Matrix:", observes, sep="\n")
print("Action Confusion Matrix:", actions, sep="\n")
print("Dialogue Act Statistics:", sep="\n")
das_stats(das)
