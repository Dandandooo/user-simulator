import numpy as np
import sys
import os

from tqdm import tqdm

from src.prompt_llm.gpt4_entire_eval import calc_baseline, baseline_das, das_stats

destpath = sys.argv[1]

overall = np.zeros((2, 2))
alls = np.zeros((2, 2))
speaks = np.zeros((2, 2))
observes = np.zeros((2, 2))
actions = np.zeros((2, 2))
das = np.zeros((20, 20))
for file in tqdm(sorted(os.listdir(destpath))):
    if "result" not in file:
        continue
    overall += calc_baseline(os.path.join(destpath, file), matrix=True)
    speaks += calc_baseline(os.path.join(destpath, file), prev="speak", matrix=True)
    observes += calc_baseline(os.path.join(destpath, file), prev="observe", matrix=True)
    actions += calc_baseline(os.path.join(destpath, file), prev="action", matrix=True)
    das += baseline_das(os.path.join(destpath, file), prev=None)
print("Overall Confusion Matrix:", overall, sep="\n")
print("Entire Confusion Matrix:", alls, sep="\n")
print("Speak Confusion Matrix:", speaks, sep="\n")
print("Observe Confusion Matrix:", observes, sep="\n")
print("Action Confusion Matrix:", actions, sep="\n")
print("Dialogue act statistics:")
das_stats(das)
