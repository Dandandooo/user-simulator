import sys
import os

from tqdm import tqdm

if len(sys.argv) > 1:
    datapaths = sys.argv[1:]
else:
    raise ValueError("Please provide a datapath")

paths = [os.path.join(datapath, folder) for datapath in datapaths for folder in os.listdir(datapath)]

for folder in tqdm(paths):
    for file in os.listdir(folder):
        if "answer" in file:
            continue
        with open(os.path.join(folder, file), "r") as f:
            # This got flagged by OpenAI
            replaced = f.read().replace("StoveBurner", "Stove")
            # Adding consistency
            replaced = replaced.replace("Imagine you ", "Imagine you, the COMMANDER, ")
            replaced = replaced.replace("instruct a robot ", "instruct a robot, named DRIVER, ")
            replaced = replaced.replace("the robot", "the DRIVER")
            replaced = replaced.replace(",,", ",")
            replaced = replaced.replace("the COMMANDER, the COMMANDER", "the COMMANDER")

        with open(os.path.join(folder, file), "w") as f:
            f.write(replaced)
