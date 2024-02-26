import json
import os
from tqdm import tqdm

os.chdir("teach-dataset/games/")
failed = 0
for filename in tqdm(os.listdir("train")):
    try:
        game = json.load(open("train/" + filename, 'r'))
        json.dump(game, open("train/" + filename, 'w+'), indent=4)
    except json.decoder.JSONDecodeError:
        failed += 1
print(f"Failed {failed} times")
