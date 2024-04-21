import json
import os
from tqdm import tqdm

os.chdir("teach-dataset/games/")
failed = 0
for folder in ["train/", "valid_seen/", "valid_unseen/"]:
    for filename in tqdm(os.listdir(folder)):
        try:
            game = json.load(open(folder + filename, 'r'))
            json.dump(game, open(folder + filename, 'w+'), indent=4)
        except json.decoder.JSONDecodeError:
            failed += 1
print(f"Failed {failed} times")

os.chdir("../all_game_files/")
failed = 0
for filename in tqdm(os.listdir(".")):
    try:
        game = json.load(open(filename, 'r'))
        json.dump(game, open(filename, 'w+'), indent=4)
    except json.decoder.JSONDecodeError:
        failed += 1
print(f"Failed {failed} times")
