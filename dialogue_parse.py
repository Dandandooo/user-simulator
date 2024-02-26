import json
import os
from tqdm import tqdm

total_appended = 0
total_skipped = 0

def parse_interactions(interactions: list[dict]) -> list[dict[str, str or list]]:
    global total_appended
    global total_skipped

    skipped = appended = 0

    data = []

    for interaction in interactions:
        if "da_metadata" not in interaction:
            skipped += 1
            continue
        agent = interaction["da_metadata"]['agent']
        utterance = interaction["da_metadata"]['utterance']
        text_segments = interaction["da_metadata"]['text_segments']
        das = interaction["da_metadata"]['das']

        data.append({
            "agent":agent, 
             "utterance":utterance, 
             "text_segments":text_segments, 
             "das":das,
        })

        appended += 1

    total_appended += appended
    total_skipped += skipped

    # print(f"Skipped {skipped} interactions") 
    # print(f"Appended {appended} interactions")
    return data

def parse_edh(filename) -> list[dict[str, str or list]]:
    # Load the individual game json file
    game: dict = json.load(open(filename, 'r'))
    return parse_interactions(game['interactions']) 

def parse_game(filename) -> list[dict[str, str or list]]:
    # Load the individual game json file
    game: dict = json.load(open(filename, 'r'))

    data = []
    
    for task in game['tasks']:
        for episode in task['episodes']:
            data += parse_interactions(episode['interactions'])
    return data

# edh_instances = [
#     "teach-dataset/edh_instances/train",
#     "teach-dataset/edh_instances/valid_seen",
#     "teach-dataset/edh_instances/valid_unseen",
# ]

game_paths = [
    ("teach-dataset/games/train", "train"),
    ("teach-dataset/games/valid_seen", "valid_seen"),
    ("teach-dataset/games/valid_unseen", "valid_unseen"),
]

save_path = "teach-dataset-parsed/"
if not os.path.exists(save_path):
    os.mkdir(save_path)

for folder, save_name in game_paths:
    print(f"\x1b[33;1mProcessing {save_name} games...\x1b[0m")
    games = []
    for game in tqdm(os.listdir(folder)):
        if game.endswith('.json'):
            games += parse_game(os.path.join(folder, game))
    json.dump(games, open(os.path.join(save_path, save_name + ".json"), 'w'), indent=4)
    print("\x1b[32;4mDone!\x1b[0m")

print(f"Total appended: {total_appended}")
print(f"Total skipped: {total_skipped}")
