import json
import os
from tqdm import tqdm

folders = [
    "teach-dataset/games/train",
    "teach-dataset/games/valid_seen",
    "teach-dataset/games/valid_unseen",
]

def convert_dialogue(dialogue_action: dict) -> dict:
    agent = dialogue_action['da_metadata']['agent'].upper()
    action = {
        agent: {
            'action': 'dialogue',
            'utterance': dialogue_action['utterance'],
            'das': [da for da in dialogue_action['da_metadata']['das'] if da]
        },
        "COMMANDER" if agent == "DRIVER" else "DRIVER": {
            'action': '<observe>',
        },
        'turn_action': 'dialogue'
    }
    return action

def convert_generic(dialogue_action: dict, action: str) -> dict:
    agent = {0: "COMMANDER", 1: "DRIVER"}[dialogue_action['agent_id']]
    action = {
        agent: {
            'action': action,
        },
        "COMMANDER" if agent == "DRIVER" else "DRIVER": {
            'action': '<observe>',
        },
        'turn_action': action
    }
    return action


def parse_game(filename: str) -> dict:
    file = json.load(open(filename, 'r'))
    data = {}
    task = file['tasks'][0]  # I am assuming there is only one task. I haven't seen more yet
    data['goal'] = task['desc']
    # Action ID 100 is a dialogue action
    turns = data['turns'] = []
    for interaction in task['episodes'][0]['interactions']:
        match interaction['action_id']:
            case 100:
                turns.append(convert_dialogue(interaction))
            case 0:
                turns.append(convert_generic(interaction, '<stop moving>'))
            case 1 | 2 | 3 | 10 | 11 | 12 | 13:
                if turns and turns[-1]['turn_action'] == '<move>':
                    continue
                turns.append(convert_generic(interaction, '<move>'))
            # case 4 | 5:
            #     if turns and turns[-1]['turn_action'] == '<turn>':
            #         continue
            #     turns.append(convert_generic(interaction, '<turn>'))
            # case 6 | 7 | 8 | 9:
            #     if turns and turns[-1]['turn_action'] == '<look around>':
            #         continue
            #     turns.append(convert_generic(interaction, '<look around>'))
            case 200:
                turns.append(convert_generic(interaction, '<pickup>'))
            case 201:
                turns.append(convert_generic(interaction, '<putdown>'))
            case 202:
                turns.append(convert_generic(interaction, '<open>'))
            case 203:
                turns.append(convert_generic(interaction, '<close>'))
            case 204:
                turns.append(convert_generic(interaction, '<toggle on>'))
            case 205:
                turns.append(convert_generic(interaction, '<toggle off>'))
            case 206:
                turns.append(convert_generic(interaction, '<slice>'))
            case 207:
                turns.append(convert_generic(interaction, '<dirty>'))
            case 208:
                turns.append(convert_generic(interaction, '<clean>'))
            case 209:
                turns.append(convert_generic(interaction, '<fill>'))
            case 210:
                turns.append(convert_generic(interaction, '<empty>'))
            case 211:
                turns.append(convert_generic(interaction, '<pour>'))
            case 212:
                turns.append(convert_generic(interaction, '<break>'))
            case _:
                # I don't know how to handle anything else yet
                pass
    return data


for folder in folders:
    save_name = folder.split("/")[-1] + "_turn"
    games = []
    for filename in tqdm(os.listdir(folder)):
        if filename.endswith(".json"):
            games.append(parse_game(os.path.join(folder, filename)))
    with open(f"teach-dataset-parsed/{save_name}.json", 'w+') as f:
        json.dump(games, f, indent=4)
