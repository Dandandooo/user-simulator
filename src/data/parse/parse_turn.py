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
            'das': [da[0].upper() + da[1:] for da in dialogue_action['da_metadata']['das'] if da]
        },
        "COMMANDER" if agent == "DRIVER" else "DRIVER": {
            'action': '<observe>',
        },
        'turn_action': 'dialogue'
    }
    return action

def convert_generic(generic_action: dict, action: str) -> dict:
    agent = {0: "COMMANDER", 1: "DRIVER"}[generic_action['agent_id']]
    act = {
        agent: {
            'action': f'<{action}>',
        },
        "COMMANDER" if agent == "DRIVER" else "DRIVER": {
            'action': '<observe>',
        },
        'turn_action': action
    }
    return act

# Pass in action without the < > tags
def convert_object(object_action: dict, action: str) -> dict:
    agent = {0: "COMMANDER", 1: "DRIVER"}[object_action['agent_id']]
    obj = object_action['oid'].split('|')[0]  # Assuming this is the case for every example
    act = {
        agent: {
            'action': f'<{action} {obj}>',
            'object': obj
        },
        "COMMANDER" if agent == "DRIVER" else "DRIVER": {
            'action': '<observe>',
        },
        'turn_action': action
    }
    return act


def parse_game(filename: str) -> dict:
    file = json.load(open(filename, 'r'))
    data = {}
    task = file['tasks'][0]  # I am assuming there is only one task. I haven't seen more yet
    data['goal'] = task['desc']
    # Action ID 100 is a dialogue action
    turns = data['turns'] = []
    # print(filename)
    for interaction in task['episodes'][0]['interactions']:
        match interaction['action_id']:
            case 100:
                turns.append(convert_dialogue(interaction))
            case 0:
                turns.append(convert_generic(interaction, 'stop moving'))
            case 1 | 2 | 3 | 10 | 11 | 12 | 13:
                if turns and turns[-1]['turn_action'] == 'move':
                    continue
                if interaction['agent_id'] == 0:  # Don't track movement for the commander
                    continue
                turns.append(convert_generic(interaction, 'move'))
            # case 4 | 5:
            #     if turns and turns[-1]['turn_action'] == '<turn>':
            #         continue
            #     turns.append(convert_generic(interaction, '<turn>'))
            # case 6 | 7 | 8 | 9:
            #     if turns and turns[-1]['turn_action'] == '<look around>':
            #         continue
            #     turns.append(convert_generic(interaction, '<look around>'))
            case 200:
                if interaction["oid"] is None or interaction["agent_id"] == 0:
                    continue
                turns.append(convert_object(interaction, 'pickup'))
            case 201:
                if interaction["oid"] is None or interaction["agent_id"] == 0:
                    continue
                turns.append(convert_object(interaction, 'putdown'))
            case 202:
                if interaction["oid"] is None or interaction["agent_id"] == 0:
                    continue
                turns.append(convert_object(interaction, 'open'))
            case 203:
                if interaction["oid"] is None or interaction["agent_id"] == 0:
                    continue
                turns.append(convert_object(interaction, 'close'))
            case 204:
                if interaction["oid"] is None or interaction["agent_id"] == 0:
                    continue
                turns.append(convert_object(interaction, 'toggle on'))
            case 205:
                if interaction["oid"] is None or interaction["agent_id"] == 0:
                    continue
                turns.append(convert_object(interaction, 'toggle off'))
            case 206:
                if interaction["oid"] is None or interaction["agent_id"] == 0:
                    continue
                turns.append(convert_object(interaction, 'slice'))
            case 207:
                if interaction["oid"] is None or interaction["agent_id"] == 0:
                    continue
                turns.append(convert_object(interaction, 'dirty'))
            case 208:
                if interaction["oid"] is None or interaction["agent_id"] == 0:
                    continue
                turns.append(convert_object(interaction, 'clean'))
            case 209:
                if interaction["oid"] is None or interaction["agent_id"] == 0:
                    continue
                turns.append(convert_object(interaction, 'fill'))
            case 210:
                if interaction["oid"] is None or interaction["agent_id"] == 0:
                    continue
                turns.append(convert_object(interaction, 'empty'))
            case 211:
                if interaction["oid"] is None or interaction["agent_id"] == 0:
                    continue
                turns.append(convert_object(interaction, 'pour'))
            case 212:
                if interaction["oid"] is None or interaction["agent_id"] == 0:
                    continue
                turns.append(convert_object(interaction, 'break'))
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
