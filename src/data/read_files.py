import json

def read_utterances(*files, labels: set = set()) -> tuple[list, list, list]:
    agents = []
    utterances = []
    utterance_labels = []
    for filename in files:
        data_file = json.load(open(filename, 'r'))

        for event in data_file:
            agents.append(event["agent"].capitalize())
            utterances.append(event["utterance"].lower())
            utterance_labels.append(
                [da[0].upper() + da[1:] for da in event["das"] if da]  # Filters out empty Dialogue Acts
            )
            labels.update(utterance_labels[-1])

    return agents, utterances, utterance_labels
