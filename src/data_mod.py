import torch
import json
import os
import sys

from datasets import Dataset
from transformers import AutoTokenizer

def read_utterances(*files, labels=set()) -> tuple[list, list, list]:
    agents = []
    utterances = []
    utterance_labels = []
    for file in files:
        agent, utterance, utterance_label = read_utters(file, labels)
        agents += agent
        utterances += utterance
        utterance_labels += utterance_label

    return agents, utterances, utterance_labels

def read_utters(file: str, labels: set) -> tuple[list, list, list]:
    data_file = json.load(open(file, 'r'))

    agents = []
    utterances = []
    utterance_labels = []
    for event in data_file:
        agents.append(event["agent"].capitalize())
        utterances.append(event["utterance"].lower())
        utterance_labels.append(
            [ da[0].upper() + da[1:] for da in event["das"] if da ]  # Filters out empty Dialogue Acts
        )
        labels.update(utterance_labels[-1])

    return agents, utterances, utterance_labels

def format_data(agents, utterances, utterance_labels, UTT=True, ST=False, DH=False, DA_E=False, trim_to=512) -> tuple[list, list]:
    data = []
    for i, (agent, utterance, _) in enumerate(zip(agents, utterances, utterance_labels)):
        cur_data = ""
        if DH and i > 0:
            cur_data += data[-1] + ' '
            if DA_E:
                cur_data += f'<<{",".join(utterance_labels[i-1])}>> '
            cur_data += '<<TURN>> '
        if ST:
            cur_data += f'<<{agent}>> '
        cur_data += utterance
        # The following slice decreased runtime from 685s to 6s
        if DH:
            data.append(" ".join(cur_data.split()[-trim_to:]))
        else:
            data.append(cur_data)

    # Remove utterances if trying to predict future dialogue acts
    if not UTT:
        for i in range(len(data)):
            data[i] = data[i][:data[i].rfind("<<TURN>>")+8]

    return data, utterance_labels

def get_dataset(data_encodings, data_labels, label_options) -> Dataset:
    def remap_labels(label_list: list[str]) -> list[int]:
        return [int(label in label_list) for label in label_options]

    data = {
        "input_ids": torch.tensor(data_encodings["input_ids"], dtype=torch.int32),
        "attention_mask": torch.tensor(data_encodings["attention_mask"], dtype=torch.int32),
        "labels": torch.tensor([remap_labels(item_labels) for item_labels in data_labels], dtype=torch.float32)
    }
    return Dataset.from_dict(data)

def make_dataset(tokenizer, *paths, label_set=set(), agent=None, split=None, max_position_embeddings=512) -> Dataset:
    agents, utterances, utterance_labels = read_utterances(*paths, labels=label_set)
    data, labels = format_data(agents, utterances, utterance_labels, trim_to=max_position_embeddings)
    match agent:
        case None:
            pass
        case "commander":
            data = [data[i] for i in range(len(data)) if agents[i] == "Commander"]
            labels = [labels[i] for i in range(len(labels)) if agents[i] == "Commander"]
        case "driver":
            data = [data[i] for i in range(len(data)) if agents[i] == "Driver"]
            labels = [labels[i] for i in range(len(labels)) if agents[i] == "Driver"]
        case _:
            raise ValueError("Invalid agent value")

    match split:
        case None:
            pass
        case 0:
            data = data[:len(data)//2]
            labels = labels[:len(labels)//2]
        case 1:
            data = data[len(data)//2:]
            labels = labels[len(labels)//2:]
        case _:
            raise ValueError("Invalid split value")
    
    encodings = tokenizer(data, truncation=True, padding=True)

    return get_dataset(encodings, labels, label_set)
    

class TeachData:
    def __init__(self,
                 tokenizer: str,
                 path="teach-dataset-parsed/",
                 UTT=True, ST=False, DH=False, DA_E=False,
                 experiment="TR-VSA-VSB",
                 ):

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            do_lower_case=True,
            use_fast=True,
            padding_side="left",
            truncation_side="left",
        )

        self.UTT = UTT
        self.ST = ST
        self.DH = DH
        self.DA_E = DA_E

        self.experiment = experiment
        self.path = path

        self.datasets = self.load_data()
        self.labels = self.datasets["labels"]

        del self.datasets["labels"]

        self.num_labels = len(self.labels)

    def load_data(self) -> dict[str, Dataset | list]:
        labels = set()

        train_path = os.path.join(self.path, "train.json")
        valid_seen_path = os.path.join(self.path, "valid_seen.json")
        valid_unseen_path = os.path.join(self.path, "valid_unseen.json")

        match self.experiment:
            case "TR-VSA-VSB":
                valid_split, test_split = 0, 1
                valid_paths = [valid_seen_path]
                test_paths = [valid_seen_path]
            case "TR-VSB-VSA":
                valid_split, test_split = 1, 0
                valid_paths = [valid_seen_path]
                test_paths = [valid_seen_path]
            case "TR-VUA-VUB":
                valid_split, test_split = 0, 1
                valid_paths = [valid_unseen_path]
                test_paths = [valid_unseen_path]
            case "TR-VUB-VUA":
                valid_split, test_split = 1, 0
                valid_paths = [valid_unseen_path]
                test_paths = [valid_unseen_path]
            case "TR-VS-VU":
                valid_split = test_split = None
                valid_paths = [valid_seen_path]
                test_paths = [valid_unseen_path]
            case "TR-VU-VS":
                valid_split = test_split = None
                valid_paths = [valid_unseen_path]
                test_paths = [valid_seen_path]
            case "TR-V-V":
                valid_split = test_split = None
                valid_paths = [valid_seen_path]
                test_paths = [valid_seen_path]
            case _:
                raise ValueError(f"Invalid experiment: {self.experiment}")

        print("Initializing datasets...")

        print("0/11: Training dataset", end="\r")
        train_dataset = make_dataset(self.tokenizer, train_path)

        sys.stdout.write("\033[K") # ]]
        print("1/11: Validation dataset", end="\r")
        valid_dataset = make_dataset(self.tokenizer, *valid_paths, label_set=labels, split=valid_split)

        sys.stdout.write("\033[K") # ]]
        print("2/11: Test dataset", end="\r")
        test_dataset = make_dataset(self.tokenizer, *test_paths, label_set=labels, split=test_split)

        sys.stdout.write("\033[K") # ]]
        print("3/11: Test dataset (Commander)", end="\r")
        test_dataset_commander = make_dataset(self.tokenizer, *test_paths, label_set=labels, agent="commander", split=test_split)

        sys.stdout.write("\033[K") # ]]
        print("4/11: Test dataset (Driver)", end="\r")
        test_dataset_driver = make_dataset(self.tokenizer, *test_paths, label_set=labels, agent="driver", split=test_split)

        sys.stdout.write("\033[K") # ]]
        print("5/11: Validation Seen dataset", end="\r")
        valid_seen_dataset = make_dataset(self.tokenizer, valid_seen_path, label_set=labels)

        sys.stdout.write("\033[K") # ]]
        print("6/11: Validation Seen dataset (Commander)", end="\r")
        valid_seen_dataset_commander = make_dataset(self.tokenizer, valid_seen_path, label_set=labels, agent="commander")

        sys.stdout.write("\033[K") # ]]
        print("7/11: Validation Seen dataset (Driver)", end="\r")
        valid_seen_dataset_driver = make_dataset(self.tokenizer, valid_seen_path, label_set=labels, agent="driver")

        sys.stdout.write("\033[K") # ]]
        print("8/11: Validation Unseen dataset", end="\r")
        valid_unseen_dataset = make_dataset(self.tokenizer, valid_unseen_path, label_set=labels)

        sys.stdout.write("\033[K") # ]]
        print("9/11: Validation Unseen dataset (Commander)", end="\r")
        valid_unseen_dataset_commander = make_dataset(self.tokenizer, valid_unseen_path, label_set=labels, agent="commander")

        sys.stdout.write("\033[K") # ]]
        print("10/11: Validation Unseen dataset (Driver)", end="\r")
        valid_unseen_dataset_driver = make_dataset(self.tokenizer, valid_unseen_path, label_set=labels, agent="driver")

        sys.stdout.write("\033[K") # ]]
        print("11/11: Done initializing datasets")

        labels = sorted(labels)

        datasets = {
            "labels": labels,

            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset,

            "test_commander": test_dataset_commander,
            "test_driver": test_dataset_driver,

            "valid_seen": valid_seen_dataset,
            "valid_seen_commander": valid_seen_dataset_commander,
            "valid_seen_driver": valid_seen_dataset_driver,

            "valid_unseen": valid_unseen_dataset,
            "valid_unseen_commander": valid_unseen_dataset_commander,
            "valid_unseen_driver": valid_unseen_dataset_driver,
        }

        return datasets

if __name__ == "__main__":
    td = TeachData("FacebookAI/roberta-base", UTT=True, ST=False, DH=False, DA_E=False, experiment="TR-V-V")
    print("Data keys:", ', '.join(td.datasets.keys()))
    print("Data shapes:")
    for key, value in td.datasets.items():
        if isinstance(value, Dataset):
            print("=>", key, ":", value.shape)
            print("  -> input_ids:", len(value["input_ids"]), len(value["input_ids"][0]), len(value["input_ids"][0][0]))
            # print("  -> attention_masks:", len(value["attention_mask"]), type(value["attention_mask"][0]), len(value["attention_mask"][0][0]))
            # print("  -> labels:", len(value["labels"]), type(value["l)
