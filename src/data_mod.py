import torch
import json
import os

from datasets import Dataset
from transformers import RobertaTokenizerFast as Tokenizer

from time import perf_counter

def read_utterances(file, labels=set()) -> tuple[list, list, list]:
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

def format_data(agents, utterances, utterance_labels, ST=False, DH=False, DA_E=False) -> tuple[list, list]:
    data = []
    for i, (agent, utterance, _) in enumerate(zip(agents, utterances, utterance_labels)):
        cur_data = ""
        if DH:
            cur_data += data[-1]
            if DA_E:
                cur_data += f' <<{",".join(utterance_labels[i-1])}>>'
        cur_data += f' <<{",".join(utterance_labels[i-1])}>>' if DA_E and DH else ""
        cur_data += ' <<TURN>> ' if DH else ""
        cur_data += f'<<{agent}>> ' if ST else ""
        cur_data += utterance
        # The following line decreased runtime from 685s to 6s
        cur_data = " ".join(cur_data.split()[-512:])  # Truncate to 512 tokens (max length of RoBERTa)
        data.append(cur_data)

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

def make_dataset(tokenizer: Tokenizer, path: str, agent=None, split=None) -> Dataset:
    agents, utterances, utterance_labels = read_utterances(path)
    data, labels = format_data(agents, utterances, utterance_labels)
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

    return get_dataset(encodings, labels, labels)
    

class TeachData:
    def __init__(self,
                 tokenizer: str,
                 path="teach-dataset-parsed/",
                 ST=False, DH=False, DA_E=False,
                 experiment="TR-VSA-VSB",
                 ):

        self.tokenizer = Tokenizer.from_pretrained(
            tokenizer,
            do_lower_case=True,
            use_fast=True,
            padding_side="left",
            truncation_side="left",
        )

        self.ST = ST
        self.DH = DH
        self.DA_E = DA_E

        self.experiment = experiment
        self.path = path

        self.labels, self.train_dataset, self.valid_dataset, self.test_dataset = self.load_data()
        self.num_labels = len(self.labels)

    def load_data(self) -> tuple[list, Dataset, Dataset, Dataset]:
        labels = set()

        train_path = os.path.join(self.path, "train.json")
        valid_seen_path = os.path.join(self.path, "valid_seen.json")
        valid_unseen_path = os.path.join(self.path, "valid_unseen.json")

        print("\x1b[33mFormatting data...\x1b[0m")
        train_data, train_labels = format_data(*read_utterances(train_path), ST=self.ST, DH=self.DH, DA_E=self.DA_E)
        valid_seen, valid_seen_labels = format_data(*read_utterances(valid_seen_path), ST=self.ST, DH=self.DH, DA_E=self.DA_E)
        valid_unseen, valid_unseen_labels = format_data(*read_utterances(valid_unseen_path), ST=self.ST, DH=self.DH, DA_E=self.DA_E)

        vsa, vsa_labels = valid_seen[:len(valid_seen)//2], valid_seen_labels[:len(valid_seen_labels)//2]
        vsb, vsb_labels = valid_seen[len(valid_seen)//2:], valid_seen_labels[len(valid_seen_labels)//2:]

        vua, vua_labels = valid_unseen[:len(valid_unseen)//2], valid_unseen_labels[:len(valid_unseen_labels)//2]
        vub, vub_labels = valid_unseen[len(valid_unseen)//2:], valid_unseen_labels[len(valid_unseen_labels)//2:]

        match self.experiment:
            case "TR-VSA-VSB":
                valid_data, valid_labels = vsa, vsa_labels
                test_data, test_labels = vsb, vsb_labels
            case "TR-VSB-VSA":
                valid_data, valid_labels = vsb, vsb_labels
                valid_data, valid_labels = vua, vua_labels
                test_data, test_labels = vub, vub_labels
            case "TR-VUB-VUA":
                valid_data, valid_labels = vub, vub_labels
                test_data, test_labels = vua, vua_labels
            case "TR-VS-VU":
                valid_data, valid_labels = valid_seen, valid_seen_labels
                test_data, test_labels = valid_unseen, valid_unseen_labels
            case "TR-VU-VS":
                valid_data, valid_labels = valid_unseen, valid_unseen_labels
                test_data, test_labels = valid_seen, valid_seen_labels
            case "TR-V-None":
                valid_data, valid_labels = valid_seen + valid_unseen, valid_seen_labels + valid_unseen_labels
                test_data, test_labels = valid_seen + valid_unseen, valid_seen_labels + valid_unseen_labels
            case _:
                raise ValueError(f"Invalid experiment: {self.experiment}")

        labels = sorted(labels)

        tokenizer_start = perf_counter()
        print("\x1b[33mTokenizing data...\x1b[0m", end=" ")
        train_encodings = self.tokenizer(train_data, truncation=True, padding=True)
        valid_encodings = self.tokenizer(valid_data, truncation=True, padding=True)
        test_encodings = self.tokenizer(test_data, truncation=True, padding=True)
        print(f"\x1b[32mDone \x1b[90m({perf_counter()-tokenizer_start:.2f}s)\x1b[0m")

        train_dataset = get_dataset(train_encodings, train_labels, labels)
        valid_dataset = get_dataset(valid_encodings, valid_labels, labels)
        test_dataset = get_dataset(test_encodings, test_labels, labels)

        return labels, train_dataset, valid_dataset, test_dataset

# Debugging purposes only when you run this file directly
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data = TeachData("FacebookAI/roberta-base", ST=(sys.argv[1].lower().strip() == "true"), DH=(sys.argv[2].lower().strip() == "true"), DA_E=(sys.argv[3].lower().strip() == "true"))
    else:
        data = TeachData("FacebookAI/roberta-base")
    print(data.labels)
    print("Last utterance:", data.train_dataset["input_ids"][-1])
    print(data.valid_dataset)
    print(data.test_dataset)
    print("Number of Labels: ",data.num_labels)
    print(data.tokenizer)

