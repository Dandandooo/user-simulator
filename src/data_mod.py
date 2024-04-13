import torch
import json
import os
import sys

from datasets import Dataset
from transformers import AutoTokenizer


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
            model_max_length=512,
        )

        self.max_tokens = self.tokenizer.model_max_length

        self.UTT = UTT
        self.ST = ST
        self.DH = DH
        self.DA_E = DA_E

        self.experiment = experiment
        self.path = path

        self.labels = set()
        self.read_utterances(os.path.join(path, "train.json"), labels=self.labels)
        self.labels = sorted(self.labels)
        self.datasets = self.load_data()

        self.num_labels = len(self.labels)

    @staticmethod
    def read_utterances(*files, labels=set()) -> tuple[list, list, list]:
        def read_utters(filename: str) -> tuple[list, list, list]:
            data_file = json.load(open(filename, 'r'))

            agents = []
            utterances = []
            utterance_labels = []
            for event in data_file:
                agents.append(event["agent"].capitalize())
                utterances.append(event["utterance"].lower())
                utterance_labels.append(
                    [da[0].upper() + da[1:] for da in event["das"] if da]  # Filters out empty Dialogue Acts
                )
                labels.update(utterance_labels[-1])

            return agents, utterances, utterance_labels
        agents = []
        utterances = []
        utterance_labels = []
        for file in files:
            agent, utterance, utterance_label = read_utters(file)
            agents += agent
            utterances += utterance
            utterance_labels += utterance_label

        return agents, utterances, utterance_labels

    def format_data(self, agents, utterances, utterance_labels) -> list:
        data = []
        for i, (agent, utterance, _) in enumerate(zip(agents, utterances, utterance_labels)):
            cur_data = ""
            if self.DH and i > 0:
                cur_data += data[-1] + ' '
                if self.DA_E:
                    cur_data += f'<<{",".join(utterance_labels[i - 1])}>> '
                cur_data += '<<TURN>> '
            if self.ST:
                cur_data += f'<<{agent}>> '
            cur_data += utterance
            # The following slice decreased runtime from 685s to 6s
            if self.DH:
                data.append(" ".join(cur_data.split()[-self.max_tokens:]))
            else:
                data.append(cur_data)

        # Remove utterances if trying to predict future dialogue acts
        if not self.UTT:
            for i in range(len(data)):
                data[i] = data[i][:data[i].rfind("<<TURN>>") + 8]

        return data

    def get_dataset(self, data_encodings, data_labels) -> Dataset:
        def remap_labels(label_list: list[str]) -> list[int]:
            return [int(label in label_list) for label in self.labels]

        data = {
            "input_ids": torch.tensor(data_encodings["input_ids"], dtype=torch.int32),
            "attention_mask": torch.tensor(data_encodings["attention_mask"], dtype=torch.int32),
            "labels": torch.tensor([remap_labels(item_labels) for item_labels in data_labels], dtype=torch.float32)
        }

        return Dataset.from_dict(data)

    def make_dataset(self, tokenizer, *paths, agent=None, split=None) -> Dataset:
        agents, utterances, labels = self.read_utterances(*paths)
        data = self.format_data(agents, utterances, labels)
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
                data = data[:len(data) // 2]
                labels = labels[:len(labels) // 2]
            case 1:
                data = data[len(data) // 2:]
                labels = labels[len(labels) // 2:]
            case _:
                raise ValueError("Invalid split value")

        encodings = tokenizer(data, truncation=True, padding=True)

        return self.get_dataset(encodings, labels)

    def load_data(self) -> dict[str, Dataset]:
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
                valid_paths = [valid_seen_path, valid_unseen_path]
                test_paths = [valid_seen_path, valid_unseen_path]
            case _:
                raise ValueError(f"Invalid experiment: {self.experiment}")

        print("Initializing datasets...")

        print("0/11: Training dataset", end="\r")
        train_dataset = self.make_dataset(self.tokenizer, train_path)

        sys.stdout.write("\033[K") # ]]
        print("1/11: Validation dataset", end="\r")
        valid_dataset = self.make_dataset(self.tokenizer, *valid_paths, split=valid_split)

        sys.stdout.write("\033[K") # ]]
        print("2/11: Test dataset", end="\r")
        test_dataset = self.make_dataset(self.tokenizer, *test_paths, split=test_split)

        sys.stdout.write("\033[K") # ]]
        print("3/11: Test dataset (Commander)", end="\r")
        test_dataset_commander = self.make_dataset(self.tokenizer, *test_paths, agent="commander", split=test_split)

        sys.stdout.write("\033[K") # ]]
        print("4/11: Test dataset (Driver)", end="\r")
        test_dataset_driver = self.make_dataset(self.tokenizer, *test_paths, agent="driver", split=test_split)

        sys.stdout.write("\033[K") # ]]
        print("5/11: Validation Seen dataset", end="\r")
        valid_seen_dataset = self.make_dataset(self.tokenizer, valid_seen_path)

        sys.stdout.write("\033[K") # ]]
        print("6/11: Validation Seen dataset (Commander)", end="\r")
        valid_seen_dataset_commander = self.make_dataset(self.tokenizer, valid_seen_path, agent="commander")

        sys.stdout.write("\033[K") # ]]
        print("7/11: Validation Seen dataset (Driver)", end="\r")
        valid_seen_dataset_driver = self.make_dataset(self.tokenizer, valid_seen_path, agent="driver")

        sys.stdout.write("\033[K") # ]]
        print("8/11: Validation Unseen dataset", end="\r")
        valid_unseen_dataset = self.make_dataset(self.tokenizer, valid_unseen_path)

        sys.stdout.write("\033[K") # ]]
        print("9/11: Validation Unseen dataset (Commander)", end="\r")
        valid_unseen_dataset_commander = self.make_dataset(self.tokenizer, valid_unseen_path, agent="commander")

        sys.stdout.write("\033[K") # ]]
        print("10/11: Validation Unseen dataset (Driver)", end="\r")
        valid_unseen_dataset_driver = self.make_dataset(self.tokenizer, valid_unseen_path, agent="driver")

        sys.stdout.write("\033[K") # ]]
        print("11/11: Done initializing datasets")

        datasets = {
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
    # model = "FacebookAI/roberta-base"
    model = "t5-base"
    td = TeachData(model, UTT=False, ST=False, DH=True, DA_E=False, experiment="TR-V-V")
    print("Data keys:", ', '.join(td.datasets.keys()))
    print("Data shapes:")
    for key, value in td.datasets.items():
        if isinstance(value, Dataset):
            print("\x1b[32m=>\x1b[0;3;4;34m", key, "\x1b[0;90m:\x1b[0m", value.shape) # ]]]]]
            for k in ["input_ids", "attention_mask", "labels"]:
                print("  \x1b[33m->\x1b[0m", k, "\x1b[90m:\x1b[0m", (*torch.tensor(value[k]).shape,)) # ]]]]
