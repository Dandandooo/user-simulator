from src.recreate.model import TeachModel2
from src.recreate.histories import SpandanaHistories
from datasets import Dataset
import os
import json

train_path = "teach-data-parsed/train_turn.json"
valid_path = "teach-data-parsed/valid_unseen_turn.json"
test_path = "teach-data-parsed/valid_seen_turn.json"

models = [
    "FacebookAI/roberta-base",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
]

model = models[0]

# utt, st, dh, dae
experiments = {
    "Utt": (True, False, False, False),
    "Utt + ST": (True, True, False, False),
    "Utt + DH": (True, False, True, False),
    "Utt + DH + DA_E": (True, False, True, True),
    "Utt + ST + DH + DA_E": (True, True, True, True),
}

experiment = "Utt"

histories = SpandanaHistories(None, *experiments[experiment])


train_dataset = Dataset.from_generator(
    histories.iter_n,
    cache_dir=".cache",
    gen_kwargs={
        "dataset": json.load(open(train_path)),
        "n": 5000
    }
)

valid_dataset = Dataset.from_generator(
    histories.iter_n,
    cache_dir=".cache",
    gen_kwargs={
        "dataset": json.load(open(valid_path)),
        "n": 1000
    }
)

test_dataset = Dataset.from_generator(
    histories.iter_n,
    cache_dir=".cache",
    gen_kwargs={
        "dataset": json.load(open(test_path)),
        "n": 1000
    }
)


