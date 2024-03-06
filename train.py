from src.data_mod import TeachData
from src.model import TeachModel
import sys
import os
import wandb

os.environ["WANDB_PROJECT"] = "TeachRecreate"
os.environ["WANDB_LOG_MODEL"] = "end"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

args = sys.argv

model_maps = {
    "Utt": {"ST": False, "DH": False, "DA_E": False},
    "Utt_ST": {"ST": True, "DH": False, "DA_E": False},
    "Utt_DH": {"ST": False, "DH": True, "DA_E": False},
    "Utt_DA-E": {"ST": False, "DH": False, "DA_E": True},
    "Utt_ST_DH": {"ST": True, "DH": True, "DA_E": False},
    "Utt_ST_DA-E": {"ST": True, "DH": False, "DA_E": True},
    "Utt_DH_DA-E": {"ST": False, "DH": True, "DA_E": True},
    "Utt_ST_DH_DA-E": {"ST": True, "DH": True, "DA_E": True},
}

if len(args) > 1:
    experiment = args[1]
    model = args[2]
else:
    experiment = "TR-VSA-VSB"
    model = "Utt"

experiments = [
    "TR-VSA-VSB",
    "TR-VSB-VSA",
    "TR-VUA-VUB",
    "TR-VUB-VUA",
    # "TR-VS-VU",
    # "TR-VU-VS",
    "TR-V-None",
]
models = [
    "Utt",
    "Utt_ST",
    "Utt_DH",
    "Utt_DA-E",
    "Utt_ST_DH",
    "Utt_ST_DA-E",
    "Utt_DH_DA-E",
    "Utt_ST_DH_DA-E",
]

MODEL = "FacebookAI/roberta-base"

wandb.init(project="TeachRecreate", name=f"{model}-{experiment}", tags=[model, experiment], group=experiment)

print(f"\x1b[33;1mTraining {model} on {experiment}\x1b[0m")
teach_model = TeachModel(model_name=MODEL, **model_maps[model], experiment=experiment)
teach_model.train()
print(f"\x1b[32;4mFinished training {model} on {experiment}\x1b[0m")
print(f"\x1b[33;1mTesting {model} on {experiment}\x1b[0m")
teach_model.test()
teach_model.save()
print(f"\x1b[32;2mModel {model} saved\x1b[0m")
