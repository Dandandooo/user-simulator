from src.model.encoder import TeachModel
import sys
import os
import wandb

os.environ["WANDB_PROJECT"] = "TeachRecreate"
os.environ["WANDB_LOG_MODEL"] = "end"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

args = sys.argv

data_flag_maps = {
    "Utt": {"UTT": True, "ST": False, "DH": False, "DA_E": False},
    "Utt_ST": {"UTT": True, "ST": True, "DH": False, "DA_E": False},
    "Utt_DH": {"UTT": True, "ST": False, "DH": True, "DA_E": False},
    "Utt_DA-E": {"UTT": True, "ST": False, "DH": False, "DA_E": True},
    "Utt_ST_DH": {"UTT": True, "ST": True, "DH": True, "DA_E": False},
    "Utt_ST_DA-E": {"UTT": True, "ST": True, "DH": False, "DA_E": True},
    "Utt_DH_DA-E": {"UTT": True, "ST": False, "DH": True, "DA_E": True},
    "Utt_ST_DH_DA-E": {"UTT": True, "ST": True, "DH": True, "DA_E": True},
    "DH": {"UTT": False, "ST": False, "DH": True, "DA_E": False},
    "DH_ST": {"UTT": False, "ST": True, "DH": True, "DA_E": False},
    "DH_DA-E": {"UTT": False, "ST": False, "DH": True, "DA_E": True},
    "DH_ST_DA-E": {"UTT": False, "ST": True, "DH": True, "DA_E": True},
}

log = True
if len(args) > 1:
    model = args[1]
    experiment = args[2]
    data_flags = args[3]
    if len(args) > 4:
        match args[4]:
            case "F" | "f" | "false" | "False" | "FALSE" | "0" | "no" | "No" | "NO" | "n" | "N":
                log = False
            case "T" | "t" | "true" | "True" | "TRUE" | "1" | "yes" | "Yes" | "YES" | "y" | "Y":
                log = True
            case _:
                print("Invalid log argument, defaulting to True")

else:
    model = "FacebookAI/roberta-base"
    experiment = "TR-VSA-VSB"
    data_flags = "Utt"

MODEL = "FacebookAI/roberta-base"

if log:
    wandb.init(project="TeachRecreate", name=f"{model.split('/')[-1]}-{data_flags}", tags=[model, data_flags, experiment], group=experiment)

print(f"\x1b[33;1mTraining {model} on {experiment}\x1b[0m")
teach_model = TeachModel(model_name=model, **data_flag_maps[data_flags], experiment=experiment, log=log)
teach_model.train()
print(f"\x1b[32;4mFinished training {model} with {data_flags} on {experiment}\x1b[0m") # ]]]]]]
print(f"\x1b[33;1mTesting {model} with {data_flags}\x1b[0m") # ]]]]]]
results = teach_model.eval_paper()

metric = "test_accuracy (multi)"
if log:
    wandb.summary["valid_seen"] = results["valid_seen"][metric]
    wandb.summary["valid_unseen"] = results["valid_unseen"][metric]

    wandb.summary["valid_seen_commander"] = results["valid_seen_commander"][metric]
    wandb.summary["valid_seen_driver"] = results["valid_seen_driver"][metric]
    wandb.summary["valid_unseen_commander"] = results["valid_unseen_commander"][metric]
    wandb.summary["valid_unseen_driver"] = results["valid_unseen_driver"][metric]
    wandb.summary.update()

teach_model.save()
print(f"\x1b[32;2mModel {model} saved\x1b[0m")
