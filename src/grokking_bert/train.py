from transformers import AutoModelForSequenceClassification, EvalPrediction
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from src.grokking_bert.prompt_process import stripped_generator, stripped_list
from datasets import Dataset
from sklearn.metrics import f1_score
import torch
import numpy as np

LABELS = [
    "OBSERVE", "Acknowledge", "Affirm", "AlternateQuestions",
    "Confirm", "Deny", "FeedbackNegative", "FeedbackPositive",
    "Greetings/Salutations", "InformationOnObjectDetails",
    "InformationOther", "Instruction", "MiscOther", "NotifyFailure",
    "OtherInterfaceComment", "RequestForInstruction",
    "RequestForObjectLocationAndOtherDetails", "RequestMore",
    "RequestOtherInfo"
]

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

model_name = "FacebookAI/roberta-base"  # Token window too small
# model_name = "distilbert/distilbert-base-cased"  # Token window too small
# model_name = "distilbert/distilgpt2"
# model_name = "unsloth/tinyllama-bnb-4bit"
# model_name = "google-t5/t5-small"  # Token window too small
# model_name = "google-t5/t5-base"  # Token window too small
# model_name = "google/gemma-2-2b-it"


# Todo: try padding_side="right" and evaluate results
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=True, truncation_side="left")

config_name = "0_no_move"

gen_kwargs = {
    "config_name": config_name,
    "tokenizer": tokenizer,
    "label2id": label2id,
}

train_dataset = Dataset.from_generator(stripped_generator, gen_kwargs={**gen_kwargs, "split": "train"})
print("Train dataset downloaded")
valid_dataset = Dataset.from_generator(stripped_generator, gen_kwargs={**gen_kwargs, "split": "validation"})
print("Validation dataset downloaded")
test_dataset = Dataset.from_generator(stripped_generator, gen_kwargs={**gen_kwargs, "split": "test"})
print("Test dataset downloaded")


save_name = f'{model_name.split("/")[-1]}_{config_name}'

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    cache_dir=".cache",
    use_cache=True,
    num_labels=len(LABELS),
    torch_dtype=torch.bfloat16,
    device_map="auto",
    label2id=label2id,
    id2label=id2label,
)
print("\x1b[35mModel\x1b[90m:\x1b[0m")
print(" \x1b[33m->\x1b[0;1;34m name\x1b[90m:\x1b[0m", model_name)
print(" \x1b[33m->\x1b[0;1;34m dtype\x1b[90m:\x1b[0m", model.dtype)
print(" \x1b[33m->\x1b[0;1;34m device\x1b[90m:\x1b[0m", model.device)


BATCH_SIZE = 40
EPOCHS = 10

RESUME = False
output_dir = f"llm_models/{save_name}"

args = TrainingArguments(
    output_dir=output_dir,
    resume_from_checkpoint=output_dir if RESUME else None,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    bf16=True,

    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"logs/{save_name}/",
    logging_strategy="epoch",
    run_name=f"grokking-{save_name}",

    push_to_hub=True,
    hub_model_id=f"Dandandooo/user_sim__{save_name}",
    hub_private_repo=True,
    hub_always_push=False,

    label_names=LABELS,
    metric_for_best_model="speak_f1",
)


def eval_metrics(p: EvalPrediction) -> dict:
    confusion_matrix = np.zeros((len(LABELS), len(LABELS)))
    true = p.label_ids
    pred = p.predictions.argmax(-1)
    for t, p in zip(true, pred):
        confusion_matrix[t, p] += 1

    speak_confusion = np.array([[confusion_matrix[0, 0], confusion_matrix[0, 1:].sum()],
                                [confusion_matrix[1:, 0].sum(), confusion_matrix[1:, 1:].sum()]])

    speak_f1 = 2 * speak_confusion[1, 1] / (2 * speak_confusion[1, 1] + speak_confusion[0, 1] + speak_confusion[1, 0])

    da_f1 = f1_score(true[1:, 1:], pred[1:, 1:], average="weighted")

    return {
        "speak_f1": speak_f1,
        "da_f1": da_f1,
    }


trainer = Trainer(
    model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=eval_metrics,
    # tokenizer=tokenizer,
)

del tokenizer

evaluate = False
if not evaluate:
    print("\x1b[35mTraining\x1b[90m...\x1b[0m")
    trainer.train()
else:
    print("\x1b[35mEvaluating\x1b[90m...\x1b[0m")
    print(trainer.evaluate(test_dataset))
