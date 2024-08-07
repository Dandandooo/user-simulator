from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import BitsAndBytesConfig
from src.grokking_bert.prompt_process import stripped_generator
from datasets import Dataset
import torch
import os

LABELS = [
    "OBSERVE", "Acknowledge", "Affirm", "AlternateQuestions",
    "Confirm", "Deny", "FeedbackNegative", "FeedbackPositive",
    "Greetings/Salutations", "InformationOnObjectDetails",
    "InformationOther", "Instruction", "MiscOther", "NotifyFailure",
    "OtherInterfaceComment", "RequestForInstruction",
    "RequestForObjectLocationAndOtherDetails", "RequestMore",
    "RequestOtherInfo"
]

config_name = "0_no_move"
train_dataset = Dataset.from_generator(stripped_generator, gen_kwargs={"config_name": config_name, "split": "train"})
print("Train dataset downloaded")
valid_dataset = Dataset.from_generator(stripped_generator, gen_kwargs={"config_name": config_name, "split": "validation"})
print("Validation dataset downloaded")
test_dataset = Dataset.from_generator(stripped_generator, gen_kwargs={"config_name": config_name, "split": "test"})
print("Test dataset downloaded")

# Todo: try RoBERTa, T5, DistilGPT2, DistilBert?,
# Todo: research about Seq2SeqLM or MaskedLM (if Sequence Classification is not enough)
model_name = "FacebookAI/roberta-base"
# model_name = "distilbert/DistilGPT2"
# model_name = "t5-small"
# model_name = "t5-base"
save_name = f'{model_name.split("/")[-1]}_{config_name}'

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    cache_dir=".cache",
    use_cache=True,
    # quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    num_labels=len(LABELS),
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("\x1b[35mModel\x1b[90m:\x1b[0m")
print(" \x1b[33m->\x1b[0;1;34m name\x1b[90m:\x1b[0m", model_name)
print(" \x1b[33m->\x1b[0;1;34m dtype\x1b[90m:\x1b[0m", model.dtype)
print(" \x1b[33m->\x1b[0;1;34m device\x1b[90m:\x1b[0m", model.device)
# Todo: try padding_side="right" and evaluate results
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=True)

BATCH_SIZE = 16
EPOCHS = 10

RESUME = False
output_dir = f"llm_models/{save_name}"

args = TrainingArguments(
    output_dir=output_dir,
    resume_from_checkpoint=output_dir if RESUME else None,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    # bf16=True,
    # fp16=True, # TODO: figure out why neither work on mac, despite model being loaded in bfloat16

    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"logs/{save_name}/",
    logging_strategy="epoch",
    run_name=f"grokking-{save_name}",

    push_to_hub=True,
    hub_model_id=f"Dandandooo/user_sim__{save_name}",
    hub_private_repo=True,

    label_names=LABELS,
)

trainer = Trainer(
    model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
)

evaluate = False
if not evaluate:
    print("\x1b[35mTraining\x1b[90m...\x1b[0m")
    trainer.train()
else:
    print("\x1b[35mEvaluating\x1b[90m...\x1b[0m")
    trainer.evaluate(test_dataset)
