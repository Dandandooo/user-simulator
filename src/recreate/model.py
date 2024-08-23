from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, EvalPrediction
from datasets import Dataset
from sklearn.metrics import f1_score
import torch
import numpy as np

LABELS = [
    "Acknowledge", "Affirm", "AlternateQuestions",
    "Confirm", "Deny", "FeedbackNegative", "FeedbackPositive",
    "Greetings/Salutations", "InformationOnObjectDetails",
    "InformationOther", "Instruction", "MiscOther", "NotifyFailure",
    "OtherInterfaceComment", "RequestForInstruction",
    "RequestForObjectLocationAndOtherDetails", "RequestMore",
    "RequestOtherInfo"
]

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

class TeachModel2:
    def __init__(self, model_name: str, train_dataset: Dataset, eval_dataset: Dataset, batch_size: int = 16):

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,

            cache_dir=".cache",
            torch_dtype=torch.bfloat16,
            device_map="auto",

            num_labels=18,
            label2id=label2id,
            id2label=id2label,
        )

        args = TrainingArguments(
            output_dir=f"llm_models/recreate/{model_name.split('/')[-1]}",

            bf16=True,

            push_to_hub=True,
            hub_model_id=f"Dandandooo/teach-recreate_{model_name.split('/')[-1]}",
            hub_always_push=False,
            hub_private_repo=True,

            evaluation_strategy="epoch",
            save_strategy="epoch",

            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,

            metric_for_best_model=""
        )

        self.trainer = Trainer(
            self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.eval_metrics,

            args=args
        )

    @staticmethod
    def eval_metrics(p: EvalPrediction):
        confusion_matrix = np.zeros((len(LABELS), len(LABELS)))
        true = p.label_ids
        pred = p.predictions.argmax(-1)
        for t, p in zip(true, pred):
            confusion_matrix[t, p] += 1

        speak_confusion = np.array([[confusion_matrix[0, 0], confusion_matrix[0, 1:].sum()],
                                    [confusion_matrix[1:, 0].sum(), confusion_matrix[1:, 1:].sum()]])

        speak_f1 = 2 * speak_confusion[1, 1] / (
                    2 * speak_confusion[1, 1] + speak_confusion[0, 1] + speak_confusion[1, 0])

        da_f1 = f1_score(true[1:, 1:], pred[1:, 1:], average="weighted")

        return {
            "speak_f1": speak_f1,
            "da_f1": da_f1,
        }

    def train(self):
        self.trainer.train()

    def test(self, test_dataset: Dataset):
        return self.trainer.evaluate(test_dataset)
