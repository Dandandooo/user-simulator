# Data
from transformers import RobertaTokenizerFast as Tokenizer

# Training
from transformers import AutoModelForSequenceClassification as SeqModel
from transformers import TrainingArguments, Trainer

# Evaluation
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score

import torch
# import numpy as np

# Import class from data_mod.py
from src.data_mod import TeachData

class TeachModel:
    def __init__(self, model_name="FacebookAI/roberta-base",
                 ST=False,
                 DH=False,
                 DA_E=False,
                 data=None,
                 experiment="TR-VSA-VSB",
                 EPOCHS=5,
                 BATCH_SIZE=16,
                 ):
        print(f"\x1b[34mInitializing TeachModel (\x1b[3{1+ST}mST, \x1b[3{1+DH}mDH, \x1b[3{1+DA_E}mDA-E\x1b[34m) for experiment \x1b[34m{experiment}\x1b[0m")


        if data is None:
            data = TeachData(model_name, ST=ST, DH=DH, DA_E=DA_E, experiment=experiment)

        self.data = data

        self.run_name = f"{model_name.split('/')[1]}_Utt{'_ST'*ST}{'_DH'*DH}{'_DA-E'*DA_E}"  # Self naming for wandb
        self.model = SeqModel.from_pretrained(model_name, num_labels=self.data.num_labels, problem_type="multi_label_classification")
        self.device = self.to_device()

        self.model.train()

        self.training_args = TrainingArguments(
            output_dir=f'./results/{self.run_name}',  # output directory
            num_train_epochs=EPOCHS,  # total number of training epochs
            per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
            per_device_eval_batch_size=BATCH_SIZE,  # batch size for evaluation

            # Optimizer Configuration
            adam_beta1=0.9,  # values for Adam optimizer
            adam_beta2=0.99,  # values for Adam optimizer
            adam_epsilon=1e-8,  # value to avoid division by zero for Adam optimizer

            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay

            learning_rate=5e-5,  # learning rate

            # WandB Logging
            report_to=["wandb"],
            run_name=self.run_name,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
        )

        self.trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=self.training_args,  # training arguments, defined above
            train_dataset=self.data.train_dataset,  # training dataset
            eval_dataset=self.data.valid_dataset,  # evaluation dataset
            compute_metrics=TeachModel.compute_metric,
        )

    def train(self):
        self.model.train()
        self.trainer.train()

    def save(self):
        path = f"models/{self.run_name}"
        self.model.save_pretrained(path)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.logits

    def to_device(self):
        if torch.backends.mps.is_available():
            torch_device = torch.device("mps")
            print("\x1b[90mUsing Metal Renderer\x1b[0m")
        elif torch.cuda.is_available():
            torch_device = torch.device("cuda")
            print("\x1b[90mUsing CUDA\x1b[0m")
        else:
            torch_device = torch.device("cpu")
            print("\x1b[90mUsing CPU\x1b[0m")
        self.model.to(torch_device)
        return torch_device

    @staticmethod
    def compute_metric(pred: EvalPrediction, threshold=0.5):
        predictions = pred.predictions
        labels = pred.label_ids

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.tensor(predictions, dtype=torch.float32))

        y_pred = probs > threshold
        y_true = torch.tensor(labels, dtype=torch.int32)
        return {'accuracy': accuracy_score(y_true, y_pred)}

    def test(self):
        self.trainer.evaluate(self.data.test_dataset, metric_key_prefix="test")
