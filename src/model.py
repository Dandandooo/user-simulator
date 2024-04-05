# Training
from transformers import AutoModelForSequenceClassification as SeqModel
from transformers import TrainingArguments, Trainer

# Evaluation
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score, f1_score

import torch
# import numpy as np

# Import class from data_mod.py
from src.data_mod import TeachData

class TeachModel:
    def __init__(self, model_name="FacebookAI/roberta-base",
                 UTT=True, ST=False, DH=False, DA_E=False,
                 data=None,
                 experiment="TR-V-V",
                 EPOCHS=5,
                 BATCH_SIZE=16,
                 ):
        print(f"\x1b[34mInitializing TeachModel (\x1b[3{1+UTT}mUtt, \x1b[3{1+ST}mST, \x1b[3{1+DH}mDH, \x1b[3{1+DA_E}mDA-E\x1b[34m) for experiment \x1b[34m{experiment}\x1b[0m")
        # ]]]]]]]])  Neovim LSP problems

        if data is None:
            data = TeachData(model_name, UTT=UTT, ST=ST, DH=DH, DA_E=DA_E, experiment=experiment)

        self.data: TeachData = data

        self.run_name = f"{model_name.split('/')[-1]}{'_Utt'*UTT}{'_ST'*ST}{'_DH'*DH}{'_DA-E'*DA_E}"  # Self naming for wandb
        self.model = SeqModel.from_pretrained(model_name, num_labels=self.data.num_labels, problem_type="multi_label_classification")
        self.device = self.to_device()

        self.model.train()

        self.training_args = TrainingArguments(
            output_dir=f'./results/{experiment}/{self.run_name}',  # output directory
            num_train_epochs=EPOCHS,  # total number of training epochs
            per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
            per_device_eval_batch_size=BATCH_SIZE,  # batch size for evaluation

            # Optimizer Configuration
            adam_beta1=0.9,  # values for Adam optimizer
            adam_beta2=0.99,  # values for Adam optimizer
            adam_epsilon=1e-8,  # value to avoid division by zero for Adam optimizer

            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay

            learning_rate=2e-5,  # learning rate

            # WandB Logging
            report_to=["wandb"],
            run_name=self.run_name,
            evaluation_strategy="epoch",
            # eval_steps=100,
            logging_strategy="epoch",
            # logging_steps=100,
            save_strategy="epoch",
        )

        self.trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=self.training_args,  # training arguments, defined above
            train_dataset=self.data.datasets["train"],  # training dataset
            eval_dataset=self.data.datasets["valid"],  # evaluation dataset
            compute_metrics=TeachModel.compute_metric,
        )

    def train(self):
        # Puts the model into training mode
        self.model.train()
        # Initiates training
        self.trainer.train()
        # Puts the model into evaluation mode
        self.model.eval()

    def save(self):
        path = f"models/{self.run_name}"
        self.model.save_pretrained(path)

    def predict(self, text):
        inputs = self.data.tokenizer(text, return_tensors="pt")
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
        # ])])]) Neovim LSP problems

        self.model.to(torch_device)
        return torch_device

    @staticmethod
    def compute_metric(pred: EvalPrediction):
        predictions = pred.predictions
        labels = pred.label_ids

        # SKLearn Accuracy
        threshold = 0.5
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.tensor(predictions, dtype=torch.float32))
        y_pred = probs > threshold
        y_true = torch.tensor(labels, dtype=torch.int32)

        acc_score = accuracy_score(y_true, y_pred)
        # bal_acc_score = balanced_accuracy_score(y_true, y_pred)
        f1_score_val = f1_score(y_true, y_pred, average='macro')

        # Argmax(es) from case
        n_maxes = y_true.count_nonzero()
        y_arg_pred = torch.zeros_like(y_pred)
        predict_indices = torch.argsort(probs, descending=True, dim=1)[:, :n_maxes]
        for i in range(len(y_true)):
            for j in predict_indices[i]:
                y_arg_pred[i][j] = 1

        argmax_score = accuracy_score(y_true, y_arg_pred)

        # Max confidence
        max_conf = torch.max(probs, dim=1)

        return {
            'accuracy': acc_score,
            'argmax_accuracy': argmax_score,
            'f1_score': f1_score_val,
            'max_confidence': max_conf.values.mean(),
        }

    def test(self, test_data="test"):
        return self.trainer.evaluate(self.data.datasets[test_data], metric_key_prefix="test")

    def eval_paper(self):
        results = {
            "valid_seen": self.test("valid_seen"),
            "valid_unseen": self.test("valid_unseen"),
            "valid_seen_commander": self.test("valid_seen_commander"),
            "valid_seen_driver": self.test("valid_seen_driver"),
            "valid_unseen_commander": self.test("valid_unseen_commander"),
            "valid_unseen_driver": self.test("valid_unseen_driver"),
        }
        return results 
