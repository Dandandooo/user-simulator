# Training
from transformers import AutoModelForSequenceClassification as SeqModel
from transformers import TrainingArguments, Trainer

# Evaluation
from transformers import EvalPrediction
from src.model.eval import metrics as calc_metrics

import torch

# Import class from data_mod.py
from src.data.dataclass import TeachData


class TeachModel:
    def __init__(self, model_name="FacebookAI/roberta-base",
                 UTT=True, ST=False, DH=False, DA_E=False,
                 data=None,
                 experiment="TR-V-V",
                 EPOCHS=5,
                 BATCH_SIZE=16,
                 log=True,
                 ):
        print(f"\x1b[34mInitializing TeachModel (\x1b[3{1+UTT}mUtt, \x1b[3{1+ST}mST, \x1b[3{1+DH}mDH, \x1b[3{1+DA_E}mDA-E\x1b[34m) for experiment \x1b[34m{experiment}\x1b[0m")

        if data is None:
            data = TeachData(model_name, UTT=UTT, ST=ST, DH=DH, DA_E=DA_E, experiment=experiment)

        self.data: TeachData = data

        self.run_name = f"{model_name.split('/')[-1]}{'_Utt'*UTT}{'_ST'*ST}{'_DH'*DH}{'_DA-E'*DA_E}"  # Self naming for wandb
        self.model = SeqModel.from_pretrained(model_name, num_labels=self.data.num_labels, problem_type="multi_label_classification")
        self.device = self.to_device()


        self.training_args = TrainingArguments(
            output_dir=f'./results/{experiment}/{self.run_name}',  # output directory
            overwrite_output_dir=True,  # overwrite the content of the output directory
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
            report_to=(["wandb"] if log else None),
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
            tokenizer=self.data.tokenizer,
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
        predictions = torch.tensor(pred.predictions)
        labels = torch.tensor(pred.label_ids)

        scores = calc_metrics(labels, predictions)

        return {
            'accuracy (single)': scores['single_argmax_accuracy'],
            'accuracy (multi)': scores['multi_argmax_accuracy'],
            'max confidence': scores['max_confidence'],
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

if __name__ == "__main__":
    model = TeachModel("t5-small", UTT=False, ST=False, DH=True, DA_E=False, experiment="TR-V-V", log=False)
    model.test()
