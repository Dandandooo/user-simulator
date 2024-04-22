from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# from src.data.dataclass import TeachData
from transformers import DataCollatorForLanguageModeling

import torch

class TeachModelLM:
    def __init__(self, model_name="bigscience/bloom-7b1", UTT=False, ST=False, DH=True, DA_E=False, data=None):
        # if data is None:
        #     data = TeachData(model_name, UTT=UTT, ST=ST, DH=DH, DA_E=DA_E)
        #
        # self.data: TeachData = data

        self.run_name = f"{model_name.split('/')[-1]}{'_Utt'*UTT}{'_ST'*ST}{'_DH'*DH}{'_DA-E'*DA_E}"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            # num_labels=self.data.num_labels,
            # problem_type="multi_label_classification",
        )
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            # init_lora_weights="gaussian",
            bias='none',
            task_type="CAUSAL_LM",
        )

        self.peft_model = get_peft_model(self.model, self.lora_config)

        # self.trainer = Trainer(
        #     model=self.peft_model,
        #     args=TrainingArguments(
        #         output_dir="results",
        #         overwrite_output_dir=True,
        #         num_train_epochs=1,
        #         per_device_train_batch_size=1,
        #         save_steps=1,
        #         save_total_limit=2,
        #     ),
        #     data_collator=DataCollatorForLanguageModeling(tokenizer=self.data.tokenizer, mlm=False),
        # )

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

if __name__ == "__main__":
    model = TeachModelLM()
    model.freeze()

    model.peft_model.print_trainable_parameters()
