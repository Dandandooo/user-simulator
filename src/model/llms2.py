from datasets import load_dataset, Dataset, DatasetDict, get_dataset_config_info
from typing import Generator, Iterable, NoReturn
import itertools as it
import json
import os
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

"""
This script is made for the second version of my User-Sim dataset.
It is typically a chat format dataset, but I have instruct format
as well. This dataset doesn't explain dialogue acts by default,
but there is an option for it.
"""


class BaseLM:
    def __init__(self, model_name: str = None, prompt_format: str = "chat"):
        self.datasets = {}
        self.model_name = model_name
        self.prompt_format = prompt_format.lower()

    def answer(self, prompt: dict, chat: bool = True) -> str:
        """
        @param prompt: A dictionary that follows either the:
        - chat format: {"messages": [{"role": "user", "content": "Hello!"}, ...]}
        - instruct format: {"prompt": "Ask a question", "completion": "What is your name?"}
        @param chat: If True, the prompt is in chat format. If False, the prompt is in instruct format.
        """
        raise NotImplementedError("Use a child of the base class!")

    def answer_batch(self, prompts: Iterable[dict], lines: list[str] = None, total_num: int = None) -> Generator[str, None, None]:
        answered_number = 0

        if lines is not None:
            answered_number = len(lines)
            print(f'Found {answered_number} existing answers')
            yield from tqdm(map(json.loads, lines), total=answered_number, desc="Reading answers")

        try:
            # will only reassign if it's None
            total_num = total_num or len(prompts)
        except TypeError:
            pass

        for i, prompt in tqdm(enumerate(prompts), total=total_num, initial=0, desc="Answering"):
            if i >= answered_number:
                yield self.answer(prompt)

    def answer_dataset(self, filename: str, dataset_name: str, split: str = "test") -> NoReturn:
        dataset = self.datasets[dataset_name][split]

        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))

        dataset_length = get_dataset_config_info("Dandandooo/user-sim2", f"{self.prompt_format}_{dataset_name}").splits[split].num_examples

        if os.path.exists(filename):
            lines = open(filename).readlines()
        else:
            lines = []

        with open(filename, "w") as f:
            for prompt, response in zip(dataset, self.answer_batch(dataset, lines=lines, total_num=dataset_length)):
                match self.prompt_format:
                    case "chat":
                        to_write = {
                            "input": prompt["messages"][:-1],
                            "truth": prompt["messages"][-1]["content"]
                        }
                    case "instruct":
                        to_write = {
                            "input": prompt["prompt"],
                            "truth": prompt["completion"]
                        }
                to_write["response"] = response
                f.write(json.dumps(to_write) + "\n")

    def fetch_dataset(self, dataset_name: str, streaming: bool = True):
        match self.prompt_format:
            case "chat" | "instruct":
                self.datasets[dataset_name] = DatasetDict({
                    "train": load_dataset("Dandandooo/user-sim2", f"{self.prompt_format}_{dataset_name}", streaming=streaming, split="train"),
                    "valid": load_dataset("Dandandooo/user-sim2", f"{self.prompt_format}_{dataset_name}", streaming=streaming, split="valid"),
                    "test": load_dataset("Dandandooo/user-sim2", f"{self.prompt_format}_{dataset_name}", streaming=streaming, split="test")
                })
            case _:
                raise ValueError("Invalid prompt format")


    def fine_tune(self, dataset_name: str, num_epochs: int, batch_size: int, save_dir: str = None):
        raise NotImplementedError()


from openai import AzureOpenAI, RateLimitError
import time
class AzureLM(BaseLM):
    def __init__(self, endpoint: str, api_key: str, model: str, dataset_streaming: bool = True):
        super().__init__(model_name = model, prompt_format = "chat")
        self.prompt_format = "chat"

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-07-01-preview"
        )


    def answer(self, prompt: dict) -> str:
        while True:
            try:
                response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = prompt["messages"][:-1]  # Exclude the answer of course
                )
                return response.choices[0].message.content
            except RateLimitError:
                time.sleep(1)

    def _batch_format(self, prompts: Iterable[dict]) -> Generator[dict, None, None]:
        for i, prompt in enumerate(prompts):
            yield {"custom_id": f'task_{i}', "url": "/chat/completions", "method": "POST", "body": {"model": self.model_name, **prompt}}

    def _save_batch_id(self, batch_id: str, exp_no: int | float, exp_name: str):
        root_path = f"experiment_results/exp{exp_no}"
        file_path = f"{root_path}/{exp_name}_batches.json"
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        if not os.path.exists(file_path):
            existing = {}
        else:
            existing = json.load(open(file_path, "r"))

        existing |= {time.strftime("%Y-%m-%d_%H-%M-%S"): batch_id}
        json.dump(existing, open(file_path, "w"))
        print(f"Batch ID ({batch_id}) saved to {file_path}")


    # Returns the input_file_id
    def _upload_dataset(self, dataset: Dataset, dataset_id: str, split: str = "test") -> str:
        already_uploaded = self.client.files.list().data
        for file in already_uploaded:
            if file.filename == f"{dataset_id}_{split}.jsonl":
                print("Found existing dataset file")
                return file.id

        filename = f"/tmp/{dataset_id}_{split}.jsonl"
        with open(filename, "w") as f:
            filename.write("\n".join(map(json.dumps, self.batch_format(dataset))))

        file_response = self.client.files.create("/tmp/user_sim2_dataset")
        status = "pending"
        while status != "processed":
            time.sleep(15)
            upload_response = self.client.files.retrieve(file_response.id)
            status = upload_response.status
            print(f"File Upload Status: {status}")


        return file_response.id



    # TODO: Implement this specifically for Azure OpenAI Batch API
    def send_batch(self, file_id: str) -> Generator[str, None, None]:
        batch_response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/chat/completions",
            completion_window="24h"
        )
        return batch_response.id



    def answer_dataset(self, filename: str, dataset_name: str, split: str = "test", send_batch: bool = False) -> NoReturn:
        if not send_batch:
            return super().answer_dataset(filename, dataset_name, split)


    def fine_tune(self, dataset_name: str, num_epochs: int, batch_size: int, save_dir: str = None):
        print("Submitting fine-tuning job...", end=" ")
        train_id, valid_id, test_id = self._get_dataset_ids(dataset_name)
        response = self.client.fine_tuning.jobs.create(
            model=self.model,
            training_file=train_id,
            validation_file=valid_id,
            hyperparameters={
                "batch_size": batch_size,
                "n_epochs": num_epochs
            }
        )
        print("done!")
        return response


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
class HugLM(BaseLM):
    def __init__(self, model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", tokenizer_name: str = None, prompt_format="chat", **model_kwargs):
        super().__init__(model_name, prompt_format)
        if tokenizer_name is None:
            tokenizer_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast = True,
            padding_side = "right",
        )

        match self.tokenizer.padding_side:
            case "right":
                self.tokenizer.pad_token = self.tokenizer.eos_token
            case "left":
                self.tokenizer.pad_token = self.tokenizer.bos_token

        model_config = {
            "use_cache": True,
            "cache_dir": ".cache",
            "device_map": "auto",
            "force_download": False,
            **model_kwargs,
        }

        if torch.cuda.is_available():
            model_config |= {
                # "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16,
            }
            if "bnb" not in model_name:
                model_config["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.config = model_config

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)

        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def answer(self, prompt: dict) -> str:
        match self.prompt_format:
            case "chat":
                return self.pipeline(prompt["messages"][:-1], return_full_text=False)[0]["generated_text"]
            case "instruct":
                return self.pipeline(prompt["prompt"], return_full_text=False)[0]["generated_text"]


from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from transformers import TrainingArguments
from peft import LoraConfig, PeftModel
class LoraLM(HugLM):
    def __init__(self, model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", dataset_name="no-move_0-shot_100pc-obs", tokenizer: str = None, adapter_name: str = None):
        super().__init__(model_name, tokenizer, prompt_format="instruct")


        save_name = f"llm_models/{model_name.split('/')[-1]}/{dataset_name}"
        save_model = f"Dandandooo/user_sim2__{model_name.split('/')[-1]}__{dataset_name}"

        # Dataset streaming turned off for proper epoch calculation
        self.fetch_dataset(dataset_name, streaming=False)

        self.tokenizer.padding_side = "right" # It keeps giving me warning about this


        collator = DataCollatorForCompletionOnlyLM(
            response_template="### Response:",
            instruction_template="### Instruction",
            tokenizer=self.tokenizer
        )

        BATCH_SIZE = 1
        GRAD_STEPS = 2
        EPOCHS = 1


        num_steps = get_dataset_config_info("Dandandooo/user-sim2", f"instruct_{dataset_name}").splits["train"].num_examples // BATCH_SIZE // GRAD_STEPS

        config = SFTConfig(
            output_dir=save_name,
            per_device_train_batch_size=BATCH_SIZE,
            max_seq_length=8000,
            gradient_accumulation_steps=GRAD_STEPS,
            num_train_epochs=EPOCHS,

            eval_steps=num_steps // 4,
            save_steps=500,
            run_name=save_model.split("/")[-1],
            report_to="wandb",
            # max_steps=num_steps
        )

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            # use_rslora=True,  # Huggingface said "shown to work better". Disabling because I don't know enough
        )

        train_dataset = self.datasets[dataset_name]["train"]
        valid_dataset = self.datasets[dataset_name]["valid"]

        if adapter_name is not None:
            self.model.load_adapter(adapter_name)

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=config,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            formatting_func=self._format_func,
            peft_config=lora_config,
            # max_seq_length=8000
        )

    def answer(self, prompt: dict) -> str:
        # Not sure if this will change anything, but I'm not risking the fine tuning
        return self.pipeline(self._format_prompt(prompt["prompt"], ""), return_full_text=False)[0]["generated_text"]

    def fine_tune(self,  epochs: int = None, batch_size: int = None):
        if epochs is not None:
            self.trainer.args.num_train_epochs = epochs
        if batch_size is not None:
            self.trainer.args.max_steps = self.trainer.args.max_steps * self.trainer.args.per_device_train_batch_size // batch_size
            self.trainer.args.per_device_train_batch_size = batch_size

        self.trainer.train()


    @staticmethod
    def _format_prompt(prompt: str, answer: str) -> str:
        return f"### Instruction: \n {prompt}\n### Response: \n {answer}"


    def _format_func(self, data: Dataset):
        return [self._format_prompt(prompt, answer) for prompt, answer in zip(data["prompt"], data["completion"])]
