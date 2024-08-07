from transformers import AutoModelForCausalLM, TFAutoModelForCausalLM, FlaxAutoModelForCausalLM, pipeline
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, prepare_model_for_kbit_training
from openai import AzureOpenAI, RateLimitError
import ollama
from datasets import Dataset, DatasetDict, load_dataset
import time

from tqdm import tqdm
import os
import re
import json

import torch
import numpy as np

from src.prompt_llm.gpt4_entire_eval import get_last_turn_type, calc_score


class LLMDataset:
    def __init__(self):
        self.data: dict[str, DatasetDict] = {}
        self.locations = {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def load(self, dataset_name):
        self.data[dataset_name] = load_dataset("Dandandooo/user-sim", dataset_name)


class BaseLM:
    def __init__(self):
        self.data = LLMDataset()

    def answer(self, prompt: str) -> str:
        raise NotImplementedError

    def answer_dataset(self, dataset_name: str, dataset_split: str = "test", destfile: str = None) -> list[tuple[str, list[dict]]]:
        print(f'Answering "{dataset_name}" dataset')
        if destfile is not None:
            answered = json.load(open(destfile, "r"))
            answered_prompts = {result["prompt"]: result["response"] for result in answered}

        for i, turn in enumerate(self.data[dataset_name][dataset_split]):
            file_id, prompt, answer = turn.values()
            if destfile is not None and prompt in answered_prompts:
                yield file_id, prompt, answered_prompts[prompt]
                continue
            response = self.answer(prompt)
            yield file_id, prompt, answer, response

    def save_answers(self, dataset_name: str, dataset_split: str, destfile: str):
        responses = []
        try:
            for i, (file_id, prompt, answer, response) in tqdm(enumerate(self.answer_dataset(dataset_name, dataset_split)), total=len(self.data[dataset_name][dataset_split])):
                responses.append({"file_id": file_id, "prompt": prompt, "answer": answer, "response": response})
        except KeyboardInterrupt:
            print(f'Execution Stopped!!!')
        except Exception as e:
            print(f'Error: {e}')
        finally:
            if not os.path.exists(os.path.dirname(destfile)):
                os.makedir(os.path.dirname(destfile))
            with open(destfile, "w") as f:
                json.dump(responses, f, indent=4)
            print(f'Answers saved to {destfile}')

    def tp_etc(self, dataset_name: str) -> tuple[int, int, int, int, int, int]:
        true_observed = 0
        false_observed = 0
        true_speak = 0
        false_speak = 0
        correct_speak = 0
        incorrect_speak = 0

        # Returns a lazy generator
        data = self.answer_dataset(dataset_name)

        print("Getting metrics for dataset")
        for _, folder in tqdm(data, total=len(self.data[dataset_name])):
            for result in folder:
                answer, response = result["answer"], result["response"]
                if "OBSERVE" in answer:
                    if "OBSERVE" in response:
                        true_observed += 1
                    else:
                        false_speak += 1
                else:
                    if "OBSERVE" in response:
                        false_observed += 1
                    else:
                        true_speak += 1
                        if response == answer:
                            correct_speak += 1
                        else:
                            incorrect_speak += 1

        return true_observed, false_observed, true_speak, false_speak, correct_speak, incorrect_speak

    def eval(self, dataset_name: str) -> dict:
        true_observed, false_observed, true_speak, false_speak, correct_speak, incorrect_speak = self.tp_etc(dataset_name)
        return {
            "accuracy": (true_observed + true_speak) / (true_observed + false_observed + true_speak + false_speak),
            "f-score": 2 * correct_speak / (2 * correct_speak + incorrect_speak),
            "conv_matrix": np.array([[true_observed, false_speak], [false_observed, true_speak]]),
            "da_accuracy": correct_speak / (correct_speak + incorrect_speak) if (incorrect_speak + correct_speak) != 0 else 0,
        }


class GPT4LM(BaseLM):
    def __init__(self, api_key=os.getenv("AZURE_OPENAI_KEY_4"),
                 azure_endpoint="https://uiuc-convai-sweden.openai.azure.com/", role="user"):
        super().__init__()
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-15-preview"
        )

        self.role = "user"

    def answer(self, user_prompt: str, system_prompt: str = None) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model="UIUC-ConvAI-Sweden-GPT4",  # model = "deployment_name"
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                )
                return completion.choices[0].message.content
            except RateLimitError:
                time.sleep(1)


class OllamaLM(BaseLM):
    def __init__(self, model_name="gemma:instruct"):
        super().__init__()
        self.model_name = model_name
        ollama.pull(model_name)

    def answer(self, prompt: str) -> str:
        return ollama.generate(prompt=prompt, model=self.model_name)['response']


class HugPipeline(BaseLM):
    def __init__(self, model_name="google/gemma-1.1-2b-it", task="text-generation"):
        super().__init__()
        self.model_name = model_name
        self.task = task

        self.pipeline = pipeline(task, model=model_name)

    def answer(self, prompt: str) -> str:
        return self.pipeline(prompt)[0]['generated_text']


# To work with huggingface models
class HugLM(BaseLM):
    def __init__(self, model_name="google/gemma-1.1-2b-it", backend="torch", no_flash=False, **model_kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, padding_side="left", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Autoconfig does not map devices correctly
        config = {
            "device_map": "auto",
            "use_cache": True,
            "cache_dir": ".cache",
            "force_download": False,
        }

        # if torch.cuda.is_available() and not no_flash:
        #     config["attn_implementation"] = "flash_attention_2"

        # moved behind to allow for overrides
        config |= model_kwargs

        # todo: try adding some Seq2SeqLM models because GPT4 said it fits better
        match backend:
            case "torch" | "pt":
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **config)
            case "tensorflow" | "tf":
                self.model = TFAutoModelForCausalLM.from_pretrained(model_name, **config)
            case "flax" | "jax":
                self.model = FlaxAutoModelForCausalLM.from_pretrained(model_name, **config)
            case "peft":
                self.model = AutoPeftModelForCausalLM.from_pretrained(model_name, **config)
            case _:
                raise ValueError(f"Backend {backend} not supported")

        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, torch_dtype=torch.bfloat16)

        print(f"Running {model_name} on {self.model.device}")

    def answer(self, prompt: str) -> str:
        # tokenized = self.tokenizer(prompt, padding=True, return_tensors="pt").to(self.model.device)
        # result = self.model.generate(**tokenized, max_new_tokens=20)
        # decoded = self.tokenizer.decode(result[0], skip_special_tokens=True).removeprefix(prompt).strip()
        # return decoded
        return self.pipeline(prompt, max_new_tokens=20)[0]['generated_text'].removeprefix(prompt).strip()


# To use the LoRA fine-tuning method for huggingface models
class LoraLM(HugLM):
    def __init__(self, model_name="google/gemma-1.1-2b-it", dataset_name="0_no_move", resume=False, no_flash=False, **extra_kwargs):
        save_name = f"llm_training_sessions/{model_name.split('/')[-1]}/{dataset_name}"
        save_model = f"Dandandooo/user-sim__{model_name.split('/')[-1]}__{dataset_name}"

        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=8,
            lora_dropout=0.1,
            bias='none',
            task_type="CAUSAL_LM",
            use_rslora=True,  # Huggingface said "shown to work better"
        )

        extra_config = {
            "model_name": model_name,
            "torch_dtype": "auto",
            "device_map": {'': torch.cuda.current_device()}
        }

        if not no_flash and torch.cuda.is_available():
            extra_config["attn_implementation"] = "flash_attention_2"

        sft_extras = {}

        if torch.backends.mps.is_available():
            sft_extras["use_mps_device"] = True

        if torch.cuda.is_available():
            extra_config["torch_dtype"] = torch.bfloat16
            sft_extras["bf16"] = True


        # Implement extra config after manual configs
        extra_config |= extra_kwargs

        # To avoid the annoying warning
        if "bnb" not in model_name:
            extra_config["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        super().__init__(**extra_config)

        self.args = TrainingArguments(
            output_dir=save_name,
            resume_from_checkpoint=save_model if resume else None,
            # torch_compile=True,
            push_to_hub=True,
            hub_model_id=save_model,
            hub_private_repo=True,
            per_device_train_batch_size=2,  # Hopefully this won't overflow the memory
            **sft_extras,
        )

        self.data.load(dataset_name)
        
        def format_func(data: Dataset):
            return [f"### Instruction: {prompt}\n ### Response: {answer}" for prompt, answer in zip(data["prompt"], data["answer"])]

        self.tokenizer.padding_side = "right"

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.data[dataset_name]["train"],
            eval_dataset=self.data[dataset_name]["validation"],
            tokenizer=self.tokenizer,
            peft_config=self.lora_config,
            formatting_func=format_func,
            args=self.args,
        )

        print(f"Initialized trainer for LoRA fine-tuning on {model_name} with dataset {dataset_name}")

    def train(self):
        self.model.train()
        self.trainer.train()


if __name__ == "__main__":
    model = LoraLM()
    model.model.print_trainable_parameters()
