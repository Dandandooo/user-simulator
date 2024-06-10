from transformers import AutoModelForCausalLM, TFAutoModelForCausalLM, FlaxAutoModelForCausalLM, pipeline
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, prepare_model_for_kbit_training
from openai import AzureOpenAI, RateLimitError
import ollama
from datasets import Dataset, load_dataset
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
        self.data: dict[str, Dataset] = {}
        self.locations = {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def load(self, dataset_name):
        self.data[dataset_name] = load_dataset(f"Dandandooo/user-sim/{dataset_name}")


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

        for i, turn in enumerate(self.data[dataset_name].split()[dataset_split]):
            file_id, prompt, answer = turn.values()
            if destfile is not None and prompt in answered_prompts:
                yield file_id, prompt, answered_prompts[prompt]
                continue
            response = self.answer(prompt)
            yield file_id, prompt, answer, response

    def save_answers(self, dataset_name: str, dataset_split: str, destfile: str):
        responses = []
        try:
            for i, (file_id, prompt, answer, response) in enumerate(self.answer_dataset(dataset_name, dataset_split)):
                responses.append({"file_id": file_id, "prompt": prompt, "answer": answer, "response": response})
        except KeyboardInterrupt:
            print(f'Execution Stopped!!!')
        except Exception as e:
            print(f'Error: {e}')
        finally:
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

    def answer(self, prompt: str) -> str:
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model="UIUC-ConvAI-Sweden-GPT4",  # model = "deployment_name"
                    messages=[{"role": self.role, "content": prompt}],
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
    def __init__(self, model_name="google/gemma-1.1-2b-it", backend="torch", **model_kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, padding_side="left", use_fast=True)

        # Autoconfig does not map devices correctly
        config = {
            "device_map": "auto",
            "use_cache": True,
            "attn_implementation": "flash_attention_2",
            "cache_dir": ".cache",
            **model_kwargs
        }

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

        print(f"Running {model_name} on {self.model.device}")

    def answer(self, prompt: str) -> str:
        tokenized = self.tokenizer(prompt, padding=True, return_tensors="pt").to(self.model.device)
        result = self.model.generate(**tokenized, max_new_tokens=32)
        decoded = self.tokenizer.decode(result[0], skip_special_tokens=True).removeprefix(prompt).strip()
        return decoded

    # This is an alternative to "answer_folder" that batches up the prompts towards the LLM.
    # Be careful with max memory you can allocate to the model, as this can cause problems.
    def answer_folder_many(self, folder: list[tuple[str, str]]) -> list[dict[str, str]]:
        prompts: list[str] = [prompt for prompt, _ in folder]
        return [{
            "prompt": prompt,
            "answer": answer,
            "response": response,
            } for response, (prompt, answer) in zip(self.answer(prompts), folder)]


# To use the LoRA fine-tuning method for huggingface models
class LoraLM(HugLM):
    def __init__(self, model_name="google/gemma-1.1-2b-it", dataset_name="0_no_move", resume=False):
        save_name = f"llm_training_sessions/{model_name.split('/')[-1]}/{dataset_name}"
        save_model = f"Dandandooo/user-sim-{model_name.split('/')[-1]}-{dataset_name}"

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
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
        }

        super().__init__(**extra_config)

        self.args = SFTConfig(
            output_dir=save_name,
            resume_from_checkpoint=save_model if resume else None,
            torch_compile=True,
            push_to_hub=True,
            push_to_hub_model_id=save_model,
        )

        self.data.load(f"Dandandooo/user-sim/{dataset_name}")

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.data[dataset_name].split()["train"],
            eval_dataset=self.data[dataset_name].split()["validation"],
            tokenizer=self.tokenizer,
            peft_config=self.lora_config,
            formatting_func=lambda x: {"prompt": x["prompt"], "completion": x["answer"]},
            args=self.args,
        )

        print(f"Initialized trainer for LoRA fine-tuning on {model_name} with dataset {dataset_name}")

    def train(self):
        self.model.train()
        self.trainer.train()


if __name__ == "__main__":
    model = LoraLM()
    model.model.print_trainable_parameters()
