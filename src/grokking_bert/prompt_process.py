from src.prompt_llm.gpt4_entire_eval import get_task
from datasets import load_dataset
from tqdm import tqdm
import torch


def _strip_not_history(prompt: str) -> str:
    return get_task(prompt)


def stripped_generator(config_name: str, tokenizer, labels, split: str):
    def label_to_tensor(label: str):
        return torch.tensor([label == l_ for l_ in labels])

    def entry_to_content(entry: dict):
        prompt = _strip_not_history(entry["prompt"])
        answer = entry["answer"]
        return {**tokenizer(prompt), "labels": label_to_tensor(answer)}
    dataset = load_dataset("Dandandooo/user-sim", config_name)
    yield from map(entry_to_content, dataset[split])


def stripped_list(config_name: str, tokenizer, labels, split: str):
    return list(tqdm(
        stripped_generator(config_name, tokenizer, labels, split),
        desc=f"Generating {split} split"
    ))

