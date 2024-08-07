from src.prompt_llm.gpt4_entire_eval import get_task
from datasets import load_dataset


def _strip_not_history(prompt: str) -> str:
    return get_task(prompt)


def stripped_generator(config_name: str, split: str = "train"):
    def entry_to_content(entry: dict):
        prompt = entry["prompt"]
        answer = entry["answer"]
        return {"prompt": _strip_not_history(prompt), "completion": answer}
    dataset = load_dataset("Dandandooo/user-sim", config_name)
    yield from map(entry_to_content, dataset[split])

