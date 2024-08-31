from src.prompt_llm.gpt4_entire_eval import get_task
from transformers.tokenization_utils import PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm


def _strip_not_history(prompt: str) -> str:
    return get_task(prompt).removesuffix("COMMANDER response:")


def stripped_generator(config_name: str, tokenizer: PreTrainedTokenizer, label2id, split: str):
    def label_to_tensor(label: str):
        # return torch.tensor([int(label == l_) for l_ in labels])
        return label2id[label]

    def entry_to_content(entry: dict):
        prompt = _strip_not_history(entry["prompt"])
        answer = entry["answer"]
        tokenized = tokenizer(prompt, truncation=True, padding='max_length', max_length=tokenizer.model_max_length)
        if tokenized["input_ids"] != tokenizer.pad_token_id:
            goal = prompt.splitlines()[0] + "..."
            goal_tokenized = tokenizer(goal)
            tokenized["input_ids"][0:len(goal_tokenized["input_ids"])] = goal_tokenized["input_ids"]
            tokenized["attention_mask"][0:len(goal_tokenized["attention_mask"])] = goal_tokenized["attention_mask"]
        return {**tokenized, "labels": label_to_tensor(answer)}
    dataset = load_dataset("Dandandooo/user-sim", config_name)
    yield from map(entry_to_content, dataset[split])


def stripped_list(config_name: str, tokenizer, label2id, split: str):
    return list(tqdm(
        stripped_generator(config_name, tokenizer, label2id, split),
        desc=f"Generating {split} split"
    ))

