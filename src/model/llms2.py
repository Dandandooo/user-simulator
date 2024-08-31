from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset

class BaseLM:
    def __init__(self):
        self.datasets = {}

    def answer(self, prompt: dict) -> str:
        raise NotImplementedError()

    def answer_batch(self, prompts: list[dict]) -> list[str]:
        yield from map(self.answer, prompts)

    def fetch_dataset(self, dataset_name: str, split: str, streaming: bool = True):
        self.datasets[dataset_name] = load_dataset("Dandandooo/user-sim2", dataset_name, streaming=streaming)[split]


class AzureLM(BaseLM):
    def __init__(self, endpoint: str, key: str, model: str):
        super().__init__()
        raise NotImplementedError()

    def answer(self, prompt: dict) -> str:
        raise NotImplementedError()

class HugLM(BaseLM):
    def __init__(self, model_name: str, tokenizer: str = None):
        super().__init__()
        if tokenizer is None:
            tokenizer = model_name
        raise NotImplementedError()

    def answer(self, prompt: dict) -> str:
        raise NotImplementedError()

class LoraLM(HugLM):
    def __init__(self, model_name: str, tokenizer: str, dataset_name: str):
        super().__init__(model_name, tokenizer)

        save_name = f"llm_training_sessions/{model_name.split('/')[-1]}/{dataset_name}"
        save_model = f"Dandandooo/user_sim2__{model_name.split('/')[-1]}__{dataset_name}"

        self.fetch_dataset(dataset_name, "train")
        del self.fetch_dataset  # To prevent accidental contamination

        self.instruct_template = "### Instruction:"
        self.response_template = "### Response:"

        collator = DataCollatorForCompletionOnlyLM(self.response_template, self.instruct_template)
        config = SFTConfig(
            output_dir=save_name
        )

    def format_func(self, data: Dataset):
        return [f"{self.instruct_template} {prompt}\n {self.response_template} {answer}"
                for prompt, answer in zip(data["prompt"], data["answer"])]