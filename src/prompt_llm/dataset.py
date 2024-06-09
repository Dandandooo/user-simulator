import re, os

from gpt4_entire_eval import get_task

from datasets import Dataset, BuilderConfig


class LLMData:
    @staticmethod
    def create_dataset(config_name, train_dir, valid_dir, test_dir, upload=False, **upload_kwargs):
        train_data = Dataset.from_generator(LLMData.create_gen, gen_kwargs={"directory": train_dir})
        valid_data = Dataset.from_generator(LLMData.create_gen, gen_kwargs={"directory": valid_dir})
        test_data = Dataset.from_generator(LLMData.create_gen, gen_kwargs={"directory": test_dir})

        if upload:
            train_data.push_to_hub(f"Dandandooo/user-sim", config_name, private=True, split="train", **upload_kwargs)
            valid_data.push_to_hub(f"Dandandooo/user-sim", config_name, private=True, split="validation", **upload_kwargs)
            test_data.push_to_hub(f"Dandandooo/user-sim", config_name, private=True, split="test", **upload_kwargs)

        return train_data, valid_data, test_data

    @staticmethod
    def create_gen(directory, prompt_regex=r".*turn_\d+\.txt"):
        prompt_re = re.compile(prompt_regex)

        for folder in os.listdir(directory):
            for file in os.listdir(os.path.join(directory, folder)):
                if prompt_re.match(file):
                    prompt = open(os.path.join(directory, folder, file)).read()
                    answer = open(os.path.join(directory, folder, file.replace(".txt", "_answer.txt"))).read()
                    task = get_task(prompt)
                    instructions = prompt.removesuffix(task).strip()
                    yield {"game_id": folder, "prompt": prompt, "answer": answer, "task": task, "instructions": instructions}


if __name__ == "__main__":
    LLMData.create_dataset(
        "0",
        "llm_prompts_data/turns/train_0",
        "llm_prompts_data/turns/valid_unseen_0",
        "llm_prompts_data/turns/valid_seen_0",
        upload=True,
    )

    LLMData.create_dataset(
        "0_no_move",
        "llm_prompts_data/turns/train_0_no_move",
        "llm_prompts_data/turns/valid_unseen_0_no_move",
        "llm_prompts_data/turns/valid_seen_0_no_move",
        upload=True,
    )

    LLMData.create_dataset(
        "5",
        "llm_prompts_data/turns/train_5",
        "llm_prompts_data/turns/valid_unseen_5",
        "llm_prompts_data/turns/valid_seen_5",
        upload=True,
    )

    LLMData.create_dataset(
        "5_no_move",
        "llm_prompts_data/turns/train_5_no_move",
        "llm_prompts_data/turns/valid_unseen_5_no_move",
        "llm_prompts_data/turns/valid_seen_5_no_move",
        upload=True,
        # set_default=True,
    )


