import re
import os

from gpt4_entire_eval import get_task

from datasets import Dataset, BuilderConfig


class LLMData:
    @staticmethod
    def create_dataset(config_name, train_dir, valid_dir, test_dir, alpaca=False, upload=False, **upload_kwargs):
        train_data = Dataset.from_generator(LLMData.create_gen, gen_kwargs={"directory": train_dir, "alpaca": alpaca})
        valid_data = Dataset.from_generator(LLMData.create_gen, gen_kwargs={"directory": valid_dir, "alpaca": alpaca})
        test_data = Dataset.from_generator(LLMData.create_gen, gen_kwargs={"directory": test_dir, "alpaca": alpaca})

        if upload and not alpaca:
            train_data.push_to_hub(f"Dandandooo/user-sim", config_name, private=True, split="train", **upload_kwargs)
            valid_data.push_to_hub(f"Dandandooo/user-sim", config_name, private=True, split="validation", **upload_kwargs)
            test_data.push_to_hub(f"Dandandooo/user-sim", config_name, private=True, split="test", **upload_kwargs)

        elif upload and alpaca:
            train_data.push_to_hub(f"Dandandooo/user-sim_alpaca__{config_name}", private=True, split="train", **upload_kwargs)
            valid_data.push_to_hub(f"Dandandooo/user-sim_alpaca__{config_name}", private=True, split="validation", **upload_kwargs)
            test_data.push_to_hub(f"Dandandooo/user-sim_alpaca__{config_name}", private=True, split="test", **upload_kwargs)

        return train_data, valid_data, test_data

    @staticmethod
    def create_gen(directory, alpaca=False, prompt_regex=r".*turn_\d+\.txt"):
        prompt_re = re.compile(prompt_regex)

        for folder in os.listdir(directory):
            for file in os.listdir(os.path.join(directory, folder)):
                if prompt_re.match(file):
                    prompt = open(os.path.join(directory, folder, file)).read()
                    answer = open(os.path.join(directory, folder, file.replace(".txt", "_answer.txt"))).read()
                    if not alpaca:
                        yield {"game_id": folder, "prompt": prompt, "answer": answer}
                    else:
                        input_ = get_task(prompt)
                        instruction = prompt.removesuffix(input_).strip()
                        text = ("Below is an instruction that describes a task, paired with an input that provides"
                                " further context. Write a response that appropriately completes the request.\n\n"
                                f"### Instruction\n{instruction}\n\n### Input\n{input_}\n\n### Response\n")
                        yield {"instruction": instruction, "input": input_, "output": answer, "text": text}


if __name__ == "__main__":
    print("Creating dataset '0'...")
    LLMData.create_dataset(
        "0",
        "llm_prompts_data/turns/train_0",
        "llm_prompts_data/turns/valid_unseen_0",
        "llm_prompts_data/turns/valid_seen_0",
        upload=True,
        alpaca=True,
    )

    print("Creating dataset '0_no_move'...")
    LLMData.create_dataset(
        "0_no_move",
        "llm_prompts_data/turns/train_0_no_move",
        "llm_prompts_data/turns/valid_unseen_0_no_move",
        "llm_prompts_data/turns/valid_seen_0_no_move",
        upload=True,
        alpaca=True,
    )

    print("Creating dataset '5'...")
    LLMData.create_dataset(
        "5",
        "llm_prompts_data/turns/train_5",
        "llm_prompts_data/turns/valid_unseen_5",
        "llm_prompts_data/turns/valid_seen_5",
        upload=True,
        alpaca=True,
    )

    print("Creating dataset '5_no_move'...")
    LLMData.create_dataset(
        "5_no_move",
        "llm_prompts_data/turns/train_5_no_move",
        "llm_prompts_data/turns/valid_unseen_5_no_move",
        "llm_prompts_data/turns/valid_seen_5_no_move",
        upload=True,
        alpaca=True,
        # set_default=True,
    )


