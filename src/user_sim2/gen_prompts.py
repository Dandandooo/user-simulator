import os

import click
import json
import random
from datasets import Dataset, DatasetDict, NamedSplit
from tqdm import tqdm


# TODO: implement ALFWORLD
class PromptMaker:
    def __init__(self, example_source: list[dict], **kwargs):
        self.kwargs = kwargs
        self.example_source = example_source

        # Defining which function to use for prompt
        match kwargs["format"]:
            case "chat":
                self.prompt_format = self.chat_prompt
            case "instruct":
                self.prompt_format = self.instruct_prompt
            case _:
                raise ValueError(f"'{kwargs['format']}' is not a recognized format.")

    def __call__(self, task: dict, length: int = None, ignore_pc: bool = False):
        if self.kwargs["no-move"]:
            task["turns"] = [turn for turn in task["turns"] if turn["turn_action"] != "move"]
        if isinstance(length, int):
            user_, answer_ = self.user_prompt(task, length)
            return self.prompt_format(user_, answer_)
        for i in range(len(task["turns"])-1):
            user_, answer_ = self.user_prompt(task, i)
            if not ignore_pc and 100 * random.random() > self.kwargs["npc-obs"] and "OBSERVE" in answer_:
                continue
            yield self.prompt_format(user_, answer_)

    def system_prompt(self):
        prompt = open("src/user_sim2/segments/system_prompt.txt").read()
        prompt += open(f"src/user_sim2/segments/das_{'expl' if self.kwargs['das-expl'] else 'list'}.txt").read()
        return prompt

    def history(self, example: dict, length: int) -> tuple[str, str]:
        hist = example["goal"] + '\n'
        for turn in example["turns"][:length]:
            if turn["turn_action"] == "move" and self.kwargs["no-move"]:
                continue
            elif turn["DRIVER"]["action"] == "dialogue":
                hist += f"COMMANDER: <observe>\n"
                hist += f"DRIVER: {turn['DRIVER']['utterance']} <<{','.join(turn['DRIVER']['das'])}>>\n"
            elif turn["COMMANDER"]["action"] == "dialogue":
                hist += f"COMMANDER: {turn['COMMANDER']['utterance']} <<{','.join(turn['COMMANDER']['das'])}>>\n"
                hist += f"DRIVER: <observe>\n"
            else:
                hist += f"COMMANDER: <observe>\n"
                hist += f"DRIVER: {turn['DRIVER']['action']}\n"

        hist += "COMMANDER response:\n"
        final = example["turns"][length]["COMMANDER"]
        answer = "OBSERVE" if final["action"] != "dialogue" else final["das"][0]
        return hist, answer

    def examples(self, num_to_make: int):
        def choose_example():
            choice = random.choice(self.example_source)
            length = random.randint(1, len(choice["turns"]) - 1)
            if choice["turns"][length]["COMMANDER"]["action"] == "<observe>" and 100 * random.random() < self.kwargs["nex_obs"]:
                return choice, length
            return choose_example()
        for i in range(num_to_make):
            choice_, length_ = choose_example()
            example = f'Example{f" {i}" * self.kwargs["enumerate"]}:\n'
            example += "\n".join(self.history(choice_, length_))
            yield example

    def user_prompt(self, task: dict, length: int) -> (str, str):
        prompt = open("src/user_sim2/segments/user_prompt.txt").read()
        if n := self.kwargs["n-shot"]:
            prompt += "\nHere are some examples:\n\n"
            prompt += "\n\n".join(self.examples(n))
            prompt += "\n"
        prompt += "\nHere is your task:\n\n"
        task_, answer_ = self.history(task, length)
        prompt += task_
        return prompt, answer_

    def chat_prompt(self, user_: str, answer_: str):
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": user_},
                {"role": "assistant", "content": answer_}
            ]
        }

    def instruct_prompt(self, user_: str, answer_: str):
        return {
            "prompt": self.system_prompt() + "\n" + user_,
            "completion": answer_
        }


class DatasetMaker:
    def __init__(self, prompt_maker=PromptMaker, **kwargs):
        def kwargs_that_are(of_type, exceptions=()):
            return {k: v for k, v in kwargs.items() if of_type == type(v) and k not in exceptions}
        flags = kwargs_that_are(bool)
        n_choices = kwargs_that_are(int, exceptions=["nex-obs"])

        dataset_id = '_'.join([kwargs["format"],
                               *filter(flags.__getitem__, flags),
                               *[k.replace('n', str(v), 1) for k, v in n_choices.items()]])
        if not dataset_id:
            dataset_id = f'{kwargs["format"]}_default'

        print("Dataset ID:", dataset_id)

        self.id = dataset_id

        self.train_source = json.load(open(f"teach-dataset-parsed/train_turn.json"))
        self.valid_source = json.load(open(f"teach-dataset-parsed/valid_unseen_turn.json"))
        self.test_source = json.load(open(f"teach-dataset-parsed/valid_seen_turn.json"))

        self.pm = prompt_maker(example_source=self.train_source, **kwargs)

    def generate(self, source, ignore_pc: bool = False):
        for file in source:
            yield from self.pm(file, ignore_pc=ignore_pc)

    def generate_dataset(self) -> DatasetDict:
        train = Dataset.from_list(list(tqdm(self.generate(self.train_source), desc="Train")))
        valid = Dataset.from_list(list(tqdm(self.generate(self.valid_source, ignore_pc=True), desc="Valid")))
        test = Dataset.from_list(list(tqdm(self.generate(self.test_source, ignore_pc=True), desc="Test")))
        # train = Dataset.from_generator(self.generate, gen_kwargs={"source": self.train_source})
        # valid = Dataset.from_generator(self.generate, gen_kwargs={"source": self.valid_source, "ignore_pc": True})
        # test = Dataset.from_generator(self.generate, gen_kwargs={"source": self.test_source, "ignore_pc": True})
        dd = DatasetDict({"train": train, "valid": valid, "test": test})
        return dd


class DatasetManager:
    def __init__(self, variations_: dict[str, list]):
        self.datasets = {}

        for variation in DatasetManager.get_variations(list(variations_.items())):
            id_, _ = self.make_dataset(variation)
            self.save_dataset(id_)

        self.upload()

    @staticmethod
    def get_variations(variation_items: list[tuple]):
        if not variation_items:
            return {}
        key, values = variation_items[0]
        for value in values:
            yielded = False
            for rest in DatasetManager.get_variations(variation_items[1:]):
                yielded = True
                yield {key: value, **rest}
            if not yielded:
                yield {key: value}

    def make_dataset(self, variation) -> tuple[str, DatasetDict]:
        dm = DatasetMaker(**variation)
        dataset = dm.generate_dataset()
        self.datasets[dm.id] = dataset
        return dm.id, dataset

    def upload(self):
        hub_id = "Dandandooo/user-sim2"
        for id_, dataset in self.datasets.items():
            print(f"Uploading dataset: {id_}")
            dataset.push_to_hub(hub_id, id_, private=True)

    def save(self):
        folder = "llm_prompts_data/user_sim2"
        for id_, dataset in self.datasets.items():
            self.save_dataset(id_, folder)

    def save_dataset(self, id_: str, folder="llm_prompts_data/user_sim2"):
        def jsonl(dataset_):
            return json.dumps(dataset_) + "\n"
        print(f"Saving dataset: {id_}")
        dataset = self.datasets[id_]
        form, _id = id_.split("_", 1)
        with open(os.path.join(folder, form, f"{_id}_train.jsonl"), "w") as file:
            file.writelines(map(jsonl, dataset["train"]))
        with open(os.path.join(folder, form, f"{_id}_valid.jsonl"), "w") as file:
            file.writelines(map(jsonl, dataset["valid"]))
        with open(os.path.join(folder, form, f"{_id}_test.jsonl"), "w") as file:
            file.writelines(map(jsonl, dataset["test"]))


@click.command()
@click.option("--das-expl", is_flag=True, help="Whether to include the DAS explanation in the system prompt.")
@click.option("--no_move", is_flag=True, help="Whether to include move turns in user prompt")
@click.option("--n-shot", type=int, default=0, help="The number of examples to include in the prompt.")
@click.option("--nex-obs", type=int, default=20, help="The percentage of observe examples generated to keep.")
@click.option("--npc-obs", type=int, default=40, help="The percentage of observe tasks to keep in training dataset.")
@click.option("--enumerate", is_flag=True, help="Whether to enumerate the examples in the prompt.")
@click.option("--format", type=click.Choice(["chat", "instruct"]), default="chat", help='Type of prompt format')
def main(**kwargs):
    dm = DatasetManager(**kwargs)
    dm.save()


if __name__ == "__main__":
    variations = {
        "das-expl": [False, True],
        "no-move": [False, True],
        "n-shot": [0],
        "nex-obs": [20],
        "npc-obs": [20, 40, 100],
        "enumerate": [False],
        "format": ["chat", "instruct"],
    }

    manager = DatasetManager(variations)
