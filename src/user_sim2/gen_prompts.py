import click
import json
import random


# TODO: export prompts as jsonl (new-line delimited json)
# TODO: auto generate names: "<dataset>__<file_id>__<turn_id>"
# TODO: generate datasets for all combinations
class PromptMaker:
    def __init__(self, **kwargs):
        def kwargs_that_are(of_type):
            return {k: v for k, v in kwargs.items() if of_type == type(v)}  # isinstance said True is of type int
        flags = kwargs_that_are(bool)
        n_choices = kwargs_that_are(int)

        self.kwargs = kwargs

        dataset_id = '_'.join([*filter(flags.__getitem__, flags), *[k.replace('n', str(v), 1) for k, v in n_choices.items()]])
        if not dataset_id:
            dataset_id = 'default'

        print("Dataset ID:", dataset_id)

        self.train_source = json.load(open(f"teach-data-parsed/train_turn.json"))
        self.valid_source = json.load(open(f"teach-data-parsed/valid_turn.json"))
        self.test_source = json.load(open(f"teach-data-parsed/test_turn.json"))

        self.example_source = self.train_source + self.valid_source

    def system_prompt(self):
        prompt = open("src/user_sim2/system_prompt.txt").read()
        if self.kwargs["das_expl"]:
            prompt += open("src/user_sim2/das_expl.txt").read()
        else:
            prompt += open("src/user_sim2/das_list.txt").read()
        return prompt

    def history(self, example, length) -> (str, str):
        hist = example["goal"] + '\n'
        for turn in example["turns"][:length]:
            if turn["turn_action"] == "move" and not self.kwargs["move"]:
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
        answer = "OBSERVE" if example["turns"][length]["COMMANDER"]["action"] != "dialogue" else example["turns"][length]["COMMANDER"]["das"][0]

        return hist, answer

    def examples(self, num_to_make: int):
        def choose_example():
            choice = random.choice(self.example_source)
            length = random.randint(1, len(choice["turns"]) - 1)
            if choice["turns"][length]["COMMANDER"]["action"] == "<observe>" and random.random() < self.kwargs["pc_ex_obs_keep"]:
                return choice, length
            return choose_example()
        for i in range(num_to_make):
            choice, length = choose_example()
            example = f'Example{f" {i}" * self.kwargs["enumerate"]}:\n'
            example += "\n".join(self.history(choice, length))
            yield example

    def user_prompt(self, task: dict, length: int) -> (str, str):
        prompt = open("src/user_sim2/user_prompt.txt").read()
        if self.kwargs["n_shot"]:
            prompt += "\nHere are some examples:\n\n"
            prompt += "\n\n".join(self.examples(self.kwargs["n_shot"]))
            prompt += "\n"
        prompt += "\nHere is your task:\n\n"
        task_, answer_ = self.history(task, length)
        prompt += task_
        return prompt, answer_

    def chat_prompt(self, task, length):
        user_, answer_ = self.user_prompt(task, length)
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": user_},
                {"role": "assistant", "content": answer_}
            ]
        }

    def instruct_prompt(self, task, length):
        user_, answer_ = self.user_prompt(task, length)
        return {
            "prompt": self.system_prompt() + "\n" + user_,
            "completion": answer_
        }

    def jsonl_batch(self, task, length):
        match self.kwargs["format"]:
            case "chat":
                message = self.chat_prompt(task, length)
            case "instruct":
                message = self.instruct_prompt(task, length)
            case _:
                raise ValueError()

        entry_name = f'{task["file_id"]}-turn{length}'
        entry = {
            "custom_id": entry_name,
            "method": "POST",
            "url": "/v1/chat/completions",

        }
        return json.dumps(entry)

    def generate_file(self, file):
        turns = [turn for turn in file["turns"] if (turn["turn_action"] != "move") or not self.kwargs["move"]]
        for length, turn in enumerate(file["turns"]):
            pass

class DatasetMaker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.pm = PromptMaker(**kwargs)


@click.command()
@click.option("--das-expl", is_flag=True, help="Whether to include the DAS explanation in the system prompt.")
@click.option("--move/--no_move", default=False, help="Whether to include move turns in user prompt")
@click.option("--n-shot", type=int, default=0, help="The number of examples to include in the prompt.")
@click.option("--ex-obs", type=float, default=.2, help="The percentage of observe examples generated to keep.")
@click.option("--npc-obs", type=int, default=40, help="The percentage of observe tasks to keep (out of 100)")
@click.option("--enumerate", is_flag=True, help="Whether to enumerate the examples in the prompt.")
@click.option("--format", type=click.Choice(["chat", "instruct"], case_sensitive=False), default="chat", help='Type of prompt format')
def main(**kwargs):
    pm = PromptMaker(**kwargs)


if __name__ == "__main__":
    main()
