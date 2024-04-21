import glob
import random
import click
import sys
import os

sys.path.append(os.getcwd())

from src.data.read_files import read_utterances

class LLMPromptMaker:
    def __init__(self, data_path="teach-dataset-parsed/", commander_name="COMMANDER", driver_name="DRIVER", **kwargs):
        self.data_path = data_path  # data should be formatted the way I started
        self.commander_name = commander_name.upper()
        self.driver_name = driver_name.upper()

        self.utterances: list[str] = []
        self.labels:     list[str] = []
        self.agents:     list[list[str]] = []
        self.load_data()
        self.n = len(self.utterances)
        self.dialogues = self.form_dialogues()

    def __getitem__(self, index: int | str):
        if isinstance(index, int):
            return self.form_dialogue(index)
        match index:
            case "utterances":
                return self.utterances
            case "labels":
                return self.labels
            case "agents":
                return self.agents
            case _:
                raise KeyError(f"Invalid key {index}")

    def _agent_name(self, index: int) -> str:
        return self.commander_name if self.agents[index].lower() == "commander" else self.driver_name

    def load_data(self):
        self.agents, self.utterances, self.labels = read_utterances(*glob.glob(f"{self.data_path}/*.json"))

    def form_dialogue(self, index) -> str:
        return f'<<{self._agent_name(index)}>> {self.utterances[index]} <<TURN>>'

    def form_dialogues(self) -> list[str]:
        return [self.form_dialogue(index) for index in range(0, len(self.utterances))]

    def get_dialogue(self, start, end) -> tuple[str, str]:
        assert start < end, "Start must be less than end!"
        assert end < self.n, "End must be less than the number of dialogues!"
        assert start >= 0, "Start must be greater than or equal to 0!"
        # Only return the first label because we care about the very next one. Compound sentences do not matter.
        return "\n".join(self.dialogues[start:end]), self.labels[end + 1][0]

    def get_random_examples(self, n: int, length: int = None) -> list[tuple[str, str]]:
        if length is not None:
            assert length > 0, "Length must be greater than 0!"
            lengths = [length] * n
            indices = random.sample(range(0, self.n-length-1), n)  # -1 to account for the fact we are predicting
        else:
            length = 6  # minimum length (arbitrarily set)
            max_length = 25  # also arbitrary, I just don't know how much is too much
            indices = random.sample(range(0, self.n-length-1), n)
            lengths = [random.randint(length, min(max_length, self.n-idx-1)) for idx in indices]

        return [self.get_dialogue(start, start + l) for start, l in zip(indices, lengths)]

    def make_prompt(self, n=10, length=None, numerate=True, num_tasks=0, **kwargs) -> str:
        prompt = ""

        with open("prompt_llm/prompt_segments/initial_instructions.txt", "r") as f:
            prompt += f.read() + '\n\n'

        with open("prompt_llm/prompt_segments/da_explain.txt", "r") as f:
            prompt += f.read() + '\n\n'

        prompt += "Below are some examples of dialogues, with the dialogue act for the following turn shown to you.\n"
        examples = self.get_random_examples(n, length)
        for idx, (dialogue, label) in enumerate(examples, start=1):
            prompt += f"Example {idx if numerate else ''}:\n{dialogue}\n\nAnswer {idx if numerate else ''}: {label}\n\n"

        with open("prompt_llm/prompt_segments/final_instructions.txt", "r") as f:
            prompt += f.read() + '\n\n'

        return prompt

    def generate_prompt(self, save_path="prompt_llm/generated.txt", **kwargs):
        with open(save_path, "w") as f:
            f.write(self.make_prompt(**kwargs))
        print(f"Prompt generated and saved to {save_path}")

@click.command()
@click.option("--data_path", "-d", default="teach-dataset-parsed/", help="Path to the data")
@click.option("--commander_name", default="COMMANDER", help="Name of the commander")
@click.option("--driver_name", default="DRIVER", help="Name of the driver")
@click.option("--n", "-n", "--num", default=10, help="Number of examples to generate")
@click.option("--length", "-l", default=None, help="Length of the examples")
@click.option("--numerate", is_flag=True, help="Whether to number the examples")
@click.option("--num_tasks", "-t", default=0, help="Number of tasks")
@click.option("--save_path", "-s", default="prompt_llm/generated.txt", help="Path to save the generated prompt")
def main(**kwargs):
    prompt_maker = LLMPromptMaker(**kwargs)
    prompt_maker.generate_prompt(**kwargs)


if __name__ == "__main__":
    import os
    print(os.getcwd())
    main()
