import click
import json
import glob
import random
import os

class TurnMaker:
    def __init__(self, **kwargs):
        self.data_path = kwargs["data_path"]
        self.commander_name = kwargs["commander_name"]
        self.driver_name = kwargs['driver_name']
        self.n = kwargs["n"]
        self.task_length = kwargs["length"]
        self.numerate = kwargs["numerate"]
        self.save_path = kwargs["save_path"]
        self.give_task = kwargs["give_task"]

        self.tasks: list[dict] = [task for datafile in glob.glob(os.path.join(self.data_path, "*_turn.json"))
                                  for task in json.load(open(datafile, 'r'))]

    def agent_name(self, original_name) -> str:
        return self.commander_name if original_name.lower() == "commander" else self.driver_name

    def process_turn(self, turn: dict) -> str:
        driver_action = turn['DRIVER']['action'] if turn['DRIVER']['action'] != 'dialogue' else turn["DRIVER"]['utterance']
        commander_action = turn['COMMANDER']['action'] if turn['COMMANDER']['action'] != 'dialogue' else turn["COMMANDER"]['utterance']
        return (f"<TURN> <{self.commander_name}> {commander_action} </{self.commander_name}>\n"
                f"       <{self.driver_name}> {driver_action} </{self.driver_name}> </TURN>")

    def get_rand_turns(self, task: dict) -> list[str]:
        length = self.task_length if self.task_length is not None else random.randint(6, min(len(task['turns']) - 2, 25))
        start = 0
        end = start + length
        return [self.process_turn(turn) for turn in task['turns'][start:end]]

    def make_prompt(self) -> str:
        prompt = ""
        prompt += open("src/prompt_llm/prompt_segments/initial_instructions.txt", "r").read() + '\n\n'

        tasks = random.sample(self.tasks, self.n + self.give_task)

        for i, task in enumerate(tasks[:self.n]):
            prompt += f"Example {i if self.numerate else ''}:\n"
            prompt += f"<goal> {task['goal']} </goal>\n"
            prompt += "\n".join(self.get_rand_turns(task)) + '\n\n'

        prompt += open("src/prompt_llm/prompt_segments/final_instructions.txt", "r").read() + '\n\n'

        if self.give_task:
            prompt += "Give your answer for the following example:\n"
            prompt += f"<goal> {tasks[-1]['goal']} </goal>\n"
            prompt += "\n".join(self.get_rand_turns(tasks[-1])) + '\n\n'
        return prompt

    def generate_prompt(self):
        with open(self.save_path, "w+") as f:
            f.write(self.make_prompt())
        print(f"Prompt generated and saved to {self.save_path}")


@click.command()
@click.option("--data_path", "-d", default="teach-dataset-parsed/", help="Path to the data")
@click.option("--commander_name", default="COMMANDER", help="Name of the commander")
@click.option("--driver_name", default="DRIVER", help="Name of the driver")
@click.option("--n", "-n", "--num", default=10, help="Number of examples to generate")
@click.option("--length", "-l", default=None, help="Length of the examples")
@click.option("--numerate", is_flag=True, help="Whether to number the examples")
@click.option("--save_path", "-s", default="src/prompt_llm/generated_turn.txt", help="Path to save the generated prompt")
@click.option("--max_length", default=25, help="Maximum length of the examples")
@click.option("--min_length", default=6, help="Minimum length of the examples")
@click.option("--give_task", "-t", is_flag=True, help="Whether to give a task to the LLM to classify at the end")
def main(**kwargs):
    tm = TurnMaker(**kwargs)
    tm.generate_prompt()


if __name__ == "__main__":
    main()
