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
        self.include_das = kwargs["include_das"]
        self.nl_ify = kwargs["nl_ify"]
        self.html_format = kwargs["html_format"]
        self.predict_das = kwargs["predict_das"]

        if self.nl_ify:
            self.das = {
                "Acknowledge": "Acknowledge",
                "Affirm": "Affirm",
                "AlternateQuestions": "Alternate Question",
                "Confirm": "Confirm",
                "Deny": "Deny",
                "FeedbackNegative": "Negative Feedback",
                "FeedbackPositive": "Positive Feedback",
                "Greetings/Salutations": "Greeting or Salutation",
                "InformationOnObjectDetails": "Information on Object Details",
                "InformationOther": "Other Information",
                "Instruction": "Instruction",
                "MiscOther": "Miscellaneous",
                "NotifyFailure": "Notify user of Failure",
                "OtherInterfaceComment": "Other Interface Comment",
                "RequestForInstruction": "Request a new Instruction",
                "RequestForObjectLocationAndOtherDetails": "Request for Object Information",
                "RequestMore": "Request More Information",
                "RequestOtherInfo": "Request Other Information",
            }
        else:
            self.das = {
                "Acknowledge": "Acknowledge",
                "Affirm": "Affirm",
                "AlternateQuestions": "AlternateQuestions",
                "Confirm": "Confirm",
                "Deny": "Deny",
                "FeedbackNegative": "FeedbackNegative",
                "FeedbackPositive": "FeedbackPositive",
                "Greetings/Salutations": "Greetings/Salutations",
                "InformationOnObjectDetails": "InformationOnObjectDetails",
                "InformationOther": "InformationOther",
                "Instruction": "Instruction",
                "MiscOther": "MiscOther",
                "NotifyFailure": "NotifyFailure",
                "OtherInterfaceComment": "OtherInterfaceComment",
                "RequestForInstruction": "RequestForInstruction",
                "RequestForObjectLocationAndOtherDetails": "RequestForObjectLocationAndOtherDetails",
                "RequestMore": "RequestMore",
                "RequestOtherInfo": "RequestOtherInfo",
            }

        self.kwargs = kwargs

        self.tasks: list[dict] = [task for datafile in glob.glob(os.path.join(self.data_path, "*_turn.json"))
                                  for task in json.load(open(datafile, 'r'))]

    def agent_name(self, original_name) -> str:
        return self.commander_name if original_name.lower() == "commander" else self.driver_name

    def process_turn(self, turn: dict) -> str:
        driver_action = turn['DRIVER']['action'] if turn['DRIVER']['action'] != 'dialogue' else turn["DRIVER"]['utterance']
        commander_action = turn['COMMANDER']['action'] if turn['COMMANDER']['action'] != 'dialogue' else turn["COMMANDER"]['utterance']
        if turn['DRIVER']['action'] == 'dialogue':
            driver_das = ','.join([self.das[da] for da in turn['DRIVER']['das']])
        else:
            driver_das = ""
        if turn['COMMANDER']['action'] == 'dialogue':
            commander_das = ','.join([self.das[da] for da in turn['COMMANDER']['das']])
        else:
            commander_das = ""
        if self.html_format:
            return (f"<TURN> <{self.commander_name}> {commander_action} {f'<<{commander_das}>> ' if self.include_das and commander_das else ''}</{self.commander_name}>\n"
                    f"       <{self.driver_name}> {driver_action} {f'<<{driver_das}>> ' if self.include_das and driver_das else ''}</{self.driver_name}> </TURN>")
        else:
            return f"{self.commander_name}: {commander_action}{f' <<{commander_das}>>' if self.include_das and commander_das else ''}\n{self.driver_name}: {driver_action}{f' <<{driver_das}>>' if self.include_das and driver_das else ''}"

    def get_rand_turns(self, task: dict) -> list[str]:
        length = self.task_length if self.task_length is not None else random.randint(6, min(len(task['turns']) - 2, 25))
        start = 0
        end = start + length
        return [self.process_turn(turn) for turn in task['turns'][start:end]]

    def make_prompt(self) -> str:
        prompt = ""
        prompt += open("src/prompt_llm/user_sim_segments/initial_instructions.txt", "r").read() + '\n\n'

        if self.include_das or self.predict_das:
            if self.nl_ify:
                prompt += open("src/prompt_llm/user_sim_segments/da_nl_explain.txt", "r").read() + '\n\n'
            else:
                prompt += open("src/prompt_llm/user_sim_segments/da_explain.txt", "r").read() + '\n\n'

        tasks = random.sample(self.tasks, self.n + self.give_task)

        for i, task in enumerate(tasks[:self.n]):
            prompt += f"Example {i if self.numerate else ''}:\n"
            prompt += f"<goal> {task['goal']} </goal>\n"
            turns = self.get_rand_turns(task)
            prompt += "\n".join(turns[:-1]) + '\n'
            prompt += "Commander's Response:\n"
            prompt += turns[-1].splitlines()[0] + '\n\n'
            prompt += '\n\n'

        if self.predict_das:
            prompt += open("src/prompt_llm/user_sim_segments/final_da_instructions.txt", "r").read() + '\n\n'
        else:
            prompt += open("src/prompt_llm/user_sim_segments/final_instructions.txt", "r").read() + '\n\n'

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
@click.option("--include_das", "-i", is_flag=True, help="Whether to include dialogue acts in the examples")
@click.option("--nl_ify", "--nl", is_flag=True, help="Make dialogue acts be more natural language-y")
@click.option("--html-format", is_flag=True, help="Whether to format the prompt in HTML")
@click.option("--predict_das", "-p", is_flag=True, help="Whether to make the task be predicting dialogue acts")
def main(**kwargs):
    tm = TurnMaker(**kwargs)
    tm.generate_prompt()


if __name__ == "__main__":
    main()
