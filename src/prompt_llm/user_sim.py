import click
import json
import random
import os
import shutil
import sys

from tqdm import tqdm


class TurnMaker:
    def __init__(self, **kwargs):
        self.data_path = kwargs["data_path"]
        self.commander_name = kwargs["commander_name"]
        self.driver_name = kwargs['driver_name']
        self.n = kwargs["n"]
        self.task_length = eval(str(kwargs["length"]))
        self.numerate = kwargs["numerate"]
        self.save_path = kwargs["save_path"]
        self.give_task = not kwargs["no_task"]
        self.include_das = kwargs["include_das"]
        self.max_length = kwargs["max_length"]
        self.min_length = kwargs["min_length"]
        self.nl_ify = kwargs["nl_ify"]
        self.html_format = kwargs["html_format"]
        self.predict_das = kwargs["predict_das"]
        self.gen_response = kwargs["gen_response"]
        self.cheat = kwargs["cheat"]
        self.ex_max_obs = kwargs["example_max_percent_observe"]
        self.entire_file = kwargs["entire_file"]
        self.no_obs = kwargs["no_obs"]
        self.split_dataset = kwargs["split_dataset"]
        self.entire_dataset = kwargs["entire_dataset"]
        self.no_move = kwargs["no_move"]

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

        def filter_move(task) -> dict:
            if self.no_move:
                task['turns'] = [turn for turn in task['turns'] if "move" not in turn['DRIVER']['action']]
            return task

        # Chosen since it has the most examples
        with open(os.path.join(self.data_path, "train_turn.json"), 'r') as f:
            train_source: list[dict] = [filter_move(task) for task in json.load(f)]

        # Chosen since it has the least number of examples
        with open(os.path.join(self.data_path, "valid_seen_turn.json"), 'r') as f:
            valid_seen_source: list[dict] = [filter_move(task) for task in json.load(f)]

        with open(os.path.join(self.data_path, "valid_unseen_turn.json"), 'r') as f:
            valid_unseen_source: list[dict] = [filter_move(task) for task in json.load(f)]

        self.example_source: list[dict] = train_source + valid_unseen_source

        match kwargs["split"]:
            case "train":
                self.task_source: list[dict] = train_source
            case "valid":
                self.task_source: list[dict] = valid_unseen_source
            case "test":
                self.task_source: list[dict] = valid_seen_source

        self.tasks: list[dict] = self.example_source + self.task_source

    def agent_name(self, original_name) -> str:
        match original_name.lower():
            case "commander":
                return self.commander_name
            case "driver":
                return self.driver_name
            case _:
                raise ValueError(f"Invalid agent name: {original_name}")

    def process_turn(self, turn: dict, is_last=False) -> str:
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
            to_ret = "<TURN> "
            if is_last or not (("observe" in commander_action.lower()) and self.no_obs):  # Either none or both
                to_ret += f"<{self.commander_name}> {commander_action} {f'<<{commander_das}>> ' if self.include_das and commander_das else ''}</{self.commander_name}>"
                if is_last:
                    return to_ret
            if not (("observe" in driver_action.lower()) and self.no_obs):
                to_ret += f"<{self.driver_name}> {driver_action} {f'<<{driver_das}>> ' if self.include_das and driver_das else ''}</{self.driver_name}>"
            to_ret += "</TURN>"
        else:
            to_ret = ""
            if is_last or not (("observe" in commander_action.lower()) and self.no_obs):  # Either none or both
                to_ret += f"{self.commander_name}: {commander_action}{f' <<{commander_das}>>' if self.include_das and commander_das else ''}" + "\n" * (not self.no_obs)
                if is_last:
                    return to_ret
            if not (("observe" in driver_action.lower()) and self.no_obs):
                to_ret += f"{self.driver_name}: {driver_action}{f' <<{driver_das}>>' if self.include_das and driver_das else ''}"
        return to_ret

    def get_rand_turns(self, task: dict, get_last=True, length=None) -> list[dict]:
        if length is not None:
            length = length
        elif len(task['turns']) < self.min_length:
            length = len(task['turns'])
        else:
            length = self.task_length if self.task_length is not None else random.randint(self.min_length, min(len(task['turns']), self.max_length))
        start = 0
        end = start + length
        return task['turns'][start:end]

    def display_examples(self) -> str:
        examples = []
        used_tasks = []
        while len(examples) < self.n:
            task = random.choice(self.tasks)
            prompt, answer = self.display_example(task)
            if not examples:
                examples.append((prompt, answer))
                used_tasks.append(task)
            elif task in used_tasks and not self.cheat:
                continue
            elif answer.startswith("OBSERVE") and sum(["OBSERVE" in ans for _, ans in examples]) >= (float(self.ex_max_obs) / 100.0) * len(examples):
                continue
            else:
                examples.append((prompt, answer))
                used_tasks.append(task)

        return "\n\n".join([f"Example {i if self.numerate else ''}:\n{prompt}\n{answer}" for i, (prompt, answer) in enumerate(examples)])

    def display_example(self, task: dict, length=None) -> tuple[str, str]:
        if self.html_format:
            prompt = f"<goal> {task['goal']} </goal>\n"
        else:
            prompt = f"Goal: {task['goal']}\n"
        turns = self.get_rand_turns(task, length=length)
        prompt += "\n".join(map(self.process_turn, turns[:-1])) + '\n'
        prompt += f"{self.commander_name} response:"
        ans = self.process_turn(turns[-1], is_last=True)
        if self.html_format:
            ans = ans[ans.find(f"<{self.commander_name}>")+2+len(self.commander_name):ans.find("</")]
        else:
            ans = ans.split(":", maxsplit=1)[1].strip()
        if not self.predict_das:
            act = "OBSERVE" if "<observe>" in ans else "SPEAK"
        else:
            try:
                act = "OBSERVE" if "observe" in ans.lower() else ans[ans.index("<<") + 2:ans.index(">>")].split(",")[0].strip()
            except ValueError:
                raise ValueError(f"Error in parsing dialogue act: {task['file_id']} {turns[-1]}")
        utt = ans.split(":")[-1][:ans.rfind("<<")].strip() if self.gen_response and "<observe>" not in ans else ""
        return prompt, f"{act}\n{utt}".strip()

    def give_instructions(self) -> str:
        prompt = ""
        prompt += open("src/prompt_llm/user_sim_segments/initial_instructions.txt", "r").read() + '\n\n'

        if self.include_das or self.predict_das:
            if self.nl_ify:
                prompt += open("src/prompt_llm/user_sim_segments/da_nl_explain.txt", "r").read() + '\n\n'
            else:
                prompt += open("src/prompt_llm/user_sim_segments/da_explain.txt", "r").read() + '\n\n'

        prompt += self.display_examples() + '\n\n'

        if self.predict_das:
            prompt += open("src/prompt_llm/user_sim_segments/final_da_instructions.txt", "r").read() + '\n'
        else:
            prompt += open("src/prompt_llm/user_sim_segments/final_instructions.txt", "r").read() + '\n'
        if self.gen_response:
            prompt += "You must also generate a response to the given example based on the dialogue act you predict.\n"
        prompt += '\n'

        return prompt

    def make_prompt(self, override=False) -> tuple[str, str]:
        prompt = self.give_instructions()

        answer = ""
        if self.give_task and not override:
            prompt += "Give your answer for the following example:\n"
            task = random.choice(self.example_source if self.split_dataset else self.tasks)
            p, a = self.display_example(task)
            prompt += p + '\n'
            answer = a

        return prompt, answer

    def generate_file(self, save_answer, save_path, task: dict = None, dont_edit_folders=False, dont_print=False):
        if task is None:
            task = random.choice(self.tasks)
        save_root, save_ext = save_path.rsplit(".", 1)
        episodes = [self.display_example(task, i) for i in range(1, len(task['turns'])+1)]
        prompts = [self.make_prompt(override=True) for _ in range(len(episodes))]

        if not dont_edit_folders:
            if os.path.exists(save_path[:save_path.rfind('/')]):
                shutil.rmtree(save_path[:save_path.rfind('/')])
            os.mkdir(save_path[:save_path.rfind('/')])

        for i, ((prompt, _), (task, answer)) in enumerate(zip(prompts, episodes)):
            with open(f"{save_root}_{i}.{save_ext}", "w+") as f:
                f.write(prompt)
                f.write(f"\n{task}")
                if save_answer:
                    with open(f"{save_root}_{i}_answer.{save_ext}", "w+") as a:
                        a.write(answer)
        if not dont_print:
            print(f"Saved to {save_path[:save_path.rfind('/')]}")

    def generate_entire_dataset(self, save_path):
        save_root = save_path
        save_file = "turn.txt"

        if os.path.exists(save_root):
            shutil.rmtree(save_root)
        os.mkdir(save_root)

        print("Generating entire dataset...")
        for target in tqdm(self.task_source):
            folder = os.path.join(save_root, target["file_id"])
            os.mkdir(folder)
            path = os.path.join(folder, save_file)
            self.generate_file(True, path, task=target, dont_edit_folders=True, dont_print=True)


    def generate_prompt(self, save_answer=False, save_path=None, num_to_gen=1):
        if save_path is None:
            save_path = self.save_path
        prompts = []
        answers = []

        if self.entire_file:
            self.generate_file(save_answer, save_path)
            return

        while len(prompts) < num_to_gen:
            prompt, answer = self.make_prompt()
            if not prompts:
                if answer.startswith("OBSERVE"):
                    continue
                prompts.append(prompt)
                answers.append(answer)
            elif answer.startswith("OBSERVE") and sum([ans.startswith("OBSERVE") for ans in answers]) < (float(self.kwargs["max_percent_observe"]) / 100.0) * len(answers):
                prompts.append(prompt)
                answers.append(answer)
            elif not answer.startswith("OBSERVE"):
                prompts.append(prompt)
                answers.append(answer)
            else:
                print("Skipping b/c too many observes")

        if num_to_gen == 1:
            prompt, answer = prompts[0], answers[0]
            with open(save_path, "w+") as f:
                f.write(prompt)
                if save_answer:
                    root, ext = save_path.rsplit(".", 1)
                    with open(f"{root}_answer.{ext}", "w+") as a:
                        a.write(answer)
            print(f"Prompt generated and saved to {save_path}")
            return
        root, ext = save_path.rsplit(".", 1)
        for i, (prompt, answer) in enumerate(zip(prompts, answers)):
            with open(f"{root}_{i}.{ext}", "w+") as f:
                f.write(prompt)
                if save_answer:
                    with open(f"{root}_{i}_answer.{ext}", "w+") as a:
                        a.write(answer)
            print(f"Prompt generated and saved to {root}_{i}.{ext}")


@click.command()
@click.option("--data_path", "-d", default="teach-dataset-parsed/", help="Path to the data")
@click.option("--commander_name", default="COMMANDER", help="Name of the commander")
@click.option("--driver_name", default="DRIVER", help="Name of the driver")
@click.option("--n", "-n", "--num", default=5, help="Number of examples to generate")
@click.option("--length", "-l", default=None, help="Length of the examples")
@click.option("--numerate", is_flag=True, help="Whether to number the examples")
@click.option("--save_path", "-s", default="src/prompt_llm/generated_turn.txt", help="Path to save the generated prompt")
@click.option("--max_length", default=10, help="Maximum length of the examples")
@click.option("--min_length", default=1, help="Minimum length of the examples")
@click.option("--no_task", "-t", is_flag=True, help="Whether to give a task to the LLM to classify at the end")
@click.option("--include_das", "-i", is_flag=True, help="Whether to include dialogue acts in the examples")
@click.option("--nl_ify", "--nl", is_flag=True, help="Make dialogue acts be more natural language-y")
@click.option("--html-format", is_flag=True, help="Whether to format the prompt in HTML")
@click.option("--predict_das", "-p", is_flag=True, help="Whether to make the task be predicting dialogue acts")
@click.option("--num_prompts", default=1, help="Number of prompts to generate")
@click.option("--save_answer", "-a", is_flag=True, help="Whether to save the answer to the prompt")
@click.option("--example_max_percent_observe", default=35, help="Maximum percentage of given examples to be observation (to balance the data)")
@click.option("--max_percent_observe", default=35, help="Maximum percentage of tasks to be observation (to balance the data)")
@click.option("--gen_response", "-r", is_flag=True, help="Whether to generate a response to the prompt in addition to the dialogue act")
# TODO setup running with images
# TODO consider visual features (don't have them yet; nl-ified)
@click.option("--cheat", "-c", is_flag=True, help="Whether to cheat and make the only example have the same answer as the examples")
@click.option("--no_obs", is_flag=True, help="Whether to omit observes in the examples provided, but not the task")
@click.option("--entire_file", "-e", is_flag=True, help="Whether to generate prompts for an entire game file")
@click.option("--split_dataset", is_flag=True, help="Whether to split the dataset")
@click.option("--split", default="test", help="Which dataset to split")
@click.option("--train", is_flag=True, help="Whether to generate prompts for the training set, not the entire dataset")
@click.option("--entire_dataset", is_flag=True, help="Whether to generate prompts for the entire dataset. Save path should be a folder for this one.")
@click.option("--no_move", is_flag=True, help="Whether to not remove the move actions")
def main(**kwargs):
    tm = TurnMaker(**kwargs)
    if kwargs["entire_dataset"]:
        tm.generate_entire_dataset(kwargs["save_path"])
    else:
        tm.generate_prompt(save_answer=kwargs["save_answer"], num_to_gen=kwargs["num_prompts"])


if __name__ == "__main__":
    main()
