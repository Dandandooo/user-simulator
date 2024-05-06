import click
import json
import glob
import random
import os
import shutil
import sys

class TurnMaker:
    def __init__(self, **kwargs):
        self.data_path = kwargs["data_path"]
        self.commander_name = kwargs["commander_name"]
        self.driver_name = kwargs['driver_name']
        self.n = kwargs["n"]
        self.task_length = kwargs["length"]
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

    def process_turn(self, turn: dict, is_last) -> str:
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
            return f"{self.commander_name}: {commander_action}{f' <<{commander_das}>>'if (self.include_das or (self.predict_das and is_last)) and commander_das else ''}\n{self.driver_name}: {driver_action}{f' <<{driver_das}>>' if self.include_das and driver_das else ''}"

    def get_rand_turns(self, task: dict, get_last=True, length=None) -> list[str]:
        if length is not None:
            length = length
        elif len(task['turns']) < self.min_length:
            length = len(task['turns'])
        else:
            length = self.task_length if self.task_length is not None else random.randint(self.min_length, min(len(task['turns']), self.max_length))
        start = 0
        end = start + length
        return [self.process_turn(turn, get_last and (i == length - 1)) for i, turn in enumerate(task['turns'][start:end])]

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
        prompt += "\n".join(turns[:-1]) + '\n'
        prompt += f"{self.commander_name} response:"
        ans = turns[-1].splitlines()[0]
        if self.html_format:
            ans = ans[ans.find(f"<{self.commander_name}>")+2+len(self.commander_name):ans.find("</")]
        else:
            ans = ans.split(":")[1].strip()
        if not self.predict_das:
            act = "OBSERVE" if "<observe>" in ans else "SPEAK"
        else:
            try:
                act = "OBSERVE" if "observe" in ans.lower() else ans[ans.index("<<") + 2:ans.index(">>")].split(",")[0].strip()
            except ValueError:
                print(ans)
                sys.exit()
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
            task = random.choice(self.tasks)
            p, a = self.display_example(task)
            prompt += p + '\n'
            answer = a

        return prompt, answer

    def generate_file(self, save_answer, save_path):
        task = random.choice(self.tasks)
        save_root, save_ext = save_path.rsplit(".", 1)
        episodes = [self.display_example(task, i) for i in range(1, len(task['turns'])+1)]
        prompts = [self.make_prompt(override=True) for _ in range(len(episodes))]
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
        print(f"Saved to {save_path[:save_path.rfind('/')]}")

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
@click.option("--n", "-n", "--num", default=2, help="Number of examples to generate")
@click.option("--length", "-l", default=None, help="Length of the examples")
@click.option("--numerate", is_flag=True, help="Whether to number the examples")
@click.option("--save_path", "-s", default="src/prompt_llm/generated_turn.txt", help="Path to save the generated prompt")
@click.option("--max_length", default=25, help="Maximum length of the examples")
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
# TODO add flag for observes to be included on
# TODO consider visual features (don't have them yet; nl-ified)
# TODO add flag on standard (random) vs cheating (don't make the only example have the same answer as the examples)
@click.option("--cheat", "-c", is_flag=True, help="Whether to cheat and make the only example have the same answer as the examples")
# TODO consider adding two examples with the same goal (maybe even from the same scenario)
# TODO measure using accuracy (split up) and f measure ->
# TODO get accuracy for entire game file sequentially
@click.option("--entire_file", "-e", is_flag=True, help="Whether to generate prompts for an entire game file")
def main(**kwargs):
    tm = TurnMaker(**kwargs)
    tm.generate_prompt(save_answer=kwargs["save_answer"], num_to_gen=kwargs["num_prompts"])
    # todo try 20-30 examples

if __name__ == "__main__":
    main()
