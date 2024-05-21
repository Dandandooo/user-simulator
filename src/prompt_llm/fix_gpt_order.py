import os, re, json
from tqdm import tqdm

datapath = "llm_prompts_data/turns"
testpath = "llm_prompt_sessions/gpt4_valid"

temp_path = os.path.join(testpath, "temp")
temp_data = os.path.join(datapath, "valid")
temp5_path = os.path.join(testpath, "temp5")
temp5_data = os.path.join(datapath, "valid_5")
temp5noobs_path = os.path.join(testpath, "temp5noobs")
temp5noobs_data = os.path.join(datapath, "valid_5_no_obs")

digits = re.compile(r"\d+")


def sorted_d(filenames: list):
    return sorted(filenames, key=lambda e: int(digits.search(e).group()))


foldername = re.compile(r"\w+(?=_result.json)")
turns = re.compile(r"(?<=\n\n\n)Goal: (?:(?!COMMANDER response:).)+", re.MULTILINE + re.DOTALL)

for result_path, data_path in [(temp_path, temp_data), (temp5_path, temp5_data), (temp5noobs_path, temp5noobs_data)]:
    for file in tqdm(list(filter(lambda e: "result" in e, os.listdir(result_path)))):
        try:
            file_id = file[:file.index("_result.json")]
        except ValueError:
            raise ValueError(f"File {file} does not match the expected format")
        data = json.load(open(os.path.join(result_path, file), "r"))
        responses = [d["response"] for d in data]
        answers = []
        prompts = []
        for filename in sorted_d(os.listdir(os.path.join(data_path, file_id))):
            if "answer" in filename:
                answers.append(open(os.path.join(data_path, file_id, filename), "r").read())
            else:
                prompts.append(open(os.path.join(data_path, file_id, filename), "r").read())

        results = [{
            "prompt": prompt,
            "answer": answer,
            "response": response if response else ""
        } for prompt, response, answer in zip(prompts, responses, answers)]

        json.dump(results, open(os.path.join(result_path, file), 'w'), indent=4)
