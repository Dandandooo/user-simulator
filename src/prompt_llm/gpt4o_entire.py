import os, sys, json, re, time, itertools as it, base64, glob
from openai import AzureOpenAI, RateLimitError
from tqdm import tqdm

client = AzureOpenAI(
    azure_endpoint="https://uiuc-convai.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_KEY_4o"),
    api_version="2024-07-01-preview"
)

# I don't really know what this does
role = "system"

digits = re.compile(r"\d+")


def run_gpt(folder="llm_prompts_data/turns/entire", system=False):
    prompts = []
    answers = []

    for filename in sorted(os.listdir(folder), key=lambda e: int(digits.search(e).group())):
        if "answer" in filename:
            answers.append(open(os.path.join(folder, filename), "r").read())
        else:
            prompts.append(open(os.path.join(folder, filename), "r").read())

    folder_name = folder.split("/")[-1]
    # !!! I didn't know what "role" did when I started, but it seems that splitting to system
    # and user will increase performance
    if system:
        message_texts = [[{"role": "system", "content": parse_images(prompt[:prompt.rfind("Goal:")], folder_name)},
                          {"role": "user", "content": parse_images(prompt[prompt.rfind("Goal:"):], folder_name)}]
                         for prompt in prompts]
    else:
        message_texts = [[{"role": "user", "content": prompt}] for prompt in prompts]
    responses = []
    for text in tqdm(message_texts):
        while True:
            try:
                completion = client.chat.completions.create(
                        model="gpt-4o-mini",  # model = "deployment_name"
                        messages=text,
                        temperature=0.2,
                        # I think this is fine even though prompts are longer b/c in docs it says "generated"
                        max_tokens=1024,
                        stop=None
                        )
                responses.append(completion.choices[0].message.content)
                break
            except RateLimitError:
                time.sleep(1)

    results = [{
            "prompt": prompt,
            "answer": answer,
            "response": response if response else ""
        } for prompt, response, answer in zip(prompts, responses, answers)
    ]

    json.dump(results, open("gpt4_result.json", 'w'), indent=4)


def parse_images(content: str, folder_name: str, targetobject=True, debug=False):
    timestamp_re = re.compile(r"<time (?P<timestamp>[1-9.]+)>")
    split_content = timestamp_re.split(content)
    timestamps = timestamp_re.finditer(content)

    base_dir = glob.glob(f"teach-dataset/images/*/{folder_name}")[0]

    message_content = []
    for content, time_regex in it.zip_longest(split_content, timestamps):

        timestamp = time_regex.group("timestamp")

        commander_path = os.path.join(base_dir, f"commander.frame.{timestamp}.jpeg")
        if os.path.exists(commander_path):
            if debug:
                print("Commander:", timestamp)
            commander_encoded = base64.b64encode(open(os.path.join(base_dir, f"commander.frame.{timestamp}.jpeg"), "rb").read()).decode("utf-8")

            message_content.append({
                "type": "text",
                "text": "What the commander sees:",
            })

            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{commander_encoded}",
                },
            })

        driver_path = os.path.join(base_dir, f"driver.frame.{timestamp}.jpeg")
        if os.path.exists(driver_path):
            if debug:
                print("Driver:", timestamp)
            driver_encoded = base64.b64encode(open(os.path.join(base_dir, f"driver.frame.{timestamp}.jpeg"), "rb").read()).decode("utf-8")


            message_content.append({
                "type": "text",
                "text": "What the driver sees:",
            })

            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{driver_encoded}",
                },
            })

        targetobject_path = os.path.join(base_dir, f"targetobject.frame.{timestamp}.jpeg")
        if targetobject and os.path.exists(targetobject_path):
            if debug:
                print("Target:", timestamp)
            targetobject_encoded = open(os.path.join(base_dir, f"targetobject.frame.{timestamp}.jpeg"), "rb").read()

            message_content.append({
                "type": "text",
                "text": "What the target object is:",
            })

            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(targetobject_encoded).decode('utf-8')}",
                },
            })

        message_content.append({
            "type": "text",
            "text": content,
        })

    return message_content


if __name__ == "__main__":
    run_gpt()
