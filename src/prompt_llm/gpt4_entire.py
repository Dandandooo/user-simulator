import os, sys, json, re, time
from openai import AzureOpenAI, RateLimitError
from tqdm import tqdm

client = AzureOpenAI(
    azure_endpoint="https://uiuc-convai-sweden.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_KEY_4"),
    api_version="2024-02-15-preview"
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

    # !!! I didn't know what "role" did when I started, but it seems that splitting to system
    # and user will increase performance
    if system:
        message_texts = [[{"role": "system", "content": prompt[:prompt.rfind("Goal:")]},
                          {"role": "user", "content": prompt[prompt.rfind("Goal:"):]}]
                         for prompt in prompts]
    else:
        message_texts = [[{"role": "user", "content": prompt}] for prompt in prompts]
    responses = []
    for text in tqdm(message_texts):
        while True:
            try:
                completion = client.chat.completions.create(
                        model="UIUC-ConvAI-Sweden-GPT4",  # model = "deployment_name"
                        messages=text,
                        temperature=0.7,
                        max_tokens=1024,
                        top_p=0.95,
                        frequency_penalty=0,
                        presence_penalty=0,
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


if __name__ == "__main__":
    run_gpt()
