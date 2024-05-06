import os, sys, json
from openai import AzureOpenAI
from tqdm import tqdm

client = AzureOpenAI(
    azure_endpoint="https://uiuc-convai-sweden.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_KEY_4"),
    api_version="2024-02-15-preview"
)

# I don't really know what this does
role = "system"

prompts = []
answers = []

folder = "llm_prompts_data/turns/entire"

for filename in os.listdir(folder):
    if "answer" in filename:
        answers.append(open(os.path.join(folder, filename), "r").read())
    else:
        prompts.append(open(os.path.join(folder, filename), "r").read())

message_texts = [{"role": "user", "content": prompt} for prompt in prompts]
responses = []
for text in tqdm(message_texts):
    completion = client.chat.completions.create(
        model="UIUC-ConvAI-Sweden-GPT4",  # model = "deployment_name"
        messages=[text],
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    responses.append(completion.choices[0].message.content)

results = [{
        "prompt": prompt,
        "answer": answer,
        "response": response
    } for prompt, response, answer in zip(prompts, responses, answers)
]

json.dump(results, open("gpt4_result.json", 'w'), indent=4)
