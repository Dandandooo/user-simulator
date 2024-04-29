import os, sys, json
from openai import AzureOpenAI

prompt = open(sys.argv[1], "r").read()
answer = open(sys.argv[2], "r").read()

client = AzureOpenAI(
  azure_endpoint="https://uiuc-convai-sweden.openai.azure.com/",
  api_key=os.getenv("AZURE_OPENAI_KEY_4"),
  api_version="2024-02-15-preview"
)

message_text = [{
  "role": "system",
  "content": prompt
}]
completion = client.chat.completions.create(
  model="UIUC-ConvAI-Sweden-GPT4",  # model = "deployment_name"
  messages=message_text,
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None
)

responses = [{"index": choice.index, "message": choice.message.content} for choice in completion.choices]
results = {
  "prompt": prompt,
  "answer": answer,
  "responses": responses
}
json.dump(results, open("gpt4_result.json", 'w'), indent=4)
