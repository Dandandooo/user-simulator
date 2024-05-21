from openai import AzureOpenAI
from functools import lru_cache
from src.model.causal import BaseLM
import os


class GPT4LM(BaseLM):
    def __init__(self, api_key=os.getenv("AZURE_OPENAI_KEY_4"),
                 azure_endpoint="https://uiuc-convai-sweden.openai.azure.com/", role="user"):
        super().__init__()
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-15-preview"
        )

        self.role = role

    @lru_cache
    def answer(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model="UIUC-ConvAI-Sweden-GPT4",  # model = "deployment_name"
            messages=[{"role": self.role, "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return completion.choices[0].message.content