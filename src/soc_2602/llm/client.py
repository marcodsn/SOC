import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMClient:
    def __init__(self, model_id: str):
        # self.client = OpenAI(
        #     base_url="https://router.huggingface.co/v1",
        #     api_key=os.getenv("HF_TOKEN"),
        # )
        self.client = OpenAI(
            base_url="http://192.168.1.67:8083/v1",
            api_key=os.getenv("HF_TOKEN"),
        )
        self.model_id = model_id

    def generate(self, prompt: list, stop_sequences: list = []):
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=prompt,
            stop=stop_sequences,
            max_tokens=8192,
            temperature=1.0,
        )
