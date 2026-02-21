import os

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()


class LLMClient:
    def __init__(self, model_id: str):
        # self.client = OpenAI(
        #     base_url="https://router.huggingface.co/v1",
        #     api_key=os.getenv("HF_TOKEN"),
        # )
        # self.client = OpenAI(
        #     # base_url="http://192.168.1.67:8083/v1",
        #     base_url="http://10.8.0.5:8083/v1",
        #     api_key=os.getenv("HF_TOKEN"),
        # )
        # self.async_client = AsyncOpenAI(
        #     base_url="http://10.8.0.5:8083/v1",
        #     api_key=os.getenv("HF_TOKEN"),
        # )
        self.client = OpenAI(
            base_url="http://10.8.0.5:8000/v1",
            api_key=os.getenv("HF_TOKEN"),
        )
        self.async_client = AsyncOpenAI(
            base_url="http://10.8.0.5:8000/v1",
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

    async def generate_async(self, prompt: list, stop_sequences: list = []):
        return await self.async_client.chat.completions.create(
            model=self.model_id,
            messages=prompt,
            stop=stop_sequences,
            max_tokens=8192,
            temperature=1.0,
        )
