import os

from huggingface_hub import InferenceClient


class LLMClient:
    def __init__(self, model_id: str):
        self.client = InferenceClient(token=os.getenv("HF_TOKEN"))
        self.model_id = model_id

    def generate(self, prompt: str, stop_sequences: list = []):
        return self.client.text_generation(
            prompt,
            model=self.model_id,
            max_new_tokens=512,
            stop_sequences=stop_sequences or ["<EOS>", "\nUser:"],
        )
