import os

from dotenv import load_dotenv
from mistralai import Mistral
from openai import AsyncOpenAI, OpenAI

load_dotenv()

PROVIDER_LOCAL = "local"
PROVIDER_HF = "huggingface"
PROVIDER_MISTRAL = "mistral"
PROVIDER_NIM = "nim"
PROVIDER_MODAL = "modal"

_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
_MODAL_BASE_URL = "https://api.us-west-2.modal.direct/v1"


class LLMClient:
    def __init__(self, model_id: str, provider: str = PROVIDER_MODAL):
        self.model_id = model_id
        self.provider = provider

        if provider == PROVIDER_MISTRAL:
            self._mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

        elif provider == PROVIDER_NIM:
            self.client = OpenAI(
                base_url=_NIM_BASE_URL,
                api_key=os.getenv("NVIDIA_API_KEY"),
            )
            self.async_client = AsyncOpenAI(
                base_url=_NIM_BASE_URL,
                api_key=os.getenv("NVIDIA_API_KEY"),
            )

        elif provider == PROVIDER_MODAL:
            base_url = _MODAL_BASE_URL
            self.client = OpenAI(
                base_url=base_url,
                api_key=os.getenv("MODAL_TOKEN"),
            )
            self.async_client = AsyncOpenAI(
                base_url=base_url,
                api_key=os.getenv("MODAL_TOKEN"),
            )

        elif provider == PROVIDER_HF:
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=os.getenv("HF_TOKEN"),
            )
            self.async_client = AsyncOpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=os.getenv("HF_TOKEN"),
            )

        else:  # local
            self.client = OpenAI(
                base_url="http://10.8.0.5:8000/v1",
                api_key=os.getenv("HF_TOKEN"),
            )
            self.async_client = AsyncOpenAI(
                base_url="http://10.8.0.5:8000/v1",
                api_key=os.getenv("HF_TOKEN"),
            )

    # ── NIM-specific params passed via extra_body ──────────────────────────
    @staticmethod
    def _nim_extra(kwargs: dict) -> dict:
        """
        Pull NIM-only keys out of kwargs and return them as extra_body.
        Supported extras: top_k, repetition_penalty, chat_template_kwargs.
        """
        nim_keys = ("top_k", "repetition_penalty", "chat_template_kwargs")
        extra_body = {k: kwargs.pop(k) for k in nim_keys if k in kwargs}
        return extra_body

    # ── Public API ─────────────────────────────────────────────────────────
    def generate(self, prompt: list, stop_sequences: list = [], **kwargs):
        if self.provider == PROVIDER_MISTRAL:
            return self._mistral.chat.complete(
                model=self.model_id,
                messages=prompt,
                stop=stop_sequences or None,
                max_tokens=8192,
                temperature=1.0,
            )

        extra_body = self._nim_extra(kwargs) if self.provider == PROVIDER_NIM else {}

        return self.client.chat.completions.create(
            model=self.model_id,
            messages=prompt,
            stop=stop_sequences or None,
            max_tokens=8192,
            temperature=1.0,
            extra_body=extra_body or None,
            **kwargs,
        )

    async def generate_async(self, prompt: list, stop_sequences: list = [], **kwargs):
        if self.provider == PROVIDER_MISTRAL:
            return await self._mistral.chat.complete_async(
                model=self.model_id,
                messages=prompt,
                stop=stop_sequences or None,
                max_tokens=8192,
                temperature=1.0,
            )

        extra_body = self._nim_extra(kwargs) if self.provider == PROVIDER_NIM else {}

        return await self.async_client.chat.completions.create(
            model=self.model_id,
            messages=prompt,
            stop=stop_sequences or None,
            max_tokens=8192,
            temperature=1.0,
            extra_body=extra_body or None,
            **kwargs,
        )
