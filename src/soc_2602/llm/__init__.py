"""
LLM client module for the SOC pipeline.

Provides a unified :class:`LLMClient` that wraps multiple inference
providers (local vLLM, HuggingFace Inference API, NVIDIA NIM, Modal,
Mistral) behind an OpenAI-compatible interface with both synchronous
and asynchronous generation methods.

Prompt Caching
--------------
The client exposes a :meth:`LLMClient.build_messages` static helper
that structures messages to maximise the byte-identical prefix across
calls within the same generation stage.  Static instructions go into
the ``system`` message; variable per-call content (few-shot examples,
persona data, conversation history) goes into the ``user`` message.
This layout gives providers with automatic prompt caching (vLLM prefix
caching, Anthropic, OpenAI, DeepSeek, etc.) the best chance of
reusing the KV cache across requests.

Provider Constants
------------------
.. data:: PROVIDER_LOCAL
.. data:: PROVIDER_HF
.. data:: PROVIDER_MISTRAL
.. data:: PROVIDER_NIM
.. data:: PROVIDER_MODAL
"""

from soc_2602.llm.client import (
    PROVIDER_HF,
    PROVIDER_LOCAL,
    PROVIDER_MISTRAL,
    PROVIDER_MODAL,
    PROVIDER_NIM,
    LLMClient,
)

__all__ = [
    "LLMClient",
    "PROVIDER_LOCAL",
    "PROVIDER_HF",
    "PROVIDER_MISTRAL",
    "PROVIDER_NIM",
    "PROVIDER_MODAL",
]
