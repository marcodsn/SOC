"""
LLM client for the SOC pipeline.

Supports multiple providers (local vLLM, HuggingFace, NVIDIA NIM, Modal,
Mistral) through an OpenAI-compatible interface, with optional prompt
caching via stable system-message prefixes.

Prompt Caching Strategy
-----------------------
Most providers that support automatic prompt caching (Anthropic, OpenAI,
DeepSeek, vLLM with --enable-prefix-caching) do so based on a shared
prefix across requests.  To exploit this we:

1. Put ALL static instruction text into the ``system`` message.
2. Keep the ``user`` message for variable-per-call content only
   (few-shot examples, persona data, conversation history).
3. Expose a ``build_messages`` helper that enforces this split so
   callers don't accidentally interleave static and dynamic content.

This maximises the byte-identical prefix across calls within the same
generation stage, giving the provider the best chance of hitting the
cache.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

# ── Provider constants ───────────────────────────────────────────────────────

PROVIDER_LOCAL = "local"
PROVIDER_HF = "huggingface"
PROVIDER_MISTRAL = "mistral"
PROVIDER_NIM = "nim"
PROVIDER_MODAL = "modal"

_DEFAULT_ENDPOINTS: Dict[str, str] = {
    PROVIDER_LOCAL: "http://10.8.0.5:8083/v1",
    PROVIDER_HF: "https://router.huggingface.co/v1",
    PROVIDER_NIM: "https://integrate.api.nvidia.com/v1",
    PROVIDER_MODAL: "https://api.us-west-2.modal.direct/v1",
}

_DEFAULT_API_KEY_ENVS: Dict[str, str] = {
    PROVIDER_LOCAL: "HF_TOKEN",
    PROVIDER_HF: "HF_TOKEN",
    PROVIDER_NIM: "NVIDIA_API_KEY",
    PROVIDER_MODAL: "MODAL_TOKEN",
    PROVIDER_MISTRAL: "MISTRAL_API_KEY",
}


class LLMClient:
    """Unified LLM client with support for sync/async generation.

    Parameters
    ----------
    model_id : str
        Model identifier as expected by the provider (e.g.
        ``"moonshotai/Kimi-K2-Instruct-0905"``).
    provider : str
        One of ``"local"``, ``"huggingface"``, ``"nim"``, ``"modal"``,
        ``"mistral"``.
    temperature : float
        Sampling temperature.  Default ``1.0``.
    max_tokens : int
        Maximum tokens to generate per call.  Default ``8192``.
    base_url : str | None
        Override the default endpoint URL for the chosen provider.
    api_key : str | None
        Explicit API key.  When ``None`` the key is read from the
        environment variable associated with the provider.
    """

    def __init__(
        self,
        model_id: str,
        provider: str = PROVIDER_LOCAL,
        temperature: float = 1.0,
        max_tokens: int = 8192,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_id = model_id
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Resolve API key
        env_var = _DEFAULT_API_KEY_ENVS.get(provider, "HF_TOKEN")
        resolved_key = api_key or os.getenv(env_var, "")

        if provider == PROVIDER_MISTRAL:
            # Lazy import — only needed when actually using Mistral
            from mistralai import Mistral

            self._mistral = Mistral(api_key=resolved_key)
            # Sync/async OpenAI clients are unused for Mistral but we
            # set them to None so attribute access doesn't explode.
            self.client: Optional[OpenAI] = None
            self.async_client: Optional[AsyncOpenAI] = None
        else:
            url = base_url or _DEFAULT_ENDPOINTS.get(
                provider, _DEFAULT_ENDPOINTS[PROVIDER_LOCAL]
            )
            self.client = OpenAI(base_url=url, api_key=resolved_key)
            self.async_client = AsyncOpenAI(base_url=url, api_key=resolved_key)
            self._mistral = None

    # ── Convenience: config-driven construction ──────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        preset_name: Optional[str] = None,
        *,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "LLMClient":
        """Create an LLMClient from the merged pipeline config.

        CLI arguments (``model_id``, ``provider``, …) take priority over
        preset values, which in turn take priority over the defaults
        section in ``config.yaml``.

        Parameters
        ----------
        cfg : dict
            Merged config as returned by ``load_config()``.
        preset_name : str | None
            Named preset from ``config.yaml → models.presets``.
        model_id, provider, temperature, max_tokens
            Explicit overrides (typically from CLI flags).
        """
        # Start with defaults from config.yaml → models.default_preset
        models_section = cfg.get("models") or {}
        default_preset_name = models_section.get("default_preset")

        # Resolve base defaults from the default_preset if it exists
        hardcoded: Dict[str, Any] = {
            "model_id": "unsloth/Qwen3.5-35B-A3B-UD-Q4_K_XL",
            "provider": PROVIDER_LOCAL,
            "temperature": 1.0,
            "max_tokens": 16384,
        }

        if default_preset_name:
            presets = models_section.get("presets") or {}
            dp = presets.get(default_preset_name) or {}
            resolved: Dict[str, Any] = {
                "model_id": dp.get("model_id", hardcoded["model_id"]),
                "provider": dp.get("provider", hardcoded["provider"]),
                "temperature": dp.get("temperature", hardcoded["temperature"]),
                "max_tokens": dp.get("max_tokens", hardcoded["max_tokens"]),
            }
        else:
            resolved = dict(hardcoded)

        # Layer on preset if requested
        if preset_name:
            from soc_2602.utils.config import get_model_preset

            preset = get_model_preset(cfg, preset_name)
            for key in ("model_id", "provider", "temperature", "max_tokens"):
                if key in preset and preset[key] is not None:
                    resolved[key] = preset[key]

        # Layer on explicit overrides (CLI flags)
        if model_id is not None:
            resolved["model_id"] = model_id
        if provider is not None:
            resolved["provider"] = provider
        if temperature is not None:
            resolved["temperature"] = temperature
        if max_tokens is not None:
            resolved["max_tokens"] = max_tokens

        # Resolve base_url from provider config
        provider_name = resolved["provider"]
        providers_cfg = cfg.get("providers") or {}
        pcfg = providers_cfg.get(provider_name) or {}
        base_url = pcfg.get("base_url")

        return cls(
            model_id=resolved["model_id"],
            provider=resolved["provider"],
            temperature=resolved["temperature"],
            max_tokens=resolved["max_tokens"],
            base_url=base_url,
        )

    # ── Message construction helpers ─────────────────────────────────────────

    @staticmethod
    def build_messages(
        system_text: str,
        user_text: str,
        *,
        cache_prefix: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build a message list optimised for prompt caching.

        When *cache_prefix* is supplied it is prepended to the system
        message (separated by two newlines), producing a longer
        byte-identical prefix across calls that share the same stage
        instructions.

        Parameters
        ----------
        system_text : str
            Static instructions for this generation stage.
        user_text : str
            Variable content (few-shot examples, persona data, etc.).
        cache_prefix : str | None
            Optional extra text to prepend to the system message for
            even longer shared prefixes.

        Returns
        -------
        list[dict]
            Ready-to-send message list.
        """
        system = f"{cache_prefix}\n\n{system_text}" if cache_prefix else system_text
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

    # ── NIM-specific params ──────────────────────────────────────────────────

    @staticmethod
    def _nim_extra(kwargs: dict) -> dict:
        """Pull NIM-only keys out of *kwargs* and return them as ``extra_body``."""
        nim_keys = ("top_k", "repetition_penalty", "chat_template_kwargs")
        extra_body = {k: kwargs.pop(k) for k in nim_keys if k in kwargs}
        return extra_body

    # ── Public generation API ────────────────────────────────────────────────

    def generate(
        self,
        prompt: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        """Synchronous text generation.

        Parameters
        ----------
        prompt : list[dict]
            Message list (``[{"role": …, "content": …}, …]``).
        stop_sequences : list[str] | None
            Optional stop strings.
        temperature : float | None
            Override instance default for this call.
        max_tokens : int | None
            Override instance default for this call.
        """
        temp = temperature if temperature is not None else self.temperature
        mtok = max_tokens if max_tokens is not None else self.max_tokens
        stop = stop_sequences or None

        if self.provider == PROVIDER_MISTRAL:
            return self._mistral.chat.complete(
                model=self.model_id,
                messages=prompt,
                stop=stop,
                max_tokens=mtok,
                temperature=temp,
            )

        extra_body = self._nim_extra(kwargs) if self.provider == PROVIDER_NIM else {}

        return self.client.chat.completions.create(
            model=self.model_id,
            messages=prompt,
            stop=stop,
            max_tokens=mtok,
            temperature=temp,
            extra_body=extra_body or None,
            **kwargs,
        )

    async def generate_async(
        self,
        prompt: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        """Asynchronous text generation.

        Same interface as :meth:`generate` but returns an awaitable.
        """
        temp = temperature if temperature is not None else self.temperature
        mtok = max_tokens if max_tokens is not None else self.max_tokens
        stop = stop_sequences or None

        if self.provider == PROVIDER_MISTRAL:
            return await self._mistral.chat.complete_async(
                model=self.model_id,
                messages=prompt,
                stop=stop,
                max_tokens=mtok,
                temperature=temp,
            )

        extra_body = self._nim_extra(kwargs) if self.provider == PROVIDER_NIM else {}

        return await self.async_client.chat.completions.create(
            model=self.model_id,
            messages=prompt,
            stop=stop,
            max_tokens=mtok,
            temperature=temp,
            extra_body=extra_body or None,
            **kwargs,
        )
