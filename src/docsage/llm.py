"""Cliente LLM com providers. Local-first: ollama e vllm falam OpenAI-compatible."""

import asyncio
import json
import re
from dataclasses import dataclass
from enum import StrEnum

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from docsage.config import get_settings

JSON_BLOCK = re.compile(r"\{.*\}", re.S)
JSON_SUPPORT = {"openai": True, "vllm": True, "ollama": False}


class Provider(StrEnum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    VLLM = "vllm"


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMError(RuntimeError):
    pass


def _content(resp) -> str:
    """O proxy pode devolver 200 com choices nulo; isso é erro, não resposta vazia."""
    if not getattr(resp, "choices", None):
        raise LLMError("resposta sem choices")
    return resp.choices[0].message.content or ""


def _extract_json(text: str) -> dict:
    if not text:
        raise LLMError("resposta vazia do modelo")
    match = JSON_BLOCK.search(text)
    if not match:
        raise LLMError(f"sem JSON na resposta: {text[:200]}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(f"JSON inválido: {exc}: {text[:200]}") from exc
    if not isinstance(data, dict):
        raise LLMError(f"JSON não é objeto: {text[:200]}")
    return data


class LLMClient:
    def __init__(self) -> None:
        s = get_settings()
        self.provider = Provider(s.llm_provider)
        self.model = s.chat_model
        self.judge_model = s.judge_model
        self._sem = asyncio.Semaphore(s.llm_concurrency)
        self._client = AsyncOpenAI(
            base_url=s.openai_base_url, api_key=s.openai_api_key or "local",
            timeout=s.request_timeout_s, max_retries=0,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(min=0.5, max=6),
        reraise=True,
    )
    async def complete(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.0,
        max_tokens: int | None = None, model: str | None = None,
    ) -> Completion:
        s = get_settings()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or s.max_answer_tokens,
        }
        if s.reasoning_effort:
            kwargs["reasoning_effort"] = s.reasoning_effort
        async with self._sem:
            resp = await self._client.chat.completions.create(**kwargs)
        usage = resp.usage
        return Completion(
            _content(resp),
            getattr(usage, "prompt_tokens", 0) if usage else 0,
            getattr(usage, "completion_tokens", 0) if usage else 0,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(min=0.5, max=6),
        reraise=True,
    )
    async def complete_json(self, prompt: str, *, model: str | None = None) -> dict:
        """JSON com fallback de prompt quando o provider não suporta response_format."""
        s = get_settings()
        use_format = JSON_SUPPORT[self.provider]
        instruction = prompt if use_format else prompt + "\n\nResponda APENAS com JSON válido."

        kwargs: dict = {
            "model": model or self.judge_model,
            "messages": [{"role": "user", "content": instruction}],
            "temperature": 0.0,
            "max_tokens": s.max_answer_tokens,
        }
        if use_format:
            kwargs["response_format"] = {"type": "json_object"}
        if s.reasoning_effort:
            kwargs["reasoning_effort"] = s.reasoning_effort

        async with self._sem:
            resp = await self._client.chat.completions.create(**kwargs)
        return _extract_json(_content(resp))


_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
