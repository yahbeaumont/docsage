import asyncio

import pytest

from docsage import llm
from docsage.config import Settings
from docsage.telemetry import NullTracer, get_tracer


class _Usage:
    prompt_tokens = 11
    completion_tokens = 7


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage()


class _FakeCompletions:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(self.replies.pop(0))


class _FakeChat:
    def __init__(self, replies: list[str]) -> None:
        self.completions = _FakeCompletions(replies)


class _FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.chat = _FakeChat(replies)


@pytest.fixture
def fake_client(monkeypatch) -> _FakeClient:
    client = _FakeClient(['ok', '{"claims": ["a"]}'])
    monkeypatch.setattr(llm.LLMClient, "__init__", lambda self: None)
    client_obj = llm.LLMClient()
    client_obj.provider = llm.Provider.OPENAI
    client_obj.model = "m"
    client_obj.judge_model = "j"
    client_obj._sem = asyncio.Semaphore(2)
    client_obj._client = client
    monkeypatch.setattr(llm, "_client", client_obj)
    return client


def test_extract_json_finds_block():
    assert llm._extract_json('lixo {"a": 1} resto') == {"a": 1}


def test_extract_json_raises_without_json():
    with pytest.raises(llm.LLMError):
        llm._extract_json("sem json")


def test_complete_returns_usage(fake_client):
    completion = llm.get_client()
    result = __import__("asyncio").run(completion.complete("pergunta", system="sys"))
    assert result.text == "ok"
    assert (result.prompt_tokens, result.completion_tokens) == (11, 7)
    sent = fake_client.chat.completions.calls[0]
    assert sent["messages"][0]["role"] == "system"


def test_complete_json_uses_response_format(fake_client):
    fake_client.chat.completions.replies = ['{"claims": ["a"]}']
    result = __import__("asyncio").run(llm.get_client().complete_json("dados"))
    assert result == {"claims": ["a"]}
    assert fake_client.chat.completions.calls[0]["response_format"] == {"type": "json_object"}


def test_ollama_skips_response_format(fake_client):
    fake_client.chat.completions.replies = ['{"ok": true}']
    llm.get_client().provider = llm.Provider.OLLAMA
    __import__("asyncio").run(llm.get_client().complete_json("dados"))
    assert "response_format" not in fake_client.chat.completions.calls[0]


def test_settings_defaults_are_provider_agnostic():
    monkeypatch_env = pytest.MonkeyPatch()
    monkeypatch_env.delenv("QDRANT_PATH", raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_provider in {"openai", "ollama", "vllm"}
    assert settings.qdrant_path == ""
    monkeypatch_env.undo()


def test_null_choices_is_an_error():
    class _Empty:
        choices = None

    with pytest.raises(llm.LLMError):
        llm._content(_Empty())


def test_settings_rejects_bad_embed_dim():
    with pytest.raises(ValueError):
        Settings(_env_file=None, embed_dim=0)


def test_null_tracer_is_noop():
    tracer = NullTracer()
    with tracer.span("x", a=1) as span:
        span.update(output="y")
    tracer.score("metric", 0.5)
    tracer.flush()
    assert tracer.trace_id is None


def test_get_tracer_without_keys_is_null(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr("docsage.telemetry._tracer", None)
    assert isinstance(get_tracer(), NullTracer)
