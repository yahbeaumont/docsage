"""Tracing opcional. No-op quando Langfuse não está configurado."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from docsage.logging import get_logger

log = get_logger(__name__)


class NullTracer:
    trace_id: str | None = None

    @contextmanager
    def span(self, name: str, **attrs: Any):
        yield _NoopSpan()

    def score(self, name: str, value: float, comment: str | None = None) -> None: ...

    def flush(self) -> None: ...


class _NoopSpan:
    def update(self, **kwargs: Any) -> None: ...


class LangfuseTracer:
    def __init__(self) -> None:
        from langfuse import Langfuse

        self._lf = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        )
        self.trace_id: str | None = None

    @contextmanager
    def span(self, name: str, **attrs: Any):
        with self._lf.start_as_current_span(name=name, input=attrs) as span:
            if self.trace_id is None:
                self.trace_id = self._lf.get_current_trace_id()
            yield _Span(span)

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        if not self.trace_id:
            return
        self._lf.create_score(trace_id=self.trace_id, name=name, value=value, comment=comment)

    def flush(self) -> None:
        self._lf.flush()


class _Span:
    def __init__(self, span: Any) -> None:
        self._span = span

    def update(self, **kwargs: Any) -> None:
        self._span.update(**kwargs)


_tracer: NullTracer | LangfuseTracer | None = None


def get_tracer() -> NullTracer | LangfuseTracer:
    global _tracer
    if _tracer is None:
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                _tracer = LangfuseTracer()
                log.info("tracing.enabled")
            except Exception as exc:
                log.warning("tracing.init_failed", error=str(exc))
                _tracer = NullTracer()
        else:
            _tracer = NullTracer()
    return _tracer
