import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import Counter, Histogram, make_asgi_app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from docsage.generate.answer import answer
from docsage.index import store
from docsage.logging import configure, get_logger
from docsage.models import QueryRequest, QueryResponse
from docsage.retrieve.pipeline import retrieve

configure()
log = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
REQUESTS = Counter("docsage_requests_total", "Requisições", ["status"])
LATENCY = Histogram("docsage_latency_seconds", "Latência", ["stage"])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    store.ensure_collection()
    log.info("api.startup", points=store.count())
    yield
    log.info("api.shutdown")


app = FastAPI(title="docsage", version="0.3.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def request_context(request: Request, call_next) -> Response:
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:16])
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    REQUESTS.labels(status=str(response.status_code)).inc()
    LATENCY.labels(stage="total").observe(time.perf_counter() - started)
    return response


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, str | int]:
    try:
        n = store.count()
    except Exception as exc:
        raise HTTPException(503, f"qdrant indisponível: {exc}") from exc
    if n == 0:
        raise HTTPException(503, "índice vazio — rode `make index`")
    return {"status": "ready", "points": n}


@app.post("/v1/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    retrieval = await retrieve(body.question, mode=body.mode, top_k=body.top_k)
    LATENCY.labels(stage="retrieval").observe(time.perf_counter() - started)
    return await answer(body.question, retrieval)
