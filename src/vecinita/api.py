"""FastAPI application factory for the Vecinita embedding service."""

from fastapi import FastAPI, HTTPException

from .constants import DEFAULT_MODEL
from .schemas import (
    BatchEmbeddingResponse,
    BatchQueryRequest,
    EmbeddingResponse,
    QueryRequest,
)
from .service import EmbeddingExecutionError, EmbeddingService, EmptyQueryError


def create_app(service: EmbeddingService) -> FastAPI:
    """Create the ASGI app with a provided embedding service dependency."""

    web_app = FastAPI(
        title="Vecinita Embedding API",
        description=(
            "Generate vector embeddings for one query or batches of queries using "
            "FastEmbed and FastAPI. Requests are protected by the Modal auth proxy, "
            "so only authorized callers can access the API."
        ),
        version="0.1.0",
        openapi_tags=[
            {
                "name": "health",
                "description": (
                    "Service liveness and readiness checks for infrastructure and "
                    "monitoring systems."
                ),
            },
            {
                "name": "embedding",
                "description": (
                    "Embedding generation endpoints for single and batched natural "
                    "language queries."
                ),
            },
        ],
    )

    @web_app.get(
        "/",
        tags=["health"],
        summary="Get service status and default model",
        description=(
            "Returns a simple service heartbeat and the default embedding model "
            "configured for this deployment."
        ),
    )
    async def root() -> dict[str, str]:
        return {"status": "ok", "model": DEFAULT_MODEL}

    @web_app.get(
        "/health",
        tags=["health"],
        summary="Get liveness status",
        description="Returns liveness information used by uptime checks and probes.",
    )
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @web_app.post(
        "/embed",
        response_model=EmbeddingResponse,
        tags=["embedding"],
        summary="Embed a single query",
        description=(
            "Generates one embedding vector for a single natural language query. "
            "The optional model field can override the server default model."
        ),
        responses={
            422: {"description": "The input query is empty or only whitespace."},
            500: {
                "description": "Embedding generation failed due to a backend/runtime error."
            },
        },
    )
    async def embed(request: QueryRequest) -> EmbeddingResponse:
        try:
            return service.embed_query(request.query, request.model)
        except EmptyQueryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except EmbeddingExecutionError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @web_app.post(
        "/embed/batch",
        response_model=BatchEmbeddingResponse,
        tags=["embedding"],
        summary="Embed a batch of queries",
        description=(
            "Generates one embedding vector per query for a non-empty list of "
            "natural language queries. The optional model field can override the "
            "server default model."
        ),
        responses={
            422: {
                "description": "The query list is invalid or includes empty/whitespace entries."
            },
            500: {
                "description": "Embedding generation failed due to a backend/runtime error."
            },
        },
    )
    async def embed_batch(request: BatchQueryRequest) -> BatchEmbeddingResponse:
        try:
            return service.embed_batch(request.queries, request.model)
        except EmptyQueryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except EmbeddingExecutionError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return web_app
