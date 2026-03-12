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
            "Fast, open-source text embedding API. "
            "Authentication is handled by the Modal auth proxy."
        ),
        version="0.1.0",
    )

    @web_app.get("/", tags=["health"])
    async def root() -> dict[str, str]:
        return {"status": "ok", "model": DEFAULT_MODEL}

    @web_app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @web_app.post("/embed", response_model=EmbeddingResponse, tags=["embedding"])
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
    )
    async def embed_batch(request: BatchQueryRequest) -> BatchEmbeddingResponse:
        try:
            return service.embed_batch(request.queries, request.model)
        except EmptyQueryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except EmbeddingExecutionError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return web_app
