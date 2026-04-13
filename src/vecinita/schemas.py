from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class QueryRequest(BaseModel):
    """Request model for embedding a single natural language query."""

    query: str | None = Field(
        default=None,
        description="Primary text field (gateway and legacy clients).",
    )
    text: str | None = Field(
        default=None,
        description="Alias used by some callers (matches backend embedding service).",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Optional embedding model name. Defaults to the server's configured model."
        ),
    )

    @model_validator(mode="after")
    def require_query_or_text(self) -> "QueryRequest":
        if self.query is None and self.text is None:
            raise ValueError("Provide query or text.")
        return self


class BatchQueryRequest(BaseModel):
    """Request model for embedding a batch of natural language queries."""

    queries: list[str] | None = Field(
        default=None,
        description="List of queries (preferred field name).",
    )
    texts: list[str] | None = Field(
        default=None,
        description=(
            "Alias used by gateway fallback (matches backend ``/embed-batch``)."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "Optional embedding model name. Defaults to the server's configured model."
        ),
    )

    @model_validator(mode="after")
    def require_queries_or_texts(self) -> "BatchQueryRequest":
        items = self.queries if self.queries is not None else self.texts
        if not items or len(items) < 1:
            raise ValueError("Provide non-empty queries or texts.")
        normalized: list[str] = list(items)
        for q in normalized:
            if not q.strip():
                raise ValueError(
                    "Each query must be non-empty and not whitespace-only."
                )
        return self.model_copy(update={"queries": normalized})


class EmbeddingResponse(BaseModel):
    """Response model for a single query embedding."""

    embedding: list[float] = Field(..., description="The embedding vector.")
    model: str = Field(..., description="The model used to generate the embedding.")
    dimensions: int = Field(
        ..., description="The number of dimensions in the embedding."
    )


class BatchEmbeddingResponse(BaseModel):
    """Response model for a batch of query embeddings."""

    embeddings: list[list[float]] = Field(
        ..., description="The list of embedding vectors."
    )
    model: str = Field(..., description="The model used to generate the embeddings.")
    dimensions: int = Field(
        ..., description="The number of dimensions in each embedding."
    )
