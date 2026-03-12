from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for embedding a single natural language query."""

    query: str = Field(..., description="The natural language query to embed.")
    model: str | None = Field(
        default=None,
        description=(
            "Optional embedding model name. Defaults to the server's configured model."
        ),
    )


class BatchQueryRequest(BaseModel):
    """Request model for embedding a batch of natural language queries."""

    queries: list[str] = Field(
        ..., min_length=1, description="A list of natural language queries to embed."
    )
    model: str | None = Field(
        default=None,
        description=(
            "Optional embedding model name. Defaults to the server's configured model."
        ),
    )


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
