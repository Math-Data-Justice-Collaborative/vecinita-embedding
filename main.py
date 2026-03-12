"""
Vecinita Embedding API

Serverless text embedding API deployed on Modal, using FastEmbed for fast,
open-source embeddings. Model weights are stored in a Modal Volume so they
are downloaded only once and reused across container restarts.
"""

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = "/models"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# Modal primitives
# ---------------------------------------------------------------------------

app = modal.App("vecinita-embedding")

# Persistent volume for model weight storage – created on first deploy.
model_volume = modal.Volume.from_name("embedding-models", create_if_missing=True)

# Container image with all required runtime dependencies.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "fastembed>=0.7.4",
            "fastapi[standard]>=0.135.1",
        ]
    )
)


# ---------------------------------------------------------------------------
# Embedding service class
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    volumes={MODEL_DIR: model_volume},
)
class EmbeddingService:
    """Serverless embedding service backed by FastEmbed and a Modal Volume."""

    @modal.build()
    def download_model(self) -> None:
        """Download model weights into the Modal Volume during image build.

        This runs once when the image is (re)built, writing the quantised
        ONNX model files to *MODEL_DIR* which is backed by the persistent
        volume. Subsequent container starts load straight from the volume,
        avoiding redundant network downloads.
        """
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=DEFAULT_MODEL, cache_dir=MODEL_DIR)
        # Trigger the download by running a warm-up pass.
        list(model.embed(["warmup"]))

    @modal.enter()
    def load_model(self) -> None:
        """Load the embedding model from the volume when a container starts."""
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=DEFAULT_MODEL, cache_dir=MODEL_DIR)

    @modal.asgi_app()
    def api(self):
        """Create and return the FastAPI ASGI application."""
        from fastapi import FastAPI, HTTPException

        from models import (
            BatchEmbeddingResponse,
            BatchQueryRequest,
            EmbeddingResponse,
            QueryRequest,
        )

        web_app = FastAPI(
            title="Vecinita Embedding API",
            description=(
                "Fast, open-source text embedding API. "
                "Authentication is handled by the Modal auth proxy."
            ),
            version="0.1.0",
        )

        @web_app.get("/", tags=["health"])
        async def root():
            """Return service status and the active model name."""
            return {"status": "ok", "model": DEFAULT_MODEL}

        @web_app.get("/health", tags=["health"])
        async def health():
            """Health check endpoint."""
            return {"status": "ok"}

        @web_app.post("/embed", response_model=EmbeddingResponse, tags=["embedding"])
        async def embed(request: QueryRequest):
            """Embed a single natural language query.

            Returns a vector of floating-point numbers representing the
            semantic content of the query.
            """
            if not request.query.strip():
                raise HTTPException(status_code=422, detail="Query must not be empty.")

            try:
                embeddings = list(self.model.embed([request.query]))
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Embedding failed: {exc}"
                ) from exc

            vec = embeddings[0].tolist()
            return EmbeddingResponse(
                embedding=vec,
                model=request.model or DEFAULT_MODEL,
                dimensions=len(vec),
            )

        @web_app.post(
            "/embed/batch", response_model=BatchEmbeddingResponse, tags=["embedding"]
        )
        async def embed_batch(request: BatchQueryRequest):
            """Embed a batch of natural language queries in a single request.

            Returns a list of embedding vectors, one per query, in the same
            order as the input.
            """
            empty_indices = [i for i, q in enumerate(request.queries) if not q.strip()]
            if empty_indices:
                raise HTTPException(
                    status_code=422,
                    detail=f"Queries at indices {empty_indices} are empty.",
                )

            try:
                embeddings = list(self.model.embed(request.queries))
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Embedding failed: {exc}"
                ) from exc

            vecs = [e.tolist() for e in embeddings]
            return BatchEmbeddingResponse(
                embeddings=vecs,
                model=request.model or DEFAULT_MODEL,
                dimensions=len(vecs[0]),
            )

        return web_app
