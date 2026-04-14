"""Modal entrypoint for the Vecinita embedding API."""

from typing import Any

import modal

from .api import create_app
from .constants import APP_NAME, DEFAULT_MODEL, MODEL_DIR, VOLUME_NAME
from .service import EmbeddingService

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    [
        "fastembed>=0.7.4",
        "fastapi[standard]>=0.135.1",
    ]
)


def create_text_embedding() -> Any:
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=DEFAULT_MODEL, cache_dir=MODEL_DIR)


def warmup_embedding_model(model: Any) -> Any:
    list(model.embed(["warmup"]))
    return model


def load_runtime_model() -> Any:
    return warmup_embedding_model(create_text_embedding())


def build_web_app(model: Any):
    return create_app(EmbeddingService(model, default_model=DEFAULT_MODEL))


def _embed_query_impl(query: str) -> dict[str, Any]:
    model = load_runtime_model()
    vector = list(model.embed([query]))[0]
    vector_data = vector.tolist()
    return {
        "embedding": vector_data,
        "model": DEFAULT_MODEL,
        "dimension": len(vector_data),
    }


def _embed_batch_impl(queries: list[str]) -> dict[str, Any]:
    model = load_runtime_model()
    vectors = list(model.embed(queries))
    serialized = [vector.tolist() for vector in vectors]
    return {
        "embeddings": serialized,
        "model": DEFAULT_MODEL,
        "dimension": len(serialized[0]) if serialized else 0,
    }


@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=600,
)
def embed_query(query: str) -> dict[str, Any]:
    """Function-style embedding endpoint for non-HTTP invocation."""
    return _embed_query_impl(query)


@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=600,
)
def embed_batch(queries: list[str]) -> dict[str, Any]:
    """Function-style batch embedding endpoint for non-HTTP invocation."""
    return _embed_batch_impl(queries)


@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=600,
)
@modal.asgi_app()
def web_app():
    """Modal HTTPS hostname matches backend deploy: ``…-embedding-web-app.modal.run``.

    A class-based ASGI method (e.g. ``…Container.api``) registers a different
    ``*.modal.run`` host than the gateway and shared env defaults expect.
    """
    model = load_runtime_model()  # pragma: no cover
    return build_web_app(model)  # pragma: no cover
