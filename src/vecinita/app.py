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


@app.cls(
    image=image,
    volumes={MODEL_DIR: model_volume},
)
class EmbeddingServiceContainer:
    """Serverless embedding service backed by FastEmbed and a Modal Volume."""

    @modal.enter()
    def load_model(self) -> None:
        self.model = load_runtime_model()  # pragma: no cover

    @modal.asgi_app()
    def api(self):
        return build_web_app(self.model)  # pragma: no cover
