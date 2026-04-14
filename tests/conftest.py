from __future__ import annotations

import os

# ``web_app`` is omitted when ``VECINITA_MODAL_INCLUDE_WEB_ENDPOINTS=0`` at import time.
os.environ.setdefault("VECINITA_MODAL_INCLUDE_WEB_ENDPOINTS", "1")

import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeEmbedder
from vecinita.api import create_app
from vecinita.service import EmbeddingService


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def service(fake_embedder: FakeEmbedder) -> EmbeddingService:
    return EmbeddingService(fake_embedder)


@pytest.fixture
def client(service: EmbeddingService) -> TestClient:
    return TestClient(create_app(service))
