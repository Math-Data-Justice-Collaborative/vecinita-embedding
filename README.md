# vecinita-embedding

A serverless text embedding API built with [FastAPI](https://fastapi.tiangolo.com/), [FastEmbed](https://github.com/qdrant/fastembed), and [Modal](https://modal.com/).

- **Embedding model** – `BAAI/bge-small-en-v1.5` (384-dim, fast ONNX)
- **Deployment** – Modal serverless (auto-scales to zero)
- **Model storage** – Modal Volume (`embedding-models`), downloaded once
- **Authentication** – handled externally by the Modal auth proxy

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service status and active model |
| `GET` | `/health` | Health check |
| `POST` | `/embed` | Embed a single query |
| `POST` | `/embed/batch` | Embed a list of queries |

### `POST /embed`

**Request body**

```json
{
  "query": "What is the capital of France?",
  "model": null
}
```

`model` is optional. Omit it (or set to `null`) to use the default model.

**Response**

```json
{
  "embedding": [0.012, -0.034, ...],
  "model": "BAAI/bge-small-en-v1.5",
  "dimensions": 384
}
```

### `POST /embed/batch`

**Request body**

```json
{
  "queries": ["First query", "Second query"],
  "model": null
}
```

**Response**

```json
{
  "embeddings": [[...], [...]],
  "model": "BAAI/bge-small-en-v1.5",
  "dimensions": 384
}
```

---

## Local development

```bash
pip install -e .
modal serve main.py      # hot-reload dev server
```

## Deploy

```bash
modal deploy main.py
```

The first deploy builds the image and downloads the model weights into the
`embedding-models` Modal Volume. Subsequent deploys reuse the cached weights.