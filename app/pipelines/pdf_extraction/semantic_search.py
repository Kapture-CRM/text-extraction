import re
import time

import numpy as np
from openai import OpenAI

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("pdf-extraction")

_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    start = time.perf_counter()
    client = get_openai_client()
    response = client.embeddings.create(model=settings.OPENAI_EMBEDDING_MODEL, input=texts)
    embeddings = [item.embedding for item in response.data]
    logger.info(
        f"Embedded {len(texts)} chunk(s) via {settings.OPENAI_EMBEDDING_MODEL} "
        f"in {time.perf_counter() - start:.2f}s"
    )
    return embeddings


def build_embedding_index(store: list[dict]) -> np.ndarray:
    texts = [f"{sec['heading']}\n\n{sec['content']}" for sec in store]
    embeddings = embed_texts(texts)
    matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return matrix / norms


def semantic_search(
    query: str,
    store: list[dict],
    normalized_embeddings: np.ndarray,
    top_k: int = 2,
    min_score: float = 0.0,
) -> list[dict]:
    if not query.strip():
        logger.info("Empty query provided to semantic_search")
        return []

    query_embedding = np.array(embed_texts([query])[0], dtype=np.float32)
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []
    query_embedding = query_embedding / query_norm

    scores = normalized_embeddings @ query_embedding

    ranked_indices = np.argsort(-scores)
    results = []
    seen_content = set()
    skipped_duplicates = 0
    for idx in ranked_indices:
        score = float(scores[idx])
        if score < min_score:
            continue
        fingerprint = re.sub(r'\s+', ' ', store[idx]['content']).strip().lower()
        if fingerprint in seen_content:
            skipped_duplicates += 1
            continue
        seen_content.add(fingerprint)
        results.append({**store[idx], "_score": score})
        if len(results) >= top_k:
            break

    logger.info(
        f"Semantic query {query!r}: {len(results)} section(s) returned, "
        f"skipped {skipped_duplicates} duplicate(s) (top_k={top_k}, min_score={min_score})"
    )
    return results
