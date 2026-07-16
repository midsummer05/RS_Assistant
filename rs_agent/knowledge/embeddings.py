from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import List, Protocol
from urllib.request import Request, urlopen


class Embedder(Protocol):
    name: str
    dimensions: int

    def embed(self, texts: List[str]) -> List[List[float]]:
        ...


@dataclass
class HashingEmbedder:
    """Deterministic offline fallback; replaceable by a remote embedding model."""

    dimensions: int = 384
    name: str = "hashing-charword-v1"

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        lowered = text.lower()
        features = re.findall(r"[a-z0-9_\-]+|[\u4e00-\u9fff]", lowered)
        features += [lowered[index : index + 3] for index in range(max(0, len(lowered) - 2))]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % self.dimensions
            sign = 1.0 if digest[0] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@dataclass
class OpenAICompatibleEmbedder:
    api_key: str
    model: str
    base_url: str
    dimensions: int = 1536
    name: str = "openai-compatible"

    def embed(self, texts: List[str]) -> List[List[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = Request(
            f"{self.base_url.rstrip('/')}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        ordered = sorted(body["data"], key=lambda item: item["index"])
        embeddings = [item["embedding"] for item in ordered]
        if embeddings:
            self.dimensions = len(embeddings[0])
        return embeddings


def build_embedder_from_env() -> Embedder:
    api_key = os.getenv("RS_AGENT_EMBEDDING_API_KEY")
    model = os.getenv("RS_AGENT_EMBEDDING_MODEL")
    base_url = os.getenv("RS_AGENT_EMBEDDING_BASE_URL")
    if api_key and model and base_url:
        return OpenAICompatibleEmbedder(
            api_key=api_key,
            model=model,
            base_url=base_url,
            name=f"openai-compatible:{model}",
        )
    return HashingEmbedder()


def cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))
