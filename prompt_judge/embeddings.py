import asyncio
import math

from openai import AsyncOpenAI


async def _embed(client: AsyncOpenAI, text: str, model: str) -> list[float]:
    resp = await client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


async def embedding_similarity(
    client: AsyncOpenAI,
    original: str,
    compressed: str,
    model: str,
) -> float:
    a, b = await asyncio.gather(
        _embed(client, original, model),
        _embed(client, compressed, model),
    )
    return _cosine(a, b)
