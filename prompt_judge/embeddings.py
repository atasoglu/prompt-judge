import asyncio
import math

from openai import AsyncOpenAI

from . import log


async def _embed(
    client: AsyncOpenAI, text: str, model: str, label: str = ""
) -> tuple[list[float], object]:
    log.req_embed(model, log.nws(text), label or "text")
    resp = await client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding, resp.usage


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
    (a, ua), (b, ub) = await asyncio.gather(
        _embed(client, original, model, "original"),
        _embed(client, compressed, model, "compressed"),
    )
    sim = _cosine(a, b)
    log.res_embed(sim, ua, ub)
    return sim
