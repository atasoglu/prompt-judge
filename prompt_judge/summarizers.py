import asyncio
import sys

from openai import AsyncOpenAI

STRATEGIES: dict[str, str] = {
    "extractive": (
        "You are a text compression expert using the EXTRACTIVE strategy.\n"
        "Compress the given text by selecting and combining the most important existing sentences or phrases.\n"
        "Do NOT paraphrase or rewrite — only use words and phrases from the original.\n"
        "Eliminate redundant, repetitive, or less critical content.\n"
        "Output ONLY the compressed text, with no preamble or explanation."
    ),
    "abstractive": (
        "You are a text compression expert using the ABSTRACTIVE strategy.\n"
        "Compress the given text by rewriting it in fewer words while preserving the core meaning.\n"
        "Use clear, concise language. You may restructure sentences.\n"
        "Output ONLY the compressed text, with no preamble or explanation."
    ),
    "structure-preserving": (
        "You are a text compression expert using the STRUCTURE-PRESERVING strategy.\n"
        "Compress the given text while keeping its structural elements intact "
        "(headings, bullet points, numbered lists, code blocks).\n"
        "Compress the content within each element, but preserve the overall organization.\n"
        "Output ONLY the compressed text, with no preamble or explanation."
    ),
    "role-preserving": (
        "You are a text compression expert using the ROLE-PRESERVING strategy.\n"
        "Keep ALL instructions, constraints, rules, and role definitions VERBATIM.\n"
        "You may aggressively compress or remove examples, explanations, and supplementary context.\n"
        "Never modify anything that defines what the model should or should not do.\n"
        "Output ONLY the compressed text, with no preamble or explanation."
    ),
}


async def _summarize_one(
    client: AsyncOpenAI,
    prompt: str,
    strategy: str,
    model: str,
    extra_instructions: str,
) -> dict:
    system = STRATEGIES[strategy]
    if extra_instructions:
        system = f"{system}\n\nAdditional instructions: {extra_instructions}"

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Compress this:\n\n{prompt}"},
        ],
        temperature=0.3,
    )
    text = (response.choices[0].message.content or "").strip()
    return {"strategy": strategy, "text": text}


async def run_summarizers(
    client: AsyncOpenAI,
    prompt: str,
    strategies: list[str],
    model: str,
    extra_instructions: str = "",
) -> list[dict]:
    tasks = [
        _summarize_one(client, prompt, s, model, extra_instructions) for s in strategies
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid = []
    for r in results:
        if isinstance(r, Exception):
            print(f"[warn] summarizer failed: {r}", file=sys.stderr)
        else:
            valid.append(r)
    return valid
