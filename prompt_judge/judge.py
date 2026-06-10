import json
import random
import re
import sys

from openai import AsyncOpenAI

_RUBRIC_FIELDS = [
    "instruction_coverage",
    "constraint_coverage",
    "format_preservation",
    "tone",
    "compression_quality",
]

_JUDGE_SYSTEM = """\
You are an expert evaluator of text compression quality.

You will be shown an original prompt and several compressed versions labeled A, B, C, etc.
Select the BEST version and score ALL versions on these criteria (0.0–1.0 each):

- instruction_coverage: how well it preserves instructions and directives
- constraint_coverage: how well it preserves constraints and rules
- format_preservation: how well it preserves structure and formatting
- tone: how well it preserves the original voice and style
- compression_quality: overall quality — penalize over-compression that loses key meaning

IMPORTANT: Do NOT favor longer summaries. Maximum compression under quality is the goal.
Scores are always against the ORIGINAL prompt, regardless of how much was already compressed.

Respond with ONLY valid JSON, no other text:
{
  "winner": "<label>",
  "scores": {
    "<label>": {
      "instruction_coverage": <0-1>,
      "constraint_coverage": <0-1>,
      "format_preservation": <0-1>,
      "tone": <0-1>,
      "compression_quality": <0-1>
    }
  },
  "reasoning": "<one sentence>"
}"""


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No valid JSON in judge response: {text[:300]!r}")


async def judge_summaries(
    client: AsyncOpenAI,
    original: str,
    summaries: list[dict],
    model: str,
    attempts: int = 3,
) -> dict:
    # Shuffle and assign anonymous labels — counters position bias
    order = list(range(len(summaries)))
    random.shuffle(order)
    labels = [chr(ord("A") + i) for i in range(len(order))]
    label_map: dict[str, dict] = {
        labels[i]: summaries[order[i]] for i in range(len(order))
    }

    parts = [f"ORIGINAL:\n{original}"]
    for label in labels:
        parts.append(f"\nVERSION {label}:\n{label_map[label]['text']}")
    user_content = "\n".join(parts)

    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=attempt * 0.1,  # slight bump on retry
            )
            raw = (resp.choices[0].message.content or "").strip()
            result = _extract_json(raw)

            # Compute totals
            for scores in result.get("scores", {}).values():
                scores["total"] = sum(scores.get(f, 0.0) for f in _RUBRIC_FIELDS) / len(
                    _RUBRIC_FIELDS
                )

            # Annotate with strategy info
            winner = result.get("winner", labels[0])
            result["winner_strategy"] = label_map.get(winner, {}).get(
                "strategy", "unknown"
            )
            result["winner_text"] = label_map.get(winner, {}).get("text", "")
            result["label_to_strategy"] = {
                lbl: s["strategy"] for lbl, s in label_map.items()
            }
            result["label_to_text"] = {lbl: s["text"] for lbl, s in label_map.items()}
            return result

        except Exception as e:
            last_err = e
            print(
                f"[warn] judge attempt {attempt + 1}/{attempts} failed: {e}",
                file=sys.stderr,
            )

    raise RuntimeError(f"Judge failed after {attempts} attempts: {last_err}")
