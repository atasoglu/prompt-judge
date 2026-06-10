import json
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from . import log
from .embeddings import embedding_similarity
from .judge import judge_summaries
from .summarizers import STRATEGIES, run_summarizers


@dataclass
class Config:
    model: str = "gpt-5.5"
    judge_model: str | None = None
    embed_model: str | None = None
    strategies: list[str] = field(default_factory=lambda: list(STRATEGIES.keys()))
    iterations: int = 10
    threshold: float = 0.85
    extra_instructions: str = ""
    base_url: str | None = None
    api_key: str = "sk-placeholder"
    reasoning_effort: str | None = "medium"


async def compress(original: str, config: Config, log_file=None) -> dict:
    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )
    judge_model = config.judge_model or config.model

    current = original
    # best: (judge_score, text, iteration_number)
    best: tuple[float, str, int] | None = None
    no_improvement = 0
    log_entries: list[dict] = []

    for i in range(config.iterations):
        log.iter_start(i + 1, config.iterations, log.nws(current))

        # Parallel summarization
        summaries = await run_summarizers(
            client,
            current,
            config.strategies,
            config.model,
            config.extra_instructions,
            config.reasoning_effort,
        )
        if not summaries:
            log.error("no summaries produced, aborting")
            break

        # Judge always scores against the original
        judge = await judge_summaries(
            client,
            original,
            summaries,
            judge_model,
            reasoning_effort=config.reasoning_effort,
        )
        winner_label = judge["winner"]
        winner_text = judge["winner_text"]
        winner_strategy = judge["winner_strategy"]
        judge_score = judge["scores"].get(winner_label, {}).get("total", 0.0)

        # Optional: embedding similarity as second fidelity signal
        embed_sim: float | None = None
        if config.embed_model:
            try:
                embed_sim = await embedding_similarity(
                    client, original, winner_text, config.embed_model
                )
                if abs(judge_score - embed_sim) > 0.15:
                    log.warn(
                        f"judge ({judge_score:.3f}) and embedding ({embed_sim:.3f}) disagree"
                    )
            except Exception as e:
                log.warn(f"embedding failed: {e}")

        log.iter_result(
            winner_label,
            winner_strategy,
            judge_score,
            log.nws(winner_text),
            log.nws(original),
            embed_sim,
        )

        # Primary similarity signal for threshold check
        primary_sim = embed_sim if embed_sim is not None else judge_score

        iter_data = {
            "iteration": i + 1,
            "input_length": log.nws(current),
            "output_length": log.nws(winner_text),
            "compression_ratio": log.nws(winner_text) / log.nws(original),
            "winner_label": winner_label,
            "winner_strategy": winner_strategy,
            "winner_text": winner_text,
            "judge_score": judge_score,
            "embed_similarity": embed_sim,
            "all_summaries": summaries,
            "judge_detail": {
                k: v for k, v in judge.items() if k not in ("label_to_text",)
            },
        }
        log_entries.append(iter_data)

        if log_file:
            log_file.write(json.dumps(iter_data) + "\n")
            log_file.flush()

        # Stop if similarity vs original dropped below threshold
        if primary_sim < config.threshold:
            log.threshold_stop(primary_sim, config.threshold)
            break

        # Track global best (score, length) — shorter wins ties
        if (
            best is None
            or judge_score > best[0]
            or (judge_score == best[0] and log.nws(winner_text) < log.nws(best[1]))
        ):
            best = (judge_score, winner_text, i + 1)
            no_improvement = 0
        else:
            no_improvement += 1

        # Plateau detection: stop if no improvement for 2 consecutive iterations
        if no_improvement >= 2:
            log.plateau_stop()
            break

        current = winner_text

    orig_nws = log.nws(original)

    if best is None:
        log.final_summary(orig_nws, orig_nws, 1.0, 0)
        return {
            "original": original,
            "compressed": original,
            "score": 1.0,
            "compression_ratio": 1.0,
            "found_at_iteration": 0,
            "iterations": log_entries,
        }

    score, text, found_at = best
    text_nws = log.nws(text)
    log.final_summary(orig_nws, text_nws, score, found_at)
    return {
        "original": original,
        "compressed": text,
        "score": score,
        "compression_ratio": text_nws / orig_nws if orig_nws else 1.0,
        "found_at_iteration": found_at,
        "iterations": log_entries,
    }
