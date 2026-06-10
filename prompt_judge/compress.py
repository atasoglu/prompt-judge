import json
import sys
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from .embeddings import embedding_similarity
from .judge import judge_summaries
from .summarizers import STRATEGIES, run_summarizers


@dataclass
class Config:
    model: str = "gpt-4o-mini"
    judge_model: str | None = None
    embed_model: str | None = None
    strategies: list[str] = field(default_factory=lambda: list(STRATEGIES.keys()))
    iterations: int = 10
    threshold: float = 0.85
    extra_instructions: str = ""
    base_url: str | None = None
    api_key: str = "sk-placeholder"


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
    log: list[dict] = []

    for i in range(config.iterations):
        print(
            f"\n[iter {i + 1}/{config.iterations}] input={len(current)} chars",
            file=sys.stderr,
        )

        # Parallel summarization
        summaries = await run_summarizers(
            client, current, config.strategies, config.model, config.extra_instructions
        )
        if not summaries:
            print("[error] no summaries produced, aborting", file=sys.stderr)
            break

        # Judge always scores against the original
        judge = await judge_summaries(client, original, summaries, judge_model)
        winner_label = judge["winner"]
        winner_text = judge["winner_text"]
        winner_strategy = judge["winner_strategy"]
        judge_score = judge["scores"].get(winner_label, {}).get("total", 0.0)

        print(
            f"  winner={winner_label}({winner_strategy}) "
            f"judge={judge_score:.3f} output={len(winner_text)} chars",
            file=sys.stderr,
        )

        # Optional: embedding similarity as second fidelity signal
        embed_sim: float | None = None
        if config.embed_model:
            try:
                embed_sim = await embedding_similarity(
                    client, original, winner_text, config.embed_model
                )
                print(f"  embed_sim={embed_sim:.3f}", file=sys.stderr)
                if abs(judge_score - embed_sim) > 0.15:
                    print(
                        f"  [warn] judge ({judge_score:.3f}) and embedding ({embed_sim:.3f}) disagree",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(f"  [warn] embedding failed: {e}", file=sys.stderr)

        # Primary similarity signal for threshold check
        primary_sim = embed_sim if embed_sim is not None else judge_score

        iter_data = {
            "iteration": i + 1,
            "input_length": len(current),
            "output_length": len(winner_text),
            "compression_ratio": len(winner_text) / len(original),
            "winner_label": winner_label,
            "winner_strategy": winner_strategy,
            "winner_text": winner_text,
            "judge_score": judge_score,
            "embed_similarity": embed_sim,
            "all_summaries": summaries,
            "judge_detail": {
                k: v
                for k, v in judge.items()
                if k not in ("label_to_text",)  # skip large fields from detail
            },
        }
        log.append(iter_data)

        if log_file:
            log_file.write(json.dumps(iter_data) + "\n")
            log_file.flush()

        # Stop if similarity vs original dropped below threshold
        if primary_sim < config.threshold:
            print(
                f"  similarity {primary_sim:.3f} < threshold {config.threshold}, stopping",
                file=sys.stderr,
            )
            break

        # Track global best (score, length) — shorter wins ties
        if (
            best is None
            or judge_score > best[0]
            or (judge_score == best[0] and len(winner_text) < len(best[1]))
        ):
            best = (judge_score, winner_text, i + 1)
            no_improvement = 0
        else:
            no_improvement += 1

        # Plateau detection: stop if no improvement for 2 consecutive iterations
        if no_improvement >= 2:
            print("  no improvement for 2 iterations, stopping early", file=sys.stderr)
            break

        current = winner_text

    if best is None:
        return {
            "original": original,
            "compressed": original,
            "score": 1.0,
            "compression_ratio": 1.0,
            "found_at_iteration": 0,
            "iterations": log,
        }

    score, text, found_at = best
    return {
        "original": original,
        "compressed": text,
        "score": score,
        "compression_ratio": len(text) / len(original),
        "found_at_iteration": found_at,
        "iterations": log,
    }
