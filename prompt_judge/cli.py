import argparse
import asyncio
import io
import os
import sys

from .compress import Config, compress
from .summarizers import STRATEGIES


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prompt-judge",
        description=(
            "Iteratively compress a prompt while preserving semantic fidelity.\n"
            "Reads from stdin if no PROMPT argument is given."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("prompt", nargs="?", help="Prompt text to compress.")

    g = p.add_argument_group("model settings")
    g.add_argument(
        "--model",
        default="gpt-4o-mini",
        metavar="MODEL",
        help="Summarizer model (default: gpt-4o-mini)",
    )
    g.add_argument(
        "--judge-model",
        default=None,
        metavar="MODEL",
        help="Judge model (defaults to --model)",
    )
    g.add_argument(
        "--embed-model",
        default=None,
        metavar="MODEL",
        help="Embedding model for cosine similarity (e.g. text-embedding-3-small); "
        "omit to skip embedding check",
    )
    g.add_argument(
        "--base-url",
        default=None,
        metavar="URL",
        help="OpenAI-compatible base URL (Ollama, vLLM, OpenRouter, …)",
    )
    g.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="API key (defaults to $OPENAI_API_KEY)",
    )

    g = p.add_argument_group("compression settings")
    g.add_argument(
        "--iterations",
        type=int,
        default=10,
        metavar="N",
        help="Max compression iterations (default: 10)",
    )
    g.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        metavar="F",
        help="Min similarity to original to continue (default: 0.85)",
    )
    g.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGIES.keys()),
        default=list(STRATEGIES.keys()),
        metavar="STRATEGY",
        help=(
            "Summarizer strategies to use (default: all). "
            f"Choices: {', '.join(STRATEGIES.keys())}"
        ),
    )
    g.add_argument(
        "--instructions",
        default="",
        metavar="TEXT",
        help="Extra instructions appended to each summarizer prompt",
    )

    g = p.add_argument_group("output")
    g.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="FILE",
        help="Write full JSONL log to FILE",
    )
    g.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output to stderr"
    )

    return p


async def _run(args) -> None:
    # ── Read prompt ──────────────────────────────────────────────────────────
    if args.prompt:
        original = args.prompt.strip()
    elif not sys.stdin.isatty():
        original = sys.stdin.read().strip()
    else:
        print("Error: provide PROMPT as argument or pipe via stdin.", file=sys.stderr)
        sys.exit(1)

    if not original:
        print("Error: prompt is empty.", file=sys.stderr)
        sys.exit(1)

    # ── Config ────────────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder")

    config = Config(
        model=args.model,
        judge_model=args.judge_model,
        embed_model=args.embed_model,
        strategies=args.strategies,
        iterations=args.iterations,
        threshold=args.threshold,
        extra_instructions=args.instructions,
        base_url=args.base_url,
        api_key=api_key,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    real_stderr = sys.stderr
    log_file = open(args.output, "w") if args.output else None

    if args.quiet:
        sys.stderr = io.StringIO()

    try:
        result = await compress(original, config, log_file)
    finally:
        sys.stderr = real_stderr
        if log_file:
            log_file.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(
        f"\n{'=' * 60}\n"
        f"Original:   {len(result['original'])} chars\n"
        f"Compressed: {len(result['compressed'])} chars\n"
        f"Ratio:      {result['compression_ratio']:.1%}\n"
        f"Score:      {result['score']:.3f}\n"
        f"Found:      iteration {result['found_at_iteration']}\n"
        f"{'=' * 60}",
        file=real_stderr,
    )

    # Compressed prompt → stdout
    print(result["compressed"])


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))
