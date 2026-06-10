# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run python main.py          # run the app
uv run pytest                  # run tests
uv add <package>               # add a dependency
uv sync                        # install all dependencies
```

## Architecture

`prompt-judge` is a CLI tool that iteratively compresses a prompt while preserving semantic fidelity. The project is in early design/implementation phase — `PLAN.md` is the authoritative design document.

**Core loop (per iteration):**
1. N summarizer agents compress the current prompt in parallel (`asyncio` + async OpenAI client), each using a different strategy: *extractive*, *abstractive*, *structure-preserving*, *role-preserving*.
2. A judge agent receives all summaries in random order with anonymous labels (A/B/C) and returns a rubric-based JSON score (instruction coverage, constraint coverage, format, tone).
3. The best-scoring summary becomes the input for the next iteration, scored **always against the original**.
4. Stop when: iteration count exhausted, similarity drops below threshold, or no improvement for 2 consecutive iterations.

**Fidelity signals:**
- Rubric-based judge score (primary)
- Embedding cosine similarity via `/embeddings` endpoint (secondary; warns on disagreement)

**Key constraints:**
- `openai` SDK + stdlib only — no extra dependencies
- Must work with any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter)
- CLI-first: the CLI is the primary interface, not a library wrapper
- All summaries, scores, and judge reasoning written to JSON/JSONL for traceability

**Design decisions to preserve:**
- Summaries randomized before judge sees them → counters position bias
- Global best (score, length) tracked across all iterations, not just the last accepted
- Behavioral test mode (run original vs. compressed against test inputs) is a planned v2 feature — reserve architecture space for it
