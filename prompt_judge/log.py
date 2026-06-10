"""Colored stderr logging for prompt-judge."""

import sys


def nws(s: str) -> int:
    """Character count excluding all whitespace (space, newline, tab, …)."""
    return sum(1 for c in s if not c.isspace())


# ANSI codes
_R = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_RED = "\033[31m"
_WHITE = "\033[97m"
_GRAY = "\033[90m"

# Box-drawing / arrow glyphs
_ARROW_OUT = f"{_CYAN}→{_R}"
_ARROW_IN = f"{_GREEN}←{_R}"
_WARN = f"{_YELLOW}⚠{_R}"
_ERR = f"{_RED}✗{_R}"
_OK = f"{_GREEN}✓{_R}"
_BULLET = f"{_GRAY}·{_R}"


def _w(*parts: str) -> None:
    print("".join(parts), file=sys.stderr)


def _truncate(s: str, n: int = 120) -> str:
    s = s.replace("\n", " ").strip()
    return s[:n] + "…" if len(s) > n else s


# ── Iteration header ────────────────────────────────────────────────────────


def iter_start(i: int, total: int, input_len: int) -> None:
    bar = f"{_BOLD}{_BLUE}{'─' * 56}{_R}"
    _w(f"\n{bar}")
    _w(
        f"  {_BOLD}Iteration {_YELLOW}{i}{_R}{_BOLD}/{total}{_R}"
        f"  {_GRAY}input={_R}{_WHITE}{input_len:,}{_R}{_GRAY} chars{_R}"
    )
    _w(f"{bar}")


def iter_result(
    winner_label: str,
    winner_strategy: str,
    judge_score: float,
    output_len: int,
    original_len: int,
    embed_sim: float | None,
) -> None:
    ratio = output_len / original_len if original_len else 1.0
    saved = 1.0 - ratio
    score_color = (
        _GREEN if judge_score >= 0.8 else _YELLOW if judge_score >= 0.6 else _RED
    )
    ratio_color = _GREEN if saved >= 0.4 else _YELLOW if saved >= 0.2 else _GRAY
    parts = [
        f"  {_OK} ",
        f"winner={_BOLD}{_CYAN}{winner_label}{_R}({_MAGENTA}{winner_strategy}{_R})  ",
        f"score={score_color}{judge_score:.3f}{_R}  ",
        f"output={ratio_color}{output_len:,}{_R}{_GRAY}chars ({saved:.0%} saved){_R}",
    ]
    if embed_sim is not None:
        sim_color = (
            _GREEN if embed_sim >= 0.8 else _YELLOW if embed_sim >= 0.6 else _RED
        )
        parts.append(f"  embed={sim_color}{embed_sim:.3f}{_R}")
    _w("".join(parts))


# ── Request / response pairs ────────────────────────────────────────────────


def req_summarize(
    strategy: str, model: str, reasoning_effort: str | None, prompt_len: int
) -> None:
    effort_str = (
        f"  {_GRAY}reasoning_effort={_R}{_YELLOW}{reasoning_effort}{_R}"
        if reasoning_effort
        else f"  {_GRAY}temperature={_R}{_YELLOW}0.3{_R}"
    )
    _w(
        f"  {_ARROW_OUT} {_BOLD}[summarize]{_R}"
        f"  {_GRAY}strategy={_R}{_MAGENTA}{strategy}{_R}"
        f"  {_GRAY}model={_R}{_CYAN}{model}{_R}"
        f"{effort_str}"
        f"  {_GRAY}input={_R}{_WHITE}{prompt_len:,}{_R}{_GRAY}c{_R}"
    )


def res_summarize(strategy: str, text: str, usage) -> None:
    usage_str = _format_usage(usage)
    _w(
        f"  {_ARROW_IN} {_BOLD}[summarize]{_R}"
        f"  {_GRAY}strategy={_R}{_MAGENTA}{strategy}{_R}"
        f"  {_GRAY}output={_R}{_WHITE}{nws(text):,}{_R}{_GRAY}c{_R}"
        f"{usage_str}"
    )
    _w(f"     {_GRAY}{_truncate(text)}{_R}")


def req_judge(
    model: str, reasoning_effort: str | None, n_versions: int, original_len: int
) -> None:
    effort_str = (
        f"  {_GRAY}reasoning_effort={_R}{_YELLOW}{reasoning_effort}{_R}"
        if reasoning_effort
        else ""
    )
    _w(
        f"  {_ARROW_OUT} {_BOLD}[judge]{_R}"
        f"  {_GRAY}model={_R}{_CYAN}{model}{_R}"
        f"  {_GRAY}versions={_R}{_WHITE}{n_versions}{_R}"
        f"  {_GRAY}original={_R}{_WHITE}{original_len:,}{_R}{_GRAY}c{_R}"
        f"{effort_str}"
    )


def res_judge(
    winner: str, winner_strategy: str, score: float, reasoning: str, usage
) -> None:
    usage_str = _format_usage(usage)
    score_color = _GREEN if score >= 0.8 else _YELLOW if score >= 0.6 else _RED
    _w(
        f"  {_ARROW_IN} {_BOLD}[judge]{_R}"
        f"  {_GRAY}winner={_R}{_BOLD}{_CYAN}{winner}{_R}({_MAGENTA}{winner_strategy}{_R})"
        f"  {_GRAY}score={_R}{score_color}{score:.3f}{_R}"
        f"{usage_str}"
    )
    if reasoning:
        _w(f"     {_GRAY}{_truncate(reasoning, 160)}{_R}")


def req_embed(model: str, text_len: int, label: str = "") -> None:
    _w(
        f"  {_ARROW_OUT} {_BOLD}[embed]{_R}"
        f"  {_GRAY}model={_R}{_CYAN}{model}{_R}"
        f"  {_GRAY}{label}={_R}{_WHITE}{text_len:,}{_R}{_GRAY}c{_R}"
    )


def res_embed(sim: float, usage_a=None, usage_b=None) -> None:
    sim_color = _GREEN if sim >= 0.8 else _YELLOW if sim >= 0.6 else _RED
    parts = [
        f"  {_ARROW_IN} {_BOLD}[embed]{_R}  {_GRAY}cosine_sim={_R}{sim_color}{sim:.4f}{_R}"
    ]
    if usage_a and usage_b:
        ta = getattr(usage_a, "total_tokens", None)
        tb = getattr(usage_b, "total_tokens", None)
        if ta is not None and tb is not None:
            parts.append(f"  {_GRAY}tokens={_R}{_MAGENTA}{ta + tb}{_R}")
    _w("".join(parts))


# ── Warnings / errors ───────────────────────────────────────────────────────


def warn(msg: str) -> None:
    _w(f"  {_WARN}  {_YELLOW}{msg}{_R}")


def error(msg: str) -> None:
    _w(f"  {_ERR}  {_RED}{msg}{_R}")


def threshold_stop(sim: float, threshold: float) -> None:
    _w(
        f"  {_WARN}  similarity {_RED}{sim:.3f}{_R} < threshold {_YELLOW}{threshold}{_R}"
        f"  {_GRAY}→ stopping{_R}"
    )


def plateau_stop() -> None:
    _w(f"  {_WARN}  {_YELLOW}no improvement for 2 iterations — stopping early{_R}")


# ── Final summary ────────────────────────────────────────────────────────────


def final_summary(
    original_len: int, compressed_len: int, score: float, found_at: int
) -> None:
    ratio = compressed_len / original_len if original_len else 1.0
    saved = 1.0 - ratio
    saved_color = _GREEN if saved >= 0.4 else _YELLOW if saved >= 0.2 else _GRAY
    score_color = _GREEN if score >= 0.8 else _YELLOW if score >= 0.6 else _RED
    bar = f"{_BOLD}{'═' * 56}{_R}"
    _w(f"\n{bar}")
    _w(f"  {_BOLD}Result{_R}")
    _w(f"  {_GRAY}original   {_R}{_WHITE}{original_len:,}{_R}{_GRAY} chars{_R}")
    _w(
        f"  {_GRAY}compressed {_R}{saved_color}{compressed_len:,}{_R}{_GRAY} chars  ({saved:.1%} saved){_R}"
    )
    _w(f"  {_GRAY}score      {_R}{score_color}{score:.3f}{_R}")
    _w(f"  {_GRAY}found at   {_R}{_WHITE}iteration {found_at}{_R}")
    _w(f"{bar}\n")


# ── Internal helpers ─────────────────────────────────────────────────────────


def _format_usage(usage) -> str:
    if usage is None:
        return ""
    p = getattr(usage, "prompt_tokens", None)
    c = getattr(usage, "completion_tokens", None)
    t = getattr(usage, "total_tokens", None)
    if p is None and t is None:
        return ""
    parts = []
    if p is not None:
        parts.append(f"{_GRAY}in={_R}{_MAGENTA}{p:,}{_R}")
    if c is not None:
        parts.append(f"{_GRAY}out={_R}{_MAGENTA}{c:,}{_R}")
    if t is not None:
        parts.append(f"{_GRAY}total={_R}{_MAGENTA}{t:,}{_R}")
    return "  " + "  ".join(parts) if parts else ""
