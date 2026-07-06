"""Regenerate the README compare SVGs from the bundled examples/ artifacts.

Usage (from the repo root, with porchbench installed):

    python docs/assets/regen_compare_svgs.py

Writes:
- ``compare-coding-basics.svg`` — full ``porchbench compare`` output,
  per-prompt table included (linked from the README hero caption)
- ``compare-coding-basics-summary.svg`` — Model Summary + Paired Comparison
  only, exported at a narrow console width so the text stays legible at
  README column width (the README hero image)
"""

from pathlib import Path

from rich.console import Console

import porchbench.compare as compare_module
from porchbench.schemas import RunResult, Scorecard

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"
ASSETS = Path(__file__).resolve().parent


class _CapturePrints:
    """Stand-in console that records print calls for selective replay."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def print(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


def _load(model_cls, filename: str):
    return model_cls.model_validate_json((EXAMPLES / filename).read_text(encoding="utf-8"))


def main() -> None:
    runs = [
        _load(RunResult, "run-result_coding-basics_ministral-3-8b.json"),
        _load(RunResult, "run-result_coding-basics_granite4.1-8b.json"),
    ]
    scorecards = [
        _load(Scorecard, "scorecard_coding-basics_ministral-3-8b.json"),
        _load(Scorecard, "scorecard_coding-basics_granite4.1-8b.json"),
    ]

    capture = _CapturePrints()
    compare_module.console = capture
    compare_module.print_comparison_table(runs, scorecards, seed=42)

    full = Console(record=True, width=150, force_terminal=True)
    for args, kwargs in capture.calls:
        full.print(*args, **kwargs)
    full.save_svg(str(ASSETS / "compare-coding-basics.svg"), title="porchbench compare")

    # calls[0] is the per-prompt table, calls[1] its spacer line — the
    # Model Summary + Paired Comparison sections start at calls[2].
    summary = Console(record=True, width=92, force_terminal=True)
    for args, kwargs in capture.calls[2:]:
        summary.print(*args, **kwargs)
    summary.save_svg(str(ASSETS / "compare-coding-basics-summary.svg"), title="porchbench compare")

    print(f"wrote {ASSETS / 'compare-coding-basics.svg'}")
    print(f"wrote {ASSETS / 'compare-coding-basics-summary.svg'}")


if __name__ == "__main__":
    main()
