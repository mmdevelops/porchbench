"""Unit tests for small render/format helpers in cli.py.

Kept separate from the heavier CliRunner-driven tests so the simple
visual surfaces (strategy table, etc.) are exercised without mocking the
inference and backend stack.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from porchbench.cli import _build_strategy_table
from porchbench.schemas import Strategy


def _render(table) -> str:
    buf = StringIO()
    Console(file=buf, width=140, force_terminal=False).print(table)
    return buf.getvalue()


def test_strategy_table_renders_name_and_message():
    """Each strategy gets a row with its system message verbatim."""
    out = _render(_build_strategy_table({
        "cot": Strategy(system_message="Think step by step."),
    }))
    assert "cot" in out
    assert "Think step by step." in out


def test_strategy_table_marks_empty_message_as_suite_default():
    """An empty system_message (e.g. `universal: {}` baseline) is labeled,
    not rendered as a blank cell — first-time users need to see that it's
    intentional."""
    out = _render(_build_strategy_table({"universal": Strategy()}))
    assert "universal" in out
    assert "suite default" in out


def test_strategy_table_truncates_long_messages():
    """Messages over 80 chars truncate with ellipsis to keep the table tidy."""
    long_msg = "X" * 200
    out = _render(_build_strategy_table({"verbose": Strategy(system_message=long_msg)}))
    # 77 chars + "..." appears; full 200-char string does not
    assert "X" * 77 + "..." in out
    assert "X" * 200 not in out


def test_strategy_table_preserves_dict_order():
    """Strategies render in the order they appear in the suite YAML —
    important so users can match the table to the per-cell run output."""
    out = _render(_build_strategy_table({
        "alpha": Strategy(system_message="A"),
        "beta": Strategy(system_message="B"),
        "gamma": Strategy(system_message="C"),
    }))
    assert out.index("alpha") < out.index("beta") < out.index("gamma")


# ---------------------------------------------------------------------------
# _suite_has_strategies — picker filter for `routes discover`
# ---------------------------------------------------------------------------


def test_suite_has_strategies_true_for_strategy_block(tmp_path):
    from porchbench.cli import _suite_has_strategies

    p = tmp_path / "with_strats.yaml"
    p.write_text(
        "suite: {name: T, version: '1.0'}\n"
        "defaults: {options: {}}\n"
        "strategies:\n"
        "  universal: {}\n"
        "  cot: {system_message: 'Think step by step.'}\n"
        "prompts: []\n",
        encoding="utf-8",
    )
    assert _suite_has_strategies(p) is True


def test_suite_has_strategies_false_when_block_missing(tmp_path):
    """A regular eval suite (coding-basics, cross-domain) has no strategies
    block — routes discover would synthesize a degenerate baseline. Filter
    must reject these."""
    from porchbench.cli import _suite_has_strategies

    p = tmp_path / "no_strats.yaml"
    p.write_text(
        "suite: {name: T, version: '1.0'}\n"
        "defaults: {options: {}}\n"
        "prompts: []\n",
        encoding="utf-8",
    )
    assert _suite_has_strategies(p) is False


def test_suite_has_strategies_false_when_block_empty(tmp_path):
    """`strategies: {}` (or null) is the same as no strategies — synthetic
    universal baseline territory. Reject."""
    from porchbench.cli import _suite_has_strategies

    p = tmp_path / "empty_strats.yaml"
    p.write_text(
        "suite: {name: T, version: '1.0'}\n"
        "defaults: {options: {}}\n"
        "strategies: {}\n"
        "prompts: []\n",
        encoding="utf-8",
    )
    assert _suite_has_strategies(p) is False


def test_suite_has_strategies_false_for_unparseable_yaml(tmp_path):
    from porchbench.cli import _suite_has_strategies

    p = tmp_path / "broken.yaml"
    p.write_text("this is: : not valid : yaml :\n", encoding="utf-8")
    assert _suite_has_strategies(p) is False
