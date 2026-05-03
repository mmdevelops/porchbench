"""Unit tests for small render/format helpers in cli.py.

Kept separate from the heavier CliRunner-driven tests so the simple
visual surfaces (strategy table, etc.) are exercised without mocking the
inference and backend stack.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from porchbench.overnight import _build_strategy_table
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
# suite_has_strategies — predicate that drives `overnight --strategies`
# eligibility checks
# ---------------------------------------------------------------------------


def test_suite_has_strategies_true_for_strategy_block(tmp_path):
    from porchbench.suite import suite_has_strategies

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
    assert suite_has_strategies(p) is True


def test_suite_has_strategies_false_when_block_missing(tmp_path):
    """A regular eval suite (coding-basics, cross-domain) has no strategies
    block — `overnight --strategies` would have nothing to expand. Predicate
    must report False so the CLI can hard-fail (single-suite) or warn-and-
    skip (multi-suite)."""
    from porchbench.suite import suite_has_strategies

    p = tmp_path / "no_strats.yaml"
    p.write_text(
        "suite: {name: T, version: '1.0'}\n"
        "defaults: {options: {}}\n"
        "prompts: []\n",
        encoding="utf-8",
    )
    assert suite_has_strategies(p) is False


def test_suite_has_strategies_false_when_block_empty(tmp_path):
    """`strategies: {}` (or null) is the same as no strategies — nothing to
    expand. Reject."""
    from porchbench.suite import suite_has_strategies

    p = tmp_path / "empty_strats.yaml"
    p.write_text(
        "suite: {name: T, version: '1.0'}\n"
        "defaults: {options: {}}\n"
        "strategies: {}\n"
        "prompts: []\n",
        encoding="utf-8",
    )
    assert suite_has_strategies(p) is False


def test_suite_has_strategies_false_for_unparseable_yaml(tmp_path):
    from porchbench.suite import suite_has_strategies

    p = tmp_path / "broken.yaml"
    p.write_text("this is: : not valid : yaml :\n", encoding="utf-8")
    assert suite_has_strategies(p) is False


# ---------------------------------------------------------------------------
# preview_eval_model — read-only label resolver for the run options picker
# ---------------------------------------------------------------------------


def test_preview_eval_model_returns_explicit_when_set():
    """CLI flag / PORCHBENCH_EVAL_MODEL wins over backend defaults."""
    from porchbench.cli import preview_eval_model

    assert preview_eval_model("ollama", "qwen3:8b") == "qwen3:8b"
    assert preview_eval_model("api", "claude-haiku-4-5-20251001") == "claude-haiku-4-5-20251001"


def test_preview_eval_model_returns_cloud_default_when_no_explicit():
    """Cloud backends (api / claude-code) have stable named defaults — the
    label resolver should surface them without firing the picker."""
    from porchbench.cli import preview_eval_model
    from porchbench.evaluator import EVAL_BACKEND_DEFAULTS

    assert preview_eval_model("api", None) == EVAL_BACKEND_DEFAULTS["api"]
    assert preview_eval_model("claude-code", None) == EVAL_BACKEND_DEFAULTS["claude-code"]


def test_preview_eval_model_returns_none_for_ollama_no_explicit():
    """Ollama with no explicit model = picker would fire — caller should
    render `(judge: pick on confirm)` in the toggle label."""
    from porchbench.cli import preview_eval_model

    assert preview_eval_model("ollama", None) is None


# ---------------------------------------------------------------------------
# resolve_eval_model_or_exit force_pick branch — drives the "Re-pick judge
# for this run" toggle. Must skip both the explicit-model and cloud-default
# shortcuts, and must NOT prompt to persist the picked model.
# ---------------------------------------------------------------------------


def test_resolve_eval_model_force_pick_skips_explicit_and_persistence(monkeypatch):
    """force_pick=True should ignore any explicit_model / env setting and
    fire the picker; the persistence prompt must not appear (one-shot
    override)."""
    from unittest.mock import MagicMock

    from porchbench import cli
    from porchbench.backend import OllamaBackend

    select_mock = MagicMock(return_value="picked-model:7b")
    confirm_mock = MagicMock(return_value=True)
    persist_mock = MagicMock()

    # interactive.select_evaluator_model is imported inside the function;
    # patch the module attr the call resolves to.
    import porchbench.interactive as interactive_mod
    monkeypatch.setattr(interactive_mod, "select_evaluator_model", select_mock)
    monkeypatch.setattr(cli.typer, "confirm", confirm_mock)
    monkeypatch.setattr(cli, "_persist_eval_model_default", persist_mock)

    backend = MagicMock(spec=OllamaBackend)
    chosen = cli.resolve_eval_model_or_exit(
        "ollama",
        explicit_model="env-default:8b",  # would be returned without force_pick
        backend=backend,
        interactive=True,
        force_pick=True,
    )

    assert chosen == "picked-model:7b"
    select_mock.assert_called_once_with(backend)
    # Critical: no persistence prompt under force_pick — override is for
    # this run only.
    confirm_mock.assert_not_called()
    persist_mock.assert_not_called()


def test_resolve_eval_model_default_path_still_prompts_to_persist(monkeypatch):
    """Regression check: the existing first-use path (no explicit, no
    force_pick) must keep prompting to persist the picked default — that's
    what saves users from re-picking on every subsequent run."""
    from unittest.mock import MagicMock

    from porchbench import cli
    from porchbench.backend import OllamaBackend

    select_mock = MagicMock(return_value="first-pick:8b")
    confirm_mock = MagicMock(return_value=True)
    persist_mock = MagicMock()

    import porchbench.interactive as interactive_mod
    monkeypatch.setattr(interactive_mod, "select_evaluator_model", select_mock)
    monkeypatch.setattr(cli.typer, "confirm", confirm_mock)
    monkeypatch.setattr(cli, "_persist_eval_model_default", persist_mock)

    backend = MagicMock(spec=OllamaBackend)
    chosen = cli.resolve_eval_model_or_exit(
        "ollama",
        explicit_model=None,
        backend=backend,
        interactive=True,
        force_pick=False,
    )

    assert chosen == "first-pick:8b"
    confirm_mock.assert_called_once()
    persist_mock.assert_called_once_with("first-pick:8b")
