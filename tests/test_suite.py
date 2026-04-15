"""Tests for suite loading, validation, and option merging."""

import tempfile
from pathlib import Path

import pytest

from feral.schemas import Message, ModelOptions, Prompt, Strategy
from feral.suite import (
    compute_suite_hash,
    load_suite,
    make_suite_reference,
    resolve_messages,
    resolve_options,
)


# ---------------------------------------------------------------------------
# Option merging
# ---------------------------------------------------------------------------


class TestResolveOptions:
    def test_defaults_preserved(self):
        defaults = ModelOptions(temperature=0, seed=42, num_predict=2048, num_ctx=4096)
        prompt = Prompt(
            id="p1", category="coding", difficulty="easy",
            messages=[Message(role="user", content="Hi")],
        )
        resolved = resolve_options(defaults, prompt)
        assert resolved.temperature == 0
        assert resolved.seed == 42
        assert resolved.num_predict == 2048

    def test_prompt_overrides(self):
        defaults = ModelOptions(temperature=0, seed=42, num_predict=2048)
        prompt = Prompt(
            id="p1", category="coding", difficulty="easy",
            messages=[Message(role="user", content="Hi")],
            options=ModelOptions(num_predict=512),
        )
        resolved = resolve_options(defaults, prompt)
        assert resolved.num_predict == 512
        assert resolved.temperature == 0  # default preserved
        assert resolved.seed == 42  # default preserved

    def test_no_overrides(self):
        defaults = ModelOptions(temperature=0.5, seed=99)
        prompt = Prompt(
            id="p1", category="coding", difficulty="easy",
            messages=[Message(role="user", content="Hi")],
        )
        resolved = resolve_options(defaults, prompt)
        assert resolved.temperature == 0.5
        assert resolved.seed == 99


# ---------------------------------------------------------------------------
# Message resolution
# ---------------------------------------------------------------------------


class TestResolveMessages:
    def test_no_system_message(self):
        prompt = Prompt(
            id="p1", category="coding", difficulty="easy",
            messages=[Message(role="user", content="Hello")],
        )
        msgs = resolve_messages(prompt)
        assert len(msgs) == 1
        assert msgs[0].role == "user"

    def test_with_system_message(self):
        prompt = Prompt(
            id="p1", category="coding", difficulty="easy",
            messages=[Message(role="user", content="Hello")],
        )
        msgs = resolve_messages(prompt, system_message="Be brief.")
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[0].content == "Be brief."
        assert msgs[1].role == "user"

    def test_empty_system_message_skipped(self):
        prompt = Prompt(
            id="p1", category="coding", difficulty="easy",
            messages=[Message(role="user", content="Hello")],
        )
        msgs = resolve_messages(prompt, system_message="")
        assert len(msgs) == 1

    def test_multi_turn_preserved(self):
        prompt = Prompt(
            id="p1", category="coding", difficulty="easy",
            messages=[
                Message(role="user", content="What is 2+2?"),
                Message(role="assistant", content="4"),
                Message(role="user", content="What about 3+3?"),
            ],
        )
        msgs = resolve_messages(prompt, system_message="Be direct.")
        assert len(msgs) == 4
        assert msgs[0].role == "system"
        assert msgs[3].role == "user"


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestLoadSuite:
    def test_load_coding_basics(self):
        suite = load_suite("suites/coding-basics.yaml")
        assert suite.suite.name == "Coding Basics"
        assert suite.suite.version == "2.0"
        assert len(suite.prompts) > 0

    def test_load_routing_discovery(self):
        suite = load_suite("suites/routing-discovery.yaml")
        assert suite.suite.name == "Routing Discovery"
        assert len(suite.strategies) == 5
        assert len(suite.prompts) >= 90

    def test_load_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            load_suite("nonexistent.yaml")

    def test_load_invalid_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("not: valid: suite: yaml: [")
            f.flush()
            with pytest.raises(Exception):
                load_suite(f.name)

    def test_suite_hash_deterministic(self):
        h1 = compute_suite_hash("suites/coding-basics.yaml")
        h2 = compute_suite_hash("suites/coding-basics.yaml")
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex

    def test_suite_reference(self):
        suite = load_suite("suites/coding-basics.yaml")
        ref = make_suite_reference("suites/coding-basics.yaml", suite)
        assert ref.name == "Coding Basics"
        assert ref.version == "2.0"
        assert len(ref.sha256) == 64
