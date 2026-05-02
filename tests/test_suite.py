"""Tests for suite loading, validation, and option merging."""

import tempfile

import pytest

from porchbench.assets import find_suite
from porchbench.schemas import Message, ModelOptions, Prompt, Suite, SuiteDefaults, SuiteMetadata
from porchbench.suite import (
    apply_option_overrides,
    compute_suite_hash,
    load_suite,
    make_suite_reference,
    required_capabilities_for_suite,
    resolve_messages,
    resolve_options,
)


def _make_suite(defaults: ModelOptions | None = None, prompts: list[Prompt] | None = None) -> Suite:
    return Suite(
        suite=SuiteMetadata(name="T", version="1.0"),
        defaults=SuiteDefaults(options=defaults or ModelOptions()),
        prompts=prompts or [Prompt(
            id="p1", category="coding", difficulty="easy",
            messages=[Message(role="user", content="Hi")],
        )],
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


class TestApplyOptionOverrides:
    def test_empty_overrides_returns_suite_unchanged(self):
        suite = _make_suite(ModelOptions(temperature=0, num_ctx=4096))
        out = apply_option_overrides(suite, {})
        assert out is suite

    def test_override_lands_in_defaults(self):
        suite = _make_suite(ModelOptions(num_ctx=4096))
        out = apply_option_overrides(suite, {"think": False, "num_ctx": 8192})
        assert out.defaults.options.think is False
        assert out.defaults.options.num_ctx == 8192

    def test_unset_defaults_preserved(self):
        suite = _make_suite(ModelOptions(temperature=0.5, seed=99))
        out = apply_option_overrides(suite, {"think": False})
        assert out.defaults.options.temperature == 0.5
        assert out.defaults.options.seed == 99

    def test_per_prompt_options_still_win(self):
        # CLI sets think=False; one prompt explicitly opts in to think=True.
        prompt_with_think = Prompt(
            id="p1", category="coding", difficulty="easy",
            messages=[Message(role="user", content="Hi")],
            options=ModelOptions(think=True),
        )
        suite = _make_suite(prompts=[prompt_with_think])
        suite = apply_option_overrides(suite, {"think": False})
        # After per-prompt resolve, the explicit per-prompt value wins.
        resolved = resolve_options(suite.defaults.options, suite.prompts[0])
        assert resolved.think is True

    def test_unknown_keys_pass_through_via_extra_allow(self):
        # ModelOptions has extra="allow" so forwarded Ollama options round-trip.
        suite = _make_suite()
        out = apply_option_overrides(suite, {"repeat_penalty": 1.1})
        dumped = out.defaults.options.model_dump()
        assert dumped.get("repeat_penalty") == 1.1

    def test_invalid_type_raises_validation_error(self):
        suite = _make_suite()
        with pytest.raises(Exception):  # pydantic ValidationError subclass
            apply_option_overrides(suite, {"num_ctx": "not-an-int"})


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
# Required-capability derivation
# ---------------------------------------------------------------------------


class TestRequiredCapabilitiesForSuite:
    def test_text_only_suite_needs_nothing(self):
        suite = _make_suite()
        assert required_capabilities_for_suite(suite) == []

    def test_tool_use_suite_needs_tools(self):
        prompt = Prompt(
            id="t1", category="coding", difficulty="easy",
            mode="tool-use",
            messages=[Message(role="user", content="Read the file.")],
        )
        suite = _make_suite(prompts=[prompt])
        assert required_capabilities_for_suite(suite) == ["tools"]

    def test_mixed_suite_needs_tools(self):
        text_prompt = Prompt(
            id="p1", category="coding", difficulty="easy",
            messages=[Message(role="user", content="Hi")],
        )
        tool_prompt = Prompt(
            id="t1", category="coding", difficulty="easy",
            mode="tool-use",
            messages=[Message(role="user", content="Read the file.")],
        )
        suite = _make_suite(prompts=[text_prompt, tool_prompt])
        assert required_capabilities_for_suite(suite) == ["tools"]


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestLoadSuite:
    def test_load_coding_basics(self):
        suite = load_suite(find_suite("coding-basics"))
        assert suite.suite.name == "Coding Basics"
        assert suite.suite.version == "2.0"
        assert len(suite.prompts) > 0

    def test_load_routing_discovery(self):
        suite = load_suite(find_suite("routing-discovery"))
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
        path = find_suite("coding-basics")
        h1 = compute_suite_hash(path)
        h2 = compute_suite_hash(path)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex

    def test_suite_reference(self):
        path = find_suite("coding-basics")
        suite = load_suite(path)
        ref = make_suite_reference(path, suite)
        assert ref.name == "Coding Basics"
        assert ref.version == "2.0"
        assert len(ref.sha256) == 64

    def test_suite_reference_file_is_portable_for_packaged(self):
        # Packaged suites should produce a portable `<bundled>/...` identifier
        # so result JSONs don't leak absolute local paths.
        path = find_suite("coding-basics")
        suite = load_suite(path)
        ref = make_suite_reference(path, suite)
        assert ref.file == "<bundled>/coding-basics.yaml"

    def test_suite_reference_file_is_basename_for_local(self, tmp_path):
        # Non-packaged paths fall back to basename only (no absolute-path leak).
        local = tmp_path / "my-custom.yaml"
        local.write_text(
            "suite:\n  name: Local\n  version: '1.0'\n"
            "defaults:\n  options: {}\nprompts: []\n",
            encoding="utf-8",
        )
        suite = load_suite(local)
        ref = make_suite_reference(local, suite)
        assert ref.file == "my-custom.yaml"
