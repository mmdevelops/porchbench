"""Tests for the asset resolver (bundled defaults + project-local overrides)."""

from __future__ import annotations

from pathlib import Path

import pytest

from porchbench.assets import (
    PACKAGED_RUBRICS_DIR,
    PACKAGED_SUITES_DIR,
    find_rubric,
    find_suite,
    is_pathlike,
    porchbench_version,
    resolve_rubric_dir,
    resolve_suite_dir,
)

# ---------------------------------------------------------------------------
# Packaged defaults are reachable
# ---------------------------------------------------------------------------


def test_packaged_suites_dir_contains_coding_basics():
    assert PACKAGED_SUITES_DIR.is_dir()
    assert (PACKAGED_SUITES_DIR / "coding-basics.yaml").is_file()


def test_packaged_rubrics_dir_contains_default():
    assert PACKAGED_RUBRICS_DIR.is_dir()
    assert (PACKAGED_RUBRICS_DIR / "default.yaml").is_file()


# ---------------------------------------------------------------------------
# is_pathlike heuristic
# ---------------------------------------------------------------------------


class TestIsPathlike:
    def test_bare_name_is_not_pathlike(self):
        assert not is_pathlike("coding-basics")

    def test_slash_makes_pathlike(self):
        assert is_pathlike("suites/coding-basics.yaml")

    def test_backslash_makes_pathlike(self):
        assert is_pathlike("suites\\coding-basics.yaml")

    def test_yaml_extension_makes_pathlike(self):
        assert is_pathlike("coding-basics.yaml")

    def test_absolute_path_is_pathlike(self):
        assert is_pathlike(str(PACKAGED_SUITES_DIR / "coding-basics.yaml"))

    def test_bare_name_in_path_object_is_not_pathlike(self):
        # typer wraps CLI args in Path even when user typed a bare name.
        # Path("coding-basics") with no separators should still resolve as a name.
        assert not is_pathlike(Path("coding-basics"))

    def test_path_object_with_separator_is_pathlike(self):
        assert is_pathlike(Path("suites/coding-basics.yaml"))


# ---------------------------------------------------------------------------
# find_suite resolution order
# ---------------------------------------------------------------------------


class TestFindSuite:
    def test_bare_name_falls_back_to_packaged(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = find_suite("coding-basics")
        assert result == PACKAGED_SUITES_DIR / "coding-basics.yaml"
        assert result.is_file()

    def test_cwd_override_wins_over_packaged(self, tmp_path, monkeypatch):
        suites_dir = tmp_path / "suites"
        suites_dir.mkdir()
        override = suites_dir / "coding-basics.yaml"
        override.write_text("suite:\n  name: Override\n  version: '1.0'\ndefaults:\n  options: {}\nprompts: []\n")

        monkeypatch.chdir(tmp_path)
        result = find_suite("coding-basics")
        assert result == override
        assert result != PACKAGED_SUITES_DIR / "coding-basics.yaml"

    def test_explicit_absolute_path_wins(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom.yaml"
        custom.write_text("# placeholder")
        monkeypatch.chdir(tmp_path)

        result = find_suite(custom)
        assert result == custom.resolve()

    def test_relative_path_form_works(self, tmp_path, monkeypatch):
        suites_dir = tmp_path / "my-suites"
        suites_dir.mkdir()
        yaml_file = suites_dir / "custom.yaml"
        yaml_file.write_text("# placeholder")

        monkeypatch.chdir(tmp_path)
        result = find_suite("my-suites/custom.yaml")
        assert result == yaml_file.resolve()

    def test_missing_bare_name_raises_with_tried_paths(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            find_suite("no-such-suite")
        msg = str(exc_info.value)
        assert "no-such-suite" in msg
        assert "suites" in msg  # both tried paths mentioned

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            find_suite("/nonexistent/path/custom.yaml")


# ---------------------------------------------------------------------------
# find_rubric parallels
# ---------------------------------------------------------------------------


class TestFindRubric:
    def test_bare_name_finds_packaged_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = find_rubric("default")
        assert result == PACKAGED_RUBRICS_DIR / "default.yaml"
        assert result.is_file()

    def test_cwd_override_wins(self, tmp_path, monkeypatch):
        rubrics_dir = tmp_path / "rubrics"
        rubrics_dir.mkdir()
        override = rubrics_dir / "default.yaml"
        override.write_text("rubric: {name: Override, criteria: []}")

        monkeypatch.chdir(tmp_path)
        assert find_rubric("default") == override

    def test_calibration_examples_resolves(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = find_rubric("calibration-examples")
        assert result.is_file()
        assert result.name == "calibration-examples.yaml"


# ---------------------------------------------------------------------------
# resolve_*_dir
# ---------------------------------------------------------------------------


class TestResolveSuiteDir:
    def test_explicit_override_passes_through(self, tmp_path):
        assert resolve_suite_dir(tmp_path) == tmp_path

    def test_cwd_suites_wins(self, tmp_path, monkeypatch):
        (tmp_path / "suites").mkdir()
        monkeypatch.chdir(tmp_path)
        assert resolve_suite_dir() == tmp_path / "suites"

    def test_falls_back_to_packaged(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resolve_suite_dir() == PACKAGED_SUITES_DIR


class TestResolveRubricDir:
    def test_explicit_override_passes_through(self, tmp_path):
        assert resolve_rubric_dir(tmp_path) == tmp_path

    def test_cwd_rubrics_wins(self, tmp_path, monkeypatch):
        (tmp_path / "rubrics").mkdir()
        monkeypatch.chdir(tmp_path)
        assert resolve_rubric_dir() == tmp_path / "rubrics"

    def test_falls_back_to_packaged(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resolve_rubric_dir() == PACKAGED_RUBRICS_DIR


# ---------------------------------------------------------------------------
# porchbench_version
# ---------------------------------------------------------------------------


def test_porchbench_version_returns_nonempty_string():
    version = porchbench_version()
    assert isinstance(version, str)
    assert len(version) > 0
