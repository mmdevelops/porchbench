"""Tests for interactive picker label rendering and capability gating."""

from porchbench.interactive import _build_run_toggles, _format_model_label


class TestFormatModelLabel:
    def test_no_caps_no_requirement_renders_bare_name(self):
        label, qualifies = _format_model_label("qwen3:8b", caps=[], required=None)
        assert label == "qwen3:8b"
        assert qualifies is True

    def test_caps_no_requirement_renders_badge(self):
        label, qualifies = _format_model_label(
            "qwen3:8b", caps=["tools", "vision", "thinking"], required=None,
        )
        assert label == "qwen3:8b  [tools, vision, thinking]"
        assert qualifies is True

    def test_required_met_renders_badge_no_marker(self):
        label, qualifies = _format_model_label(
            "qwen3:8b", caps=["tools", "vision"], required=["tools"],
        )
        assert label == "qwen3:8b  [tools, vision]"
        assert qualifies is True

    def test_required_missing_appends_marker_and_disqualifies(self):
        label, qualifies = _format_model_label(
            "medgemma:4b", caps=["vision"], required=["tools"],
        )
        assert label == "medgemma:4b  [vision]  · missing: tools"
        assert qualifies is False

    def test_required_missing_no_caps_known(self):
        # Server returned no capability info — every required cap is "missing"
        label, qualifies = _format_model_label(
            "stub-model", caps=[], required=["tools"],
        )
        assert label == "stub-model  · missing: tools"
        assert qualifies is False

    def test_multiple_required_lists_only_missing(self):
        label, qualifies = _format_model_label(
            "vision-only:7b", caps=["vision"], required=["tools", "vision"],
        )
        assert label == "vision-only:7b  [vision]  · missing: tools"
        assert qualifies is False

    def test_empty_required_list_treated_as_no_requirement(self):
        # Empty list (text-only suite path) shouldn't disqualify anything
        label, qualifies = _format_model_label(
            "any-model", caps=["vision"], required=[],
        )
        assert label == "any-model  [vision]"
        assert qualifies is True


class TestBuildRunToggles:
    """Run options picker decorates the Evaluate label with the resolved judge
    so users see which model will score before opting in. Hybrid: a sibling
    Re-pick toggle lets users override the saved default for one run without
    leaving the picker."""

    def _evaluate_label(self, toggles):
        return next(label for label, key in toggles if key == "evaluate")

    def test_resolved_judge_shows_inline(self):
        toggles = _build_run_toggles(judge_label="ollama/gemma4:e4b")
        label = self._evaluate_label(toggles)
        assert "Evaluate results in a post-phase batch" in label
        assert "(judge: ollama/gemma4:e4b)" in label
        # Rich dim markup wraps the hint so the judge identity stays
        # visually subordinate to the toggle's primary action.
        assert "[dim]" in label and "[/dim]" in label

    def test_unresolved_judge_advertises_picker(self):
        # ollama backend with no PORCHBENCH_EVAL_MODEL set — picker fires
        # post-confirm, label tells the user that's coming.
        toggles = _build_run_toggles(judge_label=None)
        label = self._evaluate_label(toggles)
        assert "(judge: pick on confirm)" in label

    def test_repick_toggle_present_and_keyed(self):
        toggles = _build_run_toggles(judge_label="ollama/gemma4:e4b")
        keys = [k for _, k in toggles]
        assert "repick_judge" in keys
        # Sits adjacent to evaluate so users find it without scanning the
        # whole list.
        assert keys.index("repick_judge") == keys.index("evaluate") + 1

    def test_strategies_toggle_preserved(self):
        # Regression check: adding the new toggle didn't drop the existing
        # ones.
        toggles = _build_run_toggles(judge_label=None)
        keys = [k for _, k in toggles]
        for required in ("verbose", "resume", "profile_vram", "profile",
                         "evaluate", "repick_judge", "strategies"):
            assert required in keys, f"missing toggle: {required}"
