"""Tests for interactive picker label rendering and capability gating."""

from porchbench.interactive import _format_model_label


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
