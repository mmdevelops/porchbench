"""Tests for statistical inference: confidence intervals and bootstrap."""

import pytest

from ollama_bench.statistics import (
    ConfidenceInterval,
    PairedTestResult,
    auto_ci,
    bootstrap_ci,
    paired_comparison,
    paired_t_test,
    parametric_ci,
    wilcoxon_signed_rank,
)


class TestParametricCI:
    def test_basic_ci(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        ci = parametric_ci(values)
        assert ci is not None
        assert ci.mean == pytest.approx(30.0)
        assert ci.ci_lower < 30.0
        assert ci.ci_upper > 30.0
        assert ci.method == "t"
        assert ci.n == 5
        assert ci.confidence == 0.95

    def test_tight_ci_for_identical_values(self):
        values = [42.0] * 10
        ci = parametric_ci(values)
        assert ci is not None
        assert ci.ci_lower == pytest.approx(42.0)
        assert ci.ci_upper == pytest.approx(42.0)

    def test_returns_none_for_single_value(self):
        assert parametric_ci([42.0]) is None

    def test_returns_none_for_empty(self):
        assert parametric_ci([]) is None

    def test_wider_ci_for_more_variance(self):
        tight = parametric_ci([10.0, 10.1, 9.9, 10.0, 10.0])
        wide = parametric_ci([1.0, 50.0, 10.0, 90.0, 20.0])
        assert tight.margin < wide.margin

    def test_as_dict(self):
        ci = parametric_ci([1.0, 2.0, 3.0])
        d = ci.as_dict()
        assert "mean" in d
        assert "ci_lower" in d
        assert "ci_upper" in d
        assert d["method"] == "t"


class TestBootstrapCI:
    def test_basic_bootstrap(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        ci = bootstrap_ci(values, seed=42)
        assert ci is not None
        assert ci.mean == pytest.approx(30.0)
        assert ci.ci_lower < 30.0
        assert ci.ci_upper > 30.0
        assert ci.method == "bootstrap"

    def test_reproducible_with_seed(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci1 = bootstrap_ci(values, seed=42)
        ci2 = bootstrap_ci(values, seed=42)
        assert ci1.ci_lower == ci2.ci_lower
        assert ci1.ci_upper == ci2.ci_upper

    def test_single_value(self):
        ci = bootstrap_ci([42.0], seed=42)
        assert ci is not None
        assert ci.mean == pytest.approx(42.0)
        # With one value, all resamples are the same
        assert ci.ci_lower == pytest.approx(42.0)
        assert ci.ci_upper == pytest.approx(42.0)

    def test_empty_returns_none(self):
        assert bootstrap_ci([]) is None


class TestAutoCI:
    def test_small_sample_uses_bootstrap(self):
        values = [1.0, 2.0, 3.0]
        ci = auto_ci(values, bootstrap_threshold=30)
        assert ci is not None
        assert ci.method == "bootstrap"

    def test_large_sample_uses_parametric(self):
        values = list(range(50))
        ci = auto_ci([float(v) for v in values], bootstrap_threshold=30)
        assert ci is not None
        assert ci.method == "t"

    def test_single_value_returns_bootstrap(self):
        ci = auto_ci([42.0])
        assert ci is not None
        assert ci.method == "bootstrap"

    def test_empty_returns_none(self):
        assert auto_ci([]) is None


# ---------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------


class TestPairedTTest:
    def test_identical_values(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_t_test(a, a)
        assert result is not None
        assert result.mean_difference == pytest.approx(0.0)
        assert not result.significant

    def test_clearly_different(self):
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_t_test(a, b)
        assert result is not None
        assert result.mean_difference > 0
        assert result.significant
        assert result.effect_magnitude == "large"

    def test_returns_none_for_single_pair(self):
        assert paired_t_test([1.0], [2.0]) is None

    def test_returns_none_for_unequal_length(self):
        assert paired_t_test([1.0, 2.0], [3.0]) is None

    def test_as_dict(self):
        a = [10.0, 20.0, 30.0]
        b = [5.0, 10.0, 15.0]
        result = paired_t_test(a, b)
        d = result.as_dict()
        assert "test_name" in d
        assert "p_value" in d
        assert "effect_size" in d


class TestWilcoxon:
    def test_identical_values(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = wilcoxon_signed_rank(a, a)
        assert result is not None
        assert result.mean_difference == pytest.approx(0.0)
        assert not result.significant

    def test_clearly_different(self):
        a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = wilcoxon_signed_rank(a, b)
        assert result is not None
        assert result.mean_difference > 0
        assert result.significant
        assert result.effect_magnitude == "large"

    def test_returns_none_for_small_n(self):
        assert wilcoxon_signed_rank([1.0, 2.0], [3.0, 4.0]) is None

    def test_returns_none_for_unequal_length(self):
        assert wilcoxon_signed_rank([1.0] * 6, [2.0] * 5) is None


class TestPairedComparison:
    def test_small_n_uses_t_test(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [0.5, 1.5, 2.5, 3.5, 4.5]
        result = paired_comparison(a, b)
        assert result is not None
        assert result.test_name == "paired_t"

    def test_large_n_uses_wilcoxon(self):
        a = [float(i) for i in range(10)]
        b = [float(i) + 0.1 for i in range(10)]
        result = paired_comparison(a, b)
        assert result is not None
        assert result.test_name == "wilcoxon"

    def test_returns_none_for_insufficient_data(self):
        assert paired_comparison([1.0], [2.0]) is None
