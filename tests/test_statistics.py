"""Tests for statistical inference: t-distribution, CIs, paired tests, power."""

import pytest

from porchbench.statistics import (
    _t_critical,
    auto_ci,
    average_score_repeats,
    bootstrap_ci,
    minimum_detectable_dz,
    paired_comparison,
    paired_t_test,
    parametric_ci,
    permutation_test_paired,
    t_two_tailed_p,
    wilcoxon_signed_rank,
)


class TestTDistribution:
    """Known-answer tests against published t-tables."""

    @pytest.mark.parametrize(
        ("t_stat", "df", "expected_p"),
        [
            # Two-tailed p at published 95% critical values -> 0.05
            (12.706, 1, 0.05),
            (2.776, 4, 0.05),
            (2.228, 10, 0.05),
            (2.086, 20, 0.05),
            (2.042, 30, 0.05),
            # Published 90% critical values -> 0.10
            (1.812, 10, 0.10),
            (1.725, 20, 0.10),
            # t=1, df=10 -> 0.3409 (published)
            (1.0, 10, 0.3409),
        ],
    )
    def test_two_tailed_p_matches_tables(self, t_stat, df, expected_p):
        assert t_two_tailed_p(t_stat, df) == pytest.approx(expected_p, abs=5e-4)

    def test_p_at_zero_is_one(self):
        assert t_two_tailed_p(0.0, 5) == pytest.approx(1.0)

    def test_p_at_infinity_is_zero(self):
        assert t_two_tailed_p(float("inf"), 5) == 0.0

    def test_negative_t_same_as_positive(self):
        assert t_two_tailed_p(-2.0, 8) == pytest.approx(t_two_tailed_p(2.0, 8))

    @pytest.mark.parametrize(
        ("df", "expected"),
        [(1, 12.706), (4, 2.776), (10, 2.228), (20, 2.086), (30, 2.042)],
    )
    def test_critical_values_match_tables(self, df, expected):
        assert _t_critical(df, 0.95) == pytest.approx(expected, abs=2e-3)

    def test_critical_approaches_z_at_large_df(self):
        assert _t_critical(100_000, 0.95) == pytest.approx(1.96, abs=1e-2)


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

    def test_known_answer_interval(self):
        # mean=30, sd=15.811, se=7.071, t_crit(df=4)=2.776 -> margin 19.63
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        ci = parametric_ci(values)
        assert ci.ci_lower == pytest.approx(30.0 - 19.63, abs=0.02)
        assert ci.ci_upper == pytest.approx(30.0 + 19.63, abs=0.02)

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
        assert ci.ci_lower == pytest.approx(42.0)
        assert ci.ci_upper == pytest.approx(42.0)

    def test_empty_returns_none(self):
        assert bootstrap_ci([]) is None


class TestAutoCI:
    def test_small_sample_uses_t(self):
        # t-CI at every n >= 2 — the percentile bootstrap undercovers at
        # small n, so it is no longer the small-sample default.
        ci = auto_ci([1.0, 2.0, 3.0])
        assert ci is not None
        assert ci.method == "t"

    def test_large_sample_uses_t(self):
        ci = auto_ci([float(v) for v in range(50)])
        assert ci is not None
        assert ci.method == "t"

    def test_single_value_falls_back_to_bootstrap(self):
        ci = auto_ci([42.0])
        assert ci is not None
        assert ci.method == "bootstrap"

    def test_empty_returns_none(self):
        assert auto_ci([]) is None


# ---------------------------------------------------------------------------
# Repeat handling
# ---------------------------------------------------------------------------


class TestAverageScoreRepeats:
    def test_averages_across_repeats(self):
        repeats = [
            {"p1": 4.0, "p2": 3.0},
            {"p1": 5.0, "p2": 4.0},
            {"p1": 4.5, "p2": 3.5},
        ]
        avg = average_score_repeats(repeats)
        assert avg["p1"] == pytest.approx(4.5)
        assert avg["p2"] == pytest.approx(3.5)

    def test_prompt_missing_from_one_repeat(self):
        repeats = [{"p1": 4.0, "p2": 2.0}, {"p1": 5.0}]
        avg = average_score_repeats(repeats)
        assert avg["p1"] == pytest.approx(4.5)
        assert avg["p2"] == pytest.approx(2.0)  # averaged over the repeats that scored it

    def test_single_map_is_identity(self):
        assert average_score_repeats([{"p1": 3.0}]) == {"p1": 3.0}

    def test_empty(self):
        assert average_score_repeats([]) == {}


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


class TestMinimumDetectableDz:
    def test_known_value_n20(self):
        # (t_{.025,19} + t_{.20,19}) / sqrt(20) = (2.093 + 0.861) / 4.472
        assert minimum_detectable_dz(20) == pytest.approx(0.6606, abs=2e-3)

    def test_approaches_z_formula_at_large_n(self):
        # (1.960 + 0.842) / sqrt(n) as df -> inf
        assert minimum_detectable_dz(5000) == pytest.approx(2.802 / 5000**0.5, abs=1e-3)

    def test_monotone_decreasing_in_n(self):
        assert minimum_detectable_dz(10) > minimum_detectable_dz(30) > minimum_detectable_dz(100)

    def test_none_below_two(self):
        assert minimum_detectable_dz(1) is None


# ---------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------


class TestPairedTTest:
    def test_identical_values(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_t_test(a, a)
        assert result is not None
        assert result.mean_difference == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)
        assert not result.significant

    def test_exact_p_at_small_n(self):
        # diffs [9,18,27,36,45]: mean 27, sd 14.23, t = 4.243, df=4
        # -> two-tailed p = 0.0132 (published t-table interpolation)
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_t_test(a, b)
        assert result is not None
        assert result.mean_difference > 0
        assert result.statistic == pytest.approx(4.243, abs=1e-3)
        assert result.p_value == pytest.approx(0.0132, abs=5e-4)
        assert result.significant
        assert result.effect_magnitude == "large"

    def test_pvalue_available_at_any_n(self):
        a = [float(i) for i in range(40)]
        b = [float(i) + 2.0 for i in range(40)]
        result = paired_t_test(a, b)
        assert result is not None
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0
        assert result.significant is not None

    def test_ci_estimand_matches_test(self):
        # The CI is a t-interval on the mean difference: p < 0.05 iff the
        # 95% CI excludes zero (same estimand, same procedure).
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_t_test(a, b)
        assert result.significant
        assert result.ci is not None
        assert result.ci.method == "t"
        assert result.ci.ci_lower > 0  # excludes zero, agreeing with p<0.05

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
        assert d["p_value"] is not None  # exact t p-value at any df >= 1


class TestPermutationTest:
    def test_exact_all_positive_equal(self):
        # n=8 identical positive diffs: only the all-plus and all-minus
        # assignments reach |sum| = 8 -> p = 2/256 exactly.
        a = [2.0] * 8
        b = [1.0] * 8
        result = permutation_test_paired(a, b)
        assert result is not None
        assert result.method == "exact"
        assert result.p_value == pytest.approx(2 / 256)

    def test_exact_hand_enumerable_n3(self):
        # diffs [1,2,3]: sign assignments give |sum| in {6,4,2,0,0,2,4,6};
        # |sum| >= 6 in 2 of 8 -> p = 0.25.
        result = permutation_test_paired([2.0, 3.0, 4.0], [1.0, 1.0, 1.0])
        assert result is not None
        assert result.method == "exact"
        assert result.p_value == pytest.approx(0.25)

    def test_identical_values_p_one(self):
        a = [1.0, 2.0, 3.0, 4.0]
        result = permutation_test_paired(a, a)
        assert result is not None
        assert result.p_value == pytest.approx(1.0)

    def test_monte_carlo_above_threshold(self):
        a = [float(i) for i in range(20)]
        b = [float(i) + 1.0 for i in range(20)]
        result = permutation_test_paired(a, b, seed=42)
        assert result is not None
        assert result.method == "monte_carlo"
        assert 0.0 < result.p_value < 0.05

    def test_monte_carlo_reproducible(self):
        a = [float(i) for i in range(20)]
        b = [float(i) + 0.5 for i in range(20)]
        r1 = permutation_test_paired(a, b, seed=42)
        r2 = permutation_test_paired(a, b, seed=42)
        assert r1.p_value == r2.p_value

    def test_returns_none_for_single_pair(self):
        assert permutation_test_paired([1.0], [2.0]) is None


class TestWilcoxon:
    # Legacy path: retained but no longer routed by paired_comparison.
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
    def test_paired_t_is_primary_at_small_n(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [0.5, 1.5, 2.5, 3.5, 4.5]
        result = paired_comparison(a, b)
        assert result is not None
        assert result.test_name == "paired_t"
        assert result.p_value is not None

    def test_paired_t_is_primary_at_larger_n(self):
        a = [float(i) for i in range(10)]
        b = [float(i) + 0.1 for i in range(10)]
        result = paired_comparison(a, b)
        assert result is not None
        assert result.test_name == "paired_t"

    def test_permutation_cross_check_attached(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        b = [0.5, 1.8, 2.4, 4.2, 4.5, 6.1, 6.5]
        result = paired_comparison(a, b)
        assert result is not None
        assert result.permutation_p is not None
        assert result.permutation_method == "exact"
        assert 0.0 <= result.permutation_p <= 1.0

    def test_monte_carlo_cross_check_at_large_n(self):
        a = [float(i) for i in range(20)]
        b = [float(i) + 0.5 for i in range(20)]
        result = paired_comparison(a, b, seed=42)
        assert result is not None
        assert result.permutation_method == "monte_carlo"

    def test_as_dict_includes_permutation(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [0.5, 1.5, 2.5, 3.5, 4.5]
        d = paired_comparison(a, b).as_dict()
        assert "permutation_p" in d
        assert d["permutation_method"] == "exact"

    def test_returns_none_for_insufficient_data(self):
        assert paired_comparison([1.0], [2.0]) is None


# ---------------------------------------------------------------------------
# Judge reliability: ICC
# ---------------------------------------------------------------------------

SHROUT_FLEISS = [
    [9.0, 2.0, 5.0, 8.0],
    [6.0, 1.0, 3.0, 2.0],
    [8.0, 4.0, 6.0, 8.0],
    [7.0, 1.0, 2.0, 6.0],
    [10.0, 5.0, 6.0, 9.0],
    [6.0, 2.0, 4.0, 7.0],
]


class TestICC:
    def test_shrout_fleiss_known_answers(self):
        # Classic Shrout & Fleiss (1979) 6x4 dataset. Published values
        # (pingouin, SPSS): ICC(A,1)=0.29, ICC(A,k)=0.62, CI95 ~[0.02, 0.76].
        from porchbench.statistics import icc_absolute_agreement

        result = icc_absolute_agreement(SHROUT_FLEISS)
        assert result is not None
        assert not result.degenerate
        assert result.icc_single == pytest.approx(0.29, abs=0.005)
        assert result.icc_mean_of_k == pytest.approx(0.62, abs=0.005)
        assert result.ci_lower == pytest.approx(0.02, abs=0.02)
        assert result.ci_upper == pytest.approx(0.76, abs=0.02)
        assert result.n_targets == 6
        assert result.k_raters == 4

    def test_perfect_agreement_with_spread(self):
        from porchbench.statistics import icc_absolute_agreement

        ratings = [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]]
        result = icc_absolute_agreement(ratings)
        assert result is not None
        assert not result.degenerate
        assert result.icc_single == pytest.approx(1.0)
        assert result.icc_mean_of_k == pytest.approx(1.0)

    def test_identical_everything_is_degenerate(self):
        # Deterministic judge on a saturated suite: all 5s. Agreement is
        # perfect but ICC is 0/0 — must flag, not report 1.0 or crash.
        from porchbench.statistics import icc_absolute_agreement

        result = icc_absolute_agreement([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
        assert result is not None
        assert result.degenerate
        assert result.icc_single is None

    def test_restricted_range_not_degenerate(self):
        # Saturated-but-noisy: tiny between-target variance, real error.
        # ICC should come out LOW (a valid finding), not degenerate.
        from porchbench.statistics import icc_absolute_agreement

        ratings = [[4.8, 5.0], [5.0, 4.9], [4.9, 5.0], [5.0, 4.8]]
        result = icc_absolute_agreement(ratings)
        assert result is not None
        assert not result.degenerate
        assert result.icc_single is not None
        assert result.icc_single < 0.5

    def test_requires_two_targets_and_two_raters(self):
        from porchbench.statistics import icc_absolute_agreement

        assert icc_absolute_agreement([[1.0, 2.0]]) is None
        assert icc_absolute_agreement([[1.0], [2.0]]) is None

    def test_ragged_matrix_returns_none(self):
        from porchbench.statistics import icc_absolute_agreement

        assert icc_absolute_agreement([[1.0, 2.0], [3.0]]) is None

    def test_as_dict(self):
        from porchbench.statistics import icc_absolute_agreement

        d = icc_absolute_agreement(SHROUT_FLEISS).as_dict()
        assert "icc_single" in d
        assert "ci_lower" in d
        assert d["n_targets"] == 6


class TestPctWithin:
    def test_all_within(self):
        from porchbench.statistics import pct_within

        assert pct_within([[4.0, 4.5], [3.0, 3.9]], tolerance=1.0) == 1.0

    def test_none_within(self):
        from porchbench.statistics import pct_within

        assert pct_within([[1.0, 5.0]], tolerance=1.0) == 0.0

    def test_mixed(self):
        from porchbench.statistics import pct_within

        # pairs: |4-4.5|=0.5 ok, |4-2|=2 no, |4.5-2|=2.5 no -> 1/3
        assert pct_within([[4.0, 4.5, 2.0]], tolerance=1.0) == pytest.approx(1 / 3)

    def test_empty_returns_none(self):
        from porchbench.statistics import pct_within

        assert pct_within([]) is None
