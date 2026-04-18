"""Statistical inference: confidence intervals, bootstrap, and paired tests.

Pure-Python implementations using only the stdlib (random, math, statistics).
No numpy/scipy dependency — designed for small-sample benchmark data where
n is typically 3-100.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate with confidence bounds."""

    mean: float
    ci_lower: float
    ci_upper: float
    confidence: float  # e.g. 0.95
    method: str  # "bootstrap" or "t"
    n: int

    @property
    def margin(self) -> float:
        return (self.ci_upper - self.ci_lower) / 2

    def as_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "confidence": self.confidence,
            "method": self.method,
            "n": self.n,
        }


# ---------------------------------------------------------------------------
# T-distribution critical values (two-tailed, 95%)
# Precomputed for df=1..30 and df=inf. For df>30 we interpolate toward z=1.96.
# ---------------------------------------------------------------------------

_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

_Z_95 = 1.96  # Normal approximation for large df


def _t_critical(df: int, confidence: float = 0.95) -> float:
    """Lookup t critical value for given degrees of freedom (95% CI only)."""
    if confidence != 0.95:
        # Only 95% CI is precomputed; fall back to z for other levels
        return _Z_95
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    if df > 30:
        # Linear interpolation toward z=1.96
        return _Z_95 + (_T_CRIT_95[30] - _Z_95) * (30 / df)
    return _Z_95


def parametric_ci(
    values: list[float],
    confidence: float = 0.95,
) -> ConfidenceInterval | None:
    """Compute a parametric confidence interval using the t-distribution.

    Requires n >= 2. Returns None for insufficient data.
    """
    n = len(values)
    if n < 2:
        return None

    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    t_crit = _t_critical(n - 1, confidence)

    return ConfidenceInterval(
        mean=mean,
        ci_lower=mean - t_crit * se,
        ci_upper=mean + t_crit * se,
        confidence=confidence,
        method="t",
        n=n,
    )


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = 42,
) -> ConfidenceInterval | None:
    """Compute a bootstrap confidence interval via percentile method.

    Works for any sample size n >= 1, but most useful for small n where
    parametric assumptions may not hold. Uses a fixed seed for reproducibility.
    """
    n = len(values)
    if n == 0:
        return None

    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choices(values, k=n))
        for _ in range(n_resamples)
    )

    alpha = 1 - confidence
    lo_idx = int(math.floor(alpha / 2 * n_resamples))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_resamples)) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))

    return ConfidenceInterval(
        mean=statistics.mean(values),
        ci_lower=means[lo_idx],
        ci_upper=means[hi_idx],
        confidence=confidence,
        method="bootstrap",
        n=n,
    )


def auto_ci(
    values: list[float],
    confidence: float = 0.95,
    bootstrap_threshold: int = 30,
    seed: int | None = 42,
) -> ConfidenceInterval | None:
    """Choose the appropriate CI method based on sample size.

    Uses parametric t-CI for n >= bootstrap_threshold, bootstrap for smaller samples.
    The seed is forwarded to the bootstrap path so the entire statistical output
    remains reproducible under a caller-supplied RNG seed.
    """
    if len(values) < 2:
        return bootstrap_ci(values, confidence, seed=seed) if values else None
    if len(values) >= bootstrap_threshold:
        return parametric_ci(values, confidence)
    return bootstrap_ci(values, confidence, seed=seed)


# ---------------------------------------------------------------------------
# Paired difference analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedTestResult:
    """Result of a paired statistical test comparing two models.

    p_value and significant are None when the sample is too small for a
    reliable p-value (paired-t requires df >= 30 because we approximate
    the t-tail with a z-tail; see _normal_tail_p). The CI on the mean
    difference and the effect size remain valid and are the recommended
    primary evidence for small n.
    """

    test_name: str  # "paired_t" or "wilcoxon"
    n_pairs: int
    mean_difference: float  # model_a - model_b (positive = A better)
    statistic: float
    p_value: float | None
    significant: bool | None  # at alpha=0.05; None when p_value is None
    effect_size: float  # Cohen's dz for paired data
    effect_magnitude: str  # negligible, small, medium, large
    ci: ConfidenceInterval | None  # CI on the mean difference

    def as_dict(self) -> dict:
        d = {
            "test_name": self.test_name,
            "n_pairs": self.n_pairs,
            "mean_difference": round(self.mean_difference, 4),
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "significant": self.significant,
            "effect_size": round(self.effect_size, 4),
            "effect_magnitude": self.effect_magnitude,
        }
        if self.ci:
            d["ci"] = self.ci.as_dict()
        return d


def _cohens_d(differences: list[float]) -> tuple[float, str]:
    """Compute Cohen's d for paired differences and classify magnitude."""
    if len(differences) < 2:
        return 0.0, "negligible"
    mean_d = statistics.mean(differences)
    sd_d = statistics.stdev(differences)
    if sd_d == 0:
        return float("inf") if mean_d != 0 else 0.0, "large" if mean_d != 0 else "negligible"
    d = abs(mean_d) / sd_d
    if d < 0.2:
        magnitude = "negligible"
    elif d < 0.5:
        magnitude = "small"
    elif d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    return d, magnitude


MIN_DF_FOR_T_PVALUE = 30
"""Minimum degrees of freedom before we report a paired-t p-value.

We approximate the t-tail with the standard normal, which is accurate to
within a few percent at df >= 30 and diverges below that (giving smaller
p-values than the true t-distribution). Rather than report a number that
understates p for small n, we return None and let callers surface the CI
and effect size instead.
"""


def paired_t_test(
    values_a: list[float],
    values_b: list[float],
    alpha: float = 0.05,
    seed: int | None = 42,
) -> PairedTestResult | None:
    """Paired t-test for the difference between two matched samples.

    Requires n >= 2 paired observations; returns None if insufficient data.
    p_value and significant are None when df < 30 (see MIN_DF_FOR_T_PVALUE).
    The seed is forwarded to the CI's bootstrap path for reproducibility.
    """
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None

    n = len(values_a)
    differences = [a - b for a, b in zip(values_a, values_b)]
    mean_d = statistics.mean(differences)
    sd_d = statistics.stdev(differences)
    se_d = sd_d / math.sqrt(n)
    df = n - 1

    p_value: float | None
    if se_d == 0:
        t_stat = float("inf") if mean_d != 0 else 0.0
        p_value = 0.0 if mean_d != 0 else 1.0
    elif df < MIN_DF_FOR_T_PVALUE:
        t_stat = mean_d / se_d
        p_value = None
    else:
        t_stat = mean_d / se_d
        p_value = _normal_tail_p(abs(t_stat))

    significant = (p_value < alpha) if p_value is not None else None
    effect_size, magnitude = _cohens_d(differences)
    ci = auto_ci(differences, seed=seed)

    return PairedTestResult(
        test_name="paired_t",
        n_pairs=n,
        mean_difference=mean_d,
        statistic=t_stat,
        p_value=p_value,
        significant=significant,
        effect_size=effect_size,
        effect_magnitude=magnitude,
        ci=ci,
    )


def _normal_tail_p(abs_stat: float) -> float:
    """Two-tailed p-value under the standard normal: 2 * (1 - Phi(|x|)).

    Uses the logistic approximation Phi(x) ~ 1/(1+exp(-1.7x)), which is
    accurate to ~1% across the relevant range. Exact values require scipy.

    Correct for genuinely z-distributed statistics (e.g., the asymptotic
    normal approximation of the Wilcoxon signed-rank W). For t-statistics,
    gate on degrees of freedom before calling (see MIN_DF_FOR_T_PVALUE):
    the normal has lighter tails than the t, so applying this directly to
    a t-stat at small df understates p (anti-conservative).
    """
    p = 2.0 / (1.0 + math.exp(1.7 * abs_stat))
    return min(1.0, max(0.0, p))


def wilcoxon_signed_rank(
    values_a: list[float],
    values_b: list[float],
    alpha: float = 0.05,
    seed: int | None = 42,
) -> PairedTestResult | None:
    """Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test. Requires n >= 6 for
    meaningful results. Returns None if insufficient data. The seed is
    forwarded to the CI's bootstrap path for reproducibility.
    """
    if len(values_a) != len(values_b) or len(values_a) < 6:
        return None

    n = len(values_a)
    differences = [a - b for a, b in zip(values_a, values_b)]

    # Remove zero differences
    nonzero = [(abs(d), d) for d in differences if d != 0]
    if not nonzero:
        return PairedTestResult(
            test_name="wilcoxon",
            n_pairs=n,
            mean_difference=0.0,
            statistic=0.0,
            p_value=1.0,
            significant=False,
            effect_size=0.0,
            effect_magnitude="negligible",
            ci=auto_ci(differences, seed=seed),
        )

    # Rank the absolute differences
    nonzero.sort(key=lambda x: x[0])
    ranks = []
    i = 0
    while i < len(nonzero):
        j = i + 1
        while j < len(nonzero) and nonzero[j][0] == nonzero[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2  # 1-based average rank for ties
        for k in range(i, j):
            ranks.append((avg_rank, nonzero[k][1]))
        i = j

    # Sum of positive and negative ranks
    w_plus = sum(r for r, d in ranks if d > 0)
    w_minus = sum(r for r, d in ranks if d < 0)
    w = min(w_plus, w_minus)
    nr = len(nonzero)

    # Normal approximation for p-value (valid for n >= 10, approximate for n >= 6)
    mean_w = nr * (nr + 1) / 4
    var_w = nr * (nr + 1) * (2 * nr + 1) / 24
    if var_w > 0:
        z = (w - mean_w) / math.sqrt(var_w)
        p_value = _normal_tail_p(abs(z))
    else:
        p_value = 1.0

    effect_size, magnitude = _cohens_d(differences)

    return PairedTestResult(
        test_name="wilcoxon",
        n_pairs=n,
        mean_difference=statistics.mean(differences),
        statistic=w,
        p_value=p_value,
        significant=p_value < alpha,
        effect_size=effect_size,
        effect_magnitude=magnitude,
        ci=auto_ci(differences, seed=seed),
    )


def paired_comparison(
    values_a: list[float],
    values_b: list[float],
    alpha: float = 0.05,
    seed: int | None = 42,
) -> PairedTestResult | None:
    """Run the appropriate paired test based on sample size.

    Uses paired t-test for n >= 2 and Wilcoxon for n >= 6 (prefers Wilcoxon
    for small samples since it doesn't assume normality). The seed is
    forwarded to the CI's bootstrap path for reproducibility.
    """
    n = len(values_a)
    if n != len(values_b) or n < 2:
        return None
    if n >= 6:
        return wilcoxon_signed_rank(values_a, values_b, alpha, seed=seed)
    return paired_t_test(values_a, values_b, alpha, seed=seed)
