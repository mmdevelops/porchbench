"""Statistical inference: confidence intervals, paired tests, and power.

Pure-Python implementations using only the stdlib (random, math, statistics).
No numpy/scipy dependency — designed for small-sample benchmark data where
n is typically 3-100.

Methodology follows Miller, "Adding Error Bars to Evals" (arXiv:2411.00640):
paired analysis on per-prompt differences with a CLT/t-based primary result,
an exact sign-flip permutation test as the assumption-light cross-check, and
repeats averaged within prompt before testing (never pooled — pooling repeats
as independent observations inflates n and manufactures significance).
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, replace
from functools import lru_cache


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate with confidence bounds."""

    mean: float
    ci_lower: float
    ci_upper: float
    confidence: float  # e.g. 0.95
    method: str  # "t" or "bootstrap"
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
# Student's t distribution (exact, via the regularized incomplete beta)
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction evaluation for the incomplete beta (Lentz's method)."""
    max_iterations = 300
    eps = 3e-12
    fpmin = 1e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), accurate to ~1e-10 across the domain."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    front = math.exp(ln_front)
    # Use the continued fraction on whichever side converges fast.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_tailed_p(t_stat: float, df: int) -> float:
    """Exact two-tailed p-value for a t statistic at any df >= 1.

    P(|T| >= |t|) = I_x(df/2, 1/2) with x = df / (df + t^2).
    """
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    t_abs = abs(t_stat)
    if math.isinf(t_abs):
        return 0.0
    x = df / (df + t_abs * t_abs)
    p = _regularized_incomplete_beta(df / 2.0, 0.5, x)
    return min(1.0, max(0.0, p))


@lru_cache(maxsize=256)
def _t_quantile_two_tailed(df: int, alpha: float) -> float:
    """t value whose two-tailed p equals alpha, by bisection (monotone in t)."""
    lo, hi = 0.0, 1.0
    while t_two_tailed_p(hi, df) > alpha:
        hi *= 2.0
        if hi > 1e9:  # pragma: no cover - alpha pathologically small
            break
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_two_tailed_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10 * max(1.0, hi):
            break
    return (lo + hi) / 2.0


def _t_critical(df: int, confidence: float = 0.95) -> float:
    """Exact t critical value for a two-sided CI at the given confidence."""
    return _t_quantile_two_tailed(df, 1.0 - confidence)


def _normal_tail_p(abs_stat: float) -> float:
    """Exact two-tailed p under the standard normal: 2*(1 - Phi(|x|)) = erfc(|x|/sqrt 2)."""
    p = math.erfc(abs(abs_stat) / math.sqrt(2.0))
    return min(1.0, max(0.0, p))


def _f_cdf(f: float, df1: float, df2: float) -> float:
    """CDF of the F distribution: P(F <= f) = I_x(df1/2, df2/2), x = df1*f/(df1*f + df2).

    Accepts non-integer df (needed for Satterthwaite approximations).
    """
    if f <= 0:
        return 0.0
    x = df1 * f / (df1 * f + df2)
    return _regularized_incomplete_beta(df1 / 2.0, df2 / 2.0, x)


def _f_quantile(p: float, df1: float, df2: float) -> float:
    """F value with CDF = p, by bisection (monotone in f)."""
    lo, hi = 0.0, 1.0
    while _f_cdf(hi, df1, df2) < p:
        hi *= 2.0
        if hi > 1e12:  # pragma: no cover - pathological df/p combinations
            break
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _f_cdf(mid, df1, df2) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10 * max(1.0, hi):
            break
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def parametric_ci(
    values: list[float],
    confidence: float = 0.95,
) -> ConfidenceInterval | None:
    """Compute a confidence interval using the t-distribution.

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
    """Percentile-bootstrap confidence interval (explicit opt-in only).

    The percentile method is known to undercover at small n (intervals too
    short for n <= ~34), which is why `auto_ci` no longer routes to it for
    n >= 2 — prefer `parametric_ci`. Retained for callers that want a
    resampling interval anyway, and for the degenerate n=1 point interval.
    Uses a fixed seed for reproducibility.
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
    seed: int | None = 42,
) -> ConfidenceInterval | None:
    """t-CI for n >= 2; degenerate single-value data falls back to bootstrap.

    The t-interval has correct small-sample coverage where the percentile
    bootstrap runs short, so it is the default at every n where it exists.
    """
    if not values:
        return None
    if len(values) < 2:
        return bootstrap_ci(values, confidence, seed=seed)
    return parametric_ci(values, confidence)


# ---------------------------------------------------------------------------
# Repeat handling
# ---------------------------------------------------------------------------


def average_score_repeats(score_maps: list[dict[str, float]]) -> dict[str, float]:
    """Collapse per-repeat score maps into one mean score per prompt.

    Repeats of the same (model, suite) run must be averaged within prompt
    BEFORE any paired test — passing K repeats x N prompts into a test as
    K*N independent observations is pseudoreplication and inflates
    significance. Prompts missing from some repeats average over the
    repeats that scored them.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for score_map in score_maps:
        for prompt_id, value in score_map.items():
            totals[prompt_id] = totals.get(prompt_id, 0.0) + value
            counts[prompt_id] = counts.get(prompt_id, 0) + 1
    return {pid: totals[pid] / counts[pid] for pid in totals}


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


def minimum_detectable_dz(
    n: int,
    power: float = 0.8,
    alpha: float = 0.05,
) -> float | None:
    """Minimum effect size (Cohen's dz) a paired test of n pairs can detect.

    MDE = (t_{alpha/2, df} + t_{power, df}) / sqrt(n). A comparison whose
    observed |dz| is below this cannot be expected to reach significance at
    the given power — "not significant" then means "underpowered", not
    "no difference". Returns None for n < 2.
    """
    if n < 2:
        return None
    df = n - 1
    t_alpha = _t_quantile_two_tailed(df, alpha)
    # One-tailed quantile at `power` == two-tailed quantile at 2*(1-power).
    t_beta = _t_quantile_two_tailed(df, 2.0 * (1.0 - power))
    return (t_alpha + t_beta) / math.sqrt(n)


# ---------------------------------------------------------------------------
# Paired difference analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedTestResult:
    """Result of a paired statistical test comparing two models.

    The primary result is a paired t-test with an exact t-distribution
    p-value (valid at any df >= 1) and a t-CI on the mean difference, so the
    p-value and the CI answer the same question about the same estimand.
    `permutation_p` carries the sign-flip permutation cross-check when
    produced via `paired_comparison`.
    """

    test_name: str  # "paired_t" or "wilcoxon"
    n_pairs: int
    mean_difference: float  # model_a - model_b (positive = A better)
    statistic: float
    p_value: float | None
    significant: bool | None  # at alpha=0.05
    effect_size: float  # Cohen's dz for paired data
    effect_magnitude: str  # negligible, small, medium, large
    ci: ConfidenceInterval | None  # CI on the mean difference
    permutation_p: float | None = None  # sign-flip cross-check (paired_comparison)
    permutation_method: str | None = None  # "exact" or "monte_carlo"

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
        if self.permutation_p is not None:
            d["permutation_p"] = round(self.permutation_p, 6)
            d["permutation_method"] = self.permutation_method
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


def paired_t_test(
    values_a: list[float],
    values_b: list[float],
    alpha: float = 0.05,
    seed: int | None = 42,
) -> PairedTestResult | None:
    """Paired t-test for the difference between two matched samples.

    Requires n >= 2 paired observations; returns None if insufficient data.
    The p-value uses the exact t-distribution (regularized incomplete beta),
    valid at any df >= 1. The CI is a t-interval on the differences, so the
    reported p and CI share the mean-difference estimand.
    """
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None

    n = len(values_a)
    differences = [a - b for a, b in zip(values_a, values_b)]
    mean_d = statistics.mean(differences)
    sd_d = statistics.stdev(differences)
    se_d = sd_d / math.sqrt(n)
    df = n - 1

    if se_d == 0:
        t_stat = float("inf") if mean_d != 0 else 0.0
        p_value = 0.0 if mean_d != 0 else 1.0
    else:
        t_stat = mean_d / se_d
        p_value = t_two_tailed_p(t_stat, df)

    significant = p_value < alpha
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


@dataclass(frozen=True)
class PermutationResult:
    """Two-sided sign-flip permutation test on the mean of paired differences."""

    p_value: float
    method: str  # "exact" (full enumeration) or "monte_carlo"
    n_pairs: int


# Full enumeration is 2^n sums; 2^15 = 32768 stays instant in pure Python.
EXACT_PERMUTATION_MAX_N = 15


def permutation_test_paired(
    values_a: list[float],
    values_b: list[float],
    seed: int | None = 42,
    n_resamples: int = 10_000,
) -> PermutationResult | None:
    """Exact paired (sign-flip) permutation test, two-sided on the mean.

    Under the null the sign of each paired difference is arbitrary, so the
    reference distribution is the mean under all 2^n sign assignments —
    enumerated exactly for n <= EXACT_PERMUTATION_MAX_N, seeded Monte Carlo
    above. Assumption-light (no normality, no symmetry-of-ranks) and handles
    ties/zeros natively, which suits discrete bounded judge scores. Requires
    n >= 2; returns None otherwise.
    """
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None

    n = len(values_a)
    differences = [a - b for a, b in zip(values_a, values_b)]
    observed = abs(sum(differences))
    # Tolerance so float round-trip noise can't exclude sums that are
    # mathematically equal to the observed one (exactness requires >=).
    tol = 1e-9 * max(1.0, observed)

    if n <= EXACT_PERMUTATION_MAX_N:
        sums = [0.0]
        for d in differences:
            sums = [s + d for s in sums] + [s - d for s in sums]
        extreme = sum(1 for s in sums if abs(s) >= observed - tol)
        return PermutationResult(
            p_value=extreme / len(sums), method="exact", n_pairs=n
        )

    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_resamples):
        flipped = sum(d if rng.random() < 0.5 else -d for d in differences)
        if abs(flipped) >= observed - tol:
            extreme += 1
    # +1 correction: the observed assignment is itself a member of the
    # permutation distribution, and it keeps a Monte-Carlo p from being 0.
    return PermutationResult(
        p_value=(extreme + 1) / (n_resamples + 1), method="monte_carlo", n_pairs=n
    )


def wilcoxon_signed_rank(
    values_a: list[float],
    values_b: list[float],
    alpha: float = 0.05,
    seed: int | None = 42,
) -> PairedTestResult | None:
    """Wilcoxon signed-rank test for paired samples (legacy).

    No longer routed by `paired_comparison`: the normal approximation used
    here is a large-sample tool that is discouraged at n < 25 with tie-heavy
    discrete data (no tie-variance or continuity correction, zeros dropped),
    which is exactly the judge-score regime. `permutation_test_paired` is the
    assumption-light replacement. Retained for callers that want it;
    removal is an open question in PRELIM-coding-discrimination-suites.
    Requires n >= 6; returns None if insufficient data.
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

    # Normal approximation for p-value (see docstring caveat)
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


# ---------------------------------------------------------------------------
# Judge reliability: intraclass correlation (two-way, absolute agreement)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ICCResult:
    """Intraclass correlation for an n-targets x k-raters ratings matrix.

    Computed as ICC(A,1) / ICC(A,k) in McGraw-Wong terms: two-way model,
    absolute agreement. For judge reliability the "raters" are repeated
    judge samples (varied seeds), treated as random draws from the seed
    population; targets are prompts. Interpret per Koo & Li (2016) against
    the 95% CI, not the point estimate: <0.5 poor, 0.5-0.75 moderate,
    0.75-0.9 good, >0.9 excellent.

    `degenerate` is True when the matrix has ~zero total variance (e.g. a
    deterministic judge repeating identical scores): agreement is perfect
    but ICC is undefined, so icc fields are None. A LOW icc with real
    error variance on a compressed scale is NOT degenerate — that pattern
    means "the suite isn't discriminating", which companions like
    `pct_within` disambiguate from a noisy judge.
    """

    icc_single: float | None  # ICC(A,1): reliability of one judge sample
    icc_mean_of_k: float | None  # ICC(A,k): reliability of the shipped mean-of-k
    ci_lower: float | None  # 95% CI on ICC(A,1) (McGraw-Wong F-based)
    ci_upper: float | None
    confidence: float
    n_targets: int
    k_raters: int
    degenerate: bool

    def as_dict(self) -> dict:
        return {
            "icc_single": round(self.icc_single, 4) if self.icc_single is not None else None,
            "icc_mean_of_k": (
                round(self.icc_mean_of_k, 4) if self.icc_mean_of_k is not None else None
            ),
            "ci_lower": round(self.ci_lower, 4) if self.ci_lower is not None else None,
            "ci_upper": round(self.ci_upper, 4) if self.ci_upper is not None else None,
            "confidence": self.confidence,
            "n_targets": self.n_targets,
            "k_raters": self.k_raters,
            "degenerate": self.degenerate,
        }


def icc_absolute_agreement(
    ratings: list[list[float]],
    confidence: float = 0.95,
) -> ICCResult | None:
    """ICC(A,1) and ICC(A,k) with an F-based CI, from an n x k ratings matrix.

    `ratings[i][j]` is rater j's score for target i. Requires n >= 2 targets
    and k >= 2 raters with a complete (rectangular) matrix; returns None
    otherwise. Mean-squares decomposition per McGraw & Wong (1996); the CI
    uses their Satterthwaite-approximated F bounds (the same construction
    Koo & Li 2016 prescribe interpreting against).
    """
    n = len(ratings)
    if n < 2:
        return None
    k = len(ratings[0])
    if k < 2 or any(len(row) != k for row in ratings):
        return None

    grand = statistics.mean(v for row in ratings for v in row)
    row_means = [statistics.mean(row) for row in ratings]
    col_means = [statistics.mean(row[j] for row in ratings) for j in range(k)]

    ss_total = sum((v - grand) ** 2 for row in ratings for v in row)
    ss_rows = k * sum((m - grand) ** 2 for m in row_means)
    ss_cols = n * sum((m - grand) ** 2 for m in col_means)
    ss_err = max(0.0, ss_total - ss_rows - ss_cols)

    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))

    # Zero total variance: identical scores everywhere. Perfect agreement,
    # but between-target variance is zero so ICC is 0/0 — undefined.
    if ss_total < 1e-12:
        return ICCResult(
            icc_single=None,
            icc_mean_of_k=None,
            ci_lower=None,
            ci_upper=None,
            confidence=confidence,
            n_targets=n,
            k_raters=k,
            degenerate=True,
        )

    denom_single = msr + (k - 1) * mse + (k / n) * (msc - mse)
    denom_mean = msr + (msc - mse) / n
    if abs(denom_single) < 1e-12 or abs(denom_mean) < 1e-12:
        return ICCResult(
            icc_single=None,
            icc_mean_of_k=None,
            ci_lower=None,
            ci_upper=None,
            confidence=confidence,
            n_targets=n,
            k_raters=k,
            degenerate=True,
        )

    icc_a1 = (msr - mse) / denom_single
    icc_ak = (msr - mse) / denom_mean

    # McGraw-Wong F-based CI on ICC(A,1), Satterthwaite df for the
    # denominator. Degenerates (mse == 0 with real row variance -> perfect
    # agreement) get a point CI at the estimate.
    ci_lower: float | None
    ci_upper: float | None
    if mse < 1e-12:
        ci_lower, ci_upper = icc_a1, icc_a1
    else:
        alpha = 1.0 - confidence
        a = (k * icc_a1) / (n * (1.0 - icc_a1)) if icc_a1 < 1.0 else float("inf")
        if math.isinf(a):
            ci_lower, ci_upper = icc_a1, icc_a1
        else:
            b = 1.0 + (k * icc_a1 * (n - 1)) / (n * (1.0 - icc_a1))
            num = (a * msc + b * mse) ** 2
            den = (a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1))
            v = num / den if den > 0 else float(n - 1)
            f_lower = _f_quantile(1.0 - alpha / 2.0, n - 1, v)
            f_upper = _f_quantile(1.0 - alpha / 2.0, v, n - 1)
            ci_lower = (
                n * (msr - f_lower * mse)
                / (f_lower * (k * msc + (k * n - k - n) * mse) + n * msr)
            )
            ci_upper = (
                n * (f_upper * msr - mse)
                / (k * msc + (k * n - k - n) * mse + n * f_upper * msr)
            )

    return ICCResult(
        icc_single=icc_a1,
        icc_mean_of_k=icc_ak,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence=confidence,
        n_targets=n,
        k_raters=k,
        degenerate=False,
    )


def pct_within(ratings: list[list[float]], tolerance: float = 1.0) -> float | None:
    """Fraction of rating pairs (per target) within `tolerance` of each other.

    Variance-independent agreement companion to ICC: restricted range
    (a saturated suite) crashes ICC even when repeated scores agree within
    rounding, so report both — high pct_within with low ICC reads
    "compressed scale", low pct_within reads "noisy judge".
    """
    n = len(ratings)
    if n == 0:
        return None
    total_pairs = 0
    within = 0
    for row in ratings:
        k = len(row)
        for i in range(k):
            for j in range(i + 1, k):
                total_pairs += 1
                if abs(row[i] - row[j]) <= tolerance:
                    within += 1
    if total_pairs == 0:
        return None
    return within / total_pairs


def paired_comparison(
    values_a: list[float],
    values_b: list[float],
    alpha: float = 0.05,
    seed: int | None = 42,
) -> PairedTestResult | None:
    """Primary paired analysis: exact paired t, permutation cross-check.

    The paired t-test (exact t p-value, t-CI on the mean difference) is the
    primary result at every n >= 2; the sign-flip permutation p rides along
    in `permutation_p` as the assumption-light cross-check. If the two
    disagree at the margin, trust the permutation p for existence and read
    the CI for magnitude. Inputs must already be one value per prompt —
    average run-repeats first (see `average_score_repeats`).
    """
    n = len(values_a)
    if n != len(values_b) or n < 2:
        return None
    result = paired_t_test(values_a, values_b, alpha, seed=seed)
    if result is None:
        return None
    perm = permutation_test_paired(values_a, values_b, seed=seed)
    if perm is not None:
        result = replace(
            result, permutation_p=perm.p_value, permutation_method=perm.method
        )
    return result
