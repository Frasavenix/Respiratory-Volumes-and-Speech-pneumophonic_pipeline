"""
Effect Size Utilities (M2 — Stratified L3 add-on)
==================================================

Standardized effect-size estimators for paired-difference comparisons,
designed for the L3 FRC-crossing analysis but applicable to any paired
setting (above vs. below FRC, baseline vs. task, etc.).

Motivation
----------
The L3 analysis currently reports Wilcoxon signed-rank p-values, which
quantify the population-level reliability of the shift sign but not its
size relative to its own dispersion. With N = 64 segments, a p < 10^-8
result can correspond to a per-segment effect that is small relative to
the standard deviation of the shifts. To address this, we report:

  * Cohen's d (paired)         = mean(diff) / sd(diff)
  * Hedges' g                  = small-sample-corrected Cohen's d
  * Robust d                   = median(diff) / MAD(diff)   (matches our
                                 Wilcoxon non-parametric paradigm)
  * Wilcoxon r                 = |Z| / sqrt(N)              (non-parametric
                                 effect-size analogue)
  * Bootstrap 95% CIs for the median and Cohen's d
  * Sign-consistency ratio     = fraction of segments whose shift sign
                                 matches the median's sign

Companion to: scripts/analyze_l3_stratified.py
"""

from dataclasses import dataclass, asdict
from typing import Callable, Optional, Tuple
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class PairedEffectSize:
    """
    Full set of effect-size statistics for a paired-difference distribution.

    Convention: diffs = below - above. Positive median = feature increases
    going from above-FRC to below-FRC.
    """
    n: int
    mean_diff: float
    median_diff: float
    sd_diff: float
    mad_diff: float
    cohen_d: float
    hedges_g: float
    robust_d: float
    wilcoxon_p: float
    wilcoxon_r: float
    sign_consistency: float
    median_ci: Tuple[float, float]
    cohen_d_ci: Tuple[float, float]

    def to_dict(self) -> dict:
        d = asdict(self)
        d['median_ci_lo'], d['median_ci_hi'] = d.pop('median_ci')
        d['cohen_d_ci_lo'], d['cohen_d_ci_hi'] = d.pop('cohen_d_ci')
        return d


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def median_absolute_deviation(x: np.ndarray, scale: float = 1.4826) -> float:
    """
    Median Absolute Deviation, scaled by 1.4826 to be a consistent
    estimator of sigma for normal data.
    """
    x = np.asarray(x, dtype=float)
    return float(scale * np.median(np.abs(x - np.median(x))))


def _bootstrap_ci(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Percentile bootstrap confidence interval for an arbitrary statistic.
    """
    rng = rng if rng is not None else np.random.default_rng(seed=0)
    n = len(x)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = statistic(x[idx])
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return float('nan'), float('nan')
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def compute_paired_effect_size(
    above: np.ndarray,
    below: np.ndarray,
    n_boot: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> PairedEffectSize:
    """
    Compute the full paired effect-size summary for one feature.

    Args:
        above:   per-segment feature value measured above FRC, length N
        below:   per-segment feature value measured below FRC, length N
        n_boot:  bootstrap replicates for confidence intervals
        rng:     numpy Generator for reproducibility

    Returns:
        PairedEffectSize with all summary statistics. NaN-handling: any
        pair containing NaN in either side is dropped before computation.
    """
    above = np.asarray(above, dtype=float)
    below = np.asarray(below, dtype=float)
    if above.shape != below.shape:
        raise ValueError(f"Shape mismatch: above {above.shape} vs below {below.shape}")

    diffs = below - above
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 3:
        raise ValueError(f"Need at least 3 paired observations, got {n}")

    mean_d = float(np.mean(diffs))
    median_d = float(np.median(diffs))
    sd_d = float(np.std(diffs, ddof=1))
    mad_d = median_absolute_deviation(diffs)

    # Parametric effect sizes
    cohen_d = (mean_d / sd_d) if sd_d > 0 else float('nan')
    correction = 1.0 - 3.0 / (4 * n - 5) if n > 1 else 1.0   # Hedges correction
    hedges_g = cohen_d * correction
    # Non-parametric effect size (matches Wilcoxon framing)
    robust_d = (median_d / mad_d) if mad_d > 0 else float('nan')

    # Wilcoxon signed-rank (already used in current L3) + r effect size
    try:
        _, w_p = stats.wilcoxon(diffs, zero_method='wilcox', alternative='two-sided')
        if w_p > 0 and w_p < 1:
            # Convert two-sided p back to |Z| via inverse normal — large-N approx.
            z_abs = float(stats.norm.isf(w_p / 2))
            wilcoxon_r = z_abs / np.sqrt(n)
        else:
            wilcoxon_r = float('nan')
    except ValueError:
        # All-zero differences — Wilcoxon undefined
        w_p = float('nan')
        wilcoxon_r = float('nan')

    # Sign consistency: fraction of obs sharing sign with median
    if median_d != 0:
        sign_match = float(np.sum(np.sign(diffs) == np.sign(median_d)) / n)
    else:
        sign_match = 0.5

    # Bootstrap CIs
    rng = rng if rng is not None else np.random.default_rng(seed=0)
    median_ci = _bootstrap_ci(diffs, np.median, n_boot=n_boot, rng=rng)

    def _cohen_stat(arr: np.ndarray) -> float:
        s = np.std(arr, ddof=1)
        return (np.mean(arr) / s) if s > 0 else float('nan')

    cohen_ci = _bootstrap_ci(diffs, _cohen_stat, n_boot=n_boot, rng=rng)

    return PairedEffectSize(
        n=n,
        mean_diff=mean_d,
        median_diff=median_d,
        sd_diff=sd_d,
        mad_diff=mad_d,
        cohen_d=cohen_d,
        hedges_g=hedges_g,
        robust_d=robust_d,
        wilcoxon_p=float(w_p),
        wilcoxon_r=wilcoxon_r,
        sign_consistency=sign_match,
        median_ci=median_ci,
        cohen_d_ci=cohen_ci,
    )


def interpret_cohen_d(d: float) -> str:
    """
    Cohen's qualitative bins for |d|:
        <0.2  negligible   (the 'shift smaller than its SD' regime)
        <0.5  small
        <0.8  medium
        >=0.8 large
    """
    if d is None or np.isnan(d):
        return "n/a"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"