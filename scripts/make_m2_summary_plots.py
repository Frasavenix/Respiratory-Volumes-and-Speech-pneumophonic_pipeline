#!/usr/bin/env python3
"""
M2 Summary Plots — Demographic Story (Feedback #1, visualization layer)
========================================================================

Generates two summary plots that visualize the demographic structure
surfaced by analyze_l3_stratified.py:

  1) cohen_d_comparison.pdf
        Grouped bar chart of Cohen's d for F0 and %RC shifts across the
        9 strata (All / Male / Female / Young / Elder / YM / YF / EM / EF),
        with Cohen's qualitative bins as horizontal reference lines.
        Reads from: frc_stratified_summary.xlsx

  2) orthogonality_f0_vs_pctrc.pdf
        Scatter plot of per-segment F0 shift (x) vs %RC shift (y),
        coloured by demographic group, with reference lines at zero on
        both axes. Each point is one sustained-phonation segment.
        Reads from: frc_per_segment.csv

Both inputs are produced automatically by analyze_l3_stratified.py.

Usage:
    python scripts/make_m2_summary_plots.py --results-dir data_target/healthy_subjects/M2_stratified
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Consistent ordering and palette for demographic groups
STRATA_ORDER = ['All', 'Male', 'Female', 'Young', 'Elder', 'YM', 'YF', 'EM', 'EF']
DEMO_COLORS = {
    'YM': '#1f77b4',   # blue
    'YF': '#ff7f0e',   # orange
    'EM': '#2ca02c',   # green
    'EF': '#d62728',   # red
}


# ---------------------------------------------------------------------------
# Plot 1: Cohen's d comparison across strata
# ---------------------------------------------------------------------------

def plot_cohen_d_comparison(summary_xlsx: Path, output_path: Path) -> None:
    """
    Grouped bar chart: Cohen's d for F0 and %RC shifts across 9 strata.
    Error bars are bootstrap 95% CIs (already in the summary file).
    """
    df = pd.read_excel(summary_xlsx)

    def _aligned(feature: str) -> pd.DataFrame:
        sub = df[df['feature'] == feature].set_index('stratum')
        return sub.reindex(STRATA_ORDER)

    f0 = _aligned('f0')
    rc = _aligned('pct_rc')

    x = np.arange(len(STRATA_ORDER))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))

    # Asymmetric error bars from the bootstrap CI
    def _yerr(sub: pd.DataFrame) -> np.ndarray:
        return np.vstack([
            sub['cohen_d'] - sub['cohen_d_ci_lo'],
            sub['cohen_d_ci_hi'] - sub['cohen_d'],
        ])

    ax.bar(x - width/2, f0['cohen_d'], width, yerr=_yerr(f0),
           label='F0 shift', color='#4c72b0', alpha=0.88,
           capsize=3.5, error_kw={'lw': 1.0})
    ax.bar(x + width/2, rc['cohen_d'], width, yerr=_yerr(rc),
           label='%RC shift', color='#dd8452', alpha=0.88,
           capsize=3.5, error_kw={'lw': 1.0})

    # Cohen's qualitative bins
    for d_val, label in [(0.2, 'small'), (0.5, 'medium'), (0.8, 'large')]:
        ax.axhline(d_val, color='gray', linestyle=':', alpha=0.55, linewidth=0.9)
        ax.text(len(x) - 0.4, d_val + 0.015, label, fontsize=8.5,
                color='dimgray', ha='left', va='bottom')

    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(STRATA_ORDER)
    ax.set_ylabel("Cohen's d  (shift below − above FRC)")
    ax.set_title(
        "Standardized effect size by demographic stratum\n"
        "F0 shift is age-modulated; %RC shift is sex-modulated",
        fontsize=11.5,
    )
    ax.legend(loc='upper left', frameon=False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # Annotate n_segments under each tick label
    n_text = [f"n={int(f0.loc[s, 'n_segments'])}" if not pd.isna(f0.loc[s, 'n_segments'])
              else "" for s in STRATA_ORDER]
    for xi, txt in zip(x, n_text):
        ax.text(xi, ax.get_ylim()[0] - 0.06, txt,
                ha='center', va='top', fontsize=8, color='dimgray',
                transform=ax.transData)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: F0 shift vs %RC shift orthogonality scatter
# ---------------------------------------------------------------------------

def plot_orthogonality_scatter(per_segment_csv: Path, output_path: Path) -> None:
    """
    Each point is one sustained-phonation segment.
    x: F0 shift (below − above FRC), Hz
    y: %RC shift (below − above FRC), percentage points
    Colour: demographic group (YM/YF/EM/EF).
    """
    df = pd.read_csv(per_segment_csv)
    df = df.dropna(subset=['demographic'])
    df['f0_shift'] = df['f0_below'] - df['f0_above']
    df['pct_rc_shift_pp'] = (df['pct_rc_below'] - df['pct_rc_above']) * 100
    df = df.dropna(subset=['f0_shift', 'pct_rc_shift_pp'])

    fig, ax = plt.subplots(figsize=(9, 7))

    for demo in ['YM', 'YF', 'EM', 'EF']:
        sub = df[df['demographic'] == demo]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub['f0_shift'], sub['pct_rc_shift_pp'],
            c=DEMO_COLORS[demo], label=f'{demo} (n={len(sub)})',
            alpha=0.78, s=70, edgecolor='black', linewidth=0.5,
        )

    # Per-group centroids — make the demographic axes visually obvious
    for demo in ['YM', 'YF', 'EM', 'EF']:
        sub = df[df['demographic'] == demo]
        if len(sub) < 3:
            continue
        cx = sub['f0_shift'].median()
        cy = sub['pct_rc_shift_pp'].median()
        ax.plot(cx, cy, marker='X', markersize=14, color=DEMO_COLORS[demo],
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)

    # Reference lines at zero on each axis (FRC dissociation = both > 0)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.6, linewidth=1)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1] * 0.97, ylim[1] * 0.97,
            'F0 ↑  &  %RC ↑\n(expected FRC dissociation)',
            ha='right', va='top', fontsize=9, color='dimgray',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='lightgray', alpha=0.85))

    ax.set_xlabel('F0 shift below FRC  (Hz)')
    ax.set_ylabel('%RC shift below FRC  (percentage points)')
    ax.set_title(
        "FRC dissociation: F0 shift vs %RC shift by demographic\n"
        "(each point = one sustained-phonation segment; X = group median)",
        fontsize=11.5,
    )
    ax.legend(title='Demographic', loc='upper left', frameon=True)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="M2 summary plots")
    parser.add_argument('--results-dir', type=Path, required=True,
                        help='Folder containing frc_per_segment.csv and '
                             'frc_stratified_summary.xlsx (output of '
                             'analyze_l3_stratified.py).')
    args = parser.parse_args()

    csv_path = args.results_dir / 'frc_per_segment.csv'
    xlsx_path = args.results_dir / 'frc_stratified_summary.xlsx'

    for p in (csv_path, xlsx_path):
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    out_cohen = args.results_dir / 'cohen_d_comparison.pdf'
    out_ortho = args.results_dir / 'orthogonality_f0_vs_pctrc.pdf'

    plot_cohen_d_comparison(xlsx_path, out_cohen)
    print(f"  Wrote {out_cohen}")
    plot_orthogonality_scatter(csv_path, out_ortho)
    print(f"  Wrote {out_ortho}")


if __name__ == '__main__':
    main()