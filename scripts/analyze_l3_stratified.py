#!/usr/bin/env python3
"""
L3 Stratified Analysis: FRC Crossing with Effect Sizes by Demographic Subgroup
==============================================================================

Extends the L3 FRC-crossing analysis (currently producing Figure 3.8 / Table 3.1)
in two directions, both requested by the post-call feedback:

  1) Standardized effect-size statistics — Cohen's d, robust d, Wilcoxon r,
     bootstrap 95% CIs, sign-consistency. Answers the "is the shift smaller
     than its SD?" question explicitly.

  2) Demographic stratification — same analysis on Young Male, Young Female,
     Elder Male, Elder Female cells, plus aggregates Young/Elder and M/F.
     Mirrors Zocco's M/F and U55/O55 splits for methodological continuity.

Inputs
------
- Paired HDF5 files (output of paired_features.PairedFeatureExtractor.extract_batch).
- Subject metadata file (Excel or CSV) with at minimum:
    * subject_id   — must match the subject IDs used in HDF5 filenames/attrs
    * sex          — 'M' or 'F'
    * age          — integer years

Outputs (written under --output-dir)
------------------------------------
- frc_per_segment.csv          : per-segment above/below feature values
                                 with demographic columns attached
- frc_stratified_summary.xlsx  : full effect-size table, one row per
                                 (feature × stratum)
- forest_<feature>.pdf         : forest plot of median shift + bootstrap CI
                                 across strata, with d annotated
- hist_strat_<feature>.pdf     : 2x2 small-multiple histograms of the per-segment
                                 shift, one panel per YM/YF/EM/EF subgroup

Usage
-----
    python scripts/analyze_l3_stratified.py \\
        --paired-dir /path/to/paired_hdf5 \\
        --metadata   /path/to/subjects_metadata.xlsx \\
        --output-dir /path/to/results/M2_stratified \\
        --age-threshold 55

The age threshold defaults to 55 to match Zocco's Under-55 / Over-55 split.

Author: M2 stratified analysis extension (Feedback #1)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow running this script directly from anywhere in the repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pneumophonic_analysis.paired_features import PairedFeatureExtractor
from pneumophonic_analysis.effect_size import (
    compute_paired_effect_size,
    interpret_cohen_d,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)


# Tasks in which the FRC crossing is meaningful (sustained phonation only).
# Matches Zocco's AB/BE analysis tasks and the existing L3.
FRC_TASKS = ['a_2', 'a_3', 'a_7']

# Features compared above vs. below FRC. Matches the existing L3 pipeline.
FEATURES = ['f0', 'energy', 'pct_rc', 'flow_cw']

UNITS = {'f0': 'Hz', 'energy': 'rms', 'pct_rc': 'fraction', 'flow_cw': 'L/s'}


# ---------------------------------------------------------------------------
# FRC splitting
# ---------------------------------------------------------------------------

def find_frc_crossing_index(
    delta_vcw: np.ndarray,
    fallback_to_midpoint: bool = True,
    end_margin: int = 20,
) -> Optional[int]:
    """
    Find the FRC crossing in a sustained-phonation segment.

    Algorithm (matches scripts/m2_correlation.py — `run_frc_analysis`):

      1. Locate the peak of delta_vcw (argmax).
      2. Search for the first descending zero crossing AFTER the peak —
         a strict positive-to-non-positive transition. If found, that index
         is the FRC crossing.
      3. FALLBACK: if no such crossing exists (e.g. segment is already
         monotonically decreasing — typical of a_7 vocal-glide tasks where
         the auto-trim has eaten the inspiratory baseline), split at the
         MIDPOINT of the post-peak region. The fallback is rejected only
         if it would land within `end_margin` frames of the segment end.

    The midpoint fallback is what kept ~64 segments in the original L3.
    Without it, segments with `peak_idx = 0` and monotonic decrease are
    all dropped, leaving only ~30 valid splits.

    Args:
        delta_vcw:            Volume relative to segment onset, shape (n_frames,).
        fallback_to_midpoint: If True (default), use the midpoint fallback
                              when no descending zero crossing is found.
        end_margin:           Reject fallback splits within this many frames
                              of the segment end. Matches m2_correlation.py
                              (n_frames - 20).

    Returns:
        Sample index of the FRC crossing, or None if no valid split is
        possible (segment too short, or fallback rejected).
    """
    if len(delta_vcw) < 2:
        return None

    peak_idx = int(np.argmax(delta_vcw))
    post_peak = delta_vcw[peak_idx:]

    # Step 2: descending zero crossing after the peak (strict positive → ≤0)
    if len(post_peak) >= 2:
        sign = np.sign(post_peak)
        descending = (sign[:-1] > 0) & (sign[1:] <= 0)
        crossings = np.where(descending)[0]
        if len(crossings) > 0:
            return int(crossings[0]) + peak_idx + 1

    # Step 3: fallback to midpoint of post-peak region
    if not fallback_to_midpoint:
        return None
    cross_idx = peak_idx + len(post_peak) // 2
    if cross_idx >= len(delta_vcw) - end_margin:
        return None
    return int(cross_idx)


def split_segment_at_frc(
    df: pd.DataFrame,
    min_frames_each_side: int = 20,   # ~0.30 s at 66 Hz, matches m2_correlation.py
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split a paired-feature DataFrame at the FRC crossing, returning the
    above-FRC and below-FRC portions. Returns None if the split is not
    valid (no crossing or too-short side).
    """
    if 'delta_vcw' not in df.columns:
        return None
    cross_idx = find_frc_crossing_index(df['delta_vcw'].values)
    if cross_idx is None:
        return None
    above = df.iloc[:cross_idx]
    below = df.iloc[cross_idx:]
    if len(above) < min_frames_each_side or len(below) < min_frames_each_side:
        return None
    return above, below


def compute_segment_features(
    above: pd.DataFrame,
    below: pd.DataFrame,
    min_voiced_frames: int = 10,
) -> Optional[Dict[str, float]]:
    """
    Reduce above/below DataFrames to the per-segment summary used by L3.

    Convention (matches scripts/m2_correlation.py — `run_frc_analysis`):
      * F0 and energy   → averaged over VOICED frames only
                          (voiced flag == 1.0, strict equality)
      * flow_cw, pct_rc → averaged over ALL frames

    Returns None if either side has fewer than `min_voiced_frames` voiced
    frames — the same 10-frame guard the original uses.
    """
    above_v = above[above['voiced'] == 1.0]
    below_v = below[below['voiced'] == 1.0]
    if len(above_v) < min_voiced_frames or len(below_v) < min_voiced_frames:
        return None

    return {
        'f0_above':      float(np.nanmean(above_v['f0'])),
        'f0_below':      float(np.nanmean(below_v['f0'])),
        'energy_above':  float(above_v['energy'].mean()),
        'energy_below':  float(below_v['energy'].mean()),
        'flow_cw_above': float(above['flow_cw'].mean()),
        'flow_cw_below': float(below['flow_cw'].mean()),
        'pct_rc_above':  float(above['pct_rc'].mean()),
        'pct_rc_below':  float(below['pct_rc'].mean()),
    }


# ---------------------------------------------------------------------------
# Metadata + demographic assignment
# ---------------------------------------------------------------------------

def _detect_csv_separator(path: Path) -> str:
    """
    Detect the most likely CSV separator by inspecting the first non-empty
    line. Returns ';' or ',' or '\\t'; defaults to ',' when ambiguous.
    """
    with open(path, 'r', encoding='utf-8-sig') as f:
        first_line = ''
        for line in f:
            if line.strip():
                first_line = line
                break
    counts = {sep: first_line.count(sep) for sep in [';', ',', '\t']}
    best = max(counts, key=counts.get)
    return best if counts[best] >= 2 else ','


def load_metadata(path: Path) -> pd.DataFrame:
    """
    Load subject metadata. Handles the project's actual CSV format:

        ID;Date of birth ;Date of acquisition;Age;Gender;;;;;
        SubjectID;05/01/1996;27/11/2025;29;M;;;;;
        ...

    Specifically copes with: semicolon separator, trailing empty columns,
    trailing whitespace in headers, and the project's 'M'/'W' (Man/Woman)
    Gender convention which is normalized to 'M'/'F' to match the analysis
    machinery.

    Excel files (.xlsx, .xls) are loaded with pandas defaults.

    Output columns (normalized): subject_id, sex ('M'/'F'), age (numeric).
    Extra columns are kept and passed through.
    """
    suffix = path.suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        meta = pd.read_excel(path)
    elif suffix == '.csv':
        sep = _detect_csv_separator(path)
        meta = pd.read_csv(path, sep=sep, engine='python', encoding='utf-8-sig')
    else:
        raise ValueError(f"Unsupported metadata extension: {suffix}")

    # Headers: strip whitespace, lowercase
    meta.columns = [str(c).strip().lower() for c in meta.columns]

    # Drop trailing empty columns (the ';;;;;' tail in semicolon CSVs)
    meta = meta.dropna(axis=1, how='all')

    # Map project-specific column names -> analysis-internal names
    rename_map = {'id': 'subject_id', 'gender': 'sex'}
    meta = meta.rename(columns={
        k: v for k, v in rename_map.items()
        if k in meta.columns and v not in meta.columns
    })

    required = {'subject_id', 'sex', 'age'}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(
            f"Metadata file {path} missing required columns after rename: "
            f"{missing}. Found columns: {list(meta.columns)}"
        )

    # Values
    meta['subject_id'] = meta['subject_id'].astype(str).str.strip()
    meta['sex'] = meta['sex'].astype(str).str.upper().str.strip().str[0]
    # Project uses 'W' (Woman); map to 'F' so downstream stratification works.
    # 'D' (Donna, Italian) included as a defensive map.
    meta['sex'] = meta['sex'].replace({'W': 'F', 'D': 'F'})
    meta['age'] = pd.to_numeric(meta['age'], errors='coerce')

    # Drop rows missing essential fields and warn
    n_before = len(meta)
    meta = meta.dropna(subset=['subject_id', 'sex', 'age'])
    n_dropped = n_before - len(meta)
    if n_dropped > 0:
        logger.warning(
            f"Dropped {n_dropped} metadata row(s) with missing "
            f"subject_id/sex/age."
        )

    # Validate sex domain
    bad_sex = set(meta['sex'].unique()) - {'M', 'F'}
    if bad_sex:
        logger.warning(
            f"Unexpected sex values after normalization: {bad_sex}. "
            f"Expected 'M' or 'F' (incoming 'W' should map to 'F')."
        )

    return meta.reset_index(drop=True)


def assign_demographic_group(df: pd.DataFrame, age_threshold: int = 55) -> pd.DataFrame:
    """
    Attach age_group and demographic (YM/YF/EM/EF) columns to a long-format DF.
    """
    df = df.copy()
    df['age_group'] = np.where(df['age'] >= age_threshold, 'Elder', 'Young')
    label_map = {('M', 'Young'): 'YM', ('M', 'Elder'): 'EM',
                 ('F', 'Young'): 'YF', ('F', 'Elder'): 'EF'}
    df['demographic'] = [
        label_map.get((s, g), 'Unknown')
        for s, g in zip(df['sex'], df['age_group'])
    ]
    return df


# ---------------------------------------------------------------------------
# Segment collection
# ---------------------------------------------------------------------------

def collect_paired_segments(
    paired_dir: Path,
    metadata: pd.DataFrame,
    tasks: List[str] = FRC_TASKS,
) -> pd.DataFrame:
    """
    Walk paired_dir for HDF5 files, split each at the FRC crossing,
    compute above/below summaries, and merge with metadata.
    """
    rows = []
    hdf5_files = sorted(paired_dir.rglob('*.h5'))
    logger.info(f"Found {len(hdf5_files)} paired HDF5 files in {paired_dir}")

    n_skipped_task = 0
    n_skipped_split = 0
    n_skipped_voiced = 0
    for h5 in hdf5_files:
        try:
            df, meta_attrs = PairedFeatureExtractor.load_hdf5(h5)
        except Exception as e:
            logger.warning(f"Could not load {h5}: {e}")
            continue

        # Resolve subject_id / task — prefer HDF5 attrs, fall back to filename
        subject_id = meta_attrs.get('subject_id')
        task = meta_attrs.get('task_name')
        if subject_id is None or task is None:
            stem = h5.stem  # e.g. 'GaBa_a_2'
            parts = stem.split('_', 1)
            if subject_id is None and len(parts) >= 1:
                subject_id = parts[0]
            if task is None and len(parts) >= 2:
                task = parts[1]

        if task not in tasks:
            n_skipped_task += 1
            continue

        split = split_segment_at_frc(df)
        if split is None:
            n_skipped_split += 1
            logger.debug(f"  Skipped (no valid FRC split): {subject_id}/{task}")
            continue
        above, below = split
        feats = compute_segment_features(above, below)
        if feats is None:
            n_skipped_voiced += 1
            logger.debug(f"  Skipped (insufficient voiced frames): {subject_id}/{task}")
            continue
        feats.update({
            'subject_id': str(subject_id),
            'task': task,
            'n_above_frames': len(above),
            'n_below_frames': len(below),
            'source_file': str(h5.name),
        })
        rows.append(feats)

    logger.info(
        f"Segments: kept={len(rows)}  skipped_task={n_skipped_task}  "
        f"skipped_split={n_skipped_split}  skipped_voiced={n_skipped_voiced}"
    )
    seg_df = pd.DataFrame(rows)
    if seg_df.empty:
        raise RuntimeError(
            "No valid segments collected — check --paired-dir and --tasks"
        )

    # Merge metadata
    merged = seg_df.merge(
        metadata[['subject_id', 'sex', 'age']],
        on='subject_id', how='left',
    )
    unmatched = merged.loc[merged['sex'].isna(), 'subject_id'].unique()
    if len(unmatched) > 0:
        logger.warning(
            f"No metadata for {len(unmatched)} subject(s): "
            f"{list(unmatched)[:10]}{'...' if len(unmatched) > 10 else ''}. "
            f"They will be excluded from stratified analysis."
        )
    return merged


# ---------------------------------------------------------------------------
# Stratified effect sizes
# ---------------------------------------------------------------------------

STRATA_ORDER = ['All', 'Male', 'Female', 'Young', 'Elder', 'YM', 'YF', 'EM', 'EF']


def stratified_effect_sizes(
    seg_df: pd.DataFrame,
    feature: str,
    age_threshold: int = 55,
) -> pd.DataFrame:
    """
    Compute paired effect sizes for one feature across all strata.
    """
    df = assign_demographic_group(seg_df.dropna(subset=['sex', 'age']), age_threshold)
    above_col, below_col = f'{feature}_above', f'{feature}_below'

    strata = {
        'All': df,
        'Male': df[df['sex'] == 'M'],
        'Female': df[df['sex'] == 'F'],
        'Young': df[df['age_group'] == 'Young'],
        'Elder': df[df['age_group'] == 'Elder'],
        'YM': df[df['demographic'] == 'YM'],
        'YF': df[df['demographic'] == 'YF'],
        'EM': df[df['demographic'] == 'EM'],
        'EF': df[df['demographic'] == 'EF'],
    }

    rows = []
    for name, sub in strata.items():
        sub = sub.dropna(subset=[above_col, below_col])
        if len(sub) < 3:
            logger.warning(
                f"  Stratum '{name}' for {feature}: only {len(sub)} segments — skipped"
            )
            continue
        try:
            es = compute_paired_effect_size(
                above=sub[above_col].values,
                below=sub[below_col].values,
            )
        except Exception as e:
            logger.error(f"  Effect size failed for {name}/{feature}: {e}")
            continue
        rows.append({
            'feature': feature,
            'stratum': name,
            'n_segments': es.n,
            'n_subjects': sub['subject_id'].nunique(),
            'median_shift': es.median_diff,
            'median_ci_lo': es.median_ci[0],
            'median_ci_hi': es.median_ci[1],
            'mean_shift': es.mean_diff,
            'sd_shift': es.sd_diff,
            'mad_shift': es.mad_diff,
            'cohen_d': es.cohen_d,
            'cohen_d_ci_lo': es.cohen_d_ci[0],
            'cohen_d_ci_hi': es.cohen_d_ci[1],
            'cohen_d_label': interpret_cohen_d(es.cohen_d),
            'hedges_g': es.hedges_g,
            'robust_d': es.robust_d,
            'wilcoxon_p': es.wilcoxon_p,
            'wilcoxon_r': es.wilcoxon_r,
            'sign_consistency': es.sign_consistency,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_forest(summary: pd.DataFrame, feature: str, output_path: Path) -> None:
    """
    Forest plot of median shift with bootstrap 95% CI, one row per stratum.
    Color-coded by Wilcoxon significance; Cohen's d annotated on the right.
    """
    sub = summary[summary['feature'] == feature].copy()
    if sub.empty:
        return
    sub['__order'] = sub['stratum'].apply(
        lambda s: STRATA_ORDER.index(s) if s in STRATA_ORDER else 99
    )
    sub = sub.sort_values('__order').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(sub) + 1.5))
    y = np.arange(len(sub))
    medians = sub['median_shift'].values
    err_lo = medians - sub['median_ci_lo'].values
    err_hi = sub['median_ci_hi'].values - medians

    # Error bars in black, marker color encodes significance
    ax.errorbar(
        medians, y,
        xerr=[err_lo, err_hi],
        fmt='none', ecolor='black', elinewidth=1.2, capsize=3, zorder=2,
    )
    colors = ['#1f77b4' if p < 0.05 else '#999999' for p in sub['wilcoxon_p']]
    ax.scatter(medians, y, c=colors, s=60, edgecolor='black',
               linewidth=0.6, zorder=3)

    ax.axvline(0, color='red', linestyle='--', alpha=0.6, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sub['stratum'])
    ax.invert_yaxis()
    ax.set_xlabel(f'Median shift (below − above), {UNITS.get(feature, "")}')
    ax.set_title(f'L3 FRC: {feature} — stratified effect (median + bootstrap 95% CI)')

    # Annotate d, n, sign-consistency on the right margin
    for i, row in enumerate(sub.itertuples()):
        text = (
            f"d={row.cohen_d:+.2f} ({row.cohen_d_label})  "
            f"n={row.n_segments}  "
            f"sign={row.sign_consistency:.0%}"
        )
        ax.text(
            1.02, y[i], text,
            transform=ax.get_yaxis_transform(),
            ha='left', va='center', fontsize=8.5, color='dimgray',
        )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_stratified_histograms(
    seg_df: pd.DataFrame,
    feature: str,
    output_path: Path,
    age_threshold: int = 55,
) -> None:
    """
    2x2 small-multiple histograms of per-segment shift, one panel per
    YM/YF/EM/EF subgroup. Median + Cohen's d annotated.
    """
    df = assign_demographic_group(seg_df.dropna(subset=['sex', 'age']), age_threshold)
    df = df.copy()
    df['shift'] = df[f'{feature}_below'] - df[f'{feature}_above']
    df = df.dropna(subset=['shift'])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, group in zip(axes.flat, ['YM', 'YF', 'EM', 'EF']):
        sub = df[df['demographic'] == group]
        if len(sub) >= 3:
            ax.hist(sub['shift'], bins=15, color='#4c72b0',
                    alpha=0.78, edgecolor='black')
            med = sub['shift'].median()
            sd = sub['shift'].std(ddof=1)
            d = (sub['shift'].mean() / sd) if sd > 0 else float('nan')
            ax.axvline(med, color='red', linestyle='--', linewidth=1.5,
                       label=f'median={med:+.3g}')
            ax.axvline(0, color='gray', linewidth=1, alpha=0.6)
            ax.set_title(f'{group}  (n={len(sub)},  d={d:+.2f})')
            ax.legend(fontsize=8, loc='upper right')
        else:
            ax.set_title(f'{group}  (n={len(sub)} — insufficient)')
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel(f'{feature} shift (below − above), {UNITS.get(feature, "")}')

    fig.suptitle(
        f'L3 FRC — {feature} shift by demographic subgroup',
        y=1.0, fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="L3 FRC stratified analysis with effect sizes",
    )
    parser.add_argument('--paired-dir', type=Path, required=True,
                        help='Directory containing paired HDF5 files')
    parser.add_argument('--metadata', type=Path, required=True,
                        help='Subject metadata Excel/CSV (subject_id, sex, age)')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Where to write results')
    parser.add_argument('--age-threshold', type=int, default=55,
                        help='Age cutoff for Young vs Elder (default: 55)')
    parser.add_argument('--tasks', nargs='+', default=FRC_TASKS,
                        help=f'Tasks to include (default: {FRC_TASKS})')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading metadata from {args.metadata}")
    metadata = load_metadata(args.metadata)
    logger.info(f"  {len(metadata)} subjects in metadata")

    logger.info(f"Collecting paired segments from {args.paired_dir}")
    seg_df = collect_paired_segments(args.paired_dir, metadata, tasks=args.tasks)
    n_subj = seg_df['subject_id'].nunique()
    logger.info(f"  {len(seg_df)} segments / {n_subj} subjects collected")

    # Save raw per-segment results for downstream use / reproducibility
    seg_df_with_groups = assign_demographic_group(
        seg_df.dropna(subset=['sex', 'age']), args.age_threshold
    )
    seg_df_with_groups.to_csv(args.output_dir / 'frc_per_segment.csv', index=False)

    # Compute stratified effect sizes per feature
    summaries = []
    for feature in FEATURES:
        logger.info(f"Computing effect sizes for: {feature}")
        s = stratified_effect_sizes(seg_df, feature, age_threshold=args.age_threshold)
        summaries.append(s)
    summary = pd.concat(summaries, ignore_index=True)

    summary_path = args.output_dir / 'frc_stratified_summary.xlsx'
    summary.to_excel(summary_path, index=False)
    logger.info(f"  Wrote {summary_path}")

    # Plots
    for feature in FEATURES:
        plot_forest(
            summary, feature,
            args.output_dir / f'forest_{feature}.pdf',
        )
        plot_stratified_histograms(
            seg_df, feature,
            args.output_dir / f'hist_strat_{feature}.pdf',
            age_threshold=args.age_threshold,
        )
        logger.info(f"  Plots written for {feature}")

    # Console summary of the headline (overall) row
    logger.info("\n=== Headline summary (overall stratum) ===")
    overall = summary[summary['stratum'] == 'All'].set_index('feature')
    for feature in FEATURES:
        if feature in overall.index:
            r = overall.loc[feature]
            logger.info(
                f"  {feature:10s}  median={r['median_shift']:+.4g}  "
                f"d={r['cohen_d']:+.3f} ({r['cohen_d_label']})  "
                f"sign_consistency={r['sign_consistency']:.0%}  "
                f"Wilcoxon p={r['wilcoxon_p']:.2e}"
            )

    logger.info(f"\nDone — see {args.output_dir}")


if __name__ == '__main__':
    main()