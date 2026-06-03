#!/usr/bin/env python3
"""
M3 — Compartmental Regression: Can Audio Predict %RC?
=====================================================

Central question of the M3 phase (per supervisor feedback): can frame-level
audio features predict the rib-cage compartmental contribution (%RC) during
phonation, and is that mapping consistent within subjects but not across the
population?

The script runs four complementary analyses:

  1) PER-SUBJECT regression  — the make-or-break test.
     For each subject, fit a regularized linear model (RidgeCV) mapping
     audio features -> %RC with leave-one-task-out cross-validation, and
     record the cross-validated R^2. If most subjects show clearly positive
     R^2, the audio->%RC mapping exists at the subject level — consistent
     with the L7 finding that the relationship is real but subject-specific.

  2) POOLED regression       — the negative control.
     One model on all subjects' frames, leave-one-subject-out CV. Expected
     to perform poorly: the L7 analysis showed the audio-%RC relationship
     has subject-heterogeneous SIGN, so a single population model cannot
     capture it. A large per-subject vs pooled R^2 gap is itself the
     evidence that subject-specific modelling is required.

  3) DEMOGRAPHIC-STRATIFIED  — pooled-within-cell (YM/YF/EM/EF),
     leave-one-subject-out CV. Identifies which cells have the most
     learnable mapping (hypothesis: Elder Male, the highest-SNR L3 cell).

  4) BINNING EDA             — the supervisors' proposed approach.
     Bin %RC into bands and test (one-way ANOVA per feature) whether
     audio-feature means differ across bands. A model-free view of which
     features carry %RC information.

Outputs (under --output-dir):
  - per_subject_r2.csv             per-subject CV R^2, n_frames, n_tasks
  - per_subject_r2_hist.pdf        distribution of per-subject R^2
  - coefficient_importance.pdf     mean |standardized coef| per feature
  - rsquared_comparison.pdf        per-subject vs pooled vs stratified
  - binning_anova.csv              per-feature ANOVA F / p across %RC bands
  - binning_feature_profiles.pdf   standardized feature means per %RC band
  - compartmental_report.txt       text summary

Usage:
    python scripts/analyze_compartmental_regression.py \\
        --paired-dir data_target/healthy_subjects/paired \\
        --metadata   data_root/healthy_subjects/subjects_metadata.csv \\
        --output-dir results/M3_compartmental \\
        --target pct_rc

Requires scikit-learn (pip install scikit-learn).

Author: M3 compartmental-prediction phase
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pneumophonic_analysis.paired_features import PairedFeatureExtractor

try:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut, GroupKFold
    from sklearn.metrics import r2_score
except ImportError:
    sys.exit("This script requires scikit-learn. Install with: "
             "pip install scikit-learn")


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')

RIDGE_ALPHAS = np.logspace(-2, 3, 10)

# Minimum data requirements for a subject to enter the per-subject analysis
MIN_TASKS_PER_SUBJECT = 4
MIN_FRAMES_PER_SUBJECT = 200


# ---------------------------------------------------------------------------
# Metadata (compact loader — same conventions as analyze_l3_stratified.py)
# ---------------------------------------------------------------------------

def _detect_csv_separator(path: Path) -> str:
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                counts = {sep: line.count(sep) for sep in [';', ',', '\t']}
                best = max(counts, key=counts.get)
                return best if counts[best] >= 2 else ','
    return ','


def load_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        meta = pd.read_excel(path)
    else:
        meta = pd.read_csv(path, sep=_detect_csv_separator(path),
                           engine='python', encoding='utf-8-sig')
    meta.columns = [str(c).strip().lower() for c in meta.columns]
    meta = meta.dropna(axis=1, how='all')
    meta = meta.rename(columns={'id': 'subject_id', 'gender': 'sex'})
    for req in ('subject_id', 'sex', 'age'):
        if req not in meta.columns:
            raise ValueError(f"Metadata missing column '{req}'. "
                             f"Found: {list(meta.columns)}")
    meta['subject_id'] = meta['subject_id'].astype(str).str.strip()
    meta['sex'] = meta['sex'].astype(str).str.upper().str.strip().str[0]
    meta['sex'] = meta['sex'].replace({'W': 'F', 'D': 'F'})
    meta['age'] = pd.to_numeric(meta['age'], errors='coerce')
    return meta.dropna(subset=['subject_id', 'sex', 'age']).reset_index(drop=True)


def assign_demographic(df: pd.DataFrame, age_threshold: int = 55) -> pd.DataFrame:
    df = df.copy()
    df['age_group'] = np.where(df['age'] >= age_threshold, 'Elder', 'Young')
    label = {('M', 'Young'): 'YM', ('M', 'Elder'): 'EM',
             ('F', 'Young'): 'YF', ('F', 'Elder'): 'EF'}
    df['demographic'] = [label.get((s, g), 'Unknown')
                         for s, g in zip(df['sex'], df['age_group'])]
    return df


# ---------------------------------------------------------------------------
# Feature-column detection
# ---------------------------------------------------------------------------

def detect_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Audio feature columns to use as predictors. Always uses f0, energy,
    spectral_centroid when present, plus every mfcc_* column found.
    """
    base = [c for c in ('f0', 'energy', 'spectral_centroid') if c in df.columns]
    mfcc = sorted([c for c in df.columns if c.lower().startswith('mfcc')],
                  key=lambda s: int(''.join(ch for ch in s if ch.isdigit()) or -1))
    feats = base + mfcc
    if not feats:
        raise RuntimeError(f"No audio feature columns found. Columns: {list(df.columns)}")
    return feats


# ---------------------------------------------------------------------------
# Frame-level dataset construction
# ---------------------------------------------------------------------------

def build_frame_dataset(
    paired_dir: Path,
    metadata: pd.DataFrame,
    target: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Walk all paired HDF5 files and build a frame-level DataFrame restricted
    to voiced frames, with audio features + target + subject/task/demographic.
    """
    rows = []
    feature_cols: Optional[List[str]] = None
    hdf5_files = sorted(paired_dir.rglob('*.h5'))
    logger.info(f"Found {len(hdf5_files)} paired HDF5 files")

    for h5 in hdf5_files:
        try:
            df, meta = PairedFeatureExtractor.load_hdf5(h5)
        except Exception as e:
            logger.warning(f"Could not load {h5.name}: {e}")
            continue

        subject_id = meta.get('subject_id') or h5.stem.split('_', 1)[0]
        task = meta.get('task_name') or (h5.stem.split('_', 1)[1]
                                         if '_' in h5.stem else h5.stem)

        if target not in df.columns:
            continue
        if feature_cols is None:
            feature_cols = detect_feature_columns(df)

        # Restrict to voiced frames (audio features defined there)
        if 'voiced' in df.columns:
            df = df[df['voiced'] == 1.0]
        # Drop frames with missing features or target
        keep_cols = feature_cols + [target]
        df = df.dropna(subset=keep_cols)
        if len(df) == 0:
            continue

        sub = df[keep_cols].copy()
        sub['subject_id'] = str(subject_id)
        sub['task'] = task
        rows.append(sub)

    if not rows:
        raise RuntimeError("No usable frames collected — check paths/target.")

    frames = pd.concat(rows, ignore_index=True)
    frames = frames.merge(metadata[['subject_id', 'sex', 'age']],
                          on='subject_id', how='left')
    frames = assign_demographic(frames.dropna(subset=['sex', 'age']))
    logger.info(f"Frame dataset: {len(frames)} voiced frames, "
                f"{frames['subject_id'].nunique()} subjects, "
                f"{len(feature_cols)} features")
    return frames, feature_cols


# ---------------------------------------------------------------------------
# 1) Per-subject regression
# ---------------------------------------------------------------------------

def per_subject_regression(
    frames: pd.DataFrame,
    features: List[str],
    target: str,
) -> pd.DataFrame:
    """
    For each subject with sufficient data, fit RidgeCV(audio -> target) with
    leave-one-task-out CV and record cross-validated R^2 + coefficients.
    """
    records = []
    for sid, sub in frames.groupby('subject_id'):
        n_tasks = sub['task'].nunique()
        if n_tasks < MIN_TASKS_PER_SUBJECT or len(sub) < MIN_FRAMES_PER_SUBJECT:
            continue
        X = sub[features].values
        y = sub[target].values
        groups = sub['task'].values

        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        try:
            y_pred = cross_val_predict(pipe, X, y, groups=groups,
                                       cv=LeaveOneGroupOut())
            r2_cv = r2_score(y, y_pred)
        except Exception as e:
            logger.warning(f"  CV failed for {sid}: {e}")
            continue

        # Refit on all data for coefficients (standardized features)
        pipe.fit(X, y)
        coefs = pipe.named_steps['ridgecv'].coef_

        rec = {'subject_id': sid, 'n_frames': len(sub), 'n_tasks': n_tasks,
               'r2_cv': r2_cv,
               'demographic': sub['demographic'].iloc[0],
               'sex': sub['sex'].iloc[0], 'age': sub['age'].iloc[0]}
        for f, c in zip(features, coefs):
            rec[f'coef_{f}'] = c
        records.append(rec)

    if not records:
        raise RuntimeError("No subjects met the minimum data requirements.")
    out = pd.DataFrame(records)
    logger.info(f"Per-subject regression: {len(out)} subjects analyzed, "
                f"median CV R^2 = {out['r2_cv'].median():.3f}")
    return out


# ---------------------------------------------------------------------------
# 2) Pooled and 3) stratified regression
# ---------------------------------------------------------------------------

def grouped_regression_r2(
    frames: pd.DataFrame,
    features: List[str],
    target: str,
    group_col: str = 'subject_id',
    n_splits: int = 5,
) -> Tuple[float, int]:
    """
    Fit a single Ridge model on the given frames with group-aware CV
    (groups = subjects, so test frames come from held-out subjects).
    Returns (cross-validated R^2, n_groups).
    """
    n_groups = frames[group_col].nunique()
    if n_groups < 2:
        return float('nan'), n_groups
    X = frames[features].values
    y = frames[target].values
    groups = frames[group_col].values

    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    splits = min(n_splits, n_groups)
    try:
        y_pred = cross_val_predict(pipe, X, y, groups=groups,
                                   cv=GroupKFold(n_splits=splits))
        return r2_score(y, y_pred), n_groups
    except Exception as e:
        logger.warning(f"  Grouped regression failed: {e}")
        return float('nan'), n_groups


def stratified_regression(
    frames: pd.DataFrame,
    features: List[str],
    target: str,
) -> pd.DataFrame:
    """Pooled-within-demographic-cell regression, leave-subjects-out CV."""
    records = []
    for cell in ['YM', 'YF', 'EM', 'EF']:
        sub = frames[frames['demographic'] == cell]
        if sub['subject_id'].nunique() < 3:
            logger.warning(f"  Stratum {cell}: <3 subjects, skipped")
            continue
        r2, n_subj = grouped_regression_r2(sub, features, target)
        records.append({'stratum': cell, 'n_subjects': n_subj,
                        'n_frames': len(sub), 'r2_cv': r2})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4) Binning EDA
# ---------------------------------------------------------------------------

def binning_analysis(
    frames: pd.DataFrame,
    features: List[str],
    target: str,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Bin the target into quantile bands and run a one-way ANOVA per audio
    feature to test whether feature means differ across bands.
    """
    frames = frames.copy()
    frames['_band'] = pd.qcut(frames[target], q=n_bins, duplicates='drop')
    bands = sorted(frames['_band'].dropna().unique(), key=lambda b: b.left)

    records = []
    for feat in features:
        groups = [frames.loc[frames['_band'] == b, feat].values for b in bands]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        F, p = stats.f_oneway(*groups)
        # eta^2 effect size: between-group SS / total SS
        all_vals = np.concatenate(groups)
        grand = all_vals.mean()
        ss_total = ((all_vals - grand) ** 2).sum()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        eta2 = ss_between / ss_total if ss_total > 0 else np.nan
        records.append({'feature': feat, 'anova_F': F, 'anova_p': p,
                        'eta_squared': eta2})

    out = pd.DataFrame(records).sort_values('eta_squared', ascending=False)
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_per_subject_r2(per_subj: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    r2 = per_subj['r2_cv'].clip(lower=-0.5)  # clip extreme negatives for display
    ax.hist(r2, bins=20, color='#4c72b0', alpha=0.8, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='R²=0 (no skill)')
    ax.axvline(per_subj['r2_cv'].median(), color='green', linestyle='-',
               label=f"median = {per_subj['r2_cv'].median():.3f}")
    frac_pos = (per_subj['r2_cv'] > 0).mean()
    ax.set_xlabel('Cross-validated R²  (audio → %RC, leave-one-task-out)')
    ax.set_ylabel('Number of subjects')
    ax.set_title(f'Per-subject audio→%RC predictability\n'
                 f'{frac_pos:.0%} of subjects have R² > 0  '
                 f'(n={len(per_subj)} subjects)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_coefficient_importance(per_subj: pd.DataFrame, features: List[str],
                                output_path: Path) -> None:
    coef_cols = [f'coef_{f}' for f in features]
    mean_abs = per_subj[coef_cols].abs().mean().sort_values(ascending=True)
    # consistency of sign: fraction of subjects sharing the modal sign
    sign_consistency = {}
    for f in features:
        c = per_subj[f'coef_{f}']
        modal = np.sign(c.median())
        sign_consistency[f] = (np.sign(c) == modal).mean() if modal != 0 else 0.5

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(features))))
    labels = [c.replace('coef_', '') for c in mean_abs.index]
    colors = ['#2ca02c' if sign_consistency[lbl] >= 0.7 else '#999999'
              for lbl in labels]
    ax.barh(range(len(mean_abs)), mean_abs.values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(mean_abs)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean |standardized Ridge coefficient| across subjects')
    ax.set_title('Audio feature importance for %RC prediction\n'
                 '(green = sign-consistent across ≥70% of subjects)')
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_rsquared_comparison(per_subj_median: float, pooled_r2: float,
                             strat_df: pd.DataFrame, output_path: Path) -> None:
    labels = ['Per-subject\n(median)', 'Pooled\n(all subjects)']
    values = [per_subj_median, pooled_r2]
    colors = ['#4c72b0', '#c44e52']
    for _, row in strat_df.iterrows():
        labels.append(f"{row['stratum']}\n(stratified)")
        values.append(row['r2_cv'])
        colors.append('#55a868')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(values)), values, color=colors, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Cross-validated R²')
    ax.set_title('Audio→%RC predictability: per-subject vs pooled vs stratified\n'
                 'A large per-subject vs pooled gap indicates subject-specific mapping')
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_binning_profiles(frames: pd.DataFrame, features: List[str],
                          target: str, anova: pd.DataFrame,
                          output_path: Path, n_bins: int = 5,
                          top_k: int = 6) -> None:
    """Standardized feature means per %RC band, for the top-k features by eta²."""
    frames = frames.copy()
    frames['_band'] = pd.qcut(frames[target], q=n_bins, duplicates='drop')
    bands = sorted(frames['_band'].dropna().unique(), key=lambda b: b.left)
    band_labels = [f'{b.left:.2f}–{b.right:.2f}' for b in bands]

    top_feats = anova.head(top_k)['feature'].tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    for feat in top_feats:
        means = [frames.loc[frames['_band'] == b, feat].mean() for b in bands]
        # standardize the profile for comparability
        means = np.array(means)
        sd = frames[feat].std()
        means_z = (means - frames[feat].mean()) / sd if sd > 0 else means
        ax.plot(range(len(bands)), means_z, marker='o', label=feat)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(band_labels, rotation=30, ha='right', fontsize=8)
    ax.set_xlabel(f'{target} band')
    ax.set_ylabel('Standardized feature mean (z)')
    ax.set_title(f'Audio feature profiles across {target} bands '
                 f'(top {top_k} by η²)')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(per_subj: pd.DataFrame, pooled_r2: float,
                 strat_df: pd.DataFrame, anova: pd.DataFrame,
                 features: List[str], target: str, output_path: Path) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("M3 — COMPARTMENTAL REGRESSION REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Target: {target}\n")
        f.write(f"Audio features ({len(features)}): {', '.join(features)}\n\n")

        f.write("1) PER-SUBJECT REGRESSION (leave-one-task-out CV)\n")
        f.write("-" * 45 + "\n")
        f.write(f"  Subjects analyzed:   {len(per_subj)}\n")
        f.write(f"  Median CV R^2:       {per_subj['r2_cv'].median():.3f}\n")
        f.write(f"  Mean CV R^2:         {per_subj['r2_cv'].mean():.3f}\n")
        f.write(f"  Subjects with R^2>0: {(per_subj['r2_cv']>0).mean():.0%}\n")
        f.write(f"  Best subject R^2:    {per_subj['r2_cv'].max():.3f}\n\n")

        f.write("2) POOLED REGRESSION (leave-one-subject-out CV)\n")
        f.write("-" * 45 + "\n")
        f.write(f"  Pooled CV R^2:       {pooled_r2:.3f}\n")
        gap = per_subj['r2_cv'].median() - pooled_r2
        f.write(f"  Per-subject − pooled gap: {gap:+.3f}\n")
        f.write("  (A large positive gap = the mapping is subject-specific,\n")
        f.write("   consistent with the L7 sign-heterogeneity finding.)\n\n")

        f.write("3) DEMOGRAPHIC-STRATIFIED REGRESSION\n")
        f.write("-" * 45 + "\n")
        if not strat_df.empty:
            for _, r in strat_df.iterrows():
                f.write(f"  {r['stratum']}: R^2={r['r2_cv']:+.3f} "
                        f"(n_subj={int(r['n_subjects'])}, "
                        f"n_frames={int(r['n_frames'])})\n")
        f.write("\n")

        f.write("4) BINNING ANOVA (feature means across %RC bands)\n")
        f.write("-" * 45 + "\n")
        f.write("  Top features by eta^2 (effect size):\n")
        for _, r in anova.head(8).iterrows():
            f.write(f"    {r['feature']:18s}  eta^2={r['eta_squared']:.4f}  "
                    f"F={r['anova_F']:.1f}  p={r['anova_p']:.2e}\n")
        f.write("\n")

        # Top audio features by mean |coef|
        coef_cols = [f'coef_{c}' for c in features]
        mean_abs = per_subj[coef_cols].abs().mean().sort_values(ascending=False)
        f.write("TOP AUDIO FEATURES (mean |standardized coef|, per-subject)\n")
        f.write("-" * 45 + "\n")
        for c, v in mean_abs.head(8).items():
            f.write(f"    {c.replace('coef_',''):18s}  {v:.4f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="M3 compartmental regression")
    parser.add_argument('--paired-dir', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--target', type=str, default='pct_rc',
                        help="Respiratory target to predict (default: pct_rc)")
    parser.add_argument('--age-threshold', type=int, default=55)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata)
    logger.info(f"Metadata: {len(metadata)} subjects")

    frames, features = build_frame_dataset(args.paired_dir, metadata, args.target)

    # 1) Per-subject
    logger.info("Running per-subject regression...")
    per_subj = per_subject_regression(frames, features, args.target)
    per_subj.to_csv(args.output_dir / 'per_subject_r2.csv', index=False)

    # 2) Pooled
    logger.info("Running pooled regression (negative control)...")
    pooled_r2, _ = grouped_regression_r2(frames, features, args.target)

    # 3) Stratified
    logger.info("Running demographic-stratified regression...")
    strat_df = stratified_regression(frames, features, args.target)

    # 4) Binning EDA
    logger.info("Running binning ANOVA...")
    anova = binning_analysis(frames, features, args.target)
    anova.to_csv(args.output_dir / 'binning_anova.csv', index=False)

    # Plots
    plot_per_subject_r2(per_subj, args.output_dir / 'per_subject_r2_hist.pdf')
    plot_coefficient_importance(per_subj, features,
                                args.output_dir / 'coefficient_importance.pdf')
    plot_rsquared_comparison(per_subj['r2_cv'].median(), pooled_r2, strat_df,
                             args.output_dir / 'rsquared_comparison.pdf')
    plot_binning_profiles(frames, features, args.target, anova,
                          args.output_dir / 'binning_feature_profiles.pdf')

    # Report
    write_report(per_subj, pooled_r2, strat_df, anova, features, args.target,
                 args.output_dir / 'compartmental_report.txt')

    # Console headline
    logger.info("\n=== HEADLINE ===")
    logger.info(f"  Per-subject median CV R^2 : {per_subj['r2_cv'].median():+.3f}")
    logger.info(f"  Subjects with R^2 > 0      : {(per_subj['r2_cv']>0).mean():.0%}")
    logger.info(f"  Pooled CV R^2              : {pooled_r2:+.3f}")
    logger.info(f"  Per-subject − pooled gap   : "
                f"{per_subj['r2_cv'].median() - pooled_r2:+.3f}")
    if not strat_df.empty:
        best = strat_df.loc[strat_df['r2_cv'].idxmax()]
        logger.info(f"  Best demographic cell      : {best['stratum']} "
                    f"(R^2={best['r2_cv']:+.3f})")
    top_feat = anova.iloc[0]
    logger.info(f"  Top feature by eta^2       : {top_feat['feature']} "
                f"(eta^2={top_feat['eta_squared']:.3f})")
    logger.info(f"\nDone — see {args.output_dir}")


if __name__ == '__main__':
    main()
