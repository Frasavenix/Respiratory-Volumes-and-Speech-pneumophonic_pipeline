#!/usr/bin/env python3
"""
M3 Option A — FRC-State Classification from Audio
==================================================

Following the negative result for continuous %RC regression, this script
tests a better-posed, evidence-backed target: can audio features classify
whether a phonation frame is ABOVE or BELOW the Functional Residual
Capacity (FRC) crossing?

Rationale: the L3 analysis proved F0 is reliably elevated below FRC
(median +4.9 Hz, Cohen's d = 0.72, 84% sign consistency). FRC state is
therefore acoustically accessible in a way continuous %RC was not. This
script tests whether that signal is strong enough to *predict* FRC state.

Design decisions:
  * Sustained tasks only (a_2, a_3, a_7) — the FRC crossing is well-defined
    for single-breath phonation, matching L3.
  * Each voiced frame labelled below_frc = 1 (after the FRC crossing) or 0
    (before), using the same peak-then-descent crossing detector as L3.
  * Features are SUBJECT-STANDARDIZED (z-scored within subject) before
    classification. This removes between-subject F0 baseline differences
    and isolates the within-subject FRC-state signal that L3 showed is
    consistent. NOTE: this assumes per-subject calibration data is available
    at deployment — consistent with the L7 subject-specificity finding.

Analyses:
  1) POOLED (primary)        leave-one-subject-out CV on subject-standardized
                             features. Tests whether a generalizable audio
                             signature of FRC state exists across subjects.
  2) F0-ONLY baseline        same, using only F0. If it nearly matches the
                             full model, F0 is the carrier (validates L3).
  3) PER-SUBJECT             leave-one-recording-out CV within each subject
                             that has >=2 usable sustained recordings.
  4) DEMOGRAPHIC-STRATIFIED  pooled-within-cell, leave-one-subject-out.
  5) FEATURE IMPORTANCE      pooled-model coefficients; F0 expected to
                             dominate and energy to be weak (matches L3).

Metric: ROC-AUC (threshold-free, robust to the above/below class imbalance),
plus balanced accuracy.

Outputs (under --output-dir):
  - frc_classification_report.txt
  - per_subject_auc.csv / per_subject_auc_hist.pdf
  - frc_clf_feature_importance.pdf
  - frc_clf_auc_comparison.pdf

Usage:
    python scripts/analyze_frc_classification.py \\
        --paired-dir data_target/healthy_subjects/paired \\
        --metadata   data_root/healthy_subjects/subjects_metadata.csv \\
        --output-dir results/M3_frc_classification

Requires scikit-learn.

Author: M3 compartmental-prediction phase (Option A)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pneumophonic_analysis.paired_features import PairedFeatureExtractor

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut, GroupKFold
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
except ImportError:
    sys.exit("Requires scikit-learn. Install: pip install scikit-learn")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')

FRC_TASKS = ['a_2', 'a_3', 'a_7']
MIN_FRAMES_PER_CLASS = 20     # min above-FRC and below-FRC voiced frames / recording


# ---------------------------------------------------------------------------
# Metadata + feature detection (shared conventions)
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
    meta = meta.dropna(axis=1, how='all').rename(
        columns={'id': 'subject_id', 'gender': 'sex'})
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


def detect_feature_columns(df: pd.DataFrame) -> List[str]:
    base = [c for c in ('f0', 'energy', 'spectral_centroid') if c in df.columns]
    mfcc = sorted([c for c in df.columns if c.lower().startswith('mfcc')],
                  key=lambda s: int(''.join(ch for ch in s if ch.isdigit()) or -1))
    return base + mfcc


# ---------------------------------------------------------------------------
# FRC crossing detector (identical to analyze_l3_stratified.py)
# ---------------------------------------------------------------------------

def find_frc_crossing_index(delta_vcw: np.ndarray, end_margin: int = 20) -> Optional[int]:
    if len(delta_vcw) < 2:
        return None
    peak_idx = int(np.argmax(delta_vcw))
    post_peak = delta_vcw[peak_idx:]
    if len(post_peak) >= 2:
        sign = np.sign(post_peak)
        descending = (sign[:-1] > 0) & (sign[1:] <= 0)
        crossings = np.where(descending)[0]
        if len(crossings) > 0:
            return int(crossings[0]) + peak_idx + 1
    cross_idx = peak_idx + len(post_peak) // 2
    if cross_idx >= len(delta_vcw) - end_margin:
        return None
    return int(cross_idx)


# ---------------------------------------------------------------------------
# Build the labelled, subject-standardized frame dataset
# ---------------------------------------------------------------------------

def build_labeled_dataset(
    paired_dir: Path,
    metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Frame-level dataset from sustained tasks. Each voiced frame labelled
    below_frc (1) or above_frc (0). Features subject-standardized (z-score
    within subject) into '<feat>_z' columns.
    """
    rows, feature_cols = [], None
    for h5 in sorted(paired_dir.rglob('*.h5')):
        try:
            df, meta = PairedFeatureExtractor.load_hdf5(h5)
        except Exception:
            continue
        sid = meta.get('subject_id') or h5.stem.split('_', 1)[0]
        task = meta.get('task_name') or (h5.stem.split('_', 1)[1]
                                         if '_' in h5.stem else h5.stem)
        if task not in FRC_TASKS or 'delta_vcw' not in df.columns:
            continue
        if feature_cols is None:
            feature_cols = detect_feature_columns(df)

        cross = find_frc_crossing_index(df['delta_vcw'].values)
        if cross is None:
            continue
        labels = np.zeros(len(df), dtype=int)
        labels[cross:] = 1   # below FRC

        sub = df.copy()
        sub['below_frc'] = labels
        # voiced only
        if 'voiced' in sub.columns:
            sub = sub[sub['voiced'] == 1.0]
        sub = sub.dropna(subset=feature_cols + ['below_frc'])
        # require enough of each class in this recording
        n_above = int((sub['below_frc'] == 0).sum())
        n_below = int((sub['below_frc'] == 1).sum())
        if n_above < MIN_FRAMES_PER_CLASS or n_below < MIN_FRAMES_PER_CLASS:
            continue

        keep = feature_cols + ['below_frc']
        out = sub[keep].copy()
        out['subject_id'] = str(sid)
        out['task'] = task
        rows.append(out)

    if not rows:
        raise RuntimeError("No usable sustained recordings with both FRC classes.")

    frames = pd.concat(rows, ignore_index=True)
    frames = frames.merge(metadata[['subject_id', 'sex', 'age']],
                          on='subject_id', how='left')
    frames = assign_demographic(frames.dropna(subset=['sex', 'age']))

    # Subject-standardize features (z-score within subject)
    z_cols = []
    for f in feature_cols:
        zc = f'{f}_z'
        frames[zc] = frames.groupby('subject_id')[f].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0)
        z_cols.append(zc)

    logger.info(f"Labelled dataset: {len(frames)} voiced frames, "
                f"{frames['subject_id'].nunique()} subjects, "
                f"below-FRC fraction = {frames['below_frc'].mean():.2f}")
    return frames, z_cols


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _logreg() :
    return LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)


def grouped_auc(frames: pd.DataFrame, features: List[str],
                group_col: str, n_splits: int = 5) -> Tuple[float, float, int]:
    """
    Group-aware CV classification. Returns (AUC, balanced_accuracy, n_groups).
    Features are assumed already subject-standardized.
    """
    n_groups = frames[group_col].nunique()
    if n_groups < 2 or frames['below_frc'].nunique() < 2:
        return float('nan'), float('nan'), n_groups
    X = frames[features].values
    y = frames['below_frc'].values
    groups = frames[group_col].values
    splits = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=splits) if splits < n_groups else LeaveOneGroupOut()
    try:
        proba = cross_val_predict(make_pipeline(StandardScaler(), _logreg()),
                                  X, y, groups=groups, cv=cv,
                                  method='predict_proba')[:, 1]
        pred = (proba >= 0.5).astype(int)
        return (roc_auc_score(y, proba),
                balanced_accuracy_score(y, pred), n_groups)
    except Exception as e:
        logger.warning(f"  grouped_auc failed: {e}")
        return float('nan'), float('nan'), n_groups


def per_subject_auc(frames: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Leave-one-recording-out classification within each subject."""
    records = []
    for sid, sub in frames.groupby('subject_id'):
        if sub['task'].nunique() < 2 or sub['below_frc'].nunique() < 2:
            continue
        X = sub[features].values
        y = sub['below_frc'].values
        groups = sub['task'].values
        try:
            proba = cross_val_predict(make_pipeline(StandardScaler(), _logreg()),
                                      X, y, groups=groups, cv=LeaveOneGroupOut(),
                                      method='predict_proba')[:, 1]
            auc = roc_auc_score(y, proba)
        except Exception:
            continue
        records.append({'subject_id': sid, 'n_recordings': sub['task'].nunique(),
                        'n_frames': len(sub), 'auc': auc,
                        'demographic': sub['demographic'].iloc[0]})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_per_subject_auc(per_subj: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_subj['auc'], bins=18, color='#55a868', alpha=0.8, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', label='0.5 (chance)')
    ax.axvline(per_subj['auc'].median(), color='green',
               label=f"median = {per_subj['auc'].median():.3f}")
    frac = (per_subj['auc'] > 0.5).mean()
    ax.set_xlabel('Per-subject ROC-AUC (leave-one-recording-out)')
    ax.set_ylabel('Number of subjects')
    ax.set_title(f'Per-subject FRC-state classification from audio\n'
                 f'{frac:.0%} of subjects above chance (n={len(per_subj)})')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_feature_importance(frames: pd.DataFrame, features: List[str],
                            output_path: Path) -> None:
    # Fit pooled model on all data for coefficient inspection
    pipe = make_pipeline(StandardScaler(), _logreg())
    pipe.fit(frames[features].values, frames['below_frc'].values)
    coefs = pipe.named_steps['logisticregression'].coef_[0]
    order = np.argsort(np.abs(coefs))
    labels = [features[i].replace('_z', '') for i in order]
    vals = coefs[order]
    colors = ['#c44e52' if v < 0 else '#4c72b0' for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(features))))
    ax.barh(range(len(vals)), vals, color=colors, alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Logistic regression coefficient (subject-standardized features)')
    ax.set_title('FRC-state classifier feature importance\n'
                 'Positive = pushes toward "below FRC"; F0 expected to dominate (matches L3)')
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_auc_comparison(pooled_auc: float, f0_only_auc: float,
                        per_subj_median: float, strat: dict,
                        output_path: Path) -> None:
    labels = ['Pooled\n(all features)', 'Pooled\n(F0 only)', 'Per-subject\n(median)']
    values = [pooled_auc, f0_only_auc, per_subj_median]
    colors = ['#4c72b0', '#8172b3', '#55a868']
    for cell, auc in strat.items():
        labels.append(f'{cell}\n(stratified)')
        values.append(auc)
        colors.append('#937860')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(values)), values, color=colors, alpha=0.85)
    ax.axhline(0.5, color='red', linestyle='--', label='chance (0.5)')
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('FRC-state classification AUC across framings')
    ax.legend()
    for i, v in enumerate(values):
        if not np.isnan(v):
            ax.text(i, v + 0.008, f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="M3 FRC-state classification")
    parser.add_argument('--paired-dir', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata)
    frames, z_features = build_labeled_dataset(args.paired_dir, metadata)

    # 1) Pooled (all features), leave-one-subject-out
    logger.info("1) Pooled classification (all features)...")
    pooled_auc, pooled_bacc, _ = grouped_auc(frames, z_features, 'subject_id')

    # 2) F0-only baseline
    f0_z = [c for c in z_features if c.startswith('f0')]
    logger.info("2) Pooled classification (F0 only)...")
    f0_auc, f0_bacc, _ = (grouped_auc(frames, f0_z, 'subject_id')
                          if f0_z else (float('nan'), float('nan'), 0))

    # 3) Per-subject
    logger.info("3) Per-subject classification...")
    per_subj = per_subject_auc(frames, z_features)
    per_subj.to_csv(args.output_dir / 'per_subject_auc.csv', index=False)
    per_subj_median = per_subj['auc'].median() if len(per_subj) else float('nan')

    # 4) Stratified
    logger.info("4) Demographic-stratified classification...")
    strat = {}
    for cell in ['YM', 'YF', 'EM', 'EF']:
        sub = frames[frames['demographic'] == cell]
        if sub['subject_id'].nunique() >= 3:
            auc, _, _ = grouped_auc(sub, z_features, 'subject_id')
            strat[cell] = auc

    # Plots
    if len(per_subj):
        plot_per_subject_auc(per_subj, args.output_dir / 'per_subject_auc_hist.pdf')
    plot_feature_importance(frames, z_features,
                            args.output_dir / 'frc_clf_feature_importance.pdf')
    plot_auc_comparison(pooled_auc, f0_auc, per_subj_median, strat,
                        args.output_dir / 'frc_clf_auc_comparison.pdf')

    # Report
    with open(args.output_dir / 'frc_classification_report.txt', 'w',
              encoding='utf-8') as f:
        f.write("M3 OPTION A — FRC-STATE CLASSIFICATION FROM AUDIO\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Sustained tasks: {FRC_TASKS}\n")
        f.write(f"Frames: {len(frames)}, subjects: {frames['subject_id'].nunique()}, "
                f"below-FRC fraction: {frames['below_frc'].mean():.2f}\n")
        f.write(f"Features (subject-standardized): {len(z_features)}\n\n")
        f.write("Reference ceiling: L3 recording-level sign consistency = 84%\n")
        f.write("(the fraction of recordings where F0 is higher below FRC)\n\n")
        f.write("RESULTS (ROC-AUC; 0.5 = chance)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Pooled, all features (LOSO):  AUC={pooled_auc:.3f}  "
                f"bal-acc={pooled_bacc:.3f}\n")
        f.write(f"  Pooled, F0 only (LOSO):       AUC={f0_auc:.3f}  "
                f"bal-acc={f0_bacc:.3f}\n")
        f.write(f"  Per-subject (median, LORO):   AUC={per_subj_median:.3f}  "
                f"(n={len(per_subj)} subjects)\n")
        for cell, auc in strat.items():
            f.write(f"  Stratified {cell}:              AUC={auc:.3f}\n")
        f.write("\nINTERPRETATION\n")
        f.write("-" * 40 + "\n")
        if pooled_auc > 0.6:
            f.write("  Audio classifies FRC state above chance — the L3 F0\n")
            f.write("  signal is predictively usable. Check feature importance:\n")
            f.write("  if F0 dominates and the F0-only AUC ~ full-model AUC,\n")
            f.write("  F0 is the carrier (consistent with L3).\n")
        else:
            f.write("  Classification near chance even on the proven-signal\n")
            f.write("  target. The L3 effect, though significant in aggregate,\n")
            f.write("  is too small per-frame to support frame-level prediction.\n")
            f.write("  Next: aggregate to coarser units (per breath segment) or\n")
            f.write("  add glottal-source features.\n")

    # Console headline
    logger.info("\n=== FRC CLASSIFICATION HEADLINE ===")
    logger.info(f"  Pooled AUC (all features): {pooled_auc:.3f}")
    logger.info(f"  Pooled AUC (F0 only):      {f0_auc:.3f}")
    logger.info(f"  Per-subject median AUC:    {per_subj_median:.3f} "
                f"(n={len(per_subj)})")
    for cell, auc in strat.items():
        logger.info(f"  Stratified {cell} AUC:        {auc:.3f}")
    logger.info(f"  (Reference: L3 sign-consistency ceiling = 0.84)")
    logger.info(f"\nDone — see {args.output_dir}")


if __name__ == '__main__':
    main()
