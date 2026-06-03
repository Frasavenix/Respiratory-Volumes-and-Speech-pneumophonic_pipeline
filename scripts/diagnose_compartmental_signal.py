#!/usr/bin/env python3
"""
M3 Diagnostic — Why Did Audio→%RC Regression Fail?
===================================================

The per-subject frame-level regression (analyze_compartmental_regression.py)
returned strongly negative cross-validated R^2 under leave-one-task-out CV.
This diagnostic determines WHY, and which reframing (if any) is viable.

It runs three analyses:

  A) VARIANCE DECOMPOSITION
     For the target (%RC), decompose total variance into within-task and
     between-task components, per subject and pooled. If %RC variance is
     mostly BETWEEN tasks, frame-level within-task prediction is ill-posed
     (there is little to predict within a recording) and the cross-task CV
     was effectively asking the model to extrapolate task baselines.

  B) WITHIN-TASK TRACKING
     For each recording, blocked (contiguous-block) leave-one-block-out CV
     of audio -> %RC. Tests whether audio tracks %RC *fluctuations within a
     single recording*, with reduced (not zero) autocorrelation leakage.
     Compared against the failed cross-task R^2: if within-task R^2 is
     positive while cross-task is negative, audio tracks moment-to-moment
     %RC but cannot predict the task-level baseline.

  C) SEGMENT-LEVEL PREDICTION
     Reframe to one observation per recording: predict the recording's MEAN
     %RC from its MEAN audio features, leave-one-subject-out CV. This is the
     better-posed problem if %RC is task-structured — and is closer to the
     supervisors' "which segments display which compartmentalisation" goal.

Outputs (under --output-dir):
  - variance_decomposition.csv     per-subject within/between %RC variance
  - variance_decomposition.pdf     distribution of within-task variance fraction
  - within_task_r2.csv             per-recording within-task CV R^2
  - framing_comparison.pdf         cross-task vs within-task vs segment-level R^2
  - diagnostic_report.txt          text summary + recommended reframing

Usage:
    python scripts/diagnose_compartmental_signal.py \\
        --paired-dir data_target/healthy_subjects/paired \\
        --metadata   data_root/healthy_subjects/subjects_metadata.csv \\
        --output-dir results/M3_compartmental \\
        --target pct_rc

Requires scikit-learn.

Author: M3 compartmental-prediction phase (diagnostic)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pneumophonic_analysis.paired_features import PairedFeatureExtractor

try:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, GroupKFold
    from sklearn.metrics import r2_score
except ImportError:
    sys.exit("Requires scikit-learn. Install: pip install scikit-learn")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')

RIDGE_ALPHAS = np.logspace(-2, 3, 10)
MIN_FRAMES_RECORDING = 60     # ~0.9 s of voiced frames
WITHIN_TASK_BLOCKS = 5


# ---------------------------------------------------------------------------
# Shared helpers (compact copies of the regression-script utilities)
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


def detect_feature_columns(df: pd.DataFrame) -> List[str]:
    base = [c for c in ('f0', 'energy', 'spectral_centroid') if c in df.columns]
    mfcc = sorted([c for c in df.columns if c.lower().startswith('mfcc')],
                  key=lambda s: int(''.join(ch for ch in s if ch.isdigit()) or -1))
    return base + mfcc


def build_frame_dataset(paired_dir: Path, metadata: pd.DataFrame,
                        target: str) -> Tuple[pd.DataFrame, List[str]]:
    rows, feature_cols = [], None
    for h5 in sorted(paired_dir.rglob('*.h5')):
        try:
            df, meta = PairedFeatureExtractor.load_hdf5(h5)
        except Exception:
            continue
        sid = meta.get('subject_id') or h5.stem.split('_', 1)[0]
        task = meta.get('task_name') or (h5.stem.split('_', 1)[1]
                                         if '_' in h5.stem else h5.stem)
        if target not in df.columns:
            continue
        if feature_cols is None:
            feature_cols = detect_feature_columns(df)
        if 'voiced' in df.columns:
            df = df[df['voiced'] == 1.0]
        keep = feature_cols + [target]
        df = df.dropna(subset=keep)
        if len(df) == 0:
            continue
        sub = df[keep].copy()
        sub['subject_id'] = str(sid)
        sub['task'] = task
        # preserve frame order within recording for blocked CV
        sub['frame_order'] = np.arange(len(sub))
        rows.append(sub)
    frames = pd.concat(rows, ignore_index=True)
    frames = frames.merge(metadata[['subject_id', 'sex', 'age']],
                          on='subject_id', how='left')
    logger.info(f"Frame dataset: {len(frames)} frames, "
                f"{frames['subject_id'].nunique()} subjects, "
                f"{len(feature_cols)} features")
    return frames, feature_cols


# ---------------------------------------------------------------------------
# A) Variance decomposition
# ---------------------------------------------------------------------------

def variance_decomposition(frames: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Per subject: decompose %RC variance into within-task and between-task.
    within_fraction = within-task variance / total variance.
    """
    records = []
    for sid, sub in frames.groupby('subject_id'):
        if sub['task'].nunique() < 2:
            continue
        grand = sub[target].mean()
        total_var = ((sub[target] - grand) ** 2).mean()
        if total_var == 0:
            continue
        # between-task: variance of task means (weighted by task size)
        between = 0.0
        within = 0.0
        for _, g in sub.groupby('task'):
            w = len(g) / len(sub)
            between += w * (g[target].mean() - grand) ** 2
            within += w * g[target].var(ddof=0)
        records.append({
            'subject_id': sid,
            'n_tasks': sub['task'].nunique(),
            'n_frames': len(sub),
            'total_var': total_var,
            'between_task_var': between,
            'within_task_var': within,
            'within_fraction': within / total_var,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# B) Within-task tracking (blocked CV)
# ---------------------------------------------------------------------------

def within_task_r2(frames: pd.DataFrame, features: List[str],
                   target: str) -> pd.DataFrame:
    """
    For each recording, blocked leave-one-block-out CV of audio -> %RC.
    Tests within-recording tracking with reduced autocorrelation leakage.
    """
    records = []
    for (sid, task), rec in frames.groupby(['subject_id', 'task']):
        if len(rec) < MIN_FRAMES_RECORDING:
            continue
        rec = rec.sort_values('frame_order')
        y = rec[target].values
        if np.var(y) < 1e-8:   # essentially constant %RC — nothing to predict
            records.append({'subject_id': sid, 'task': task,
                            'n_frames': len(rec), 'r2_within': np.nan,
                            'target_var': float(np.var(y)),
                            'note': 'constant_target'})
            continue
        X = rec[features].values
        # contiguous blocks as CV groups
        blocks = np.floor(np.linspace(0, WITHIN_TASK_BLOCKS,
                                      len(rec), endpoint=False)).astype(int)
        n_blocks = len(np.unique(blocks))
        if n_blocks < 2:
            continue
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        try:
            y_pred = cross_val_predict(pipe, X, y, groups=blocks,
                                       cv=GroupKFold(n_splits=n_blocks))
            r2 = r2_score(y, y_pred)
        except Exception:
            continue
        records.append({'subject_id': sid, 'task': task, 'n_frames': len(rec),
                        'r2_within': r2, 'target_var': float(np.var(y)),
                        'note': ''})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# C) Segment-level prediction
# ---------------------------------------------------------------------------

def segment_level_r2(frames: pd.DataFrame, features: List[str],
                     target: str) -> Tuple[float, int, pd.DataFrame]:
    """
    One row per recording: mean audio features -> mean %RC.
    Leave-one-subject-out CV (GroupKFold by subject).
    """
    agg = frames.groupby(['subject_id', 'task']).agg(
        {**{f: 'mean' for f in features}, target: 'mean'}).reset_index()
    if len(agg) < 10:
        return float('nan'), len(agg), agg
    X = agg[features].values
    y = agg[target].values
    groups = agg['subject_id'].values
    n_splits = min(5, agg['subject_id'].nunique())
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    y_pred = cross_val_predict(pipe, X, y, groups=groups,
                               cv=GroupKFold(n_splits=n_splits))
    return r2_score(y, y_pred), len(agg), agg


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_variance(decomp: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(decomp['within_fraction'], bins=20, color='#4c72b0',
            alpha=0.8, edgecolor='black')
    med = decomp['within_fraction'].median()
    ax.axvline(med, color='red', linestyle='--',
               label=f'median = {med:.2f}')
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.7,
               label='50% (equal within/between)')
    ax.set_xlabel('Within-task fraction of %RC variance')
    ax.set_ylabel('Number of subjects')
    ax.set_title('How much %RC variance lives WITHIN a recording?\n'
                 'Low values = %RC is task-structured (between-task dominates)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_framing_comparison(cross_task_r2: float, within_task_median: float,
                            segment_r2: float, output_path: Path) -> None:
    labels = ['Cross-task\n(frame, per-subj)',
              'Within-task\n(frame, blocked)',
              'Segment-level\n(per recording)']
    values = [cross_task_r2, within_task_median, segment_r2]
    colors = ['#c44e52', '#4c72b0', '#55a868']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(3), values, color=colors, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Cross-validated R²')
    ax.set_title('Audio→%RC predictability under three framings')
    for i, v in enumerate(values):
        if not np.isnan(v):
            ax.text(i, v + (0.01 if v >= 0 else -0.03), f'{v:+.3f}',
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(decomp: pd.DataFrame, within_df: pd.DataFrame,
                 cross_task_r2: float, segment_r2: float,
                 target: str, output_path: Path) -> None:
    within_med = within_df['r2_within'].dropna().median()
    n_constant = (within_df['note'] == 'constant_target').sum()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("M3 DIAGNOSTIC — WHY DID AUDIO→%RC REGRESSION FAIL?\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Target: {target}\n\n")

        f.write("A) VARIANCE DECOMPOSITION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Subjects analyzed:            {len(decomp)}\n")
        f.write(f"  Median within-task fraction:  "
                f"{decomp['within_fraction'].median():.2f}\n")
        f.write(f"  Median between-task fraction: "
                f"{1 - decomp['within_fraction'].median():.2f}\n")
        if decomp['within_fraction'].median() < 0.4:
            f.write("  => %RC is TASK-STRUCTURED: most variance is between\n")
            f.write("     tasks. Frame-level within-task prediction is\n")
            f.write("     ill-posed; segment-level framing is appropriate.\n")
        else:
            f.write("  => Substantial within-task variance exists; frame-level\n")
            f.write("     tracking is worth pursuing.\n")
        f.write("\n")

        f.write("B) WITHIN-TASK TRACKING (blocked CV)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Recordings tested:           {within_df['r2_within'].notna().sum()}\n")
        f.write(f"  Recordings w/ constant %RC:  {n_constant}\n")
        f.write(f"  Median within-task R^2:      {within_med:+.3f}\n")
        f.write(f"  Recordings w/ R^2 > 0:       "
                f"{(within_df['r2_within'] > 0).mean():.0%}\n\n")

        f.write("C) FRAMING COMPARISON (cross-validated R^2)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Cross-task (frame, per-subject): {cross_task_r2:+.3f}\n")
        f.write(f"  Within-task (frame, blocked):    {within_med:+.3f}\n")
        f.write(f"  Segment-level (per recording):   {segment_r2:+.3f}\n\n")

        f.write("RECOMMENDED REFRAMING\n")
        f.write("-" * 40 + "\n")
        best = max([('within-task frame', within_med),
                    ('segment-level', segment_r2),
                    ('cross-task frame', cross_task_r2)],
                   key=lambda kv: (kv[1] if not np.isnan(kv[1]) else -99))
        f.write(f"  Best-performing framing: {best[0]} (R^2={best[1]:+.3f})\n")
        if best[1] < 0.05:
            f.write("  WARNING: even the best framing has negligible R^2.\n")
            f.write("  Linear audio->%RC prediction does not work in any\n")
            f.write("  framing. Next options: (1) nonlinear models, (2) richer\n")
            f.write("  audio features (formants, HNR, modulation spectrum),\n")
            f.write("  (3) reconsider whether %RC is recoverable from voice.\n")
        else:
            f.write(f"  Pursue the {best[0]} framing for M3 modelling.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="M3 compartmental diagnostic")
    parser.add_argument('--paired-dir', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--target', type=str, default='pct_rc')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata)
    frames, features = build_frame_dataset(args.paired_dir, metadata, args.target)

    logger.info("A) Variance decomposition...")
    decomp = variance_decomposition(frames, args.target)
    decomp.to_csv(args.output_dir / 'variance_decomposition.csv', index=False)

    logger.info("B) Within-task tracking (blocked CV)...")
    within_df = within_task_r2(frames, features, args.target)
    within_df.to_csv(args.output_dir / 'within_task_r2.csv', index=False)
    within_med = within_df['r2_within'].dropna().median()

    logger.info("C) Segment-level prediction...")
    segment_r2, n_seg, _ = segment_level_r2(frames, features, args.target)

    # Cross-task number from the main script (recompute quickly here for the plot)
    # Use the per-subject median we already know is negative; recompute cheaply
    # by pooling: this is only for the comparison bar. We reuse the within-task
    # machinery's frames for a quick per-subject leave-one-task-out estimate.
    from sklearn.model_selection import LeaveOneGroupOut
    cross_r2s = []
    for sid, sub in frames.groupby('subject_id'):
        if sub['task'].nunique() < 4 or len(sub) < 200:
            continue
        X, y, g = sub[features].values, sub[args.target].values, sub['task'].values
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        try:
            yp = cross_val_predict(pipe, X, y, groups=g, cv=LeaveOneGroupOut())
            cross_r2s.append(r2_score(y, yp))
        except Exception:
            continue
    cross_task_r2 = float(np.median(cross_r2s)) if cross_r2s else float('nan')

    plot_variance(decomp, args.output_dir / 'variance_decomposition.pdf')
    plot_framing_comparison(cross_task_r2, within_med, segment_r2,
                            args.output_dir / 'framing_comparison.pdf')
    write_report(decomp, within_df, cross_task_r2, segment_r2,
                 args.target, args.output_dir / 'diagnostic_report.txt')

    logger.info("\n=== DIAGNOSTIC HEADLINE ===")
    logger.info(f"  Median within-task variance fraction: "
                f"{decomp['within_fraction'].median():.2f}")
    logger.info(f"  Cross-task frame R^2  (failed):       {cross_task_r2:+.3f}")
    logger.info(f"  Within-task frame R^2 (blocked CV):   {within_med:+.3f}")
    logger.info(f"  Segment-level R^2 (per recording):    {segment_r2:+.3f}")
    logger.info(f"\nDone — see {args.output_dir}/diagnostic_report.txt")


if __name__ == '__main__':
    main()
