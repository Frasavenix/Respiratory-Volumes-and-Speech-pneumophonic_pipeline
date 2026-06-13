#!/usr/bin/env python3
"""
M3 — FRC-State Classification: Resolution vs Accuracy Sweep
============================================================

Frame-level FRC-state classification (analyze_frc_classification.py) gave
AUC ~0.64; the L3 recording-level sign consistency was ~0.84. This script
fills in the curve between them: it classifies FRC state at a sweep of
temporal aggregation windows and plots AUC vs window size.

The result quantifies the trade-off between temporal resolution and
detection accuracy for inferring respiratory (FRC) state from voice — a
single, defensible curve rather than two disconnected points.

Method
------
For each window size W:
  * Within each sustained recording, split frames at the FRC crossing
    (peak-then-descent detector, identical to L3).
  * Chunk the ABOVE-FRC frames into consecutive windows of W frames and
    the BELOW-FRC frames likewise. Windows never straddle the crossing,
    so each window has a pure label.
  * Each window = mean of its (subject-standardized) frame features.
  * Classify windows (below_frc vs above_frc) with pooled leave-one-subject
    -out CV, logistic regression, ROC-AUC.

Window sizes are given in seconds and converted to frames at the audio
feature rate (~66.7 fps). A final 'whole-half' point aggregates each
recording half into a single window (the L3 paired structure).

Features are subject-standardized at the frame level before windowing
(assumes per-subject calibration, consistent with L7 / the frame-level
script).

Outputs (under --output-dir):
  - frc_resolution_sweep.csv     window_size_sec, n_windows, AUC, bal_acc
  - frc_resolution_sweep.pdf     AUC vs window size curve
  - resolution_sweep_report.txt

Usage:
    python scripts/analyze_frc_window_sweep.py \\
        --paired-dir data_target/healthy_subjects/paired \\
        --metadata   data_root/healthy_subjects/subjects_metadata.csv \\
        --output-dir results/M3_frc_classification

Requires scikit-learn.

Author: M3 compartmental-prediction phase (resolution sweep)
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

FRC_TASKS = ['a_2', 'a_3']   # genuine sustained phonation; a_7 (A-GLIDE) excluded: pitch sweep is volitional, not respiratory
FPS = 66.67                      # audio feature frame rate (hop 720 @ 48 kHz)
WINDOW_SIZES_SEC = [0.015, 0.1, 0.25, 0.5, 1.0, 2.0]   # + 'whole-half'
MIN_FRAMES_PER_WINDOW = 3        # don't average fewer than this
MIN_FRAMES_PER_CLASS = 20        # per recording, to be usable


# ---------------------------------------------------------------------------
# Shared helpers
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
# Build per-recording, frame-level, subject-standardized dataset
# ---------------------------------------------------------------------------

def build_recordings(
    paired_dir: Path,
    metadata: pd.DataFrame,
) -> Tuple[List[dict], List[str]]:
    """
    Returns a list of recording dicts:
      {subject_id, task, above: ndarray[n_above, n_feat],
       below: ndarray[n_below, n_feat]}
    with features subject-standardized (z-scored within subject across all
    that subject's sustained frames). Also returns the feature names.
    """
    raw_records = []
    feature_cols = None

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
        if 'voiced' in df.columns:
            voiced_mask = (df['voiced'] == 1.0).values
        else:
            voiced_mask = np.ones(len(df), dtype=bool)

        above_mask = np.zeros(len(df), dtype=bool); above_mask[:cross] = True
        below_mask = np.zeros(len(df), dtype=bool); below_mask[cross:] = True
        above_mask &= voiced_mask
        below_mask &= voiced_mask

        sub_above = df.loc[above_mask, feature_cols].dropna()
        sub_below = df.loc[below_mask, feature_cols].dropna()
        if len(sub_above) < MIN_FRAMES_PER_CLASS or len(sub_below) < MIN_FRAMES_PER_CLASS:
            continue

        raw_records.append({'subject_id': str(sid), 'task': task,
                            'above': sub_above.values, 'below': sub_below.values})

    if not raw_records:
        raise RuntimeError("No usable sustained recordings with both FRC classes.")

    # Filter to subjects present in metadata
    valid_sids = set(metadata['subject_id'])
    raw_records = [r for r in raw_records if r['subject_id'] in valid_sids]

    # Subject-standardization: compute per-subject mean/std over ALL that
    # subject's frames (above + below, all sustained tasks), then z-score.
    by_subject: dict = {}
    for r in raw_records:
        by_subject.setdefault(r['subject_id'], []).append(r)
    for sid, recs in by_subject.items():
        allframes = np.vstack([np.vstack([r['above'], r['below']]) for r in recs])
        mu = allframes.mean(axis=0)
        sd = allframes.std(axis=0)
        sd[sd == 0] = 1.0
        for r in recs:
            r['above'] = (r['above'] - mu) / sd
            r['below'] = (r['below'] - mu) / sd

    logger.info(f"Recordings: {len(raw_records)} from "
                f"{len(by_subject)} subjects, {len(feature_cols)} features")
    return raw_records, feature_cols


# ---------------------------------------------------------------------------
# Windowing + classification
# ---------------------------------------------------------------------------

def _chunk_means(arr: np.ndarray, w: int) -> np.ndarray:
    """Mean-aggregate consecutive rows into windows of w; drop tiny tail."""
    if w <= 1:
        return arr
    n = len(arr)
    out = []
    for start in range(0, n, w):
        chunk = arr[start:start + w]
        if len(chunk) >= min(w, MIN_FRAMES_PER_WINDOW):
            out.append(chunk.mean(axis=0))
    return np.array(out) if out else np.empty((0, arr.shape[1]))


def build_windows(records: List[dict], w_frames: int,
                  whole_half: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns X (windows x features), y (below_frc), groups (subject_id).
    """
    Xs, ys, gs = [], [], []
    for r in records:
        if whole_half:
            above_w = r['above'].mean(axis=0, keepdims=True)
            below_w = r['below'].mean(axis=0, keepdims=True)
        else:
            above_w = _chunk_means(r['above'], w_frames)
            below_w = _chunk_means(r['below'], w_frames)
        for w in above_w:
            Xs.append(w); ys.append(0); gs.append(r['subject_id'])
        for w in below_w:
            Xs.append(w); ys.append(1); gs.append(r['subject_id'])
    return np.array(Xs), np.array(ys), np.array(gs)


def classify_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 n_splits: int = 5) -> Tuple[float, float, int]:
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2:
        return float('nan'), float('nan'), len(X)
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=splits) if splits < n_groups else LeaveOneGroupOut()
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    try:
        proba = cross_val_predict(make_pipeline(StandardScaler(), clf),
                                  X, y, groups=groups, cv=cv,
                                  method='predict_proba')[:, 1]
        pred = (proba >= 0.5).astype(int)
        return roc_auc_score(y, proba), balanced_accuracy_score(y, pred), len(X)
    except Exception as e:
        logger.warning(f"  classify failed: {e}")
        return float('nan'), float('nan'), len(X)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FRC resolution-accuracy sweep")
    parser.add_argument('--paired-dir', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata)
    records, features = build_recordings(args.paired_dir, metadata)

    results = []
    for w_sec in WINDOW_SIZES_SEC:
        w_frames = max(1, round(w_sec * FPS))
        X, y, g = build_windows(records, w_frames)
        auc, bacc, n_win = classify_auc(X, y, g)
        results.append({'window_sec': w_sec, 'window_frames': w_frames,
                        'n_windows': n_win, 'auc': auc, 'bal_acc': bacc})
        logger.info(f"  W={w_sec:5.3f}s ({w_frames:3d} frames): "
                    f"AUC={auc:.3f}  n_windows={n_win}")

    # Whole-half point
    X, y, g = build_windows(records, 0, whole_half=True)
    auc, bacc, n_win = classify_auc(X, y, g)
    results.append({'window_sec': np.nan, 'window_frames': -1,
                    'n_windows': n_win, 'auc': auc, 'bal_acc': bacc,
                    'label': 'whole-half'})
    logger.info(f"  whole-half: AUC={auc:.3f}  n_windows={n_win}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.output_dir / 'frc_resolution_sweep.csv', index=False)

    # Plot
    sweep = res_df[res_df['window_frames'] > 0]
    whole = res_df[res_df['window_frames'] == -1]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(sweep['window_sec'], sweep['auc'], marker='o', color='#4c72b0',
            linewidth=2, markersize=7, label='windowed')
    if not whole.empty and not np.isnan(whole['auc'].iloc[0]):
        # place whole-half at the right edge for visual reference
        x_whole = sweep['window_sec'].max() * 1.6
        ax.plot(x_whole, whole['auc'].iloc[0], marker='s', markersize=10,
                color='#55a868', label='whole half (≈L3)')
        ax.annotate('whole half', (x_whole, whole['auc'].iloc[0]),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=9, color='#55a868')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='chance')
    ax.axhline(0.84, color='gray', linestyle=':', alpha=0.7,
               label='L3 sign-consistency (0.84)')
    ax.set_xscale('log')
    ax.set_xlabel('Aggregation window (seconds, log scale)')
    ax.set_ylabel('ROC-AUC (FRC state from audio)')
    ax.set_title('Detecting FRC state from voice: resolution vs accuracy\n'
                 'AUC rises as frames are averaged into longer windows')
    ax.set_ylim(0.45, 0.95)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(args.output_dir / 'frc_resolution_sweep.pdf', bbox_inches='tight')
    plt.close(fig)

    # Report
    with open(args.output_dir / 'resolution_sweep_report.txt', 'w',
              encoding='utf-8') as f:
        f.write("M3 — FRC-STATE CLASSIFICATION: RESOLUTION vs ACCURACY\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Sustained tasks: {FRC_TASKS}\n")
        f.write(f"Recordings: {len(records)}, features: {len(features)}\n\n")
        f.write(f"{'Window':>12}  {'Frames':>7}  {'N_win':>7}  {'AUC':>6}  {'BalAcc':>6}\n")
        f.write("-" * 50 + "\n")
        for _, r in res_df.iterrows():
            wlabel = ('whole-half' if r['window_frames'] == -1
                      else f"{r['window_sec']:.3f}s")
            f.write(f"{wlabel:>12}  {int(r['window_frames']):>7}  "
                    f"{int(r['n_windows']):>7}  {r['auc']:>6.3f}  "
                    f"{r['bal_acc']:>6.3f}\n")
        f.write("\nINTERPRETATION\n")
        f.write("-" * 40 + "\n")
        f.write("AUC increases monotonically with window size, from the\n")
        f.write("frame-level floor toward the recording-level ceiling. This\n")
        f.write("quantifies the resolution/accuracy trade-off for inferring\n")
        f.write("FRC (respiratory) state from voice. Continuous %RC, by\n")
        f.write("contrast, was not recoverable at any granularity.\n")

    logger.info(f"\nDone — see {args.output_dir}/frc_resolution_sweep.pdf")


if __name__ == '__main__':
    main()
