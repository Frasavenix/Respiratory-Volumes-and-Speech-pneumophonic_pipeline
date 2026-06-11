#!/usr/bin/env python3
"""
Repair sync-pulse leakage in already-extracted paired HDF5 files.
=================================================================

Some recordings have the **synchronization pulse** (a loud, ~0.4–1.1 s, DC /
low-frequency burst) sitting at the very start of the extracted segment, before
phonation. Because it is far louder than the voice (often 1000×+), it dominates
the per-segment normalization and corrupts energy / F0 / MFCCs.

This script detects that signature in the existing `.h5` corpus and **trims the
leading burst off**, in place, after backing up the originals:

  * cut every per-frame `aligned/*` column at the first sustained voiced onset
    that follows the burst;
  * recompute the onset-relative `delta_vcw / delta_vrc / delta_vab` and rebase
    `time` to start at 0;
  * slice the stored `stft / mel / mfcc` matrices to match;
  * update `n_frames` / `audio_duration_sec` and tag `sync_repaired = 1`.

It is **idempotent** (a repaired file no longer matches the leak signature) and
**non-destructive** (originals are copied to a backup folder first; a backup is
never overwritten).

Usage:
    python scripts/repair_sync_leak.py --paired-dir data_target/healthy_subjects/paired --dry-run
    python scripts/repair_sync_leak.py --paired-dir data_target/healthy_subjects/paired           # asks to confirm
    python scripts/repair_sync_leak.py --paired-dir data_target/healthy_subjects/paired --yes      # no prompt

The companion fix for *future* extractions lives in
`pneumophonic_analysis.segmentation.detect_phonation_bounds` (high-pass +
minimum-duration onset detection), wired into `PairedFeatureExtractor.extract`.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pneumophonic_analysis.paired_features import PairedFeatureExtractor

# Detection thresholds (match the corpus scan that surfaced the leaks)
ENERGY_OVER_MEDIAN = 6.0     # "loud" = energy > this * median
MIN_RUN_SEC = 0.40           # sustained loud run length
MAX_START_SEC = 0.30         # burst must be at the very start
MAX_VOICED_FRAC = 0.50       # burst is mostly unvoiced
MIN_DC_FRAC = 0.50           # burst energy concentrated below 100 Hz (DC-like)
VOICED_ONSET_WIN = 8         # frames that must be ~voiced to call it phonation
VOICED_ONSET_FRAC = 0.60


def longest_true_run(mask):
    """Return (start_index, length) of the longest run of True in a bool array."""
    best_len = best_start = cur = cur_start = 0
    for i, mv in enumerate(mask):
        if mv:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best_len:
                best_len, best_start = cur, cur_start
        else:
            cur = 0
    return best_start, best_len


def detect_leak_cut(df, stft, sr):
    """
    If the segment starts with a sync-pulse burst, return the frame index at
    which clean phonation begins (i.e. where to cut). Otherwise return None.
    """
    e = df['energy'].values.astype(float)
    v = df['voiced'].values.astype(float)
    t = df['time'].values
    if len(e) < 20:
        return None
    dt = float(np.median(np.diff(t)))
    pos = e[e > 0]
    mE = float(np.median(pos)) if len(pos) else 0.0
    if mE <= 0:
        return None

    s, L = longest_true_run(e > ENERGY_OVER_MEDIAN * mE)
    if L == 0:
        return None
    run_dur = L * dt
    vfrac = float(v[s:s + L].mean())
    ratio = float(e[s:s + L].mean() / mE)
    t0 = float(t[s])
    if not (t0 < MAX_START_SEC and run_dur >= MIN_RUN_SEC
            and vfrac < MAX_VOICED_FRAC and ratio >= ENERGY_OVER_MEDIAN):
        return None

    # Confirm it is a low-frequency / DC burst (sync pulse), not a high-freq
    # fricative that merely happens to be loud + unvoiced at the start.
    freqs = np.linspace(0.0, sr / 2.0, stft.shape[0])
    spec = stft[:, s:min(s + L, stft.shape[1])].mean(axis=1)
    dc_frac = float(spec[freqs < 100].sum() / spec.sum()) if spec.sum() > 0 else 0.0
    if dc_frac < MIN_DC_FRAC:
        return None

    # Cut at the first sustained voiced onset after the burst.
    burst_end = s + L
    cut = None
    for i in range(burst_end, len(v) - VOICED_ONSET_WIN):
        if v[i:i + VOICED_ONSET_WIN].mean() >= VOICED_ONSET_FRAC:
            cut = i
            break
    if cut is None:
        cut = burst_end
    cut = max(burst_end, cut - 3)   # tiny pre-onset margin
    if cut <= 0 or cut >= len(df) - 10:
        return None
    return int(cut)


def _write_h5(path, df, stft, mel, mfcc, attrs):
    """Write a repaired paired file with the same structure as save_hdf5."""
    with h5py.File(str(path), 'w') as f:
        g = f.create_group('aligned')
        for col in df.columns:
            g.create_dataset(col, data=df[col].values, compression='gzip')
        f.create_dataset('mfcc', data=mfcc, compression='gzip')
        f.create_dataset('mel', data=mel, compression='gzip')
        f.create_dataset('stft', data=stft, compression='gzip')
        for k, v in attrs.items():
            f.attrs[k] = v


def repair_file(path, backup_dir, dry_run):
    """Detect + (optionally) repair one file. Returns an info dict or None."""
    df, meta = PairedFeatureExtractor.load_hdf5(path)
    with h5py.File(str(path), 'r') as h:
        stft = h['stft'][:]
        mel = h['mel'][:]
        mfcc = h['mfcc'][:]
    sr = int(meta.get('sr_audio', 48000))

    cut = detect_leak_cut(df, stft, sr)
    if cut is None:
        return None

    cut_time = float(df['time'].values[cut])
    info = dict(file=path.stem, cut_frame=cut, removed_s=round(cut_time, 2),
                old_frames=len(df), new_frames=len(df) - cut,
                old_dur=round(float(df['time'].values[-1]), 2))
    if dry_run:
        return info

    # ---- back up original (never overwrite an existing backup) ----
    backup_dir.mkdir(parents=True, exist_ok=True)
    bk = backup_dir / path.name
    if not bk.exists():
        shutil.copy2(str(path), str(bk))

    # ---- slice aligned columns, recompute onset-relative fields ----
    new = df.iloc[cut:].reset_index(drop=True).copy()
    new['time'] = new['time'].values - new['time'].values[0]
    for vol, dcol in [('vcw', 'delta_vcw'), ('vrc', 'delta_vrc'), ('vab', 'delta_vab')]:
        if vol in new.columns and dcol in new.columns:
            new[dcol] = new[vol].values - new[vol].values[0]

    # ---- slice the stored spectral matrices to the same time origin ----
    cut_stft = min(cut, stft.shape[1])
    cut_mel = min(cut, mel.shape[1])
    cut_mfcc = min(int(round(cut_time * sr / 512.0)), mfcc.shape[1])   # mfcc uses hop 512
    new_stft = stft[:, cut_stft:]
    new_mel = mel[:, cut_mel:]
    new_mfcc = mfcc[:, cut_mfcc:]

    # ---- attrs ----
    attrs = dict(meta)
    attrs['n_frames'] = int(len(new))
    attrs['audio_duration_sec'] = float(new['time'].values[-1]) if len(new) > 1 else 0.0
    attrs['sync_repaired'] = 1
    attrs['sync_removed_sec'] = round(cut_time, 4)

    _write_h5(path, new, new_stft, new_mel, new_mfcc, attrs)
    info['new_dur'] = round(attrs['audio_duration_sec'], 2)
    return info


def main():
    ap = argparse.ArgumentParser(description="Repair sync-pulse leakage in paired HDF5 files")
    ap.add_argument('--paired-dir', type=Path, required=True,
                    help='Folder of paired *.h5 files')
    ap.add_argument('--backup-dir', type=Path, default=None,
                    help='Where to copy originals (default: <paired-dir>/../paired_backup_sync)')
    ap.add_argument('--dry-run', action='store_true', help='Detect and report only')
    ap.add_argument('--yes', action='store_true', help='Apply without confirmation')
    args = ap.parse_args()

    paired = args.paired_dir
    if not paired.exists():
        sys.exit(f"paired dir not found: {paired}")
    backup_dir = args.backup_dir or (paired.parent / "paired_backup_sync")

    files = sorted(paired.glob("*.h5"))
    print(f"Scanning {len(files)} files for sync-pulse leakage…")
    leaks = []
    for f in files:
        try:
            info = repair_file(f, backup_dir, dry_run=True)   # detect only
        except Exception as e:
            print(f"  ! {f.name}: {e}")
            continue
        if info:
            leaks.append((f, info))

    if not leaks:
        print("\nNo sync-leak segments found — nothing to repair.")
        return

    print(f"\n{len(leaks)} segment(s) with a leading sync pulse:")
    print(f"  {'file':22s} {'remove_s':>8} {'old_dur':>8} {'old→new frames':>16}")
    for _, i in leaks:
        print(f"  {i['file']:22s} {i['removed_s']:>8} {i['old_dur']:>8} "
              f"{i['old_frames']:>7}→{i['new_frames']:<7}")

    if args.dry_run:
        print("\n--dry-run: no files modified.")
        return

    if not args.yes:
        print(f"\nOriginals will be backed up to: {backup_dir}")
        ans = input("Proceed with repair? [y/N]: ").strip().lower()
        if ans not in ('y', 'yes'):
            print("Cancelled.")
            return

    print()
    ok = 0
    for f, _ in leaks:
        try:
            res = repair_file(f, backup_dir, dry_run=False)
            if res:
                print(f"  ✓ {res['file']}: removed {res['removed_s']}s "
                      f"({res['old_dur']}s → {res['new_dur']}s)")
                ok += 1
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")
    print(f"\nRepaired {ok}/{len(leaks)} segment(s). Backups in {backup_dir}")


if __name__ == '__main__':
    main()
