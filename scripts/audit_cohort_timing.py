#!/usr/bin/env python3
"""
Cohort audit: timing (sync-in-window / truncation) + OEP sync verification.
===========================================================================

For each subject folder, for each task:
  * TIMING — analyse the raw render to find the sync-pulse end and the true
    phonation end; flag `sync_in_window` (Excel start lands inside the sync) and
    `trunc_s` (seconds of phonation past the Excel stop).
  * SYNC   — detect the sync peaks in the task's OEP CSV and compare the nearest
    one to the Excel `falling edge` (`sync_disc_s`); a large discrepancy means
    the OEP↔audio alignment value is suspect.

Writes a per-(subject,task) CSV and prints a per-subject summary. Read-only.

Usage:
    python scripts/audit_cohort_timing.py --acq-dir "M:/.../voice acquisitions" \
        --subjects 20251112_FrRo 20251127_AlMo 20251112_AnGu \
        --out data_target/healthy_subjects/cohort_timing_audit.csv
    # all subjects:
    python scripts/audit_cohort_timing.py --acq-dir "M:/.../voice acquisitions" --all --out ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pneumophonic_analysis import create_config, DataLoader, Synchronizer
from batch_extract import TASK_MAP
from suggest_timing import analyse_render

TRUNC_FLAG = 0.5   # seconds of lost phonation to flag
SYNC_FLAG = 0.5    # seconds of falling-edge discrepancy to flag


def audit_subject(subj_dir, cfg, sync):
    sid = subj_dir.name.split('_', 1)[1] if '_' in subj_dir.name else subj_dir.name
    xlsx = subj_dir / f"{sid}_audio.xlsx"
    renders = subj_dir / "renders"
    if not xlsx.exists():
        return []
    try:
        t = pd.read_excel(xlsx, sheet_name='Timing')
    except Exception:
        return []
    lab = t.columns[0]
    t[lab] = t[lab].astype(str).str.strip()
    loader = DataLoader(subj_dir, cfg)
    oep_cache = {}
    rows = []
    for _, r in t.iterrows():
        task = str(r[lab]).strip()
        if task not in TASK_MAP:
            continue
        audio_file, suffix = TASK_MAP[task]
        try:
            ex_start = float(r['start']); ex_stop = float(r['stop']); fe = float(r['falling edge'])
        except Exception:
            ex_start = ex_stop = fe = np.nan

        # ---- timing audit (render) ----
        wav = renders / audio_file
        if not wav.exists():
            alt = renders / audio_file.replace('_2.wav', '.wav')
            wav = alt if alt.exists() else wav
        sync_in = None; trunc = np.nan; sync_end = phon_end = None
        if wav.exists():
            try:
                y, srr = librosa.load(str(wav), sr=cfg.audio.sample_rate, mono=True)
                d = analyse_render(y, srr)
                sync_end, phon_end = d['sync_end'], d['phon_end']
                if not np.isnan(ex_start) and sync_end is not None:
                    sync_in = bool(ex_start < sync_end)
                if not np.isnan(ex_stop) and phon_end is not None:
                    trunc = max(0.0, phon_end - ex_stop)
            except Exception:
                pass

        # ---- sync audit (OEP CSV) ----
        sync_disc = np.nan; nearest = np.nan; n_peaks = np.nan
        oep_csv = f"csv/{sid}_{suffix}.csv"
        if (subj_dir / oep_csv).exists():
            if suffix not in oep_cache:
                try:
                    oep = loader.load_oep_data(oep_csv)
                    dt_oep = np.median(np.diff(oep['time'].values))
                    fs_csv = int(round(1 / dt_oep)) if dt_oep > 0 else cfg.oep.fs_kinematic
                    pk = sync.detect_sync_onsets_oep(oep) / fs_csv
                    oep_cache[suffix] = pk
                except Exception:
                    oep_cache[suffix] = None
            pk = oep_cache[suffix]
            if pk is not None and len(pk):
                n_peaks = len(pk)
                if not np.isnan(fe):
                    nearest = float(pk[np.argmin(np.abs(pk - fe))])
                    sync_disc = abs(nearest - fe)

        rows.append(dict(
            subject=sid, task=task,
            ex_start=round(ex_start, 2), ex_stop=round(ex_stop, 2),
            sync_end=round(sync_end, 2) if sync_end is not None else np.nan,
            phon_end=round(phon_end, 2) if phon_end is not None else np.nan,
            sync_in_window=sync_in, trunc_s=round(trunc, 2) if trunc == trunc else np.nan,
            falling_edge=round(fe, 2) if fe == fe else np.nan,
            nearest_oep_peak=round(nearest, 2) if nearest == nearest else np.nan,
            n_oep_peaks=int(n_peaks) if n_peaks == n_peaks else np.nan,
            sync_disc_s=round(sync_disc, 2) if sync_disc == sync_disc else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Cohort timing + sync audit")
    ap.add_argument('--acq-dir', type=Path, required=True)
    ap.add_argument('--subjects', nargs='*', default=None, help='folder names (default: all)')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--out', type=Path, default=Path('data_target/healthy_subjects/cohort_timing_audit.csv'))
    args = ap.parse_args()

    cfg = create_config()
    sync = Synchronizer(cfg)

    if args.subjects:
        dirs = [args.acq_dir / s for s in args.subjects]
    else:
        dirs = sorted([d for d in args.acq_dir.iterdir()
                       if d.is_dir() and '_' in d.name
                       and (d / f"{d.name.split('_',1)[1]}_audio.xlsx").exists()])
    all_rows = []
    for d in dirs:
        if not d.exists():
            print(f"  ! missing: {d.name}"); continue
        print(f"auditing {d.name} …", flush=True)
        all_rows += audit_subject(d, cfg, sync)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("no rows"); return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n=== PER-SUBJECT SUMMARY ===")
    print(f"{'subject':10s} {'sync_in':>8} {'trunc>0.5':>10} {'sync_mismatch>0.5s':>18}")
    for sid, g in df.groupby('subject'):
        n_sync = int(g['sync_in_window'].fillna(False).sum())
        n_tr = int((g['trunc_s'] > TRUNC_FLAG).sum())
        n_sd = int((g['sync_disc_s'] > SYNC_FLAG).sum())
        flag = "  <-- review" if (n_sync or n_tr or n_sd) else ""
        print(f"{sid:10s} {n_sync:>8} {n_tr:>10} {n_sd:>18}{flag}")
    print(f"\ndetailed table -> {args.out}")


if __name__ == '__main__':
    main()
