#!/usr/bin/env python3
"""
Re-extract one subject from the raw renders with corrected timings.
===================================================================

Runs the M1 paired extraction for every task of a subject, using either the
render-derived **suggested** start/stop (default) or the Excel `start`/`stop`.
The OEP `falling edge` is always read from the Excel `Timing` sheet.

Writes the .h5 to --out-dir (use a review folder to keep the live corpus
untouched, or point it at data_target/<batch>/paired to commit).

Usage:
    python scripts/reextract_subject.py \
        --subject-dir "M:/.../20251112_FrRo" \
        --out-dir data_target/healthy_subjects/paired_FrRo_reextract \
        --timing suggested
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa  # noqa: F401  (ensures the audio backend is importable)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pneumophonic_analysis import create_config
from pneumophonic_analysis.paired_features import PairedFeatureExtractor
from batch_extract import TASK_MAP                  # task -> (audio_file, oep_suffix)
from suggest_timing import analyse_render           # render -> sync/phonation bounds


def load_timing(xlsx, sheet='Timing'):
    df = pd.read_excel(xlsx, sheet_name=sheet)
    lab = df.columns[0]
    df[lab] = df[lab].astype(str).str.strip()
    return df, lab


def main():
    ap = argparse.ArgumentParser(description="Re-extract a subject with corrected timings")
    ap.add_argument('--subject-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--timing', choices=['suggested', 'excel'], default='suggested')
    ap.add_argument('--start-margin', type=float, default=0.10)
    ap.add_argument('--stop-margin', type=float, default=0.15)
    ap.add_argument('--tasks', nargs='*', default=None, help='subset of task labels')
    args = ap.parse_args()

    subj = args.subject_dir
    sid = subj.name.split('_', 1)[1] if '_' in subj.name else subj.name
    xlsx = subj / f"{sid}_audio.xlsx"
    renders = subj / "renders"
    if not xlsx.exists():
        sys.exit(f"timing workbook not found: {xlsx}")

    tdf, lab = load_timing(xlsx)
    fe = {str(r[lab]).strip(): r.get('falling edge') for _, r in tdf.iterrows()}
    ex = {str(r[lab]).strip(): (r.get('start'), r.get('stop')) for _, r in tdf.iterrows()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = create_config(data_root=subj.parent, output_root=args.out_dir.parent)
    extractor = PairedFeatureExtractor(config)

    tasks = args.tasks or [t for t in tdf[lab].tolist() if t in TASK_MAP]
    print(f"Subject {sid}: re-extracting {len(tasks)} task(s) [{args.timing} timings] -> {args.out_dir}")
    rows = []
    for task in tasks:
        if task not in TASK_MAP:
            print(f"  - {task}: not in TASK_MAP, skip"); continue
        audio_file, oep_suffix = TASK_MAP[task]
        # Resolve the render, tolerating the '_2' take-suffix naming variants seen
        # in the raw data (e.g. u -> u_2.wav, phrase_2 -> phrase_2_2.wav, o -> o_2.wav).
        stem = audio_file[:-4] if audio_file.endswith('.wav') else audio_file
        candidates = [audio_file, audio_file.replace('_2.wav', '.wav'), f"{stem}_2.wav"]
        wav = next((renders / c for c in candidates if (renders / c).exists()), None)
        if wav is None:
            print(f"  - {task}: render missing ({audio_file}), skip"); continue
        audio_file = wav.name
        oep_csv = f"csv/{sid}_{oep_suffix}.csv"
        if not (subj / oep_csv).exists():
            print(f"  - {task}: OEP csv missing ({oep_csv}), skip"); continue
        falling = fe.get(task)
        if falling is None or (isinstance(falling, float) and np.isnan(falling)):
            print(f"  - {task}: no falling edge, skip"); continue

        # timing
        if args.timing == 'excel':
            start, stop = ex[task]
        else:
            y, sr = librosa.load(str(wav), sr=config.audio.sample_rate, mono=True)
            d = analyse_render(y, sr)
            if d['phon_start'] is None:
                print(f"  - {task}: no phonation detected, skip"); continue
            floor = (d['sync_end'] or 0.0) + 0.05
            start = round(max(floor, d['phon_start'] - args.start_margin), 2)
            stop = round(d['phon_end'] + args.stop_margin, 2)

        try:
            paired = extractor.extract(
                subject_folder=subj, task_name=task, audio_filename=audio_file,
                oep_csv_path=oep_csv, audio_start_sec=float(start),
                audio_end_sec=float(stop), oep_falling_edge_sec=float(falling))
            out_h5 = args.out_dir / f"{sid}_{task}.h5"
            PairedFeatureExtractor.save_hdf5(paired, out_h5)
            df = paired.dataframe
            voiced = float((df['voiced'] == 1.0).mean()) * 100
            err = float((df['vrc'] + df['vab'] - df['vcw']).abs().mean())
            dur = float(paired.metadata.get('audio_duration_sec', stop - start))
            rows.append(dict(task=task, win=f"{start}-{stop}", dur=round(dur, 2),
                             frames=df.shape[0], voiced_pct=round(voiced, 0),
                             comp_err=f"{err:.1e}"))
            print(f"  ✓ {task}: {df.shape[0]} frames, {dur:.1f}s, voiced {voiced:.0f}%")
        except Exception as e:
            print(f"  ✗ {task}: {e}")

    if rows:
        print("\n--- summary ---")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
