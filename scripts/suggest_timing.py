#!/usr/bin/env python3
"""
Suggest corrected start/stop timings from the raw renders.
==========================================================

For a subject's raw acquisition folder, this:
  * reads the existing `Timing` sheet (start / stop / falling edge);
  * for each task, analyses the raw render to find the **sync-pulse end**
    (loud low-frequency <120 Hz plateau) and the **true phonation start/end**
    (speech-band 300-4000 Hz energy, after the sync);
  * proposes start = just after the sync / just before phonation, and
    stop = just after phonation ends;
  * prints a review table (Excel vs suggested) and flags sync-in-window and
    truncated-phrase cases; optionally saves an overview figure.

It does NOT modify anything (review only) unless --apply is given, in which
case it writes the suggested start/stop back to the Timing sheet (after backing
up the workbook). The `falling edge` column is never touched.

Usage:
    python scripts/suggest_timing.py --subject-dir "M:/.../20251112_FrRo" --out-fig results/FrRo_timing.png
    python scripts/suggest_timing.py --subject-dir "M:/.../20251112_FrRo" --apply   # write corrected sheet
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt

# task -> render filename (matches batch_extract.TASK_MAP; a_7 file is phonema_a_7.wav)
TASK_RENDER = {
    'a': 'a.wav', 'e': 'e.wav', 'i': 'i.wav', 'o': 'o.wav', 'u': 'u.wav',
    'a_2': 'phonema_a_2.wav', 'a_3': 'phonema_a_3.wav', 'a_7': 'phonema_a_7.wav',
    'r': 'r.wav', 'f_1': 'phrase_1.wav', 'f_2': 'phrase_2.wav', 'f_3': 'phrase_3.wav',
    'f_4': 'phrase_4.wav', 'f_5': 'phrase_5.wav', 'testo': 'testo.wav',
}
HOP = 512
SR = 48000


def _band_rms(y, sr, lo, hi):
    nyq = sr / 2.0
    if lo <= 0:
        b, a = butter(4, hi / nyq, 'low')
    elif hi >= nyq:
        b, a = butter(4, lo / nyq, 'high')
    else:
        b, a = butter(4, [lo / nyq, hi / nyq], 'band')
    yf = filtfilt(b, a, y)
    return librosa.feature.rms(y=yf, frame_length=2048, hop_length=HOP)[0]


def _runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def analyse_render(y, sr):
    """Return detected sync window + phonation bounds (seconds) for one render."""
    low = _band_rms(y, sr, 0, 120)       # sync pulse lives here
    sp = _band_rms(y, sr, 300, 4000)     # speech/voice formants
    t = np.arange(len(low)) * HOP / sr

    # sync = first low-freq run >0.2 s above 0.5*max (the pulse dominates <120 Hz)
    sync_start = sync_end = None
    if low.max() > 0:
        for a, b in _runs(low > 0.5 * low.max()):
            if (b - a) * HOP / sr > 0.2:
                sync_start, sync_end = float(t[a]), float(t[min(b, len(t) - 1)])
                break

    # phonation = speech-band energy after the sync
    after = 0 if sync_end is None else int(np.searchsorted(t, sync_end + 0.05))
    phon_start = phon_end = None
    if sp.max() > 0:
        idx = np.where(sp[after:] > 0.12 * sp.max())[0]
        if len(idx):
            phon_start = float(t[after + idx[0]])
            phon_end = float(t[after + idx[-1]])
    return dict(low=low, sp=sp, t=t, sync_start=sync_start, sync_end=sync_end,
                phon_start=phon_start, phon_end=phon_end)


def main():
    ap = argparse.ArgumentParser(description="Suggest corrected start/stop from raw renders")
    ap.add_argument('--subject-dir', type=Path, required=True)
    ap.add_argument('--sheet', default='Timing')
    ap.add_argument('--out-fig', type=Path, default=None)
    ap.add_argument('--apply', action='store_true',
                    help='Write suggested start/stop back into the Timing sheet (backs up first)')
    ap.add_argument('--start-margin', type=float, default=0.10)
    ap.add_argument('--stop-margin', type=float, default=0.15)
    args = ap.parse_args()

    subj_dir = args.subject_dir
    sid = subj_dir.name.split('_', 1)[1] if '_' in subj_dir.name else subj_dir.name
    xlsx = subj_dir / f"{sid}_audio.xlsx"
    renders = subj_dir / "renders"
    if not xlsx.exists():
        sys.exit(f"timing workbook not found: {xlsx}")
    if not renders.exists():
        sys.exit(f"renders folder not found: {renders}")

    tdf = pd.read_excel(xlsx, sheet_name=args.sheet)
    label_col = tdf.columns[0]
    tdf[label_col] = tdf[label_col].astype(str).str.strip()

    rows, panels = [], []
    for _, r in tdf.iterrows():
        task = str(r[label_col]).strip()
        fname = TASK_RENDER.get(task)
        if fname is None:
            continue
        wav = renders / fname
        if not wav.exists():
            alt = renders / fname.replace('.wav', '_2.wav')
            wav = alt if alt.exists() else wav
        ex_start = float(r['start']) if pd.notna(r.get('start')) else np.nan
        ex_stop = float(r['stop']) if pd.notna(r.get('stop')) else np.nan
        if not wav.exists():
            rows.append(dict(task=task, render='MISSING', ex_start=ex_start, ex_stop=ex_stop))
            continue
        y, sr = librosa.load(str(wav), sr=SR, mono=True)
        d = analyse_render(y, sr)
        sug_start = sug_stop = np.nan
        if d['phon_start'] is not None:
            floor = (d['sync_end'] or 0.0) + 0.05
            sug_start = round(max(floor, d['phon_start'] - args.start_margin), 2)
            sug_stop = round(d['phon_end'] + args.stop_margin, 2)
        sync_in = (not np.isnan(ex_start)) and d['sync_end'] is not None and ex_start < d['sync_end']
        trunc = (not np.isnan(ex_stop)) and d['phon_end'] is not None and d['phon_end'] > ex_stop + 0.1
        flags = []
        if sync_in:
            flags.append('SYNC-IN-WINDOW')
        if trunc:
            flags.append(f"TRUNCATED(-{d['phon_end']-ex_stop:.1f}s)")
        rows.append(dict(
            task=task, render=wav.name,
            sync=f"{d['sync_start']:.2f}-{d['sync_end']:.2f}" if d['sync_end'] else "—",
            phon=f"{d['phon_start']:.2f}-{d['phon_end']:.2f}" if d['phon_end'] else "—",
            ex_start=round(ex_start, 2), ex_stop=round(ex_stop, 2),
            sug_start=sug_start, sug_stop=sug_stop, flags=" ".join(flags)))
        panels.append((task, d, ex_start, ex_stop, sug_start, sug_stop))

    out = pd.DataFrame(rows)
    pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
    print(f"\nSubject: {sid}   ({xlsx})")
    print(out.to_string(index=False))
    n_sync = out['flags'].str.contains('SYNC', na=False).sum() if 'flags' in out else 0
    n_trunc = out['flags'].str.contains('TRUNC', na=False).sum() if 'flags' in out else 0
    print(f"\nflags: SYNC-IN-WINDOW={n_sync}  TRUNCATED={n_trunc}")

    if args.out_fig and panels:
        import matplotlib
        matplotlib.use('Agg'); import matplotlib.pyplot as plt
        n = len(panels); ncol = 3; nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.4 * nrow))
        for ax, (task, d, es, ep, ss, sp_) in zip(np.atleast_1d(axes).ravel(), panels):
            ax.plot(d['t'], d['low'], color='crimson', lw=0.8, label='<120Hz (sync)')
            ax.plot(d['t'], d['sp'], color='steelblue', lw=0.8, label='speech 300-4kHz')
            if not np.isnan(es):
                ax.axvspan(es, ep, color='orange', alpha=0.15)
                ax.axvline(es, color='orange', ls='--', lw=1); ax.axvline(ep, color='orange', ls='--', lw=1)
            if not np.isnan(ss):
                ax.axvline(ss, color='green', ls='-', lw=1.2); ax.axvline(sp_, color='green', ls='-', lw=1.2)
            xmax = (sp_ if not np.isnan(sp_) else (ep if not np.isnan(ep) else d['t'][-1]))
            ax.set_xlim(0, min(d['t'][-1], (xmax or 6) + 2))
            ax.set_title(task, fontsize=9); ax.set_yticks([])
        for ax in np.atleast_1d(axes).ravel()[n:]:
            ax.set_visible(False)
        fig.suptitle(f"{sid}: orange dashed = Excel window, green solid = suggested", fontsize=11)
        plt.tight_layout(); args.out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_fig, dpi=100, bbox_inches='tight'); plt.close(fig)
        print(f"figure -> {args.out_fig}")

    if args.apply:
        bak = xlsx.with_suffix('.xlsx.bak')
        if not bak.exists():
            shutil.copy2(str(xlsx), str(bak))
        sug = {row['task']: (row['sug_start'], row['sug_stop'])
               for row in rows if 'sug_start' in row and not pd.isna(row.get('sug_start'))}
        new = tdf.copy()
        for i, rr in new.iterrows():
            tk = str(rr[label_col]).strip()
            if tk in sug:
                new.at[i, 'start'], new.at[i, 'stop'] = sug[tk]
        # rewrite just the Timing sheet, preserve the others
        all_sheets = pd.read_excel(xlsx, sheet_name=None)
        all_sheets[args.sheet] = new
        with pd.ExcelWriter(xlsx, engine='openpyxl') as w:
            for name, sdf in all_sheets.items():
                sdf.to_excel(w, sheet_name=name, index=False)
        print(f"\nAPPLIED suggested start/stop to {xlsx} (backup: {bak.name})")


if __name__ == '__main__':
    main()
