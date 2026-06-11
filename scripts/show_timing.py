#!/usr/bin/env python3
"""
Show & validate a subject's timing Excel (the source of start / stop / falling edge).
=====================================================================================

Opens a `<SubjectID>_audio.xlsx` (file browser if no --file given), prints the
`Timing` sheet with its real column headers, shows which columns the extractor
actually reads, and flags problems (missing columns, non-numeric / NaN values,
start >= stop, unknown task labels).

Usage:
    python scripts/show_timing.py                         # pick the .xlsx in a file browser
    python scripts/show_timing.py --file /path/FrRo_audio.xlsx
    python scripts/show_timing.py --file ... --task f_1   # only that task row
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pneumophonic_analysis import select_file_gui

# task labels the pipeline understands (must match batch_extract.TASK_MAP keys)
TASK_KEYS = {'a', 'e', 'i', 'o', 'u', 'a_2', 'a_3', 'a_7', 'r',
             'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'testo'}


def _as_float(v):
    """Parse a value as float, tolerating European comma decimals / text."""
    try:
        return float(v)
    except Exception:
        try:
            return float(str(v).strip().replace(' ', '').replace(',', '.'))
        except Exception:
            return None


def main():
    ap = argparse.ArgumentParser(description="Show/validate a subject timing Excel")
    ap.add_argument('--file', type=Path, default=None)
    ap.add_argument('--sheet', default='Timing')
    ap.add_argument('--task', default=None, help='Only show this task row')
    args = ap.parse_args()

    path = args.file
    if path is None:
        print("Select a <SubjectID>_audio.xlsx file…")
        path = select_file_gui("Select a subject timing Excel",
                               [("Excel", "*.xlsx *.xls"), ("All files", "*.*")])
    if not path or not Path(path).exists():
        sys.exit("No file selected / not found.")
    path = Path(path)

    xl = pd.ExcelFile(path)
    print(f"\nFile  : {path}")
    print(f"Sheets: {xl.sheet_names}")
    sheet = args.sheet if args.sheet in xl.sheet_names else xl.sheet_names[0]
    if sheet != args.sheet:
        print(f"  (sheet '{args.sheet}' not found — using '{sheet}')")

    df = pd.read_excel(path, sheet_name=sheet)
    print(f"\n[{sheet}] {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"COLUMNS: {list(df.columns)}")

    # Which columns does the extractor read? (exact names, case-insensitive)
    norm = {str(c).strip().lower(): c for c in df.columns}
    c_start, c_stop, c_fe = norm.get('start'), norm.get('stop'), norm.get('falling edge')
    label_col = df.columns[0]
    print("\nThe extractor reads, BY HEADER NAME:")
    print(f"   task label : first column  -> {label_col!r}")
    print(f"   start       : {c_start!r}")
    print(f"   stop        : {c_stop!r}")
    print(f"   falling edge: {c_fe!r}")
    extras = [c for c in df.columns if c not in (label_col, c_start, c_stop, c_fe)]
    if extras:
        print(f"   (ignored extra columns: {extras})")

    print("\n--- rows ---")
    show = df
    if args.task:
        show = df[df[label_col].astype(str).str.strip() == args.task]
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 60)
    print(show.to_string(index=False))

    # ---- validation ----
    print("\n--- checks ---")
    missing = [n for n, c in [('start', c_start), ('stop', c_stop), ('falling edge', c_fe)] if c is None]
    if missing:
        print(f"  ✗ MISSING expected column(s): {missing}")
        print("    Extraction reads these by exact name — rename your columns to "
              "'start', 'stop', 'falling edge' (falling edge lowercase, with the space).")
    if c_start and c_stop:
        any_issue = False
        for _, r in df.iterrows():
            lab = str(r[label_col]).strip()
            s, e = _as_float(r[c_start]), _as_float(r[c_stop])
            flags = []
            if s is None or e is None:
                flags.append(f"start/stop not numeric ({r[c_start]!r}, {r[c_stop]!r})")
            else:
                if s >= e:
                    flags.append(f"start>=stop ({s} >= {e})")
                if e - s > 120:
                    flags.append(f"window {e - s:.0f}s long — are these seconds?")
            if lab not in TASK_KEYS and lab.lower() != 'nan':
                flags.append("task label not in TASK_MAP")
            if flags:
                any_issue = True
                print(f"  ⚠ {lab}: {'; '.join(flags)}")
        if not any_issue:
            print("  ✓ start/stop look numeric and ordered; labels recognized.")
    print()


if __name__ == '__main__':
    main()
