"""
Diagnostic Tools 
================================================

utilities for inspecting data, verifying sync alignment and checking data integrity. 

Then select a tool from the menu.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pneumophonic_analysis import create_config, DataLoader, Synchronizer

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data_root"
DATA_TARGET  = PROJECT_ROOT / "data_target"

BATCHES = ["healthy_subjects", "pathological_subjects"]


# ==================================================================
# Helpers
# ==================================================================

def select_from_list(items, label):
    print(f"\n{label}:")
    for idx, item in enumerate(items):
        print(f"  [{idx}] {item}")
    while True:
        sel = input("Select by number: ")
        if sel.isdigit() and 0 <= int(sel) < len(items):
            return int(sel)
        print("Invalid selection.")

def select_batch():
    idx = select_from_list(BATCHES, "Available batches")
    return BATCHES[idx]

def select_subject(batch_name):
    source_dir = DATA_ROOT / batch_name
    subjects = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir() and (d / "renders").exists()
    ])
    names = [s.name for s in subjects]
    idx = select_from_list(names, f"Subjects in {batch_name}")
    return subjects[idx]


# ==================================================================
# Tool 1: Inspect OEP header
# ==================================================================

def inspect_oep_header():
    """Load an OEP CSV and print the first rows + column verification."""
    batch = select_batch()
    subj = select_subject(batch)
    sid = subj.name.split('_')[1] if '_' in subj.name else subj.name

    csv_dir = subj / "csv"
    if not csv_dir.exists():
        print(f"  No csv/ folder in {subj.name}")
        return

    csvs = sorted(csv_dir.glob("*.csv"))
    names = [f.name for f in csvs]
    idx = select_from_list(names, "Available CSVs")
    csv_path = csvs[idx]

    loader = DataLoader(subj)
    oep = loader.load_oep_data(f"csv/{csv_path.name}")

    fs = int(round(1 / np.median(np.diff(oep['time'].values)))) if len(oep) > 1 else 50
    print(f"\n--- {csv_path.name} ---")
    print(f"Shape: {oep.shape[0]} rows x {oep.shape[1]} columns")
    print(f"Sample rate: {fs} Hz (from time column)")
    print(f"Duration: {oep['time'].iloc[-1]:.1f}s")
    print(f"Columns: {oep.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{oep.head()}")

    # Verify A + B + C = tot_vol
    err = (oep['A'] + oep['B'] + oep['C'] - oep['tot_vol']).abs().mean()
    print(f"\nVerification |A + B + C - tot_vol| = {err:.6f} L")
    if err < 0.01:
        print("  Compartmental mapping OK")
    else:
        print("  WARNING: columns may not match expected layout")


# ==================================================================
# Tool 2: Check sync peaks
# ==================================================================

def check_sync_peaks():
    """Detect and display sync peaks in an OEP file."""
    batch = select_batch()
    subj = select_subject(batch)
    sid = subj.name.split('_')[1] if '_' in subj.name else subj.name

    csv_dir = subj / "csv"
    csvs = sorted(csv_dir.glob("*.csv"))
    names = [f.name for f in csvs]
    idx = select_from_list(names, "Available CSVs")

    loader = DataLoader(subj)
    oep = loader.load_oep_data(f"csv/{csvs[idx].name}")
    sync = Synchronizer()
    peaks = sync.detect_sync_onsets_oep(oep)
    fs = int(round(1 / np.median(np.diff(oep['time'].values)))) if len(oep) > 1 else 50

    print(f"\n--- Sync peaks in {csvs[idx].name} ---")
    print(f"OEP duration: {oep['time'].iloc[-1]:.1f}s  (rate {fs} Hz)")
    print(f"Number of peaks: {len(peaks)}")
    print(f"Peak positions (samples): {peaks}")
    print(f"Peak times (s): {peaks / fs}")

    # Compare with Excel if available
    excel_path = subj / f"{sid}_audio.xlsx"
    if excel_path.exists():
        try:
            timing = pd.read_excel(excel_path, sheet_name="Timing")
            print(f"\nExcel falling edges:")
            label_col = timing.columns[0]
            for _, row in timing.iterrows():
                task = row[label_col]
                fe = row.get('falling edge', '?')
                print(f"  {task}: {fe}s")
        except Exception as e:
            print(f"\nCould not read Excel: {e}")


# ==================================================================
# Tool 3: List subject inventory
# ==================================================================

def list_inventory():
    """List all subjects and their available data files."""
    batch = select_batch()
    source_dir = DATA_ROOT / batch

    subjects = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir() and (d / "renders").exists()
    ])

    print(f"\n{'Subject':<30} {'Audio':<8} {'CSVs':<8} {'Sync':<6} {'Excel':<6}")
    print("-" * 60)

    for subj in subjects:
        sid = subj.name.split('_')[1] if '_' in subj.name else subj.name

        n_audio = len(list((subj / "renders").glob("*.wav"))) if (subj / "renders").exists() else 0
        n_csv = len(list((subj / "csv").glob("*.csv"))) if (subj / "csv").exists() else 0
        has_sync = "Yes" if (subj / "sync_signal.wav").exists() else "No"
        has_excel = "Yes" if (subj / f"{sid}_audio.xlsx").exists() else "No"

        print(f"{subj.name:<30} {n_audio:<8} {n_csv:<8} {has_sync:<6} {has_excel:<6}")

    print(f"\nTotal: {len(subjects)} subjects")


# ==================================================================
# Tool 4: Check extraction coverage
# ==================================================================

def check_coverage():
    """Compare what's been extracted (HDF5) vs what's available."""
    batch = select_batch()
    paired_dir = DATA_TARGET / batch / "paired"

    if not paired_dir.exists():
        print(f"\nNo paired/ folder found. Run batch_extract.py first.")
        return

    h5_files = sorted(paired_dir.glob("*.h5"))
    print(f"\n{len(h5_files)} HDF5 files in {paired_dir.relative_to(PROJECT_ROOT)}")

    # Group by subject
    by_subject = {}
    for h5 in h5_files:
        parts = h5.stem.rsplit('_', 1)
        sid = parts[0] if len(parts) > 1 else h5.stem
        task = parts[1] if len(parts) > 1 else '?'
        by_subject.setdefault(sid, []).append(task)

    print(f"\n{'Subject':<20} {'Tasks extracted'}")
    print("-" * 60)
    for sid in sorted(by_subject):
        tasks = sorted(by_subject[sid])
        print(f"{sid:<20} {len(tasks):>2} tasks: {', '.join(tasks)}")

    # Check summary CSV
    summary_path = paired_dir / "extraction_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        statuses = summary['status'].value_counts()
        print(f"\nExtraction summary:")
        for status, count in statuses.items():
            print(f"  {status}: {count}")


# ==================================================================
# Tool 5: Quick HDF5 inspector
# ==================================================================

def inspect_h5():
    """Load and display info about an HDF5 paired file."""
    batch = select_batch()
    paired_dir = DATA_TARGET / batch / "paired"

    if not paired_dir.exists():
        print(f"\nNo paired/ folder found.")
        return

    h5_files = sorted(paired_dir.glob("*.h5"))
    if not h5_files:
        print(f"\nNo .h5 files found.")
        return

    names = [f.name for f in h5_files]
    idx = select_from_list(names, "Available HDF5 files")

    from pneumophonic_analysis.paired_features import PairedFeatureExtractor
    df, meta = PairedFeatureExtractor.load_hdf5(h5_files[idx])

    print(f"\n--- {h5_files[idx].name} ---")
    print(f"\nMetadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print(f"\nDataFrame: {df.shape[0]} frames x {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Duration: {df['time'].iloc[-1]:.2f}s")

    voiced = df[df['voiced'] == 1.0]
    print(f"Voiced frames: {len(voiced)} ({len(voiced)/len(df)*100:.1f}%)")

    if len(voiced) > 0 and 'f0' in voiced.columns:
        f0_valid = voiced['f0'].dropna()
        if len(f0_valid) > 0:
            print(f"F0: {f0_valid.mean():.1f} +/- {f0_valid.std():.1f} Hz "
                  f"(range: {f0_valid.min():.1f}-{f0_valid.max():.1f})")

    err = (df['vrc'] + df['vab'] - df['vcw']).abs().mean()
    print(f"Compartmental check |Vrc+Vab-Vcw|: {err:.6f} L")

    print(f"\nFirst 3 rows:\n{df.head(3)}")


# ==================================================================
# Menu
# ==================================================================

TOOLS = {
    '1': ('Inspect OEP header (columns, values, compartmental check)', inspect_oep_header),
    '2': ('Check sync peaks in OEP file', check_sync_peaks),
    '3': ('List subject inventory (files available)', list_inventory),
    '4': ('Check extraction coverage (HDF5 vs available)', check_coverage),
    '5': ('Inspect HDF5 paired file', inspect_h5),
}


def main():
    print("\n" + "=" * 50)
    print("  PNEUMOPHONIC PIPELINE — DIAGNOSTIC TOOLS")
    print("=" * 50)

    for key, (desc, _) in TOOLS.items():
        print(f"  [{key}] {desc}")
    print(f"  [q] Quit")

    while True:
        choice = input("\nSelect tool: ").strip().lower()
        if choice == 'q':
            break
        if choice in TOOLS:
            TOOLS[choice][1]()
            print("\n" + "-" * 50)
        else:
            print("Invalid selection.")


if __name__ == '__main__':
    main()