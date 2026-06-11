"""
Pneumophonic Pipeline — Control Center
======================================

One entry point for the whole project. Run from the project root:

    python MAIN_UX.py

From the menu you can reach every step of the pipeline — extraction, plotting,
M2 correlation, M3 modelling, acoustic/single-subject analysis, and diagnostics.
Press  q  (or Ctrl+C at the menu) to quit at any time; Ctrl+C while a task is
running aborts it and returns you to the menu.

The argparse-based analyses (L3, M3) are launched with the standard project
paths for the *current batch* shown at the top of the menu (change it with `b`).
The interactive scripts ask their own questions.
"""
import logging
import subprocess
import sys
from pathlib import Path

# Make unicode safe on Windows consoles (cp1252) — no-op elsewhere.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_ROOT = PROJECT_ROOT / "data_root"
DATA_TARGET = PROJECT_ROOT / "data_target"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"

BATCHES = ["healthy_subjects", "pathological_subjects"]

# Session state — the batch the path-based analyses operate on.
session = {"batch": "healthy_subjects"}


# ---------------------------------------------------------------------------
# Path helpers (current batch)
# ---------------------------------------------------------------------------

def paired_dir() -> Path:
    return DATA_TARGET / session["batch"] / "paired"


def metadata_path() -> Path:
    return DATA_ROOT / session["batch"] / "subjects_metadata.csv"


def n_paired() -> int:
    p = paired_dir()
    return len(list(p.glob("*.h5"))) if p.exists() else 0


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

def run_script(script: str, args=None, long_running: bool = False) -> None:
    """Launch a script as a subprocess (inherits this console, so file
    browsers and prompts work). Ctrl+C aborts it back to the menu."""
    args = [str(a) for a in (args or [])]
    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        script_path = PROJECT_ROOT / script
    if not script_path.exists():
        print(f"  ✗ Script not found: {script}")
        return

    print(f"\n▶ {script_path.name} {' '.join(args)}".rstrip())
    if long_running:
        print("  (this can take a while — Ctrl+C aborts back to the menu)")
    print("-" * 64)
    try:
        subprocess.run([sys.executable, str(script_path), *args], cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n  ⏹ Aborted — returning to menu.")
    print("-" * 64)


# ---------------------------------------------------------------------------
# Input resolvers (with graceful fallbacks / file browser)
# ---------------------------------------------------------------------------

def resolve_paired_dir():
    """Return a paired/ dir containing .h5 files, or None."""
    p = paired_dir()
    if p.exists() and any(p.glob("*.h5")):
        return p
    print(f"  ✗ No paired/*.h5 found at {p}")
    print("    Run extraction (option 2) first, or point me at a paired/ folder.")
    if input("    Browse for a paired/ folder? [y/N]: ").strip().lower() == "y":
        from pneumophonic_analysis import select_folders_gui
        picked = select_folders_gui("Select the paired/ folder (with .h5 files)",
                                    initialdir=DATA_TARGET, multiple=False)
        if picked and any(picked[0].glob("*.h5")):
            return picked[0]
        print("    No valid folder selected.")
    return None


def resolve_metadata():
    """Return the metadata CSV/XLSX path, or None."""
    m = metadata_path()
    if m.exists():
        return m
    print(f"  ✗ Metadata not found at {m}")
    if input("    Browse for the metadata CSV/XLSX? [y/N]: ").strip().lower() == "y":
        from pneumophonic_analysis import select_file_gui
        f = select_file_gui("Select subjects metadata",
                            [("CSV/Excel", "*.csv *.xlsx *.xls"), ("All", "*.*")],
                            initialdir=DATA_ROOT)
        if f:
            return f
        print("    No file selected.")
    return None


# ---------------------------------------------------------------------------
# Handlers — M1 extraction
# ---------------------------------------------------------------------------

def h_extract_single():
    run_script("test_paired.py")


def h_extract_batch():
    run_script("batch_extract.py")


# ---------------------------------------------------------------------------
# Handlers — visualization
# ---------------------------------------------------------------------------

def h_plot_single():
    run_script("plot_paired_features.py")


def h_plot_batch():
    run_script("batch_plot_paired.py", long_running=True)


# ---------------------------------------------------------------------------
# Handlers — M2 correlation
# ---------------------------------------------------------------------------

def h_m2_correlation():
    run_script("m2_correlation.py", long_running=True)


def h_l3_stratified():
    pdir, meta = resolve_paired_dir(), resolve_metadata()
    if not (pdir and meta):
        return
    out = DATA_TARGET / session["batch"] / "M2_stratified"
    run_script("analyze_l3_stratified.py",
               ["--paired-dir", pdir, "--metadata", meta, "--output-dir", out],
               long_running=True)
    if (out / "frc_stratified_summary.xlsx").exists():
        print("\n  → generating summary plots…")
        run_script("make_m2_summary_plots.py", ["--results-dir", out])


# ---------------------------------------------------------------------------
# Handlers — M3 modelling
# ---------------------------------------------------------------------------

def h_m3_compartmental():
    pdir, meta = resolve_paired_dir(), resolve_metadata()
    if not (pdir and meta):
        return
    out = DATA_TARGET / session["batch"] / "M3_compartmental"
    run_script("analyze_compartmental_regression.py",
               ["--paired-dir", pdir, "--metadata", meta, "--output-dir", out],
               long_running=True)
    run_script("diagnose_compartmental_signal.py",
               ["--paired-dir", pdir, "--metadata", meta, "--output-dir", out],
               long_running=True)


def h_m3_frc():
    pdir, meta = resolve_paired_dir(), resolve_metadata()
    if not (pdir and meta):
        return
    out = DATA_TARGET / session["batch"] / "M3_frc_classification"
    run_script("analyze_frc_classification.py",
               ["--paired-dir", pdir, "--metadata", meta, "--output-dir", out],
               long_running=True)
    run_script("analyze_frc_window_sweep.py",
               ["--paired-dir", pdir, "--metadata", meta, "--output-dir", out],
               long_running=True)


# ---------------------------------------------------------------------------
# Handlers — acoustic / single-subject
# ---------------------------------------------------------------------------

def h_single_subject():
    from pneumophonic_analysis import select_subject_folders_gui
    print("  Select ONE subject folder to analyze (file browser)…")
    picked = select_subject_folders_gui("Select a subject folder to analyze",
                                        initialdir=DATA_ROOT if DATA_ROOT.exists() else PROJECT_ROOT,
                                        multiple=False)
    if not picked:
        print("  No folder selected.")
        return
    out = RESULTS_DIR / "acoustic" / picked[0].name
    run_script("analyze_single_subject.py", [picked[0], "-o", out], long_running=True)


def h_trill():
    from pneumophonic_analysis import select_file_gui
    print("  Select a trill .wav file (file browser)…")
    f = select_file_gui("Select a trill .wav file",
                        [("WAV audio", "*.wav"), ("All", "*.*")],
                        initialdir=DATA_ROOT if DATA_ROOT.exists() else PROJECT_ROOT)
    if not f:
        print("  No file selected.")
        return
    args = [f, "-o", RESULTS_DIR / "trill", "--no-show"]
    frc = input("  FRC crossing time in seconds (blank to skip): ").strip()
    if frc:
        args += ["--frc-time", frc]
    run_script("analyze_trill_modulation.py", args)


def h_legacy():
    run_legacy_analysis()


# ---------------------------------------------------------------------------
# Handlers — diagnostics & project
# ---------------------------------------------------------------------------

def h_tools():
    run_script("tools.py")


def h_counts():
    run_script("count_check.py")


def h_h5_diag():
    run_script("tool_h5_id_match.py")


def h_repair_sync():
    pdir = resolve_paired_dir()
    if not pdir:
        return
    run_script("repair_sync_leak.py", ["--paired-dir", pdir])


def h_show_timing():
    run_script("show_timing.py")


def h_jupyter():
    print(f"\n▶ Launching Jupyter Lab on {NOTEBOOKS_DIR.relative_to(PROJECT_ROOT)}")
    print("  (Ctrl+C here stops the server and returns to the menu)")
    print("-" * 64)
    try:
        subprocess.run([sys.executable, "-m", "jupyter", "lab", str(NOTEBOOKS_DIR)],
                       cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n  ⏹ Jupyter stopped — returning to menu.")
    except FileNotFoundError:
        print("  ✗ Jupyter not installed in this environment (pip install jupyterlab).")
    print("-" * 64)


# ---------------------------------------------------------------------------
# Legacy Zocco acoustic pipeline (in-process)
# ---------------------------------------------------------------------------

def run_legacy_analysis():
    """Original acoustic-only analysis via PneumophonicPipeline."""
    from pneumophonic_analysis import run_pipeline

    batch = change_batch(announce=False) or session["batch"]
    data_root = DATA_ROOT / batch
    data_target = DATA_TARGET / batch

    available = sorted([d.name for d in data_root.glob("*_*") if d.is_dir()]) if data_root.exists() else []
    subjects_to_run = None
    if available:
        print(f"\n{len(available)} subjects found in {batch}.")
        raw = input("Subjects to EXCLUDE (comma-separated), or Enter for all: ").strip()
        if raw:
            excluded = {s.strip() for s in raw.split(",")}
            subjects_to_run = [s for s in available if s not in excluded]
            print(f"Running with {len(subjects_to_run)} subject(s).")
    else:
        print(f"\n  ✗ No subject folders found in {data_root} — restore data or use extraction first.")
        return

    results = run_pipeline(
        data_root=data_root,
        output_root=data_target,
        subjects=subjects_to_run,
        tasks=["vowel", "trill", "glide"],
    )
    print(f"\nAnalyzed {results.n_subjects} subjects | "
          f"Success: {results.n_successful}, Failed: {results.n_failed}")


# ---------------------------------------------------------------------------
# Menu definition
# ---------------------------------------------------------------------------

SECTIONS = [
    ("M1 · EXTRACTION  (produce .h5 from raw recordings)", [
        ("Extract paired features — SINGLE   (file browser)", h_extract_single),
        ("Extract paired features — BATCH    (file browser)", h_extract_batch),
    ]),
    ("VISUALIZATION  (plot the .h5 corpus)", [
        ("Plot paired features — SINGLE      (interactive)", h_plot_single),
        ("Plot paired features — BATCH       (all recordings)", h_plot_batch),
    ]),
    ("M2 · CORRELATION", [
        ("M2 correlation analysis (7 levels)", h_m2_correlation),
        ("L3 stratified effect sizes + summary plots", h_l3_stratified),
    ]),
    ("M3 · MODELLING  (needs scikit-learn)", [
        ("Compartmental regression (audio→%RC) + diagnostic", h_m3_compartmental),
        ("FRC-state classification + resolution sweep", h_m3_frc),
    ]),
    ("ACOUSTIC / SINGLE-SUBJECT", [
        ("Analyze a single subject            (file browser)", h_single_subject),
        ("Trill modulation analysis           (file browser)", h_trill),
        ("Legacy Zocco acoustic batch pipeline", h_legacy),
    ]),
    ("DIAGNOSTICS & PROJECT", [
        ("Diagnostic tools (OEP/sync/inventory/HDF5)", h_tools),
        ("Quick counts (cohort + extraction tallies)", h_counts),
        ("FRC-crossing / ΔVcw diagnostic", h_h5_diag),
        ("Check a subject's timing sheet (file browser)", h_show_timing),
        ("Repair sync-pulse leakage in .h5 (trim leading sync burst)", h_repair_sync),
        ("Open the notebooks in Jupyter Lab", h_jupyter),
    ]),
]


def build_index():
    """Flatten sections into an ordered [(label, handler), …] list."""
    return [entry for _, entries in SECTIONS for entry in entries]


def print_menu(items):
    p = paired_dir()
    print("\n" + "=" * 64)
    print("  PNEUMOPHONIC PIPELINE — CONTROL CENTER")
    print("=" * 64)
    print(f"  Batch: {session['batch']:<22}  paired .h5: {n_paired():<5}  "
          f"metadata: {'✓' if metadata_path().exists() else '✗'}")
    n = 1
    for section, entries in SECTIONS:
        print(f"\n  {section}")
        for label, _ in entries:
            print(f"    [{n:>2}] {label}")
            n += 1
    print("\n  [b] Change batch    [q] Quit")


def change_batch(announce: bool = True):
    print("\n  Batches:")
    for i, b in enumerate(BATCHES):
        print(f"    [{i}] {b}")
    sel = input("  Select batch (Enter to cancel)> ").strip()
    if sel.isdigit() and 0 <= int(sel) < len(BATCHES):
        session["batch"] = BATCHES[int(sel)]
        if announce:
            print(f"  → batch set to {session['batch']}")
        return session["batch"]
    return None


def main():
    items = build_index()
    while True:
        print_menu(items)
        try:
            choice = input("\n  Select> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            break

        if choice in ("q", "quit", "exit", "0"):
            print("  Bye.")
            break
        if choice == "b":
            change_batch()
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(items):
            label, handler = items[int(choice) - 1]
            try:
                handler()
            except KeyboardInterrupt:
                print("\n  ⏹ Aborted — returning to menu.")
            except Exception as e:
                print(f"  ✗ Error: {e}")
            try:
                input("\n  [Enter] to return to the menu…")
            except (EOFError, KeyboardInterrupt):
                print("\n  Bye.")
                break
        else:
            print("  Invalid selection — type a number, 'b', or 'q'.")


if __name__ == "__main__":
    main()
