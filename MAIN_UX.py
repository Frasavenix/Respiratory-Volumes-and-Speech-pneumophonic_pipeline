"""
Quick Test — Pneumophonic Pipeline
====================================

Entry point for running different analysis modes.
Run from the project root:

    python run_quick_test.py
"""
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"

# ---- Available modes ----
MODES = {
    '1': ('Extract paired features (SINGLE ) -> gen .h5 data files' ,       'test_paired.py'),
    '2': ('Extract paired features (BATCH ) -> gen .h5 data files',         'batch_extract.py'),
    '3': ('Plot paired features and Save (SINGLE) -> gen PDFs + open plot tabs',        'plot_paired_features.py'),
    '4': ('Plot paired features and Save (BATCH) !-LONG-!',  'batch_plot_paired.py'),
    '5': ('M2 correlation analysis',  'm2_correlation.py'),
    '6': ('Diagnostic tools (OEP headers, sync, inventory)','tools.py'),
    '7': ('Run legacy acoustic analysis (Zocco pipeline)',  None),
}


def run_legacy_analysis():
    """Original acoustic-only analysis via PneumophonicPipeline."""
    from pneumophonic_analysis import run_pipeline

    # Batch selection
    batches = ["healthy_subjects", "pathological_subjects"]
    print("\nWhich subjects to analyse?")
    for idx, name in enumerate(batches):
        print(f"  [{idx}] {name}")
    while True:
        sel = input("Select by number: ").strip()
        if sel.isdigit() and 0 <= int(sel) < len(batches):
            batch = batches[int(sel)]
            break
        print("Invalid selection.")

    data_root = PROJECT_ROOT / "data_root" / batch
    data_target = PROJECT_ROOT / "data_target" / batch

    # Subject discovery
    available = sorted([d.name for d in data_root.glob("*_*") if d.is_dir()])
    subjects_to_run = None

    if available:
        print(f"\n{len(available)} subjects found in {batch}.")
        print("Enter subject names to EXCLUDE (comma-separated), or press Enter for all:")
        raw = input("> ").strip()
        if raw:
            excluded = {s.strip() for s in raw.split(",")}
            subjects_to_run = [s for s in available if s not in excluded]
            print(f"Running with {len(subjects_to_run)} subject(s).")
        else:
            print(f"Running with all {len(available)} subject(s).")

    results = run_pipeline(
        data_root=data_root,
        output_root=data_target,
        subjects=subjects_to_run,
        tasks=['vowel', 'trill', 'glide']
    )
    print(f"\nAnalyzed {results.n_subjects} subjects")
    print(f"Success: {results.n_successful}, Failed: {results.n_failed}")


def main():
    print("\n" + "=" * 50)
    print("  PNEUMOPHONIC PIPELINE")
    print("=" * 50)



    while True:
        for key, (desc, _) in MODES.items():
            print(f"  [{key}] {desc}")
        print(f"  [q] Quit")
        choice = input("\nSelect mode: ").strip().lower()

        if choice == 'q':
            break

        if choice not in MODES:
            print("Invalid selection.")
            continue

        desc, script = MODES[choice]

        if script is None:
            # Legacy mode runs in-process
            run_legacy_analysis()
        else:
            script_path = SCRIPTS_DIR / script
            if not script_path.exists():
                # Fallback: check project root (in case scripts haven't been moved yet)
                script_path = PROJECT_ROOT / script
            if not script_path.exists():
                print(f"  Script not found: {script}")
                print(f"  Looked in: {SCRIPTS_DIR} and {PROJECT_ROOT}")
                continue

            print(f"\n  Running: {script_path.name}")
            print("-" * 50)
            subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))

        print("\n" + "-" * 50)


if __name__ == '__main__':
    main()