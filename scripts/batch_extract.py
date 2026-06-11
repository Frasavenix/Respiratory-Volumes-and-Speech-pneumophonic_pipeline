"""
Batch extraction of paired features for all subjects × all tasks.
Reads each subject's Excel timing file, resolves OEP CSV + falling edge,
and produces HDF5 files in data_target/<batch>/paired/


"""
import logging
import sys
from pathlib import Path
import pandas as pd
from pneumophonic_analysis import create_config, select_subject_folders_gui
from pneumophonic_analysis.paired_features import PairedFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data_root"
DATA_TARGET  = PROJECT_ROOT / "data_target"

BATCHES = ["healthy_subjects", "pathological_subjects"]

# ---- Mapping: task label → (audio filename, OEP csv suffix) ----
# take_number is no longer needed — we use the Excel falling edge directly
TASK_MAP = {
    'a':     ('a.wav',              'Vocali'),
    'e':     ('e.wav',              'Vocali'),
    'i':     ('i.wav',              'Vocali'),
    'o':     ('o.wav',              'Vocali'),
    'u':     ('u.wav',              'Vocali'),
    'a_2':   ('phonema_a_2.wav',    'phonema_a_2'),
    'a_3':   ('phonema_a_3.wav',    'phonema_a_3'),
    'a_7':   ('phonema_a_7_2.wav',  'phonema_a_7'),
    'r':     ('r.wav',              'r'),
    'f_1':   ('phrase_1.wav',       'frasi'),
    'f_2':   ('phrase_2.wav',       'frasi'),
    'f_3':   ('phrase_3.wav',       'frasi'),
    'f_4':   ('phrase_4.wav',       'frasi'),
    'f_5':   ('phrase_5.wav',       'frasi'),
    'testo': ('testo.wav',          'testo'),
}

# ---- Output batch label ----
def prompt_batch_label(default="healthy_subjects"):
    """Ask for the output batch label that organises data_target/<label>/paired/."""
    print("\nOutput batch label (organises data_target/<label>/paired/).")
    print(f"  Suggestions: {', '.join(BATCHES)}")
    label = input(f"  Batch label [{default}]: ").strip()
    return label or default

# ---- Subject discovery ----
def discover_subjects(batch_dir):
    """Find all subject folders that have a renders/ subfolder."""
    return sorted([
        d for d in batch_dir.iterdir()
        if d.is_dir() and (d / "renders").exists()
    ])

def expand_selection(folders):
    """
    Normalize the user's picks into a flat list of subject folders.

    Each chosen folder is treated as a subject folder; but if a chosen folder
    instead *contains* subject subfolders (e.g. the user picked a dataset root),
    it is expanded to those. Keeps both 'pick subjects' and 'pick a root' working.
    """
    out = []
    for f in folders:
        f = Path(f)
        if (f / "renders").exists() or (f / "csv").exists():
            out.append(f)
        else:
            found = discover_subjects(f)
            out.extend(found if found else [f])
    seen, uniq = set(), []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq

# ---- Load timing ----
def load_timing(subject_folder, subject_id):
    """Load the Timing sheet from the subject's Excel file."""
    excel_path = subject_folder / f"{subject_id}_audio.xlsx"
    if not excel_path.exists():
        return None
    try:
        df = pd.read_excel(excel_path, sheet_name="Timing")
        label_col = df.columns[0]
        df[label_col] = df[label_col].astype(str).str.strip()
        return df
    except Exception as e:
        logger.warning(f"  Failed to read Excel: {e}")
        return None

# ---- Main batch extraction ----
def run_batch(subjects, batch_name, skip_existing=True):
    """Extract paired features for an explicit list of subject folders.

    Args:
        subjects:      list of subject folder Paths (chosen via the file browser)
        batch_name:    output label -> data_target/<batch_name>/paired/
        skip_existing: skip subject/task pairs whose .h5 already exists
    """
    output_dir = DATA_TARGET / batch_name / "paired"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting {len(subjects)} subject folder(s) "
          f"-> {output_dir.relative_to(PROJECT_ROOT)}")

    # data_root is unused by the extractor (each subject folder is passed
    # explicitly); only output_root matters here.
    config = create_config(
        data_root=None,
        output_root=DATA_TARGET / batch_name
    )
    extractor = PairedFeatureExtractor(config)

    # Summary tracking
    summary = []
    total_tasks = 0
    total_ok = 0
    total_skip = 0
    total_fail = 0

    for subj_folder in subjects:
        subject_id = subj_folder.name.split('_')[1] if '_' in subj_folder.name else subj_folder.name
        print(f"\n{'='*60}")
        print(f"Subject: {subj_folder.name}")
        print(f"{'='*60}")

        # Load timing Excel
        timing_df = load_timing(subj_folder, subject_id)
        if timing_df is None:
            logger.warning(f"  No timing Excel found — skipping subject")
            summary.append({
                'subject': subj_folder.name, 'task': 'ALL',
                'status': 'no_excel', 'n_frames': None, 'duration': None
            })
            continue

        label_col = timing_df.columns[0]

        # Process each task in the timing file
        for _, row in timing_df.iterrows():
            task_label = str(row[label_col]).strip()
            total_tasks += 1

            # Check if task is in our mapping
            if task_label not in TASK_MAP:
                logger.warning(f"  [{task_label}] Unknown task — skipping")
                summary.append({
                    'subject': subj_folder.name, 'task': task_label,
                    'status': 'unknown_task', 'n_frames': None, 'duration': None
                })
                total_fail += 1
                continue

            # Check if already extracted
            h5_path = output_dir / f"{subject_id}_{task_label}.h5"
            if skip_existing and h5_path.exists():
                logger.info(f"  [{task_label}] Already exists — skipping")
                summary.append({
                    'subject': subj_folder.name, 'task': task_label,
                    'status': 'skipped', 'n_frames': None, 'duration': None
                })
                total_skip += 1
                continue

            # Resolve paths and parameters
            audio_file, csv_suffix = TASK_MAP[task_label]
            oep_csv = f"csv/{subject_id}_{csv_suffix}.csv"

            # Validate timing values (some subjects have '??' or NaN)
            try:
                start = float(row['start'])
                stop = float(row['stop'])
                falling_edge = float(row['falling edge'])
            except (ValueError, TypeError):
                logger.warning(f"  [{task_label}] Invalid timing data "
                               f"(start={row['start']}, stop={row['stop']}, "
                               f"falling edge={row['falling edge']}) — skipping")
                summary.append({
                    'subject': subj_folder.name, 'task': task_label,
                    'status': 'invalid_timing', 'n_frames': None, 'duration': None
                })
                total_fail += 1
                continue

            if pd.isna(start) or pd.isna(stop) or pd.isna(falling_edge):
                logger.warning(f"  [{task_label}] NaN in timing data — skipping")
                summary.append({
                    'subject': subj_folder.name, 'task': task_label,
                    'status': 'nan_timing', 'n_frames': None, 'duration': None
                })
                total_fail += 1
                continue

            # Verify files exist
            oep_path = subj_folder / oep_csv
            if not oep_path.exists():
                logger.warning(f"  [{task_label}] OEP CSV not found: {oep_csv}")
                summary.append({
                    'subject': subj_folder.name, 'task': task_label,
                    'status': f'missing_csv: {oep_csv}', 'n_frames': None, 'duration': None
                })
                total_fail += 1
                continue

            audio_path = subj_folder / "renders" / audio_file
            if not audio_path.exists():
                # Try without the _2 suffix for a_7
                alt_audio = audio_file.replace('_2.wav', '.wav')
                alt_path = subj_folder / "renders" / alt_audio
                if alt_path.exists():
                    audio_file = alt_audio
                else:
                    logger.warning(f"  [{task_label}] Audio not found: {audio_file}")
                    summary.append({
                        'subject': subj_folder.name, 'task': task_label,
                        'status': f'missing_audio: {audio_file}', 'n_frames': None, 'duration': None
                    })
                    total_fail += 1
                    continue

            # Extract
            try:
                paired = extractor.extract(
                    subject_folder=subj_folder,
                    task_name=task_label,
                    audio_filename=audio_file,
                    oep_csv_path=oep_csv,
                    take_number=1,  # ignored when falling edge is provided
                    audio_start_sec=start,
                    audio_end_sec=stop,
                    oep_falling_edge_sec=falling_edge,
                )

                PairedFeatureExtractor.save_hdf5(paired, h5_path)

                n_frames = paired.dataframe.shape[0]
                duration = paired.metadata.get('audio_duration_sec', stop - start)

                summary.append({
                    'subject': subj_folder.name, 'task': task_label,
                    'status': 'ok', 'n_frames': n_frames, 'duration': round(duration, 2)
                })
                total_ok += 1
                print(f"  ✅ {task_label}: {n_frames} frames, {duration:.1f}s")

            except Exception as e:
                logger.error(f"  [{task_label}] FAILED: {e}")
                summary.append({
                    'subject': subj_folder.name, 'task': task_label,
                    'status': f'error: {e}', 'n_frames': None, 'duration': None
                })
                total_fail += 1

    # ---- Summary report ----
    print(f"\n{'='*60}")
    print(f"BATCH EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total tasks:   {total_tasks}")
    print(f"  Successful:    {total_ok}")
    print(f"  Skipped:       {total_skip}")
    print(f"  Failed:        {total_fail}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary)
    summary_path = DATA_TARGET / batch_name / "paired" / "extraction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved: {summary_path.relative_to(PROJECT_ROOT)}")

    # Show failures if any
    if summary_df.empty or 'status' not in summary_df.columns:
        return summary_df
    failures = summary_df[~summary_df['status'].isin(['ok', 'skipped'])]
    if len(failures) > 0:
        print(f"\n  ⚠️  Failures:")
        for _, row in failures.iterrows():
            print(f"     {row['subject']} / {row['task']}: {row['status']}")

    return summary_df

# ---- Entry point ----
if __name__ == '__main__':
    print("=" * 60)
    print("BATCH PAIRED EXTRACTION")
    print("=" * 60)
    print("A file browser will open. Select each subject folder to extract,")
    print("then press Cancel when you are done (you can pick several).")

    initial = DATA_ROOT if DATA_ROOT.exists() else PROJECT_ROOT
    subjects = select_subject_folders_gui(
        title="Select subject folder(s) to extract",
        initialdir=initial,
    )
    subjects = expand_selection(subjects)

    if not subjects:
        print("\nNo folders selected — nothing to do.")
        sys.exit(0)

    print(f"\nSelected {len(subjects)} subject folder(s):")
    for s in subjects:
        print(f"  - {s}")

    batch_name = prompt_batch_label()
    skip = '--force' not in sys.argv
    if not skip:
        print("  (--force: re-extracting all, including existing files)")
    run_batch(subjects, batch_name, skip_existing=skip)