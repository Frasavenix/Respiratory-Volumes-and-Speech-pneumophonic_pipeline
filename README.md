# Pneumophonic Analysis Pipeline

Modular Python pipeline for the integrated analysis of respiratory-phonatory functions.
Combines Optoelectronic Plethysmography (OEP) chest wall kinematics with acoustic voice signals.

## Context

This pipeline builds on the Master's thesis:
> **"Integrated Analysis of Respiratory-Phonatory Functions: Normative Patterns Across Sex and Age"**
> Bianca Zocco, Politecnico di Milano, 2024-2025

In order to run the scripts, Zocco's Dataset is needed.

The current work extends the original analysis toward **respiratory-acoustic correlation modeling**: extracting time-aligned paired features from OEP and audio signals, computing cross-domain correlations, and preparing the ground for predictive models (audio-to-respiratory and respiratory-to-audio).

## Pipeline Overview

The pipeline operates in milestones:

| Milestone | Status | Description |
|-----------|--------|-------------|
| **M1** | Done | Paired feature extraction (time-aligned audio + OEP matrices) |
| **M2** | Done | Exploratory correlation analysis (global, time-resolved, FRC-aligned) |

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd pneumophonic_pipeline

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Dependencies

Core: numpy, scipy, pandas, librosa, soundfile, noisereduce, praat-parselmouth, matplotlib, seaborn, openpyxl, tqdm, h5py

## Data Structure

```
pneumophonic_pipeline/
├── data_root/                          # Source data (read-only)
│   ├── healthy_subjects/
│   │   └── YYYYMMDD_SubjectID/         # One folder per subject
│   │       ├── csv/                    # OEP data files
│   │       │   ├── SubjectID_Vocali.csv
│   │       │   ├── SubjectID_phonema_a_2.csv
│   │       │   ├── SubjectID_frasi.csv
│   │       │   ├── SubjectID_testo.csv
│   │       │   ├── SubjectID_r.csv
│   │       │   └── ...
│   │       ├── renders/                # Audio files rendered from Reaper
│   │       │   ├── a.wav
│   │       │   ├── phonema_a_2.wav
│   │       │   ├── testo.wav
│   │       │   └── ...
│   │       ├── sync_signal.wav         # Synchronization pulse
│   │       └── SubjectID_audio.xlsx    # Timing sheet (start, stop, falling edge)
│   └── pathological_subjects/
│       └── ...
│
├── data_target/                        # Outputs
│   ├── healthy_subjects/
│   │   ├── paired/                     # HDF5 paired feature files (M1)
│   │   │   ├── SubjectID_taskname.h5
│   │   │   └── extraction_summary.csv
│   │   ├── figures/
│   │   │   └── paired/                 # Per-subject PDF plots
│   │   └── m2_correlation/             # M2 analysis outputs
│   │       ├── global_summary.csv
│   │       ├── global_correlation_matrix.pdf
│   │       ├── frc_shifts.pdf
│   │       ├── time_resolved/
│   │       └── m2_report.txt
│   └── pathological_subjects/
│       └── ...
│
├── pneumophonic_analysis/              # Core Python package
│   ├── config.py                       # Centralized configuration
│   ├── io_utils.py                     # File I/O (OEP, audio, Excel)
│   ├── sync.py                         # OEP-Audio synchronization
│   ├── audio_processing.py             # Audio processing (noise, STFT, F0)
│   ├── acoustic_features.py            # Praat-based feature extraction
│   ├── segmentation.py                 # FRC / novelty segmentation
│   ├── task_analyzers.py               # Task-specific analyzers
│   ├── paired_features.py              # M1: Paired feature extraction
│   ├── visualization.py                # Plotting utilities
│   └── pipeline.py                     # Orchestration
│
├── scripts/                            # Standalone analysis scripts
│   ├── test_paired.py                  # Interactive single extraction
│   ├── batch_extract.py                # Batch paired extraction (all subjects)
│   ├── explore_paired.py               # Interactive HDF5 exploration + plots
│   ├── batch_plot_paired.py            # Batch PDF generation from HDF5
│   ├── m2_correlation.py               # M2 correlation analysis
│   └── tools.py                        # Diagnostic utilities
│
└── README.md
```

## OEP Column Mapping

The `.csv`/`.dat` files contain space-separated columns loaded with these labels:

| Column | Label | Physical quantity |
|--------|-------|-------------------|
| 1 | `time` | Time (s) |
| 2 | `A` | Vrcp — Pulmonary rib cage volume (L) |
| 3 | `B` | Vrca — Abdominal rib cage volume (L) |
| 4 | `C` | Vab — Abdominal volume (L) |
| 5 | `tot_vol` | Vcw — Total chest wall volume (L) |
| 6 | `sync` | Synchronization signal |

Two-compartment model (Zocco thesis): **Vrc = A + B**, **Vab = C**, verified by A + B + C = tot_vol.

## Vocal Tasks (Zocco Protocol)

| Task label | Audio file | OEP CSV suffix | Description |
|------------|------------|----------------|-------------|
| `a` | `a.wav` | `Vocali` | Sustained /a/ (5s) |
| `e`, `i`, `o`, `u` | `{vowel}.wav` | `Vocali` | Sustained vowels (5s each) |
| `a_2` | `phonema_a_2.wav` | `phonema_a_2` | Maximum phonation time /a/ |
| `a_3` | `phonema_a_3.wav` | `phonema_a_3` | Soft phonation /a/ |
| `a_7` | `phonema_a_7.wav` | `phonema_a_7` | Vocal glide |
| `r` | `r.wav` | `r` | Sustained alveolar trill |
| `f_1`..`f_5` | `phrase_{n}.wav` | `frasi` | Sentence reading |
| `testo` | `testo.wav` | `testo` | Balanced text reading |

## Quick Start

### 1. Extract paired features (single subject)

```bash
python scripts/test_paired.py
```

Interactive prompts guide you through batch, subject, and task selection. Produces an HDF5 file in `data_target/<batch>/paired/`.

### 2. Batch extraction (all subjects)

```bash
python scripts/batch_extract.py
```

Processes all subjects and tasks automatically. Skips already-extracted files. Produces `extraction_summary.csv`.

### 3. Generate plots

```bash
# Single file (interactive)
python scripts/explore_paired.py

# All HDF5 files at once
python scripts/batch_plot_paired.py
```

### 4. Run correlation analysis (M2)

```bash
python scripts/m2_correlation.py
```

Produces correlation heatmaps, scatter plots, time-resolved analysis, and FRC crossing analysis.

### 5. Diagnostics

```bash
python scripts/tools.py
```

Utility commands: inspect OEP headers, check sync peaks, verify data integrity.

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Audio sample rate | 48 kHz | Acquisition protocol |
| OEP kinematic rate | 50 Hz | OEP system |
| STFT hop length | 720 samples (~15 ms) | Config |
| Audio feature rate | ~66 fps | 48000 / 720 |
| OEP flow LP filter | 4th-order Butterworth, 10 Hz | Zocco thesis |
| Flow calibration factor | k = 0.916 | Zocco thesis (Section 4.1.3) |
| F0 range (cleanup) | 60-350 Hz | Physiological bounds |

## Synchronization Method

Audio and OEP are synchronized via a 1-second rectangular pulse recorded on both systems.
The `falling edge` column in each subject's Excel timing file provides the OEP time (in seconds) of the sync pulse for each task. This is the primary sync method, bypassing unreliable peak-pairing heuristics.

## References

```bibtex
@mastersthesis{zocco2025pneumophonic,
  title   = {Integrated Analysis of Respiratory-Phonatory Functions:
             Normative Patterns Across Sex and Age},
  author  = {Zocco, Bianca},
  year    = {2025},
  school  = {Politecnico di Milano},
  advisor = {Lo Mauro, Antonella}
}
```

## License

MIT