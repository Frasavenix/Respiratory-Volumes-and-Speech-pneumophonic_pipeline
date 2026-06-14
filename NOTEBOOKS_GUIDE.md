# Pneumophonic Pipeline — Notebook Guide

This guide documents the Jupyter-notebook edition of the **Pneumophonic Analysis Pipeline**. Every
standalone script and core module is split into a runnable, self-documenting notebook under
[`notebooks/`](notebooks/), and **at least one example of every plot type the pipeline produces** is
embedded so you can see how the data is processed end-to-end without running anything.

> **TL;DR** — open [`notebooks/00_overview_and_config.ipynb`](notebooks/00_overview_and_config.ipynb)
> and read downward, or jump to the milestone you care about. The notebooks already contain their
> output figures, so you can read them like an illustrated report.

---

## 1. What is in the pipeline

The pipeline jointly analyses **Optoelectronic Plethysmography (OEP)** chest-wall kinematics and
**acoustic voice** signals to study respiratory–phonatory coupling (continuation of the Zocco 2025
thesis). It runs in milestones:

| Milestone | Meaning |
|-----------|---------|
| **M1** | Paired feature extraction — build time-aligned `[audio \| OEP]` matrices (HDF5) |
| **M2** | Exploratory correlation — does the voice covary with the breathing signal? |
| **M3** | Modelling — can audio *predict* respiratory state (%RC / FRC state)? |

Plus an audio-only acoustic/task layer, an operatic-singer feature extension, and diagnostics.

---

## 2. Quick start

### Environment
```bash
python -m venv venv && source venv/bin/activate     # or conda
pip install -r requirements.txt
pip install -e .                                     # installs the pneumophonic_analysis package
```
Core dependencies: `numpy, scipy, pandas, librosa, soundfile, noisereduce, praat-parselmouth,
matplotlib, seaborn, openpyxl, h5py, scikit-learn` (the last is needed only for the M3 notebooks 05/06).

### Running
Open any notebook and **Run All**. They are ordered `00 → 09` and each is self-contained.

```bash
jupyter lab            # or: jupyter notebook
```

To re-execute headlessly and re-embed the figures:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/03_M2_correlation_analysis.ipynb
```

### ⚠️ Kernel note
The notebooks declare the generic `python3` kernel. Make sure the kernel you select is the **Python
environment where you installed the dependencies** (e.g. your Anaconda env). If your global Jupyter's
`python3` kernel points at a bare interpreter without the packages, register the right one:

```bash
python -m ipykernel install --user --name pneumo --display-name "Python (pneumophonic)"
```
then pick **Python (pneumophonic)** in Jupyter. (These notebooks were executed with exactly such a
kernel.)

---

## 3. Data requirements

| Data | Location | Committed? | Used by |
|------|----------|-----------|---------|
| Paired HDF5 corpus (M1 output, 546 recordings) | `data_target/healthy_subjects/paired/*.h5` | yes (local) | 01, 02, 03, 04, 05, 06, 09 |
| Subject metadata | `data_root/healthy_subjects/subjects_metadata.csv` | yes (local) | 04, 05, 06, 09 |
| Raw audio `.wav` / OEP `.csv` / sync / Excel | `data_root/<batch>/<subject>/…` | **no** | the *real* M1 extraction; otherwise synthetic |

Because raw `.wav`/`.csv` recordings are **not** committed:

* Notebooks **01, 07, 08** that need raw audio use **synthetic but physiologically plausible signals**
  (sustained vowel, vibrato vowel, vocal glide, alveolar trill, synthetic OEP). The *exact same calls*
  work on real audio via `DataLoader(subject_folder).load_audio("a.wav")`.
* Every downstream notebook (02–06, 09) runs on the **real** 546-recording HDF5 corpus. The paired
  HDF5 files conveniently also store the STFT / mel / MFCC matrices, so spectrograms come from real
  data too.

### Extracting new datasets via a file browser
The raw datasets no longer have to live inside `data_root/`. The extraction entry points open a
**native folder browser** so you can point the pipeline at wherever the subject folders are stored
(e.g. an external drive):

* `python scripts/batch_extract.py` — pick **one or more** subject folders (the dialog reopens after
  each pick; press **Cancel** when done), then type a batch label. Outputs go to
  `data_target/<label>/paired/`. If you accidentally pick a *parent* folder that contains subject
  subfolders, it is expanded automatically.
* `python scripts/test_paired.py` — pick **one** subject folder, then extract a single task.
* `python MAIN_UX.py` — menu options **1** / **2** launch the two scripts above.
* **Notebook 01** has a guarded *"Run M1 on your data"* cell: set `RUN_EXTRACTION = True` and run it to
  open the same browser from within Jupyter.

A *subject folder* is one containing `renders/`, `csv/`, `sync_signal.wav`, and `<ID>_audio.xlsx`.
The picker is implemented in `pneumophonic_analysis.io_utils.select_subject_folders_gui`
(stdlib `tkinter`; falls back to typed paths if no GUI/display is available).

---

## 4. Notebook catalogue

| # | Notebook | Covers (scripts / modules) | Data | Figures |
|---|----------|----------------------------|------|---------|
| 00 | [overview_and_config](notebooks/00_overview_and_config.ipynb) | orientation, `config.py`, layout, task map | real | 1 |
| 01 | [M1_paired_feature_extraction](notebooks/01_M1_paired_feature_extraction.ipynb) | `paired_features.py`, `test_paired.py`, `batch_extract.py` | synthetic OEP + real HDF5 | 3 |
| 02 | [paired_feature_visualization](notebooks/02_paired_feature_visualization.ipynb) | `plot_paired_features.py`, `batch_plot_paired.py`, `visualization.py` | real | 9 |
| 03 | [M2_correlation_analysis](notebooks/03_M2_correlation_analysis.ipynb) | `m2_correlation.py` (7 levels) | real | 15 |
| 04 | [M2_L3_stratified_effect_sizes](notebooks/04_M2_L3_stratified_effect_sizes.ipynb) | `analyze_l3_stratified.py`, `effect_size.py`, `make_m2_summary_plots.py` | real | 5 |
| 05 | [M3_compartmental_regression](notebooks/05_M3_compartmental_regression.ipynb) | `analyze_compartmental_regression.py`, `diagnose_compartmental_signal.py` | real | 6 |
| 06 | [M3_frc_classification](notebooks/06_M3_frc_classification.ipynb) | `analyze_frc_classification.py`, `analyze_frc_window_sweep.py` | real | 4 |
| 07 | [acoustic_features_and_tasks](notebooks/07_acoustic_features_and_tasks.ipynb) | `acoustic_features.py`, `segmentation.py`, `task_analyzers.py`, `visualization.py`, `analyze_single_subject.py`, `analyze_trill_modulation.py` | synthetic | 11 |
| 08 | [M3_multivariate_coupling](notebooks/08_M3_multivariate_coupling.ipynb) | `analyze_multivariate_coupling.py` (CCA, PLS) | real | 3 |
| 09 | [singer_acoustic_features](notebooks/09_singer_acoustic_features.ipynb) | `singer_acoustc_features.py` (+ `operatic_acoustic_analysis.ipynb`) | synthetic | 9 |
| 10 | [diagnostics_and_inspection](notebooks/10_diagnostics_and_inspection.ipynb) | `tools.py`, `count_check.py`, `tool_h5_id_match.py` | real | 4 |

There is also the pre-existing [`notebooks/operatic_acoustic_analysis.ipynb`](notebooks/operatic_acoustic_analysis.ipynb)
— the **real-data** operatic study; it requires singer recordings under
`data_root/singer_subjects/` and the [`OPERATIC_COHORT_PROTOCOL.md`](OPERATIC_COHORT_PROTOCOL.md)
task labels. Notebook 08 is its runnable, synthetic-data twin.

---

## 5. Per-notebook detail

### 00 — Overview & configuration
Package version, the centralized `config.py` parameters (sample rate, hop, mel/MFCC, OEP rate, F0
range), the data layout, the OEP two-compartment column model, the vocal-task protocol, and the
script→notebook map. Ends with a teaser of one aligned audio⊕OEP recording.

### 01 — M1: Paired feature extraction
The fusion step. Demonstrates `compute_oep_features_native` (volumes → LP-filtered flows ×k=0.916 →
compartmental ratios) and `interpolate_oep_to_audio_frames` (50 Hz → ~66.7 fps) on a synthetic OEP
recording, the `batch_extract.py` task map, then loads a **real** extracted HDF5 to show the final
aligned matrix and the `Vrc+Vab=Vcw` sanity check. Ends with a **guarded file-browser cell**
(`RUN_EXTRACTION`) for extracting your own folders from within Jupyter. *To run on real data:*
`python scripts/test_paired.py` or `python scripts/batch_extract.py` — both open a folder browser.

### 02 — Paired-feature visualization
Everything the per-recording plotting scripts produce, on a real recording: energy-vs-volume, F0-vs-flow,
compartmental strategy, voiced-frame correlation matrix, STFT spectrogram, mel-spectrogram, MFCC
heatmap (all three read from the HDF5), plus the package `Visualizer` F0 trace and chest-wall volume
with FRC zones.

### 03 — M2: Correlation analysis (7 levels)
Reproduces `m2_correlation.py` end-to-end on the full corpus, importing the script's own helpers:
**(1)** global cross-subject correlation heatmap + scatter grid + within-segment histograms;
**(2)** time-resolved sliding-window coupling + aggregate; **(3)** FRC-crossing shift histograms +
paired scatter; **(4)** sex-stratified heatmaps + scatter; **(5)** flow→energy lag cross-correlation
+ peak-lag histogram; **(6)** breath-group evolution + correlation heatmap; **(7)** MFCC↔respiratory
heatmap + boxplots.

### 04 — M2/L3: Stratified effect sizes
Adds standardized effect sizes (Cohen's d, robust d, Wilcoxon r, bootstrap CIs, sign-consistency)
and Young/Elder × Male/Female stratification to the FRC-crossing analysis: forest plot, 2×2
stratified histograms, grouped Cohen's-d bar chart, and the F0-shift-vs-%RC-shift orthogonality
scatter.

### 05 — M3: Compartmental regression
Tries to predict continuous `%RC` from audio: per-subject vs pooled vs demographic-stratified Ridge
regression, coefficient importance, binning ANOVA + feature profiles, then the diagnostic
(variance decomposition, within-task vs segment-level framing). A characterised **negative result**.

### 06 — M3: FRC-state classification
Reframes the target as binary FRC state (above/below) — which *is* decodable: per-subject AUC
distribution, feature importance, AUC across framings, and the resolution↔accuracy sweep curve.

### 07 — Acoustic features & task analyzers
Exercises the whole audio-only layer on synthetic signals: every `Visualizer` method (waveform,
audio+sync, spectrogram, mel, F0 trace, OEP volume+FRC, FRC segment, glide novelty, stacked task
result, metrics bar), `PraatAnalyzer` clinical metrics (jitter/shimmer/HNR/DSI/formants),
`FRCSegmenter`/`GlideSegmenter`/`ModulationAnalyzer`, and the trill modulation spectrum.

### 08 — Multivariate audio↔OEP coupling (CCA + PLS)
Beyond pairwise correlation: stratified CCA (symmetric audio↔OEP) and PLS (audio→%RC) on
within-subject z-scored voiced frames from sustained phonation (glissando `a_7` excluded),
cross-validated by subject. The coupling is real but low-dimensional (first canonical
correlation ≈ 0.39), carried by spectral level (MFCC-0, energy) rather than pitch; audio→%RC
is weakly predictive (CV R² ≈ 0.06 to 0.10).

### 09 — Singer-specific acoustic features
The operatic extension (`singer_acoustc_features.py`): vibrato (rate/extent/regularity), singer's-formant
cluster ratio, LTAS slope, formant-band HNR, F1/F2 vowel space, MFCC fingerprint, mel+F0, and the
*voce girata* vs *non-girata* contrast — on a synthetic cohort engineered to show the expected
differences.

### 10 — Diagnostics & inspection
Cohort age distribution by sex, extraction-coverage heatmap (subject × task), per-task recording
counts, HDF5 inspection, and the ΔVcw FRC-crossing diagnostic that validates the L3/M3 splitter.

---

## 6. Script → notebook map

| Original script | Notebook |
|-----------------|----------|
| `scripts/test_paired.py`, `scripts/batch_extract.py` | 01 |
| `scripts/plot_paired_features.py`, `scripts/batch_plot_paired.py` | 02 |
| `scripts/m2_correlation.py` | 03 |
| `scripts/analyze_l3_stratified.py`, `scripts/make_m2_summary_plots.py` | 04 |
| `scripts/analyze_compartmental_regression.py`, `scripts/diagnose_compartmental_signal.py` | 05 |
| `scripts/analyze_frc_classification.py`, `scripts/analyze_frc_window_sweep.py` | 06 |
| `scripts/analyze_single_subject.py`, `scripts/analyze_trill_modulation.py` | 07 |
| `scripts/analyze_batch.py`, `pipeline.py` (`run_pipeline`) | 07 (metrics bar) + 00 |
| `scripts/analyze_multivariate_coupling.py` | 08 |
| `scripts/tools.py`, `count_check.py`, `tool_h5_id_match.py` | 10 |
| `MAIN_UX.py` (control center) | one menu that launches every script below |

Core modules `config`, `io_utils`, `sync`, `audio_processing`, `acoustic_features`, `segmentation`,
`task_analyzers`, `paired_features`, `visualization`, `effect_size`, `singer_acoustc_features` are all
imported and exercised across the notebooks (notably 00, 01, 02, 07, 09).

---

## 7. Plot-type catalogue

One example of each is embedded in the notebook(s) listed.

| Plot type | Notebook(s) |
|-----------|-------------|
| Waveform | 07 |
| Audio + sync pulse (falling edge marked) | 07 |
| STFT spectrogram (log-frequency) | 02, 07 |
| Mel-spectrogram | 02, 07, 08 |
| Mel-spectrogram + F0 overlay | 08 |
| MFCC heatmap (full matrix / mean-per-task) | 02, 08 |
| F0 trace with mean ± SD band | 02, 07 |
| Stacked F0 contours per task | 08 |
| Dual-axis line: energy vs volume | 00, 01, 02 |
| Dual-axis line: F0 vs flow (voiced) | 02 |
| Compartmental %RC/%AB line plot | 02 |
| OEP volumes (Vcw/Vrc/Vab) | 01 |
| OEP flows (dV/dt) | 01 |
| Native-vs-interpolated resampling overlay | 01 |
| Chest-wall volume with FRC zones | 02, 07 |
| Correlation heatmap | 02, 03 |
| Scatter grid with r/p (task-coloured) | 03 |
| Within-segment correlation histograms | 03 |
| Time-resolved dual-axis + sliding-r panel | 03 |
| Aggregate sliding-r histogram | 03 |
| FRC shift histograms (Wilcoxon) | 03 |
| Above-vs-below paired scatter (identity line) | 03 |
| Sex-stratified side-by-side heatmaps | 03 |
| Sex-coloured scatter grid | 03 |
| Lag cross-correlation curve (mean ± SD) | 03 |
| Peak-lag histogram | 03 |
| Breath-group evolution scatter + binned means | 03 |
| MFCC × OEP correlation heatmap | 03 |
| MFCC per-coefficient boxplots | 03 |
| Forest plot (median + bootstrap CI) | 04 |
| 2×2 stratified histograms | 04 |
| Grouped Cohen's-d bar chart (reference lines) | 04 |
| Orthogonality scatter (group medians) | 04 |
| Per-subject R² histogram | 05 |
| Coefficient-importance bar | 05 |
| R² comparison bar (framings) | 05 |
| Binning feature-profile line plot | 05 |
| Variance-decomposition histogram | 05 |
| Framing-comparison bar | 05 |
| Per-subject AUC histogram | 06 |
| Signed feature-importance bar | 06 |
| AUC comparison bar | 06 |
| Resolution-sweep curve (log-x, reference lines) | 06 |
| FRC segment (2-panel: zones + separated) | 07 |
| Glide analysis (3-panel novelty) | 07 |
| Trill RMS envelope | 07 |
| Trill modulation spectrum (band-shaded) | 07 |
| Stacked task-result (waveform + spectrogram + F0) | 07 |
| Metrics-comparison horizontal bar | 07 |
| Vibrato strip plot (rate/extent/regularity) | 08 |
| Singer's-formant ratio bar | 08 |
| LTAS slope strip plot | 08 |
| Formant-band HNR boxplot | 08 |
| F1/F2 vowel-space scatter (inverted axes) | 08 |
| CCA canonical-correlation bars (audio↔OEP) | 08 |
| Audio-feature loading heatmap (canonical variate) | 08 |
| PLS audio→%RC CV-R² bars | 08 |
| Girata vs non-girata comparison bars | 09 |
| Cohort age histogram by sex | 10 |
| Extraction-coverage heatmap (subject × task) | 10 |
| Per-task recording-count bar | 10 |
| ΔVcw FRC-crossing diagnostic traces | 10 |

---

## 8. Regenerating the full script outputs (PDF reports)

The notebooks render figures inline (sometimes on a representative subset for speed). To regenerate
the complete on-disk PDF/CSV/XLSX reports, run the original scripts (they prompt for the batch or take
CLI args):

```bash
python scripts/batch_extract.py                       # M1 -> data_target/<batch>/paired/*.h5
python scripts/batch_plot_paired.py                   # per-recording PDFs
python scripts/m2_correlation.py                      # M2 (all 7 levels) -> m2_correlation/
python scripts/analyze_l3_stratified.py    --paired-dir data_target/healthy_subjects/paired \
    --metadata data_root/healthy_subjects/subjects_metadata.csv --output-dir results/M2_stratified
python scripts/make_m2_summary_plots.py    --results-dir results/M2_stratified
python scripts/analyze_compartmental_regression.py --paired-dir data_target/healthy_subjects/paired \
    --metadata data_root/healthy_subjects/subjects_metadata.csv --output-dir results/M3_compartmental
python scripts/diagnose_compartmental_signal.py    --paired-dir data_target/healthy_subjects/paired \
    --metadata data_root/healthy_subjects/subjects_metadata.csv --output-dir results/M3_compartmental
python scripts/analyze_frc_classification.py --paired-dir data_target/healthy_subjects/paired \
    --metadata data_root/healthy_subjects/subjects_metadata.csv --output-dir results/M3_frc_classification
python scripts/analyze_frc_window_sweep.py   --paired-dir data_target/healthy_subjects/paired \
    --metadata data_root/healthy_subjects/subjects_metadata.csv --output-dir results/M3_frc_classification
```

`python MAIN_UX.py` is the **control center**: a single grouped menu that reaches every step above
(extraction, plotting, M2, M3, acoustic/single-subject, diagnostics, and launching Jupyter Lab). It
shows the current *batch* (change with `b`) and builds the standard paths for the argparse analyses;
press `q` (or Ctrl+C) to quit any time, and Ctrl+C while a task runs returns you to the menu.
