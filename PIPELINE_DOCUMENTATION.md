# Pneumophonic Pipeline — Complete Documentation

**Author:** Mateo Vitalone (Master of Acoustics, Politecnico di Milano)
**Basis:** Bianca Zocco Master's Thesis, 2024-2025
**Purpose:** Synchronized OEP-acoustic analysis of voice-breathing kinematics

---

## 1. Project Overview

This pipeline investigates the correlation between **respiratory kinematics** (measured with Opto-Electronic Plethysmography, OEP) and **acoustic voice signals**. It builds on Bianca Zocco's thesis — *"Integrated Analysis of Respiratory-Phonatory Functions: Normative Patterns Across Sex and Age"* — and extends it toward deeper acoustic analysis, statistical stratification, and predictive modeling.

### Milestone Roadmap

| Milestone | Status | Description |
|-----------|--------|-------------|
| **M1** | Done | Paired feature extraction — time-aligned audio + OEP matrices saved as HDF5 |
| **M2** | Done | Exploratory correlation analysis — global, time-resolved, FRC-aligned, sex-stratified |
| **M3** | Planned | Baseline regression models (audio → respiratory) |
| **M4** | Planned | Sequence models (LSTM / 1D-CNN) |
| **M5** | Planned | Compartmental body mapping from audio alone |

---

## 2. Core Concepts

### 2.1 OEP — Opto-Electronic Plethysmography

OEP measures chest wall kinematics without contact, using infrared markers tracked at **50 Hz**. It produces volumetric signals of the chest wall compartments.

**Two-compartment model:**

```
Vrc  = Vrcp + Vrca     (Rib cage = pulmonary + abdominal halves)
Vcw  = Vrc + Vab       (Total chest wall = rib cage + abdomen)
```

**OEP CSV/DAT column mapping:**

| Column | Label | Physical quantity |
|--------|-------|-------------------|
| 1 | `time` | Time (s) |
| 2 | `A` | Vrcp — Pulmonary rib cage volume (L) |
| 3 | `B` | Vrca — Abdominal rib cage volume (L) |
| 4 | `C` | Vab — Abdominal volume (L) |
| 5 | `tot_vol` | Vcw — Total chest wall volume (L) |
| 6 | `sync` | Synchronization signal |

Relationship: `A + B + C = tot_vol` (verified at runtime).

### 2.2 Synchronization

Audio (48 kHz) and OEP (50 Hz) are synchronized via a **1-second rectangular pulse** recorded on both systems simultaneously. The falling edge of this pulse is detected in both signals:

- **OEP side**: prominence-based peak finding on the `sync` column
- **Audio side**: `librosa.onset.onset_detect` — 2nd onset event
- **Primary method**: the Excel timing file for each subject contains an `oep_falling_edge_sec` column giving the OEP time of the sync pulse directly; this overrides automatic detection

The result is a `SyncResult` object carrying `time_offset_sec` used to map audio timestamps → OEP sample indices.

### 2.3 FRC — Functional Residual Capacity

FRC is the lung volume at rest after a normal expiration. It serves as a **physiological threshold** to split phonation into two regimes:

- **Above FRC**: speaking or singing on a lung volume greater than rest → easier phonation, more airflow reserve
- **Below FRC**: phonation below rest level → requires expiratory muscle effort, acoustically different

Every segment in the L3 analysis (M2 extended) is split at the FRC crossing to compare acoustic features between these two respiratory states.

### 2.4 Acoustic Features Computed

| Feature Group | Features |
|--------------|----------|
| **Pitch (F0)** | pYIN estimate (Hz), cleaned with physiological bounds 60–350 Hz + median smoothing kernel=5 |
| **Harmonics** | 1st, 2nd, 3rd harmonic energy |
| **Perturbation (Praat)** | Jitter: local, RAP, PPQ5, DDP; Shimmer: local, local_dB, APQ3, APQ5, APQ11, DDA |
| **HNR** | Harmonics-to-Noise Ratio (Praat, autocorrelation) |
| **DSI** | Dysphonia Severity Index: `0.13·MPT + 0.0053·F0_high − 0.26·I_low − 1.18·jitter% + 12.4` |
| **Formants** | F1, F2, F3 (Praat Burg method) |
| **MFCCs** | 13 coefficients, z-score normalized |
| **Spectral** | Spectral centroid (Hz), RMS energy |
| **Singer-specific** | Singer's formant ratio (Cabrera 2011), LTAS slope (Sundberg 2012), vibrato rate/extent/regularity, formant-aware HNR (Ikuma 2025) |

### 2.5 Statistical Methods

| Method | Purpose |
|--------|---------|
| **Cohen's d (paired)** | `mean(diff) / sd(diff)` — parametric effect size |
| **Hedges' g** | Small-sample-corrected Cohen's d: `d × (1 − 3/(4n − 5))` |
| **Robust d** | `median(diff) / MAD(diff)` — non-parametric equivalent |
| **Wilcoxon signed-rank** | Two-sided test; `r = |Z| / √N` as effect size |
| **Bootstrap CI** | 5000 resamples; percentile method on median and Cohen's d |
| **Sign consistency** | Fraction of differences sharing sign with the median |
| **MAD** | Median Absolute Deviation, scaled by 1.4826 for normality consistency |

**Effect size interpretation (Cohen's |d|):**

| Range | Label |
|-------|-------|
| < 0.2 | Negligible |
| 0.2–0.5 | Small |
| 0.5–0.8 | Medium |
| ≥ 0.8 | Large |

### 2.6 Demographic Stratification

Subjects are split into 4 subgroups for M2/L3 analysis:

```
Young Female (YF) | Young Male (YM)
Elder Female (EF) | Elder Male (EM)
```

Age threshold: **55 years** (elder ≥ 55).

---

## 3. Vocal Task Protocol

| Task label | Audio file | OEP CSV suffix | Description |
|------------|------------|----------------|-------------|
| `a` | `a.wav` | `Vocali` | Sustained /a/ (5 s) |
| `e`, `i`, `o`, `u` | `{vowel}.wav` | `Vocali` | Sustained vowels (5 s each) |
| `a_2` | `phonema_a_2.wav` | `phonema_a_2` | Maximum phonation time /a/ (MPT) |
| `a_3` | `phonema_a_3.wav` | `phonema_a_3` | Soft phonation /a/ |
| `a_7` | `phonema_a_7.wav` | `phonema_a_7` | Vocal glide (P1 → P2 pitch sweep) |
| `r` | `r.wav` | `r` | Sustained alveolar trill /r/ |
| `f_1`..`f_5` | `phrase_{n}.wav` | `frasi` | Sentence reading (5 sentences) |
| `testo` | `testo.wav` | `testo` | Balanced text reading |

---

## 4. Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Audio sample rate | 48,000 Hz | Acquisition protocol |
| OEP kinematic rate | 50 Hz | OEP system |
| Audio frame length | 30 ms | `AudioConfig.frame_length_ms` |
| STFT hop length | 720 samples (~15 ms) | `AudioConfig.hop_length_ratio = 0.5` |
| Audio feature rate | ~66.67 fps | 48000 / 720 |
| Noise reduction | Stationary, 85% prop decrease | `noisereduce` library |
| Pre-emphasis coefficient | 0.97 | `AudioConfig.pre_emphasis_coef` |
| Mel bands | 64 | `AudioConfig.n_mels` |
| MFCCs | 13 (z-score normalized) | `AudioConfig.n_mfcc` |
| F0 pYIN range | 50–500 Hz | `PitchConfig` |
| F0 cleanup bounds | 60–350 Hz | Physiological; `clean_f0()` in `paired_features.py` |
| F0 median smoothing | kernel = 5 | `clean_f0()` |
| OEP LP filter | 4th-order Butterworth, 10 Hz, zero-phase | Zocco thesis |
| Flow calibration factor | k = 0.916 | Zocco thesis §4.1.3 |
| Trill modulation band | 10–35 Hz (fallback 5–40 Hz) | `ModulationConfig` |
| Vibrato search band | 3–8 Hz | `singer_acoustc_features.py` |
| Singer's formant band | 2–4 kHz | Cabrera et al. 2011 |
| LTAS slope band | 700–6000 Hz | Sundberg et al. 2012 |
| FRC age threshold | 55 years | Stratification protocol |
| Bootstrap resamples | 5000 | `effect_size.py` |

---

## 5. Data Directory Structure

```
pneumophonic_pipeline/
├── data_root/                          # Source data (read-only)
│   ├── healthy_subjects/
│   │   └── YYYYMMDD_SubjectID/
│   │       ├── csv/                    # OEP files (.csv or .dat, space-separated)
│   │       │   ├── SubjectID_Vocali.csv
│   │       │   ├── SubjectID_phonema_a_2.csv
│   │       │   ├── SubjectID_r.csv
│   │       │   ├── SubjectID_frasi.csv
│   │       │   └── SubjectID_testo.csv
│   │       ├── renders/                # Audio WAVs from Reaper
│   │       │   ├── a.wav, e.wav, ...
│   │       │   └── testo.wav
│   │       ├── sync_signal.wav         # Sync pulse WAV
│   │       └── SubjectID_audio.xlsx    # Timing sheet: task start/stop + oep_falling_edge_sec
│   └── pathological_subjects/
│       └── (same structure)
│
├── data_target/                        # Outputs (generated by pipeline)
│   ├── healthy_subjects/
│   │   ├── paired/                     # M1 HDF5 files
│   │   │   ├── SubjectID_taskname.h5
│   │   │   └── extraction_summary.csv
│   │   ├── figures/paired/             # PDF plots per subject
│   │   ├── m2_correlation/             # M2 outputs
│   │   │   ├── global_summary.csv
│   │   │   ├── global_correlation_matrix.pdf
│   │   │   ├── frc_shifts.pdf
│   │   │   ├── time_resolved/
│   │   │   └── m2_report.txt
│   │   └── M2_stratified/              # L3 stratified outputs
│   │       ├── frc_per_segment.csv
│   │       ├── frc_stratified_summary.xlsx
│   │       └── forest/histogram plots
│   └── pathological_subjects/
│       └── (same structure)
```

---

## 6. Core Package — `pneumophonic_analysis/`

### 6.1 `config.py` — Centralized Configuration

**Role:** Single source of truth for all numeric parameters. Everything that is not hardcoded in mathematical formulas lives here.

**Architecture:** Python `@dataclass` hierarchy — all fields have defaults that match the Zocco thesis and extended analysis requirements.

**Dataclasses:**

| Class | Governs |
|-------|---------|
| `AudioConfig` | Sample rate, noise reduction, pre-emphasis, STFT frame/hop, Mel/MFCC counts |
| `PitchConfig` | F0 range (pYIN), Praat unit |
| `OEPConfig` | OEP sample rates (50 Hz / 200 Hz), column names, sync peak detection thresholds |
| `SegmentationConfig` | Silence top-dB, onset delta, novelty window for glide detection |
| `FormantConfig` | Praat Burg: time step, max formants, frequency ceiling, window length |
| `JitterShimmerConfig` | Praat period/amplitude limits for perturbation analysis |
| `DSIConfig` | DSI formula coefficients |
| `ModulationConfig` | RMS envelope, Savitzky-Golay filter, modulation band for trill |
| `OutputConfig` | Excel engine, sheet names, figure DPI and format |
| `PipelineConfig` | Container for all sub-configs + `data_root` / `output_root` paths |

**Key functions:**
- `get_config()` — returns the module-level singleton `DEFAULT_CONFIG`
- `create_config(**kwargs)` — creates a custom `PipelineConfig` for experiments

---

### 6.2 `io_utils.py` — File I/O

**Role:** All disk access is routed through this module so the rest of the pipeline never imports `pathlib` or `pandas` read functions directly.

**`DataLoader` class:**
- `load_audio(path, sr)` — loads WAV with `soundfile`, resamples to target SR
- `load_sync_signal(path)` — loads the sync WAV
- `load_oep_data(path)` — reads space-separated OEP CSV/DAT, applies `OEPConfig.dat_columns` labels
- `load_timing_excel(path)` — reads subject Excel timing sheet
- `get_task_timing(subject_id, task)` — returns (start_sec, stop_sec, oep_falling_edge_sec)
- `list_audio_files(folder)` — discovers WAV files for a subject session

**`ResultsWriter` class:** exports analysis results to Excel sheets.

**Module-level helpers:**
- `save_audio(arr, path, sr)` — write WAV
- `discover_subjects(data_root)` — walks `data_root` and returns list of subject session folders
- `load_master_excel(path)` — reads the cohort-level Excel (demographics, pathology status)

---

### 6.3 `sync.py` — OEP-Audio Temporal Alignment

**Role:** Computes the time offset between the audio clock and the OEP clock using the shared sync pulse, and provides index-conversion utilities.

**`SyncResult` dataclass:** holds `time_offset_sec`, `audio_falling_edge_sec`, `oep_falling_edge_sec`, and a `valid` flag.

**`Synchronizer` class:**

| Method | Description |
|--------|-------------|
| `detect_falling_edge_audio(audio, sr)` | Uses `librosa.onset.onset_detect`; picks the 2nd onset as the falling edge |
| `detect_falling_edge_threshold(sync_signal)` | Threshold crossing for the OEP sync channel |
| `detect_sync_onsets_oep(sync_col)` | Prominence-based peak finder on OEP sync column |
| `get_take_falling_edge_oep(excel_value)` | Reads the override value from the Excel timing file |
| `synchronize(audio, oep_sync, excel_edge)` | Master method: returns `SyncResult`; prefers Excel override |
| `convert_audio_time_to_oep_sample(t_audio, sync_result, fs_oep)` | Converts audio-domain seconds → OEP sample index |
| `extract_oep_segment(oep_df, t_start, t_stop, sync_result)` | Slices the OEP DataFrame for a given audio-time window |

**Module helpers:** `detect_onset_in_phonation()`, `detect_end_of_phonation()`, `detect_phonation_bounds()` — detect the voiced interval within an audio segment.

---

### 6.4 `audio_processing.py` — Signal Processing Pipeline

**Role:** Pre-processing and frame-level feature extraction for audio signals.

**`AudioFeatures` dataclass:** holds all frame-level outputs: `stft`, `mel_spectrogram`, `mfcc`, `f0`, `f0_harmonics`, `spectral_centroid`, `rms_envelope`.

**`AudioProcessor` class:**

| Method | Description |
|--------|-------------|
| `reduce_noise(audio, sr)` | `noisereduce` stationary mode, 85% reduction |
| `apply_pre_emphasis(audio)` | 1st-order FIR high-pass: `y[n] = x[n] - 0.97·x[n-1]` |
| `compute_stft(audio, sr)` | Short-Time Fourier Transform with configured frame/hop |
| `compute_mel_spectrogram(audio, sr)` | 64-band Mel spectrogram |
| `compute_mfcc(audio, sr)` | 13 MFCCs, z-score normalized across time |
| `estimate_f0(audio, sr)` | pYIN algorithm (librosa), returns Hz contour with NaN for unvoiced |
| `compute_f0_harmonics(audio, sr, f0)` | Energy at 1st, 2nd, 3rd harmonics via STFT bins |
| `extract_features(audio, sr)` | Runs all feature extractors, returns `AudioFeatures` |
| `process_full_pipeline(audio, sr)` | Includes noise reduction + pre-emphasis before `extract_features` |

**Module utilities:** `to_db()`, `compute_spectral_centroid()`, `compute_rms_envelope()`.

---

### 6.5 `acoustic_features.py` — Praat-based Voice Quality

**Role:** Rigorous voice quality metrics via Praat, accessed through the `parselmouth` Python interface.

**`PraatAnalyzer` class:**

| Method | Description |
|--------|-------------|
| `analyze_file(path)` | Load WAV → full Praat analysis |
| `analyze_signal(audio, sr)` | In-memory array → full Praat analysis |
| `extract_pitch_metrics(sound)` | F0 mean, SD, range, min, max |
| `extract_perturbation_metrics(sound)` | Jitter (local, RAP, PPQ5, DDP) + Shimmer (local, local_dB, APQ3, APQ5, APQ11, DDA) |
| `extract_formant_metrics(sound)` | F1, F2, F3 means via Burg method |
| `extract_hnr_metrics(sound)` | Mean HNR via autocorrelation method |
| `compute_dsi(metrics)` | DSI = 0.13·MPT + 0.0053·F0_high − 0.26·I_low − 1.18·jitter% + 12.4 |

**Result dataclasses:** `PitchMetrics`, `PerturbationMetrics`, `FormantMetrics`, `VoiceQualityMetrics`, `AcousticAnalysisResult`.

**Helper:** `quick_analysis(path_or_array, sr)` — single-call convenience wrapper.

---

### 6.6 `segmentation.py` — Signal Segmentation

**Role:** Splits signals at physiologically meaningful boundaries (FRC crossings, glide transitions, modulation cycles).

**`FRCSegmenter` class:**

| Method | Description |
|--------|-------------|
| `find_frc_crossing(vcw, time, frc_level)` | Detects zero-crossings of `vcw − frc_level`; returns crossing times |
| `segment_by_frc(vcw, time, features, frc_level)` | Splits feature arrays into above/below FRC regions |
| `segment_by_time(array, time, t_start, t_stop)` | Simple time-window slicing |

**`GlideSegmenter` class (for task `a_7`):**

| Method | Description |
|--------|-------------|
| `compute_novelty_function(spectral_centroid, sr)` | Spectral difference function to detect pitch register transitions |
| `find_peak_auto(novelty, margin_pct)` | Finds the dominant novelty peak (P1→P2 boundary) |
| `segment_glide(audio, sr, spectral_centroid)` | Returns before/after the transition point |

**`ModulationAnalyzer` class (for trill `/r/`):**

| Method | Description |
|--------|-------------|
| `compute_modulation_frequency(audio, sr)` | RMS envelope → Savitzky-Golay detrend → FFT peak in 10–35 Hz band |
| `analyze_with_frc(audio, sr, vcw, time)` | Combines modulation analysis with FRC segmentation |

**Result dataclasses:** `FRCSegment`, `GlideSegment`, `ModulationResult`.

---

### 6.7 `task_analyzers.py` — Task-specific Analysis

**Role:** Provides one concrete analyzer class per vocal task type, each implementing a common `BaseTaskAnalyzer` interface. This allows `pipeline.py` to dispatch without knowing which task is being analyzed.

**`BaseTaskAnalyzer` (ABC):** defines `analyze(audio, oep_df, timing, config) → TaskResult`.

**Concrete classes:**

| Class | Task type | Key behavior |
|-------|-----------|-------------|
| `VowelAnalyzer` | Sustained vowels (a, e, i, o, u, a_3) | Praat perturbation + FRC segmentation |
| `PhraseAnalyzer` | Sentences (f_1..f_5), text (testo) | Phrase-level RMS, F0 contour, FRC alignment |
| `TrillAnalyzer` | Alveolar trill (`r`) | Modulation frequency via FFT of RMS envelope |
| `GlideAnalyzer` | Vocal glide (`a_7`) | Novelty-based transition detection + P1/P2 features |

**`TaskResult` dataclass:** `task_name`, `subject_id`, `audio_features`, `oep_features`, `segment_data`, `metrics_dict`, with `.to_dataframe()` method.

**Factory:** `get_analyzer_for_task(task_name)` — maps task label → analyzer class.

---

### 6.8 `paired_features.py` — M1 Core: Paired Feature Extraction

**Role:** The most central module. Produces time-aligned matrices combining audio features (~66 Hz) and OEP kinematics (50 Hz, upsampled by interpolation). Output is saved as HDF5.

**`OEPFrameFeatures` dataclass fields:**

```
time, vcw, vrc, vab, vrcp, vrca,
flow_cw, flow_rc, flow_ab,          # 4th-order Butterworth 10 Hz LP → differentiate → ×k
pct_rc, pct_ab,                     # rib cage / abdomen fraction of total
delta_vcw, delta_vrc, delta_vab     # sample-to-sample increments
```

**`PairedFrame` dataclass:** `subject_id`, `task_name`, `audio_features`, `oep_features`, `dataframe` (merged), `metadata` dict.

**`PairedFeatureExtractor.extract()` pipeline:**

1. Load audio → compute AudioFeatures (~66 fps)
2. Load OEP CSV → synchronize with `Synchronizer`
3. Extract OEP segment for task time window
4. `compute_oep_features_native()`:
   - Compute Vcw = A + B + C, Vrc = A + B, Vab = C
   - Differentiate to flows using `_butterworth_lowpass(fc=10 Hz, order=4)` + calibration `k = 0.916`
   - Compute compartmental ratios and increments
5. `interpolate_oep_to_audio_frames()` — resample OEP (50 Hz) to audio frame times (~66 Hz) via linear interpolation
6. `_build_dataframe()` — merge all columns into one pandas DataFrame
7. Save as HDF5 via `save_hdf5()`

**Key helper functions:**

| Function | Description |
|----------|-------------|
| `_butterworth_lowpass(data, fc, fs, order)` | `scipy.signal.butter` + `filtfilt` (zero-phase) |
| `clean_f0(f0, f_min=60, f_max=350, kernel=5)` | Clip to physiological range + median smoothing |
| `interpolate_oep_to_audio_frames(oep_df, audio_times)` | `np.interp` per OEP column onto audio frame timestamps |
| `save_hdf5(pf, path)` | Saves `PairedFrame` with metadata to `.h5` |
| `load_hdf5(path)` | Reconstructs `PairedFrame` from `.h5` |

---

### 6.9 `effect_size.py` — Statistical Effect Sizes

**Role:** Computes a comprehensive paired-difference effect-size summary. Used by the L3 stratified analysis (`analyze_l3_stratified.py`) to quantify above-FRC vs. below-FRC feature differences.

**Convention:** `diffs = below − above`. Positive median means the feature increases going from above-FRC to below-FRC.

**`PairedEffectSize` dataclass fields:**

```
n, mean_diff, median_diff, sd_diff, mad_diff,
cohen_d, hedges_g, robust_d,
wilcoxon_p, wilcoxon_r,
sign_consistency,
median_ci (tuple), cohen_d_ci (tuple)
```

`.to_dict()` flattens tuples to `_lo`/`_hi` suffixed keys for DataFrame export.

**`compute_paired_effect_size(above, below, n_boot=5000)`** — main API. Drops NaN pairs, computes all statistics, and runs bootstrap CIs.

**`interpret_cohen_d(d)`** — returns "negligible" / "small" / "medium" / "large".

---

### 6.10 `singer_acoustc_features.py` — Singer-Specific Metrics

**Role:** Adds four literature-grounded acoustic features designed for singing voice analysis and motivated by the need to find stronger correlates with OEP respiratory flow.

**`SingerAcousticFeatures` dataclass fields:**

| Field | Description |
|-------|-------------|
| `singers_formant_ratio` | Energy(2–4 kHz) / Energy(80–8000 Hz) — linear |
| `singers_formant_db` | Same in dB |
| `ltas_slope_db_per_octave` | Spectral tilt slope in 700–6000 Hz band |
| `ltas_slope_r2` | R² of the linear fit (quality indicator) |
| `vibrato_rate_hz` | Dominant F0 modulation frequency (3–8 Hz search) |
| `vibrato_extent_cents` | Peak-to-peak amplitude of vibrato in cents |
| `vibrato_regularity` | Spectral peak prominence (0–1, higher = more regular) |
| `hnr_f1_band_db` | HNR computed within F1 band (300–1000 Hz) |
| `hnr_f2_band_db` | HNR computed within F2 band (800–2500 Hz) |
| `hnr_full_db` | HNR in 80–5000 Hz (reference) |

**Functions:**

| Function | Source | Description |
|----------|--------|-------------|
| `compute_singers_formant_ratio(audio, sr)` | Cabrera, Davis & Connolly (J. Voice, 2011) | Welch PSD → band energy ratio |
| `compute_ltas_slope(audio, sr)` | Sundberg et al. (J. Voice, 2012) | Linear regression on log-frequency LTAS |
| `compute_vibrato(f0_hz, frame_rate_hz)` | Sundberg et al. (2012) | Welch PSD on detrended cents-domain F0 contour |
| `compute_formant_band_hnr(audio, sr)` | Ikuma et al. (J. Voice, 2025) | Butterworth bandpass → Praat harmonicity |
| `compute_singer_features(audio, sr, f0_hz, frame_rate_hz)` | — | Convenience wrapper, returns full `SingerAcousticFeatures` |

All functions return `np.nan` gracefully when inputs are too short or Parselmouth is unavailable.

---

### 6.11 `visualization.py` — Plotting

**Role:** Centralized plotting utilities; all figures in the pipeline are created here to ensure consistent style.

**`Visualizer` class methods:**

| Method | Output |
|--------|--------|
| `plot_waveform(audio, sr)` | Time-domain waveform |
| `plot_audio_with_sync(audio, sr, sync)` | Waveform + sync pulse overlay |
| `plot_spectrogram(audio, sr)` | Log-power Mel spectrogram |
| `plot_f0_contour(f0, times)` | F0 over time with voiced/unvoiced marking |
| `plot_task_result(task_result)` | Multi-panel: waveform, F0, OEP volumes, flows |
| `plot_frc_segment(segment)` | FRCSegment visualization with crossing marker |
| `plot_glide_segment(segment)` | GlideSegment P1/P2 spectrogram comparison |
| `save_figure(fig, path, dpi)` | Saves to PDF/PNG using `OutputConfig.figure_dpi` |

`COLORS` dict provides a consistent palette across all plots.

---

### 6.12 `pipeline.py` — High-level Orchestrator

**Role:** Wires all modules together. For standalone runs, use `run_pipeline()`. For interactive sessions, use `MAIN_UX.py`.

**Classes:**

| Class | Role |
|-------|------|
| `PneumophonicPipeline` | Holds config, data loaders, and analyzers; exposes `analyze_subject(subject_id, tasks)` |
| `SubjectAnalysis` | Result container for one subject: all task results + summary statistics |
| `BatchAnalysis` | Iterable container for all subjects; provides aggregation methods |

**`run_pipeline(data_root, output_root, subjects, tasks, config)`** — end-to-end function: discover subjects → extract paired features → run M2 correlation → save outputs.

---

## 7. Scripts

### 7.1 `scripts/test_paired.py` — Single-subject HDF5 Extraction

**Role:** Interactive debug runner. Prompts for batch (healthy/pathological), subject, and task, then calls `PairedFeatureExtractor.extract()` and saves the HDF5. Good for verifying sync alignment and feature shapes before running batch.

---

### 7.2 `scripts/batch_extract.py` — Batch M1 Extraction

**Role:** Processes all subjects and all tasks. Reads Excel timing files, skips already-extracted HDF5s, and writes `extraction_summary.csv` with success/failure status per subject-task combination.

**Output:** `data_target/{batch}/paired/SubjectID_task.h5`, `extraction_summary.csv`.

---

### 7.3 `scripts/plot_paired_features.py` — Single HDF5 Visualization

**Role:** Loads one HDF5 file and produces a multi-panel PDF: waveform, F0, OEP compartmental volumes, respiratory flows, and overlay of acoustic features on the OEP timeline.

---

### 7.4 `scripts/batch_plot_paired.py` — Batch PDF Generation

**Role:** Iterates over all `.h5` files in `data_target/paired/` and calls the single-subject plot for each. Produces one PDF per subject-task.

---

### 7.5 `scripts/m2_correlation.py` — M2 Seven-level Correlation Analysis

**Role:** Implements the full M2 analysis on the paired HDF5 database. Seven analysis levels:

1. **Global** — Pearson/Spearman correlation matrix: all audio vs. all OEP columns, averaged across subjects
2. **Time-resolved** — windowed correlation showing how audio-OEP coupling changes through the task
3. **Event-aligned (FRC)** — features averaged in windows before/after FRC crossing
4. **Sex-stratified** — levels 1–3 repeated separately for M and F groups
5. **Cross-correlation with lags** — identifies temporal lead/lag between audio and OEP signals
6. **Breath-group analysis** — segments by inhalation/exhalation cycles, computes per-cycle statistics
7. **MFCC-respiratory correlation** — each MFCC coefficient vs. each OEP flow component

**Outputs:** `global_summary.csv`, `global_correlation_matrix.pdf`, `frc_shifts.pdf`, `time_resolved/` folder, `m2_report.txt`.

---

### 7.6 `scripts/analyze_l3_stratified.py` — L3 Stratified FRC Analysis

**Role:** Extended M2 analysis with 4-way demographic stratification (Young/Elder × Male/Female). For each subgroup and each feature, computes the complete `PairedEffectSize` (Cohen's d, Hedges' g, robust d, Wilcoxon, bootstrap CI) comparing above-FRC vs. below-FRC segments.

**Outputs:**
- `frc_per_segment.csv` — raw segment-level data
- `frc_stratified_summary.xlsx` — effect sizes per feature × subgroup
- Forest plots (effect size comparison across subgroups)
- Histogram plots (distribution of differences per feature)

---

### 7.7 `scripts/analyze_trill_modulation.py` — Trill `/r/` Modulation Analysis

**Role:** Specialized analysis of the alveolar trill task. Uses `ModulationAnalyzer.compute_modulation_frequency()` (FFT on Savitzky-Golay-detrended RMS envelope, 10–35 Hz band) to extract trill rate, and correlates it with OEP flow modulation frequency.

---

### 7.8 `scripts/make_m2_summary_plots.py` — Cross-Batch M2 Aggregation

**Role:** Reads M2 output folders from both `healthy_subjects/` and `pathological_subjects/` and creates comparative plots (violin, box) contrasting the two cohorts on key correlation metrics.

---

### 7.9 `scripts/analyze_single_subject.py` — Legacy Single-subject Analysis

**Role:** Pre-M1 legacy script from the Zocco pipeline era. Performs OEP + acoustic analysis on one subject without creating paired HDF5 matrices. Kept for reference and backward comparison.

---

### 7.10 `scripts/analyze_batch.py` — Legacy Batch Analysis

**Role:** Pre-M1 batch version of `analyze_single_subject.py`. Runs on all subjects and produces the original Zocco-style Excel summary. Superseded by `batch_extract.py + m2_correlation.py`.

---

### 7.11 `scripts/tools.py` — Diagnostic Utilities

**Role:** Ad-hoc inspection commands for data integrity checking.

| Function | Description |
|----------|-------------|
| `inspect_oep_headers(path)` | Prints column names and first rows of an OEP file |
| `check_sync_peaks(subject_folder)` | Plots sync signal and detected peaks/edges for visual verification |
| `inventory_subjects(data_root)` | Lists all subject folders, checks for required files, reports missing data |

---

### 7.12 `MAIN_UX.py` — Interactive Menu System

**Role:** Terminal UI for running any pipeline component interactively. Numbered menu options launch scripts or pipeline stages without memorizing command-line arguments. Useful for day-to-day operation during thesis work.

---

### 7.13 Root Utilities

| File | Role |
|------|------|
| `count_check.py` | Counts subjects and files; verifies expected data presence |
| `tool_h5_id_match.py` | Matches subject IDs in HDF5 files against the master Excel cohort list; flags missing or mismatched entries |
| `requirements.txt` | Python dependency list |
| `setup.py` | `pip install -e .` package installer |

---

## 8. HDF5 File Format (M1 Output)

Each `.h5` file corresponds to one (subject, task) pair. Internal structure:

```
SubjectID_task.h5
├── dataframe/                  # Merged audio+OEP DataFrame stored as HDF5 dataset
│   └── columns: [time, f0, mfcc_0..12, spectral_centroid, rms,
│                 vcw, vrc, vab, vrcp, vrca,
│                 flow_cw, flow_rc, flow_ab,
│                 pct_rc, pct_ab,
│                 delta_vcw, delta_vrc, delta_vab, ...]
└── metadata/                   # Attributes: subject_id, task_name, sr, fs_oep,
                                #             sync_offset_sec, extraction_date, ...
```

Load with `load_hdf5(path)` → returns `PairedFrame`. The DataFrame index is audio frame time in seconds.

---

## 9. Analysis Workflow — Step by Step

```
Raw Data (WAV + OEP CSV + Excel timing)
        │
        ▼
[io_utils.py]  ──► load audio, load OEP, read Excel timing
        │
        ▼
[sync.py]  ──► detect falling edge → compute time_offset_sec (SyncResult)
        │
        ▼
[audio_processing.py]  ──► noise reduction, pre-emphasis, STFT, Mel, MFCC, pYIN F0
        │
        ▼
[acoustic_features.py]  ──► Praat: jitter, shimmer, HNR, formants, DSI
[singer_acoustc_features.py]  ──► singer's formant ratio, LTAS slope, vibrato, band HNR
        │
        ▼
[paired_features.py]  ──► compute OEP volumes/flows → interpolate to audio rate → merge
        │
        ▼  (M1 output)
HDF5 file: SubjectID_task.h5
        │
        ▼
[segmentation.py]  ──► FRC crossing detection, glide segmentation, trill modulation
        │
        ▼
[effect_size.py]  ──► Cohen's d, Hedges' g, robust d, Wilcoxon, bootstrap CI
        │
        ▼  (M2 output)
[m2_correlation.py / analyze_l3_stratified.py]
     Global + time-resolved + FRC + sex+age stratified correlation reports
        │
        ▼
[visualization.py / batch_plot_paired.py]
     PDF reports per subject-task
```

---

## 10. References

| Citation | Used for |
|----------|---------|
| Zocco, B. (2025). *Integrated Analysis of Respiratory-Phonatory Functions: Normative Patterns Across Sex and Age*. Politecnico di Milano. | OEP protocol, flow calibration (k=0.916), LP filter parameters, two-compartment model |
| Cabrera, D., Davis, P., & Connolly, A. (2011). *Long-term average spectral features of operatic singing*. J. Voice. | Singer's formant band (2–4 kHz), energy ratio formula |
| Sundberg, J., Gu, L., Huang, Q., & Huang, M. (2012). *Acoustics of Peking opera singing*. J. Voice. | LTAS slope (700–6000 Hz), vibrato rate norms (3–8 Hz) |
| Ikuma, T., McWhorter, A., Oral, K., & Kunduk, M. (2025). *Formant-aware HNR*. J. Voice. | Band-limited HNR within F1 and F2 ranges |
| Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). | Cohen's d thresholds, effect size interpretation |
| Hedges, L. V., & Olkin, I. (1985). *Statistical Methods for Meta-Analysis*. | Hedges' g small-sample correction factor |
