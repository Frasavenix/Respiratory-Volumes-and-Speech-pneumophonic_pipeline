#!/usr/bin/env python3
"""One-off: apply the reviewer's notebook style corrections (markdown + titles).

Rules applied:
- "**Observe:**/**Observation:**" -> "**Observations:**" header then a new line.
- concise prose, no em-dashes ("—"/"--") in sentences.
- "nbNN"/"(MN)" -> "notebook n°N" / "Milestone N (MN)".
- figure titles anonymised: real subject_id -> stable alias S01..S39 (anon()).
- observations rely only on the present plot or earlier ones (no forward refs).
- every plot recalls its task/dataset.

Idempotent: markdown cells are set to absolute text; code replacements vanish
after first pass; the anon() helper is appended only if missing.
"""
import re
import sys
from pathlib import Path
import nbformat

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def hb(text: str) -> str:
    """Force markdown hard line breaks: append a trailing backslash to every
    non-blank, non-header, non-table line that is immediately followed by
    another non-blank line. Without it, consecutive lines collapse into one
    rendered line in Jupyter. The last line of each paragraph keeps no break.
    """
    lines = text.split("\n")
    out = []
    for i, ln in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        collapses = bool(nxt.strip())                 # next line would merge in
        is_header = ln.lstrip().startswith("#")
        is_table = "|" in ln
        is_list = bool(re.match(r"\s*(\d+\.|[-*+])\s", ln))
        if ln.strip() and collapses and not is_header and not is_table and not is_list \
                and not ln.rstrip().endswith("\\"):
            out.append(ln.rstrip() + "\\")
        else:
            out.append(ln)
    return "\n".join(out)

HELPER = '''

# --- Presentation-safe subject aliases (figures may be reused in talks) -----
# Real IDs stay in code; titles show a stable generic alias ("S01".."S39"),
# assigned by alphabetical order of the corpus and identical across notebooks.
def _build_alias():
    _pdir = REPO / "data_target" / "healthy_subjects" / "paired"
    _sids = sorted({f.stem.split('_', 1)[0] for f in _pdir.glob('*.h5')})
    return {s: f"S{i+1:02d}" for i, s in enumerate(_sids)}
ALIAS = _build_alias()
def anon(sid): return ALIAS.get(sid, sid)'''

# ---------------------------------------------------------------- notebook 02
NB02_MD = {
 "d97be7f8": """## 2. Audio loudness vs chest-wall volume
Task `a_2` (maximum phonation time on /a/), one example subject.
Acoustic loudness (left axis) against the volume excursion `ΔVcw` (right axis).

The loudness curve is `rms_audio`: the RMS of the *raw* audio.
The pipeline's `energy` feature is computed on the pre-emphasised, L2-normalised, denoised signal. That chain annihilates the steady low-frequency vowel and collapses a long phonation to a single onset spike (drawn faint and dotted for contrast). `rms_audio` shows the truth: loudness stays broadly sustained while the chest wall deflates.

**Observations:**
Loudness is roughly flat: a small onset peak, then a plateau.
`ΔVcw` shows a large, steady decrease over the same window.
In steady phonation the audio links to volume depletion, not to a matching loudness change.
The single-peak `energy` curve is a computation artifact, not weak phonation.""",

 "c184c914": """## 3. F0 vs expiratory flow (voiced frames only)
Task `a_2` (maximum phonation time on /a/), one example subject.
Fundamental frequency against chest-wall flow `flow_cw`. Restricting to voiced frames removes the unvoiced gaps where F0 is undefined.""",

 "e2ca586e": """**Observations:**
Task `a_2` (maximum phonation time on /a/), one example subject.
F0 (purple) holds essentially flat: a steady held pitch.
`flow_cw` (green) stays below the zero line for ~95 % of the vowel (≈ −0.19 L/s here): a small, roughly steady expiratory flow, not an oscillation about zero.
The sign is negative by the pipeline's convention. `flow_cw = d(Vcw)/dt`, and the chest wall deflates monotonically (the same `ΔVcw` decrease seen in §2), so the time-derivative is negative throughout a held vowel.
It reads small and only grazes zero because the deflation is slow (~3.4 L over ~16 s, so ~0.2 L/s mean), while a ±0.15 L/s ripple (heart-beat plus numerical differentiation) occasionally pokes a few samples above the axis.
F0 and `flow_cw` do not track each other on this recording: pitch stays put while flow holds its steady expiratory level.""",

 "15e056ee": """## 4. Compartmental strategy over time
Task `a_2` (maximum phonation time on /a/), one example subject.
Instantaneous contribution of the rib cage (`%RC`) and abdomen (`%AB`) to total chest-wall volume, with `%RC + %AB = 1`. The split is the breathing strategy: rib-cage versus abdomen dominant phonation.""",

 "cb077666": """**Observations:**
Task `a_2` (maximum phonation time on /a/), one example subject.
`%RC` sits well above 0.5 (rib-cage-leaning) and drifts slightly as the lung empties.
The split is fairly stable within this held vowel.
The next cell compares the split across several subjects.""",

 "224be85d": """## 6. Spectral representations stored in the HDF5
`PairedFeatureExtractor.save_hdf5` also persists the full **STFT power spectrogram**, the **mel-spectrogram**, and the **MFCC** matrix, so the raw `.wav` is not needed to inspect the time-frequency content. We read them directly with `h5py`.""",

 "216b6436": "### 6a. STFT spectrogram (log-frequency)",
 "6c9b2f4e": "### 6b. Mel-spectrogram",
 "ab072680": """### 6c. MFCC heatmap
13 mel-frequency cepstral coefficients over time (z-normalised per coefficient).""",
 "cc90c3c8": """## 7. Reusing the package `Visualizer`
The two plots below come straight from `pneumophonic_analysis.visualization.Visualizer`, the class used across the legacy pipeline. First the **F0 trace** with its mean ± 1 SD band.""",

 "a2056f3d": """### 4b. Compartmental strategy varies across subjects
Task `a_2` (maximum phonation time on /a/), three example subjects.
The split is a personal breathing strategy. Across the cohort the rib-cage share of a sustained /a/ ranges from ~56 % (abdomen-leaning) to ~78 % (strongly rib-cage). All subjects are rib-cage-majority, but the degree differs markedly.""",

 "d103e1d3": """## 5. Within-recording correlation matrix
Task `a_2` (maximum phonation time on /a/), one example subject.
Pearson correlations between the main audio and OEP features on voiced frames. This is the per-segment building block of the Milestone 2 (M2) analysis in notebook n°3.""",

 "02b00f4c": """**Observations:**
Task `a_2` (maximum phonation time on /a/), one example subject.
Block structure is visible. `energy` tracks `vcw` and `delta_vcw` (loudness with volume).
`vcw`, `delta_vcw` and `pct_rc` move together: the kinematics are near-redundant under monotonic depletion.
`f0` and `flow_cw` sit largely independent of the rest, consistent with §3.
This is within a single recording. The cohort picture is built in notebook n°3.""",

 "0d73dd82": """**Observations:**
Task `a_2` (maximum phonation time on /a/), one example subject.
Horizontal harmonic bands (a periodic vowel) sit at integer multiples of F0, and their slow drift is the F0 contour.
Energy concentrates low, in the formants.
The broadband vertical streak at onset is the attack that the `energy` feature over-weights.""",

 "a4b45aa8": """**Observations:**
Task `a_2` (maximum phonation time on /a/), one example subject.
The mel scale compresses the highs, so harmonic detail collapses into the formant envelope.
That perceptually-weighted spectral shape is what the MFCCs summarise.""",

 "779d21d7": """**Observations:**
Task `a_2` (maximum phonation time on /a/), one example subject.
MFCC-0 (log-energy) dominates the range.
Higher coefficients (spectral shape) are far more stable across the sustained vowel: a steady articulatory posture.
These feed the Milestone 2 (M2) MFCC and respiratory analysis in notebook n°3.""",

 "1056c747": """**Observations:**
Task `a_2` (maximum phonation time on /a/), one example subject.
The `Visualizer` F0 trace for this steady /a/ is near-flat, with a small vibrato-like ripple.
The mean ± 1 SD band is narrow, confirming the held pitch seen in §3.""",

 "0a2d9248": """### 7b. Chest-wall volume with FRC zones
Task `a_2` (maximum phonation time on /a/), one example subject.
`Visualizer.plot_oep_volume` shades the recording into above-FRC (elastic recoil) and below-FRC (active expiratory effort) regions. The FRC crossing is located with the genuine baseline-derived detector also used by the L3 and FRC analyses in notebook n°4.""",

 "dfb8bedc": """**Observations:**
Task `a_2` (maximum phonation time on /a/), one example subject.
`Vcw` falls monotonically through the resting FRC level (dashed, baseline-derived).
Phonation starts above FRC (inspiratory reserve) and continues below it (expiratory reserve).
This above/below-FRC split is carried into the effect-size analysis in notebook n°4.""",

 "090b0443": """## Recap
From a single paired HDF5 file we reproduced every per-recording plot the Milestone 1 visualization scripts produce, plus the stored time-frequency representations, with no raw audio required.
The next notebook (n°3) aggregates these recordings across the whole cohort for the Milestone 2 (M2) correlation study.""",
}

NB02_GLOBAL = [
 ("# 02 — Paired-Feature Visualization (M1 output)",
  "# Notebook n°2: Paired-Feature Visualization (Milestone 1 output)"),
 ("# Notebook n°2 — Paired-Feature Visualization (Milestone 1 output)",
  "# Notebook n°2: Paired-Feature Visualization (Milestone 1 output)"),
 ("{SID} — {TASK}:", "{anon(SID)} · {TASK}:"),
 ('f"{sid} a_2  (mean %RC = {rc:.0%})"', 'f"{anon(sid)} · a_2  (mean %RC = {rc:.0%})"'),
 ("abdomen-leaning (SaMa) → balanced (AlMo) → rib-cage-dominant (LoFa)",
  "abdomen-leaning → balanced → rib-cage-dominant"),
 ("dir at {PAIRED_DIR} — run scripts/batch_extract.py first.",
  "dir at {PAIRED_DIR}: run scripts/batch_extract.py first."),
 ("onset spike — shown faint/scaled for contrast only.",
  "onset spike, shown faint/scaled for contrast only."),
]

# ---------------------------------------------------------------- notebook 00
NB00_MD = {
 "3b902373": """# Notebook n°0: Pipeline Overview & Configuration

**Pneumophonic Analysis Pipeline**, notebook n°0. This pipeline jointly analyses **Optoelectronic Plethysmography (OEP)** chest-wall kinematics and **acoustic voice** signals to study respiratory-phonatory coupling (continuation of Bianca Zocco's 2025 thesis).

These notebooks serve both a CAPSTONE project and the foundation of the next thesis works.

This notebook covers configuration, data layout, the OEP column model, the vocal-task protocol, and the map from the original scripts to these notebooks. See **`NOTEBOOKS_GUIDE.md`** for the full guide.

## Milestones
| used for | Milestone | Notebook(s) | Question |
|-----------|-----------|-------------|----------|
|**CAPSTONE + thesis**| Milestone 1 (M1): paired extraction | n°1 | build time-aligned `[audio \\| OEP]` matrices |
|**CAPSTONE + thesis**| visualization | n°2 | inspect one recording end-to-end |
|**CAPSTONE + thesis**| Milestone 2 (M2): correlation | n°3, n°4 | does the voice covary with breathing? (plus effect sizes) |
|**CAPSTONE + thesis**| Milestone 3 (M3): modelling | n°5, n°6 | can audio *predict* respiratory state? |
|**CAPSTONE + thesis**| acoustic / tasks | n°7 | Praat metrics, segmentation, `Visualizer` |
|**thesis**| operatic singers | n°8 | singer-specific features (vibrato, singer's formant) |
|**thesis**| diagnostics | n°9 | cohort, coverage, HDF5 sanity checks |

## How to run
1. Use a Python env with the project deps (`pip install -r requirements.txt`) and the package installed (`pip install -e .`).
2. Open any notebook and **Run All**. Notebooks read the committed HDF5 corpus under `data_target/healthy_subjects/paired/`; audio-input notebooks (n°7, n°8) use synthetic signals because raw `.wav` files are not committed.
3. They are roughly ordered n°0 to n°9; each is self-contained.""",

 "145ca406": """## 6. Teaser: one recording, two systems on one axis
The whole point of Milestone 1 (M1): audio energy and chest-wall volume on a shared time axis.
Task `a_2` (maximum phonation time on /a/), one example subject.""",

 "04055c7d": """**Observations:**
Task `a_2` (maximum phonation time on /a/), one example subject. The subject held the vowel for as long as possible.
Loudness (`rms_audio`) holds roughly steady while `Vcw` falls in a near-linear decline.
This audio ⊕ OEP alignment is the foundation everything downstream is built on.""",
}

NB00_GLOBAL = [
 ("{meta['subject_id']} — {meta['task_name']}:",
  "{anon(meta['subject_id'])} · {meta['task_name']}:"),
 ('f"{SUBJ} — {TASK}: source waveform', 'f"{anon(SUBJ)} · {TASK}: source waveform'),
 ("Raw source WAV for {SUBJ}/{TASK}", "Raw source WAV for {anon(SUBJ)}/{TASK}"),
 ("HDF5-only) — skipping", "HDF5-only): skipping"),
 ("No HDF5 corpus found — run scripts/batch_extract.py.",
  "No HDF5 corpus found: run scripts/batch_extract.py."),
 (', "—",', ', "(none)",'),
]

# ---------------------------------------------------------------- notebook 01
NB01_MD = {
 "49562f68": """# Notebook n°1: Milestone 1 (M1) Paired Feature Extraction (Audio ⊕ OEP)

**Pipeline stage:** Milestone 1 (M1) turns a raw recording (audio `.wav` + OEP `.csv` + sync pulse + Excel timing) into a single **time-aligned `[audio | OEP]` feature matrix**, saved as HDF5. Every later notebook consumes those HDF5 files.

**Scripts / module reproduced**
| Script / module | Role |
|-----------------|------|
| `pneumophonic_analysis/paired_features.py` | `PairedFeatureExtractor`, OEP feature computation, interpolation, HDF5 I/O |
| `scripts/test_paired.py` | interactive single-recording extraction |
| `scripts/batch_extract.py` | batch extraction across all subjects (the `TASK_MAP`) |

**The extraction pipeline** (`PairedFeatureExtractor.extract`):
1. load audio, sync pulse, and OEP CSV (`io_utils.DataLoader`)
2. synchronize via the Excel **falling-edge** time (`sync.py`)
3. auto-trim to the actual phonation window
4. extract frame-level audio features (pYIN F0, energy, spectral centroid, MFCCs) at ~66.7 fps
5. compute OEP volumes and flows at 50 Hz, then **interpolate onto the audio frame grid**
6. merge into one DataFrame and write HDF5 (plus STFT/mel/MFCC matrices)

> The raw `.wav`/`.csv` are not committed, so steps 4 to 6 are shown on **synthetic OEP** plus a **real extracted HDF5** output. Plot types: OEP volumes, OEP flows, native-vs-interpolated resampling overlay, and the final aligned audio ⊕ OEP matrix.""",

 "37f77d20": """## 2. OEP feature computation (native rate, auto-detected)
`compute_oep_features_native` derives the compartmental volumes, then the **flows** (numerical derivative of a 10 Hz Butterworth-filtered volume × the k=0.916 calibration factor), and the instantaneous compartmental ratios.

> **OEP sample rate.** Acquisitions are not uniform: some OEP CSVs are 50 Hz, others 100 Hz. `extract()` now reads the true rate from each CSV's `time` column. Using the wrong rate would time-stretch the OEP and desync it from the audio.""",

 "355189c3": """## 1. The OEP column model
The OEP `.csv` is space-separated with these columns. The two-compartment model splits the chest wall into rib cage (`Vrc = A + B`) and abdomen (`Vab = C`), and `A + B + C = tot_vol` (Vcw).

| Column | Label | Quantity |
|--------|-------|----------|
| 1 | `time` | time (s) |
| 2 | `A` | Vrcp: pulmonary rib-cage volume (L) |
| 3 | `B` | Vrca: abdominal rib-cage volume (L) |
| 4 | `C` | Vab: abdominal volume (L) |
| 5 | `tot_vol` | Vcw: total chest-wall volume (L) |
| 6 | `sync` | synchronization signal |""",

 "25c7dce3": """## 5. A real extracted HDF5: the Milestone 1 (M1) output
Task `a_2` (maximum phonation time on /a/), one example subject.
Loading a real extracted recording shows exactly what `extract()` + `save_hdf5` produce: the aligned matrix, the metadata (including the sync `sync_time_offset_sec`), and the compartmental sanity check `Vrc + Vab = Vcw`.""",

 "7a380014": """**Observations:**
Synthetic OEP recording (notebook n°1 demo), not a real subject.
`Vrc + Vab = Vcw`, and `%RC + %AB = 1`, by construction.
Flows are the time-derivatives of the measured volumes, crossing zero at the volume turning points (middle panel).
Breathing strategy varies between subjects: the same task can lean more on the abdomen or the rib cage.
This figure shows one example strategy only.""",

 "b9475d1c": """**Observations:**
Synthetic OEP recording (notebook n°1 demo), zoomed to 3 to 5 s.
The linear interpolation rides exactly through the native 50/100 Hz samples (gray dots) and fills the ~66 fps audio grid, with no smoothing or phase shift.
The OEP is the limiting resolution: breathing kinematics stay below ~10 Hz.
Upsampling only aligns it to the audio frames; it adds no information.""",

 "f11e8aa1": """### 5b. The aligned result
Task `a_2` (maximum phonation time on /a/), one example subject.
**Audio energy and chest-wall volume now share one time axis.** This coupling is studied deeper in Milestone 2 (M2), notebook n°3.""",

 "197df821": """**Observations:**
Task `a_2` (maximum phonation time on /a/), one example subject.
Loudness (`rms_audio`) stays roughly sustained while `Vcw` falls steadily: a held vowel driven by controlled expiration.
This shared time axis (audio ⊕ OEP) is the Milestone 1 (M1) output that everything downstream sits on.
We plot `rms_audio`, the raw-audio loudness, rather than the onset-spiking `energy` feature.""",

 "ec9cc720": """## 6. Run Milestone 1 (M1) on *new* data: file browser

The raw `.wav`/`.csv` recordings are not committed to this repo (they were removed because the folders were too large).

To extract new datasets wherever they live on disk, this cell opens a native **folder browser**: pick each subject folder (one containing `renders/`, `csv/`, `sync_signal.wav`, `<ID>_audio.xlsx`), press **Cancel** when done, then enter a batch label. It reuses `scripts/batch_extract.py` so the task map, Excel-timing sync, and HDF5 writing are identical.

> The cell is **guarded** by `RUN_EXTRACTION = False` so *Run All* and headless execution never block on a dialog. Flip it to `True` and run just this cell to extract. Equivalent terminal command: `python scripts/batch_extract.py`.""",

 "28fe1d05": """## Recap
Milestone 1 (M1) merges two acquisition systems running at different rates (OEP and audio) into one analysis-ready matrix. To run it for real:

```bash
python scripts/test_paired.py      # one recording, interactive
python scripts/batch_extract.py    # whole cohort -> data_target/<batch>/paired/*.h5
```

Then continue with notebook n°2 (visualization) and notebook n°3 (Milestone 2 (M2) correlation).""",
}

NB01_GLOBAL = [
 ("{meta['subject_id']} — {meta['task_name']}:",
  "{anon(meta['subject_id'])} · {meta['task_name']}:"),
 ("aligned audio ⊕ OEP (M1 output)", "aligned audio ⊕ OEP (Milestone 1 output)"),
]

# ---------------------------------------------------------------- notebook 03
NB03_MD = {
 "e0593a24": """# Notebook n°3: Milestone 2 (M2) Exploratory Respiratory-Acoustic Correlation Analysis

**Pipeline stage:** Milestone 2 (M2): does anything in the voice covary with the breathing signal?

**Script this notebook is based on:** `scripts/m2_correlation.py` (all seven analysis *levels*).\\
We import the script's own computation helpers so the numbers match exactly, and render every figure inline instead of writing PDFs.

| Level | Question | Plot type(s) |
|-------|----------|--------------|
| 1 · Global | Do per-segment summaries correlate across the cohort? | correlation heatmap, scatter grid, within-segment histograms |
| 2 · Time-resolved | Does coupling fluctuate within a recording? | dual-axis + sliding-r panel, aggregate histogram |
| 3 · FRC event | Do features shift across the FRC crossing? | shift histograms, above-vs-below paired scatter |
| 4 · Sex-stratified | Are correlations different by sex? | side-by-side heatmaps, sex-coloured scatter |
| 5 · Lag | Does flow lead the acoustics? | mean cross-correlation curve, peak-lag histogram |
| 6 · Breath group | How do features evolve across a read text? | evolution scatter + binned means, correlation heatmap |
| 7 · MFCC | Does spectral shape track respiration? | MFCC×OEP heatmap, per-coefficient boxplots |""",

 "a4591384": """**Summary (derived in the levels below).**
After the OEP sample-rate fix (about a third of recordings were 100 Hz read as 50 Hz, which time-stretched the OEP), the cross-domain correlations are coherent: the corrected 100 Hz subjects now match the never-affected 50 Hz reference, and the previously inflated flow correlations (a 2× time-stretch artifact) are gone.
In steady phonation the coupling is modest: energy with chest-wall volume is the strongest (a touch higher in elderly males), spectral-centroid with volume is weaker, and acoustic with flow is about zero.
Voice tracks lung volume (position), not flow (rate); a strong coupling needs a task that actively modulates the feature.
The one strong coupling, f0 with volume in the vocal glide `a_7`, plus the stratified breakdown and the FRC example, are in notebook n°4.""",

 "c02a680b": """## Level 1: Global per-segment correlations
All 550 recordings (39 subjects, every task pooled).
Each recording is reduced to summary statistics (`m2.compute_segment_summary`); we then correlate those summaries *across* the cohort.""",

 "36630923": """### 1a. Cross-subject correlation heatmap (audio × OEP summaries)
All 550 recordings, per-segment summaries pooled across subjects.""",

 "6750951b": """**Observations:**
Between subjects, audio summaries correlate only weakly to moderately with OEP summaries.
The strongest cell is `energy_mean` × `delta_vcw_range` (loudness with volume excursion).
No audio feature is a strong linear proxy for a respiratory summary across subjects.""",

 "d8eb1f6d": """### 1b. Key scatter relationships (coloured by task family)
All 550 recordings, per-segment summaries (sustained vs speech vs vowels).""",

 "2c6af02f": """**Observations:**
The clearest trend is volume-excursion with loudness: a bigger `ΔVcw` span goes with louder phonation.
The cloud is broad and splits by task family (sustained vs speech).
The link is real but modest; subject identity explains much of the spread.""",

 "a9a76b45": """### 1c. Distribution of within-segment correlations
All 550 recordings; each contributes its own internal Pearson r between coupled signals.""",

 "cc2b9336": """**Observations:**
Within-recording correlation distributions sit modestly off zero.
`energy↔ΔVcw` leans positive (loudness tracks volume); `f0↔flow` and `energy↔flow` sit near zero.
After the rate fix these are coherent; the previously inflated flow correlations (desync artifact) are gone.""",

 "b52e4bcf": """## Level 2: Time-resolved coupling
Task `a_2` (maximum phonation time on /a/), one example recording.
Sliding-window Pearson correlation (`m2.compute_sliding_correlation`, 0.5 s window) between audio energy and `ΔVcw`.""",

 "7f7bd8f0": """**Observations:**
Task `a_2` (maximum phonation time on /a/), one example recording.
The sliding-window correlation wanders across the recording: coupling is not stationary.
It strengthens and weakens within one phonation, so a single segment-level r hides this.""",

 "b86c2328": """### 2b. Aggregate distribution of mean sliding-r (sustained tasks)
All sustained recordings (a, a_2, a_3, a_7, r).""",

 "07893970": """**Observations:**
All sustained recordings.
The pooled mean sliding-r straddles zero with a modest positive lean.
Instantaneous energy↔volume coupling is weak and variable, not a fixed strong relationship.""",

 "ccc48441": """## Level 3: FRC-crossing event analysis
Maximum-phonation tasks `a_2`, `a_3`, `a_7`.
Each recording is split at the FRC crossing and audio features are compared above vs below FRC (the same detector used in notebook n°4).""",

 "07e783c5": """**Observations:**
Maximum-phonation tasks `a_2`, `a_3`, `a_7` (79 segments).
At the FRC crossing, `f0` and `%RC` shift in a consistent direction (medians off zero).
`energy` and `flow` shifts are smaller and noisier.
`%RC` is the most consistent shifter: the rib cage re-weights as the lung empties.
Standardized genuine-FRC effect sizes follow in notebook n°4.""",

 "3032eb9e": """### 3b. Above-vs-below paired scatter (identity line)
Maximum-phonation tasks `a_2`, `a_3`, `a_7`; one point per recording.""",

 "eb95cbf1": """**Observations:**
Maximum-phonation tasks `a_2`, `a_3`, `a_7`.
Points sit off the identity line in a consistent direction (for example F0 higher below FRC).
The shift is systematic, not scatter; the spread within each panel is between-subject variability.""",

 "bf8d7cff": """## Level 4: Sex-stratified correlations
All 550 recordings, split by subject sex from `subjects_metadata.csv`.
We recompute the global heatmap separately for each sex, then overlay sex-coloured scatter.""",

 "4fb8952f": """**Observations:**
All 550 recordings, split by sex.
The correlation structure is broadly similar in males and females: modest magnitude differences, no sign flips.
Sex alone does not reorganise the coupling; age is examined in notebook n°4.""",

 "6a21cb51": """**Observations:**
All 550 recordings, split by sex.
The flow↔F0 and related scatters overlap heavily across sexes.
Any separation is small relative to the within-group spread.""",

 "3c153c4e": """## Level 5: Cross-correlation with time lags
All sustained recordings.
`m2.compute_xcorr_lag` finds the lag (±0.5 s) at which chest-wall flow best predicts energy and F0. A negative lag means flow leads the acoustic signal.""",

 "d402e6ed": """**Observations:**
All sustained recordings.
The lagged cross-correlation peaks near zero lag: no consistent lead or lag between acoustic and respiratory signals at the frame scale.
They co-vary roughly instantaneously, when they co-vary at all.""",

 "1a030163": """**Observations:**
All sustained recordings.
Peak lags cluster around 0 (a few frames).
The occasional large lag is a recording with weak coupling where the peak is ill-defined, not a real temporal offset.""",

 "ba31ee7a": """## Level 6: Breath-group analysis (connected speech)
Task `testo` (balanced text reading).
The text is segmented into breath groups at inspiratory peaks of `ΔVcw`; we track how features evolve from the first to the last breath group.""",

 "35c53499": """**Observations:**
Task `testo` (balanced text reading), 623 breath groups.
Within a breath group, F0 and loudness decline as the lung empties (declination), tracking the falling `ΔVcw`.
This is the clearest natural audio↔respiratory coupling, driven by sub-glottal pressure falling with volume.""",

 "299bda10": """**Observations:**
Task `testo` (balanced text reading).
Aggregating by breath group strengthens correlations by smoothing frame-level noise and exposing declination coupling.""",

 "102821b8": """## Level 7: MFCC ↔ respiratory correlations
A 150-recording subset of the corpus (all tasks).
For each recording we correlate the first 8 MFCCs with three respiratory targets on voiced frames, then summarise across the subset as a heatmap and per-coefficient boxplots.""",

 "7c8cb51f": """**Observations:**
150-recording subset.
Most MFCCs correlate weakly with respiratory features.
The strongest are low-order coefficients (overall energy and tilt), echoing the energy↔volume link rather than adding independent articulatory information.""",

 "313b45a7": """**Observations:**
150-recording subset.
The MFCC↔OEP picture is mostly pale (|r| < 0.2): no MFCC is a strong respiratory marker.
Spectral shape carries little respiratory information beyond what loudness already gives.""",

 "e01190d0": """## Recap
All seven Milestone 2 (M2) levels run here on the real 550-recording corpus, reusing `m2_correlation.py`'s own computation functions. To regenerate the full PDF report instead, run:

```bash
python scripts/m2_correlation.py     # choose "healthy_subjects"
```

Notebook n°4 drills into Level 3 (FRC crossing) with standardized effect sizes and demographic stratification.""",
}

NB03_GLOBAL = [
 ("{meta['subject_id']} — {meta['task_name']}:",
  "{anon(meta['subject_id'])} · {meta['task_name']}:"),
 ("a1.plot(df['time'], df['energy'], color='steelblue', alpha=0.8, label='Energy')",
  "a1.plot(df['time'], df['energy'], color='steelblue', alpha=0.8, label='Energy')\n"
  "if 'rms_audio' in df.columns:\n"
  "    a1.plot(df['time'], df['rms_audio'], color='green', alpha=0.5, lw=1.2, label='loudness (rms_audio)')"),
 ("Level 3 — ", "Level 3: "),
 ("Level 4 — ", "Level 4: "),
 ("Level 5 — ", "Level 5: "),
 ("Level 6 — ", "Level 6: "),
 ("Level 7 — ", "Level 7: "),
 ("MFCC–respiratory", "MFCC-respiratory"),
]

# ---------------------------------------------------------------- notebook 04
NB04_MD = {
 "a18e4adc": """# Notebook n°4: Milestone 2 (M2) / Level 3 FRC Crossing with Standardized Effect Sizes by Demographic

**Pipeline stage:** the inferential follow-up to Milestone 2 (M2) Level 3. A Wilcoxon p-value tells you the *sign* of the above-to-below-FRC shift is reliable, but not whether the shift is large relative to its own spread. This notebook adds **Cohen's d**, robust d, bootstrap CIs, sign-consistency, and a **demographic stratification** (Young/Elder × Male/Female).

**Scripts reproduced**
| Script | Role |
|--------|------|
| `scripts/analyze_l3_stratified.py` | segment collection, effect sizes, forest & stratified histograms |
| `pneumophonic_analysis/effect_size.py` | `compute_paired_effect_size` (Cohen's d, robust d, Wilcoxon r, bootstrap CIs) |
| `scripts/make_m2_summary_plots.py` | Cohen's-d comparison bar chart, orthogonality scatter |

> **Plot types:** forest plot (median + bootstrap CI), 2×2 stratified histograms, grouped Cohen's-d bar chart with qualitative reference lines, orthogonality scatter with group medians.""",

 "31da563a": """## 1. Collect FRC-split segments
Maximum-phonation tasks `a_2`, `a_3`, `a_7` (85 segments, 37 subjects).
`l3.collect_paired_segments` walks the sustained-task HDF5 files, splits each at the FRC crossing, computes above/below feature means, and joins subject sex and age.""",

 "fc65b770": """## 2. Stratified effect-size table
Maximum-phonation tasks `a_2`, `a_3`, `a_7`, all 85 segments.
`l3.stratified_effect_sizes` runs `compute_paired_effect_size` for each feature across the nine strata. Below, the overall (`All`) row for every feature.""",

 "ed61f9f7": """## 3. Forest plot: F0 shift across strata
Maximum-phonation tasks `a_2`, `a_3`, `a_7`.
Median below-above shift with bootstrap 95% CI; blue markers are Wilcoxon-significant (p<0.05), with Cohen's d and sign-consistency annotated on the right. This is the headline Level 3 figure.""",

 "fbc7693f": """**Observations:**
Maximum-phonation tasks `a_2`, `a_3`, `a_7`.
Every stratum's CI sits right of zero: f0 rises below FRC.
The effect is medium (d ≈ 0.5 to 0.75), with wide CIs in the small cells (young females).
Direction is consistent, magnitude modest.""",

 "076cc4a5": """### 3b. Forest plot: %RC shift
Maximum-phonation tasks `a_2`, `a_3`, `a_7`.
The rib-cage contribution shift; contrast its demographic pattern with the F0 forest above.""",

 "07150cdb": """**Observations:**
Maximum-phonation tasks `a_2`, `a_3`, `a_7`.
The CIs sit clearly right of zero with large effects (d ≈ 1.0 to 1.5), strongest in young males and the elderly.
`%RC` is the robust FRC-crossing marker: the rib cage re-weights as the lung empties.""",

 "d5400610": """## 4. Per-segment shift histograms by subgroup (F0)
Maximum-phonation tasks `a_2`, `a_3`, `a_7`, split Young/Elder × Male/Female.
A 2×2 grid (one panel per demographic cell) of the raw per-segment below-above shift, with median and Cohen's d annotated.""",

 "3a1afef2": """**Observations:**
Maximum-phonation tasks `a_2`, `a_3`, `a_7`, by demographic.
The f0 and %RC shifts are predominantly positive (below > above FRC), with tight same-sign distributions in the strong cells.
Young females are broader and noisier (small n, weakest coupling).""",

 "e7914cac": """## 5. Cohen's d comparison across strata
Maximum-phonation tasks `a_2`, `a_3`, `a_7`, nine strata.
Grouped bars (F0 vs %RC) with Cohen's qualitative bins (small/medium/large) as dotted reference lines. The headline of `make_m2_summary_plots.py`.""",

 "7ee13e3b": """**Observations:**
Maximum-phonation tasks `a_2`, `a_3`, `a_7`, nine strata.
`%RC` (d > 1, large) towers over f0 (d ≈ 0.6, medium) in every stratum.
The FRC crossing is marked far more by which compartment drives the breath than by a pitch change.
Both are weakest in young females.""",

 "67a06fa6": """## 6. Orthogonality scatter: F0 shift vs %RC shift
Maximum-phonation tasks `a_2`, `a_3`, `a_7`; one point per segment.
Each point is one sustained-phonation segment, coloured by demographic; the X markers are group medians.""",

 "1e83b89b": """**Observations:**
Maximum-phonation tasks `a_2`, `a_3`, `a_7`.
F0-shift and %RC-shift are largely orthogonal: no strong diagonal.
The demographic medians (X) separate along different axes (F0 with age, %RC with sex).
They carry independent respiratory information, so both are worth keeping.""",

 "cb0f1ee9": """## Recap
Standardized effect sizes turn the Level 3 "is the shift real?" question into "how big, and for whom?".
To regenerate the full Excel summary, PDF forest/histogram set, and the summary plots:

```bash
python scripts/analyze_l3_stratified.py --paired-dir data_target/healthy_subjects/paired     --metadata data_root/healthy_subjects/subjects_metadata.csv --output-dir results/M2_stratified
python scripts/make_m2_summary_plots.py --results-dir results/M2_stratified
```

Notebook n°5 and notebook n°6 move to Milestone 3 (M3): trying to *predict* the respiratory state from audio.""",

 "a1e9517b": """## Stratified cross-domain correlations: which feature and task couple?

Beyond the FRC effect sizes, we ask **which acoustic feature and which task give a consistent audio↔physiology coupling**, per demographic stratum. For every recording we take the within-segment Pearson *r* between each acoustic feature (f0, energy, spectral centroid) and each OEP feature (Δvcw, flow, %RC, %AB) over voiced frames, then aggregate by task × stratum (`scripts/analyze_stratified_crossdomain.py`, written to `M2_crossdomain/`).

**Headline.**
In steady phonation the coupling is modest: energy↔volume ≈ +0.24 (strongest steady coupling, a touch higher in elderly males), spectral-centroid↔volume ≈ +0.15, and acoustic↔flow ≈ 0.
The one strong, universal coupling is f0 ↔ chest-wall volume in the vocal glide `a_7` (r ≈ −0.88, 100% sign-consistent in every stratum), which is active pitch modulation.
Take-away: voice tracks lung volume (position), not flow (rate); strong couplings appear only when the task deliberately modulates the feature.""",

 "d51e143c": """### One example on all three domains: the genuine FRC crossing (glide `a_7`)
Task `a_7` (vocal glide), one example subject.
With the baseline-referenced FRC, 67 of 86 sustained segments now cross the true resting volume (only 19 fall back to the midpoint proxy). The honest effect sizes at the genuine crossing: f0-shift is medium (d=0.60 overall, stronger in elderly males at d=0.75; f0 rises below FRC), while `%RC` is the large-effect marker (d=1.17, up to 1.46 in young males, 100% sign-consistent).
The figure below shows one glide: chest-wall volume crossing FRC (column 1, dashed line = resting FRC level), the loudness envelope (column 2), and F0 sweeping with the harmonics (column 3).""",
}

NB04_GLOBAL = [
 ("L3 FRC: {feature} — stratified", "L3 FRC: {feature} · stratified"),
 ("(n={len(s)} — insufficient)", "(n={len(s)}, insufficient)"),
 ("L3 FRC — {feature} shift", "L3 FRC: {feature} shift"),
 ("OEP — volume crosses", "OEP: volume crosses"),
 ("Audio — loudness", "Audio: loudness"),
 ("Audio — frequency", "Audio: frequency"),
 ("Cross-domain mean r (All subjects) — task", "Cross-domain mean r (All subjects): task"),
 ("fig.suptitle('Glide /a_7/ (GaRo): F0 sweeps as chest-wall volume crosses true FRC and rib-cage % rises',y=1.03,fontsize=11)",
  "fig.suptitle(f'Glide /a_7/ ({anon(meta[\"subject_id\"])}): F0 sweeps as chest-wall volume crosses true FRC and rib-cage % rises',y=1.03,fontsize=11)"),
]

# ---------------------------------------------------------------- notebook 05
NB05_MD = {
 "c9da5433": """# Notebook n°5: Milestone 3 (M3) Can Audio Predict the Compartmental Contribution (%RC)?

**Pipeline stage:** Milestone 3 (M3), moving from *correlation* (Milestone 2) to *prediction*. The central question: can frame-level audio features predict the rib-cage volume contribution `pct_rc`, and is that mapping consistent within a subject but not across the population?

**Scripts reproduced**
| Script | Role |
|--------|------|
| `scripts/analyze_compartmental_regression.py` | per-subject vs pooled vs stratified Ridge regression, binning ANOVA |
| `scripts/diagnose_compartmental_signal.py` | variance decomposition, within-task tracking, segment-level reframing |

> **Plot types:** per-subject R² histogram, coefficient-importance bar, R² comparison bar, binning feature-profile line plot, variance-decomposition histogram, framing-comparison bar.
>
> ⚠️ Requires scikit-learn. This notebook fits many cross-validated models over the full corpus, so it takes a couple of minutes.""",

 "36630d9f": """**Summary.**
Frame-level acoustic features do **not** linearly predict compartmental volumes: the per-subject median cross-validated R² is about −0.44 (negative means worse than predicting the mean; only ~3% of subjects reach R² > 0).
This is a real result, not a pipeline artifact: it held after the OEP rate fix.
Acoustics and respiratory kinematics are coupled but not frame-wise predictive; the relationship is task- and timing-dependent, not a fixed instantaneous mapping.""",

 "79b389bc": """## 1. Build the frame-level dataset
All 550 recordings, voiced frames only (193,629 frames, 38 subjects).
`cr.build_frame_dataset` loads every recording, keeps voiced frames, and attaches the audio predictors (`f0`, `energy`, `spectral_centroid`, `mfcc_0..12`) plus the target and demographics.""",

 "1ebdbf02": """## 2. Per-subject regression: the make-or-break test
All 550 recordings, per subject.
For each subject, `RidgeCV(audio → %RC)` with leave-one-task-out cross-validation. If most subjects show positive R², the mapping exists at the subject level.""",

 "f8f8455c": """**Observations:**
All 550 recordings, per subject.
The per-subject CV R² distribution sits almost entirely below zero (median ≈ −0.44; only ~3% positive).
Even subject-specific models cannot predict frame-level %RC from acoustics better than the subject's own mean.
There is no stable instantaneous mapping.""",

 "0aef1523": """## 3. Which audio features carry %RC information?
All 550 recordings, per-subject Ridge coefficients.
Mean |standardized Ridge coefficient| across subjects; green bars are sign-consistent across ≥70% of subjects.""",

 "b5eafa5f": """**Observations:**
All 550 recordings.
No single audio feature dominates, and coefficients are inconsistent in sign across subjects: a feature that helps one hurts another.
There is no universal acoustic-to-%RC predictor.""",

 "37837674": """## 4. Per-subject vs pooled vs stratified
All 550 recordings.
The negative control: one pooled model (leave-one-subject-out) should do *worse* than per-subject models if the mapping is subject-specific. A large positive gap would be the evidence.""",

 "be3aa658": """**Observations:**
All 550 recordings.
Pooled and stratified R² are also negative (pooled ≈ −0.10); stratifying by demographic cell does not rescue it.
The failure is structural, not a sample-size problem.""",

 "95f532a2": """## 5. Binning EDA (model-free)
All 550 recordings, voiced frames.
Bin %RC into quantile bands and ask, per feature, whether the feature mean differs across bands (one-way ANOVA, η² effect size). Then plot the standardized feature profiles of the top features.""",

 "86b185ed": """**Observations:**
All 550 recordings (model-free).
Binning the target against each feature shows flat-to-noisy trends; the ANOVA captures little structure.
No hidden non-linear relationship is being missed.""",

 "605d8d4c": """## 6. Diagnostic: why does cross-task prediction fail?
All 550 recordings, per subject.
`diagnose_compartmental_signal` decomposes %RC variance into within-recording vs between-recording. If %RC is mostly *between* tasks, frame-level within-task prediction is ill-posed.""",

 "f0022487": """**Observations:**
All 550 recordings.
Most %RC variance is between-task, not within-task: a cross-task model largely learns task identity, which does not transfer.
Within a task the predictable signal is tiny.""",

 "aa0e5e97": """### 6b. Framing comparison
All 550 recordings, three framings.
Three ways to pose the prediction problem. Within-task (blocked CV) and segment-level (one row per recording) are better posed than the failed cross-task framing.""",

 "906fd21e": """**Observations:**
All 550 recordings.
Across all framings the best CV R² is still about 0: frame-level acoustics do not predict compartmental volume.
The coupling seen in notebook n°3 and notebook n°4 is real but not predictive.""",

 "aa9d8b21": """## Recap
Continuous %RC is **not** linearly recoverable from audio at the frame level in any framing: a genuine, well-characterised negative result. That motivates notebook n°6, which reframes the target as a binary **FRC-state** classification (above vs below FRC), which *is* accessible.

Full scripts:
```bash
python scripts/analyze_compartmental_regression.py --paired-dir data_target/healthy_subjects/paired     --metadata data_root/healthy_subjects/subjects_metadata.csv --output-dir results/M3_compartmental
python scripts/diagnose_compartmental_signal.py     --paired-dir data_target/healthy_subjects/paired     --metadata data_root/healthy_subjects/subjects_metadata.csv --output-dir results/M3_compartmental
```""",
}

# ---------------------------------------------------------------- notebook 06
NB06_MD = {
 "1f95653b": """# Notebook n°6: Milestone 3 (M3) FRC-State Classification from Audio + Resolution Sweep

**Pipeline stage:** the productive reframing after the %RC-regression negative result (notebook n°5). Instead of predicting *continuous* %RC, we ask a better-posed binary question: is a phonation frame **above** or **below** the FRC crossing? Level 3 (notebook n°4) showed F0 is reliably elevated below FRC, so FRC state should be acoustically accessible.

**Scripts reproduced**
| Script | Role |
|--------|------|
| `scripts/analyze_frc_classification.py` | pooled / F0-only / per-subject / stratified classification, feature importance |
| `scripts/analyze_frc_window_sweep.py` | AUC vs temporal aggregation window (resolution↔accuracy trade-off) |

> **Plot types:** per-subject AUC histogram, signed feature-importance bar, AUC-comparison bar, resolution-sweep curve (log-x with reference lines).
>
> ⚠️ Requires scikit-learn.""",

 "934000fe": """**Summary.**
Classifying above- vs below-FRC state from acoustics is modestly above chance: pooled AUC ≈ 0.66 (F0 alone ≈ 0.62), per-subject median ≈ 0.53 (near chance).
It is strongest in the elderly strata, where the rib-cage compartment is most engaged (consistent with the large `%RC` FRC-crossing effect in notebook n°4).
So FRC state leaves a weak but real, F0-dominated acoustic signature, most detectable in older speakers.""",

 "8581d27f": """## 1. Build the labelled, subject-standardized frame dataset
Maximum-phonation tasks `a_2`, `a_3`, `a_7` (37,120 voiced frames, 36 subjects).
Each voiced frame of a sustained recording is labelled `below_frc` (1) or `above_frc` (0) at the FRC crossing; features are z-scored within subject (removing baseline differences, isolating the within-subject FRC signal).""",

 "3a98e6f0": """## 2. Classification across framings
Maximum-phonation tasks `a_2`, `a_3`, `a_7`.
ROC-AUC under leave-one-subject-out CV: pooled (all features), pooled (F0 only, to test whether F0 is the carrier), and per-subject (leave-one-recording-out).""",

 "b14b6478": """### 2a. Per-subject AUC distribution
Maximum-phonation tasks `a_2`, `a_3`, `a_7`, per subject.
Leave-one-recording-out AUC within each subject; the dotted line is chance (0.5).""",

 "5fe38d04": """**Observations:**
Maximum-phonation tasks `a_2`, `a_3`, `a_7`, per subject.
Per-subject FRC-state AUC clusters around 0.53 (near chance) with a long tail.
Individually, audio barely separates above-FRC from below-FRC frames.""",

 "7b94c393": """### 2b. Feature importance
Maximum-phonation tasks `a_2`, `a_3`, `a_7`, pooled.
Pooled logistic-regression coefficients on subject-standardized features. A positive value pushes toward "below FRC"; F0 is expected to dominate (consistent with Level 3).""",

 "ccc9959c": """**Observations:**
Maximum-phonation tasks `a_2`, `a_3`, `a_7`, pooled.
F0 dominates the classifier (F0-only AUC ≈ full model); other features add little.
The weak FRC signature is essentially a pitch effect (F0 rises below FRC).""",

 "0b439a2f": """### 2c. AUC comparison across framings
Maximum-phonation tasks `a_2`, `a_3`, `a_7`.""",

 "e207712f": """**Observations:**
Maximum-phonation tasks `a_2`, `a_3`, `a_7`.
Pooled AUC ≈ 0.66 beats per-subject (0.53) because pooling exploits between-subject F0 and FRC structure.
The honest within-subject signal is weaker.""",

 "e3eef383": """## 3. Resolution vs accuracy sweep
Maximum-phonation tasks `a_2`, `a_3`, `a_7` (73 recordings).
Frame-level AUC (~0.65) and the recording-level sign-consistency ceiling (~0.84) are two points; this sweep fills in the curve by classifying FRC state at a range of temporal aggregation windows.""",

 "107752be": """**Observations:**
Maximum-phonation tasks `a_2`, `a_3`, `a_7`.
AUC is flat across window sizes (~0.62 to 0.65) but jumps to ~0.72 for the whole-half split.
Coarser, breath-scale framing carries more FRC information than any single frame.
FRC state is a slow, breath-level property.""",

 "763a05e8": """## Recap
FRC state **is** decodable from voice (modestly per frame, strongly once aggregated), unlike continuous %RC. The sweep gives a single defensible resolution-vs-accuracy curve.

Full scripts:
```bash
python scripts/analyze_frc_classification.py --paired-dir data_target/healthy_subjects/paired     --metadata data_root/healthy_subjects/subjects_metadata.csv --output-dir results/M3_frc_classification
python scripts/analyze_frc_window_sweep.py   --paired-dir data_target/healthy_subjects/paired     --metadata data_root/healthy_subjects/subjects_metadata.csv --output-dir results/M3_frc_classification
```""",
}

# ---------------------------------------------------------------- notebook 07
NB07_MD = {
 "5f960571": """# Notebook n°7: Acoustic Features, Segmentation & Task Analyzers

**Pipeline stage:** the audio-only, per-task analysis layer that feeds the legacy Zocco pipeline and the Milestone 1 (M1) extractor: `audio_processing`, `acoustic_features` (Praat), `segmentation`, `task_analyzers`, and the full `visualization.Visualizer`.

**Scripts / modules reproduced**
| Script / module | Role |
|-----------------|------|
| `scripts/analyze_single_subject.py` | per-task Praat metrics + waveform figure |
| `scripts/analyze_trill_modulation.py` | trill modulation-frequency analysis |
| `pneumophonic_analysis/visualization.py` | every `Visualizer` plot method |
| `pneumophonic_analysis/segmentation.py` | FRC / glide / modulation segmenters |
| `pneumophonic_analysis/task_analyzers.py` | Vowel / Trill / Glide analyzers |

> **Note on data.** The raw `.wav` recordings are not committed to the repo, so this notebook builds **synthetic but physiologically plausible** signals (sustained vowel, vibrato vowel, vocal glide, alveolar trill) to exercise every plot. On a real dataset the identical calls work via `DataLoader(subject_folder).load_audio("a.wav")`.
>
> **Plot types:** waveform; audio + sync (falling edge); STFT spectrogram; mel-spectrogram; F0 trace; chest-wall volume with FRC zones; FRC segment (2-panel); glide analysis (3-panel novelty); trill RMS envelope + modulation spectrum; stacked task-result; metrics-comparison bar.""",

 "d618f6ad": """## 1. Waveform
Synthetic sustained /a/ at 120 Hz.
`Visualizer.plot_waveform` on the synthetic vowel.""",

 "5b27343b": """**Observations:**
Synthetic sustained /a/ at 120 Hz.
A periodic vowel with a slow amplitude envelope and 5 Hz vibrato.
It is used to exercise the extractors deterministically, since raw subject audio is not committed.""",

 "3255171d": """## 2. Audio with synchronization pulse
Synthetic /a/ plus a 1-second sync pulse.
`Visualizer.plot_audio_with_sync` shows the audio above the sync pulse and marks its falling edge, the event used to align audio with the OEP system (see `sync.py`).""",

 "d0933f1f": """**Observations:**
Synthetic /a/ plus a 1-second sync pulse.
The 1 s rectangular pulse (amplitude 1.0) is the marker that aligns audio with OEP.
Its sharp edges are what the falling-edge detector locks onto.""",

 "ac70877c": """## 3. Spectrogram & mel-spectrogram
Synthetic sustained /a/.
`Visualizer.plot_spectrogram` (log-frequency STFT) and `plot_mel_spectrogram`.""",

 "b440fa2b": """**Observations:**
Synthetic sustained /a/.
The STFT (first figure) resolves individual harmonics; the mel (second figure) compresses them into the perceptual envelope.
The vibrato shows as a gentle wobble of the harmonic bands.""",

 "6f0e101d": """## 4. F0 trace (pYIN)
Synthetic sustained /a/.
Estimate F0 with `AudioProcessor.estimate_f0` (pYIN) and plot it with `Visualizer.plot_f0_trace`, which overlays the mean ± 1 SD band.""",

 "cb81dc2b": """**Observations:**
Synthetic sustained /a/.
pYIN tracks F0 through the voiced vowel and returns NaN in unvoiced frames (the gaps).
The vibrato ripple around 120 Hz is recovered.""",

 "146f369e": """## 5. Praat acoustic metrics
Synthetic sustained /a/.
`PraatAnalyzer.analyze_signal` returns the full clinical voice profile (pitch, jitter, shimmer, HNR, DSI, formants), the engine behind `analyze_single_subject.py`.""",

 "41e8cbc4": """## 6. Chest-wall volume with FRC zones
Synthetic deflating volume trace.
Plotted with `Visualizer.plot_oep_volume`: above-FRC (elastic recoil) vs below-FRC (active expiration) zones plus the crossing line.""",

 "57433645": """**Observations:**
Synthetic deflating volume trace.
The demo volume is shaded into above-FRC (inspiratory reserve) and below-FRC zones.
The marker is where Vcw passes the resting level.""",

 "c54c7509": """## 7. FRC segmentation of the audio
Synthetic /a/, split at a 2.3 s crossing.
`FRCSegmenter.segment_by_time` splits a phonation into above/below-FRC parts; `Visualizer.plot_frc_segment` shows the full segment with zones (top) and the two parts separated (bottom).""",

 "40a45fde": """**Observations:**
Synthetic /a/, split at the FRC crossing.
The FRC segmenter splits the audio at the volume crossing into above-FRC and below-FRC spans.
This is the segmentation the Level 3 analysis (notebook n°4) uses for above-vs-below feature shifts.""",

 "7065e672": """## 8. Vocal glide: novelty-based P1/P2 segmentation
Synthetic glide, F0 sweeping 110 to 320 Hz.
`GlideSegmenter` builds a spectral-centroid novelty function and splits the glide at its peak; `Visualizer.plot_glide_analysis` shows waveform, spectrogram and novelty (3 panels).""",

 "c86edf97": """**Observations:**
Synthetic glide, F0 sweeping 110 to 320 Hz.
Novelty-based segmentation finds the P1 and P2 phases of the pitch sweep.
For a real `a_7` this is where F0 and the harmonics ramp through their range.""",

 "3aa45fef": """## 9. Alveolar trill: modulation-frequency analysis
Synthetic trill, envelope modulated at 26 Hz.
The trill modulates the acoustic envelope at the tongue-tip rate (~20 to 30 Hz). We reproduce `analyze_trill_modulation.py`: RMS envelope plus FFT of the detrended envelope, with the modulation band shaded. `ModulationAnalyzer.compute_modulation_frequency` returns the peak.""",

 "b2abd2f2": """**Observations:**
Synthetic trill, envelope modulated at 26 Hz.
The amplitude-modulation spectrum peaks at the trill rate (~20 to 30 Hz tongue oscillation).
This is the signature distinguishing a true alveolar trill (relevant to the low `voiced%` of `/r/`).""",

 "89ed354f": """## 10. Task analyzers: a complete `TaskResult`
Synthetic vibrato /a/ at 130 Hz.
`VowelAnalyzer.analyze` runs preprocessing, Praat and spectral features and returns a `TaskResult`. `Visualizer.plot_task_result` renders the stacked waveform, spectrogram and F0 summary figure.""",

 "025086a8": """**Observations:**
Synthetic vibrato /a/ at 130 Hz.
A complete `TaskResult`: F0, formants and metrics end-to-end for one task.
This is the object the batch task-analyzers emit.""",

 "86ef267e": """## 11. Cross-subject metrics comparison
Five synthetic "subjects" (mean F0 110 to 240 Hz).
`Visualizer.plot_metrics_comparison` draws a ranked horizontal bar chart, here mean F0, the figure `analyze_batch.py` / `pipeline.generate_report` produces.""",

 "3a9e8833": """**Observations:**
Five synthetic "subjects".
The cross-subject metrics table demonstrates the analyzers' consistent schema.
The absolute values are synthetic here.""",

 "4ae861fe": """## Recap
Every `Visualizer` plot method and the Praat, segmentation and task-analyzer modules are exercised here. On real recordings, replace the `synth_*` calls with `DataLoader(subject_folder).load_audio("a.wav")`, for example:

```bash
python scripts/analyze_single_subject.py data_root/healthy_subjects/<YYYYMMDD_SubjectID> -o results/<SubjectID>
python scripts/analyze_trill_modulation.py path/to/r.wav --frc-time 3.5
```""",
}

REVISIONS = {
 "00_overview_and_config.ipynb": dict(
     helper_cell="be7dd44a", md=NB00_MD, glob=NB00_GLOBAL),
 "01_M1_paired_feature_extraction.ipynb": dict(
     helper_cell="03860e88", md=NB01_MD, glob=NB01_GLOBAL),
 "02_paired_feature_visualization.ipynb": dict(
     helper_cell="d30827cc", md=NB02_MD, glob=NB02_GLOBAL),
 "03_M2_correlation_analysis.ipynb": dict(
     helper_cell="a648b76b", md=NB03_MD, glob=NB03_GLOBAL),
 "04_M2_L3_stratified_effect_sizes.ipynb": dict(
     helper_cell="5106f1e6", md=NB04_MD, glob=NB04_GLOBAL),
 "05_M3_compartmental_regression.ipynb": dict(md=NB05_MD),
 "06_M3_frc_classification.ipynb": dict(md=NB06_MD),
 "07_acoustic_features_and_tasks.ipynb": dict(md=NB07_MD),
}


def apply(nb_name, spec):
    path = NB_DIR / nb_name
    nb = nbformat.read(path, as_version=4)
    glob = spec.get("glob", [])
    md = spec.get("md", {})
    helper_cell = spec.get("helper_cell")
    n_glob = n_md = 0
    for cell in nb.cells:
        # global string replacements (titles, nb refs) on every cell
        for old, new in glob:
            if old in cell.source:
                cell.source = cell.source.replace(old, new)
                n_glob += 1
        # append anon() helper to the setup cell once
        cid = cell.get("id", "")
        if helper_cell and (cid == helper_cell or cid.startswith(helper_cell + "-")) \
                and "def anon" not in cell.source:
            cell.source = cell.source.rstrip() + "\n" + HELPER
    # absolute markdown replacements (after globals so they win).
    # cell ids may be short (8-char) or full UUIDs, so match by prefix too.
    for cell in nb.cells:
        cid = cell.get("id", "")
        key = next((k for k in md if cid == k or cid.startswith(k + "-")), None)
        if key is not None:
            text = md[key]
            # skip hard-breaks on structural cells (tables / fenced code) so we
            # don't inject backslashes into table rows or bash blocks.
            skip_hb = ("```" in text) or any(
                ln.lstrip().startswith("|") for ln in text.split("\n"))
            cell.source = text if skip_hb else hb(text)
            n_md += 1
    nbformat.write(nb, path)
    print(f"{nb_name}: {n_md} markdown cells set, {n_glob} global replacements")


if __name__ == "__main__":
    targets = sys.argv[1:] or list(REVISIONS)
    for nb_name in targets:
        apply(nb_name, REVISIONS[nb_name])
