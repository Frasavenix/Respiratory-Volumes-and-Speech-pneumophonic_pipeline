# M2 — Exploratory Correlation Analysis: Key Findings

**Study**: Respiratory-Acoustic Coupling during Phonation
**Data**: 550 recordings, 39 healthy subjects, 15 vocal tasks
**Method**: Time-aligned OEP (chest wall kinematics) + audio features at ~66 fps

> **Note (current state).** Section 3 (FRC) and the Implications below reflect the corrected analysis: the OEP sample-rate fix, the baseline-derived genuine FRC, and the reclassification of `a_7` (A-GLIDE) as a *glissando* excluded from sustained phonation. Sections 1–2 retain numbers from the earlier 20-subject snapshot; the current verified cross-domain numbers live in notebooks n°3 and n°4.

---

## 1. Global Cross-Domain Correlations

### Strongest finding: Energy ↔ Volume Excursion

Across all subjects and tasks, mean vocal energy is negatively correlated with the total volume excursion of the chest wall (**r = −0.49, p ≈ 10⁻¹⁷**).

Interpretation: subjects who exhale more deeply (longer sustained phonations) produce lower average energy — they spread their air supply over a longer duration. Sustained tasks cluster in the high-volume / low-energy region; connected speech tasks occupy the opposite quadrant.

> **Plot reference**: `global_scatter_plots.pdf` , `global_correlation_matrix.pdf`

### Compartmental strategy variability scales with volume use

The variability of rib cage contribution (%RC standard deviation) is strongly correlated with volume excursion (**r = 0.76**). When subjects use more of their lung capacity, they shift between rib-cage-dominant and abdomen-dominant strategies during phonation. This is consistent with the known physiological adaptation near FRC.

### Speech vs Sustained: different coupling regimes

| Correlation pair | All tasks | Sustained only | Speech only |
|---|---|---|---|
| Energy ↔ ΔVcw range | −0.49 | −0.30 | **−0.65** |
| Energy ↔ %RC std | −0.44 | −0.23 | **−0.59** |
| Flow std ↔ %RC std | 0.20 | 0.11 | **0.67** |

Connected speech shows systematically stronger respiratory-acoustic coupling than sustained vowels. During speech, the subject actively modulates intensity (stressed vs unstressed syllables), and this modulation is more tightly linked to respiratory dynamics than the steady-state phonation of a sustained vowel.

This suggests that **speech tasks are more informative than sustained vowels for studying respiratory-acoustic coupling** — a counterintuitive finding since sustained vowels are traditionally considered the "cleaner" signal.

> **Plot reference**: `global_correlation_sustained.pdf`, `global_correlation_speech.pdf`

---

## 2. Within-Segment Correlations

### Frame-level Energy ↔ Volume: weak but positive

| Pair | Median r | Mean r | Range | n |
|---|---|---|---|---|
| Energy ↔ ΔVcw | **0.187** | 0.141 ± 0.357 | [−0.89, 0.88] | 247 |
| F0 ↔ Flow CW | −0.004 | −0.010 ± 0.280 | [−0.80, 0.83] | 247 |
| Energy ↔ Flow CW | −0.041 | −0.036 ± 0.213 | [−0.85, 0.55] | 247 |

F0 and flow are essentially uncorrelated within recordings (median r ≈ 0). This is physiologically expected: healthy subjects maintain stable pitch despite declining lung volume — that's active laryngeal control, not passive coupling.

The wide spread in Energy ↔ ΔVcw (from −0.89 to +0.88) reveals large inter-subject variability. Subjects with strong negative correlations fail to maintain intensity as volume depletes — potentially a marker of poorer phonatory control.

> **Plot reference**: `within_segment_correlation_distributions.pdf`

### Time-resolved sliding correlation

Using a 0.5-second sliding window on sustained tasks, the **median correlation between energy and volume is 0.014** — essentially zero on short timescales.

This dissociation between macro-level coupling (across subjects: r = −0.49) and micro-level independence (within recordings: r ≈ 0) reflects the subject's active regulation of subglottal pressure. Volume declines steadily, but energy remains relatively flat until the respiratory reserve is exhausted.

> **Plot reference**: `time_resolved_aggregate.pdf`

---

## 3. FRC Crossing Analysis

Genuine sustained phonation (`a_2` = A-LONG, `a_3` = A-SOFT) was split at the FRC crossing (chest-wall volume returning to the baseline-derived resting level), and audio/respiratory features compared above vs below FRC across 58 segments (51 with a genuine crossing). The vocal glide `a_7` (A-GLIDE) is excluded: its deliberate pitch sweep confounds any F0 "shift".

### Results (standardized, Cohen's d)

| Feature | Cohen's d (All) | Direction |
|---|---|---|
| **%RC** | **1.28 (large)** | Rib-cage contribution increases below FRC |
| F0 | 0.15 (negligible) | No systematic pitch shift in sustained phonation |
| Energy | ≈ 0 | No change |
| Flow CW | ≈ 0 | No change |

### Interpretation

**%RC increases below FRC** (d = 1.28, sign-consistency 91%, large in every stratum): the headline result. Below FRC the diaphragm can no longer generate passive recoil, so the rib-cage (intercostal) muscles take over. Virtually all subjects sit on the same side of the identity line: a universal, consistent effect.

**F0 does NOT systematically shift** (d = 0.15, negligible): once the A-GLIDE glissando is excluded, the apparent pitch shift collapses. The earlier "F0 rises below FRC" reading was an artifact of pooling the glissando, where the subject sweeps pitch on purpose. The one exception is elderly males (d ≈ 0.90), who do raise F0 below FRC, plausibly reduced elastic recoil forcing laryngeal compensation.

**The FRC dissociation is compartmental, not pitch-based.** Respiratory state below FRC is marked by *which compartment drives the breath* (%RC), not by pitch. The strong f0↔volume coupling (r ≈ −0.87) appears only in the glissando `a_7`, where pitch is modulated volitionally.

> **Plot reference**: `frc_shifts.pdf`, `frc_paired_scatter.pdf` (and notebook n°4)

---

## Implications for Predictive Modeling (M3/M4)

These results inform feature selection for the next milestones:

**Audio → Respiratory prediction**: continuous %RC is NOT recoverable frame-by-frame from audio (per-subject CV R² < 0; notebook n°5). But the binary FRC state (above/below) IS decodable (notebook n°6): pooled AUC ≈ 0.70, rising to ≈ 0.96 at the breath scale. The carrier is **spectral level / loudness (MFCC-0, energy), not pitch** — F0-only AUC ≈ 0.57 (near chance). As the lung empties below FRC, sub-glottal pressure falls and loudness drops; that is the detectable signature.

**Respiratory → Audio prediction**: compartmental strategy (%RC) is the most informative respiratory feature, and its FRC shift is large and consistent (d ≈ 1.28). Pitch carries little FRC information in sustained phonation.

**Task-type distinction**: sustained vowels, the glissando (`a_7`), and connected speech have different coupling regimes. The glissando is the only task with strong f0↔volume coupling (a volitional pitch sweep) and is analysed separately.

---

## File Reference

All outputs are in `data_target/healthy_subjects/m2_correlation/`:

| File | Content |
|---|---|
| `global_summary.csv` | Summary statistics per subject × task |
| `global_correlation_matrix.pdf` | Audio ↔ OEP correlation heatmap (all tasks) |
| `global_correlation_sustained.pdf` | Heatmap — sustained tasks only |
| `global_correlation_speech.pdf` | Heatmap — connected speech only |
| `global_scatter_plots.pdf` | 4 key scatter plots with regression stats |
| `within_segment_correlation_distributions.pdf` | Per-recording correlation histograms |
| `time_resolved_aggregate.pdf` | Sliding-window correlation distribution |
| `time_resolved/*.pdf` | Per-subject time-resolved plots |
| `frc_shifts.pdf` | Above vs below FRC shift histograms |
| `frc_paired_scatter.pdf` | Paired above/below FRC comparisons |
| `frc_analysis.csv` | Raw FRC crossing data |
| `m2_report.txt` | Text summary |