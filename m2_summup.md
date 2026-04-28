# M2 — Exploratory Correlation Analysis: Key Findings

**Study**: Respiratory-Acoustic Coupling during Phonation
**Data**: 286 segments, 20 healthy subjects, 15 vocal tasks
**Method**: Time-aligned OEP (chest wall kinematics) + audio features at ~66 fps

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

Sustained phonation tasks (a_2, a_3, a_7) were split at the point where chest wall volume returns to its starting level (FRC crossing). Audio and respiratory features were compared above vs below FRC across 38 segments.

### Results

| Feature | Median shift (Below − Above) | Wilcoxon p | Direction |
|---|---|---|---|
| **F0** | **+2.7 Hz** | **5.4 × 10⁻⁵** | Pitch rises below FRC |
| **%RC** | **+0.35%** | **4.2 × 10⁻⁸** | Rib cage contribution increases |
| Energy | −0.003 | 0.93 | No significant change |
| Flow CW | −0.012 L/s | 0.24 | No significant change |

### Interpretation

**F0 rises below FRC** (p = 5.4 × 10⁻⁵): when the subject crosses into expiratory reserve volume, active muscular recruitment increases laryngeal tension, causing a small but highly significant upward pitch shift. This confirms Zocco's thesis finding with time-aligned frame-level data.

**%RC increases below FRC** (p = 4.2 × 10⁻⁸): this is the most statistically significant result. Below FRC, the diaphragm can no longer generate passive recoil, so the intercostal muscles (rib cage) take over. The paired scatter plot shows virtually all subjects above the identity line — this is a universal, consistent effect.

**Energy does NOT change** (p = 0.93): the subject successfully compensates for the loss of passive recoil, maintaining the same vocal intensity despite increased respiratory effort. The respiratory state changes are "hidden" in intensity but "leak through" in pitch.

This dissociation — respiratory effort changes acoustically detectable in F0 but not in energy — has direct implications for predictive modeling: **pitch carries respiratory information that intensity does not**.

> **Plot reference**: `frc_shifts.pdf`, `frc_paired_scatter.pdf`

---

## Implications for Predictive Modeling (M3/M4)

These results inform feature selection for the next milestones:

**Audio → Respiratory prediction**: F0 alone is weakly predictive (r ≈ 0 within segments), but the FRC analysis demonstrates it carries real respiratory-state information. A model using the *combination* of F0 trajectory, energy dynamics, and spectral features over time — especially with sequence architectures (LSTM, 1D-CNN) — should capture the slow respiratory trends.

**Respiratory → Audio prediction**: compartmental strategy (%RC) and flow are the most informative respiratory features. The consistent %RC shift at FRC crossing suggests that respiratory state can predict F0 trajectory changes.

**Task-type distinction**: sustained vowels and connected speech have fundamentally different coupling patterns. Models should either be trained separately per task type or include task category as a feature.

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