# Operatic Singer Cohort — Measurement Protocol

**Study:** OEP / Audio correlations in professional lyrical / operatic singers  
**Pre-cohort subjects:** 2 (protocol validation; full cohort to follow)  
**Target cohort balance:** fair sex / age disparity; professional singers only

---

## Voice Types & Tessitura

| Voice type | Approximate tessitura |
|---|---|
| Soprano | C4 – C6 |
| Mezzo-Soprano | A3 – A5 |
| Tenore | C3 – C5 |
| Baritone | G2 – G4 |

All scale note references below are given for **Mezzo-Soprano**; they are transposed accordingly for each voice type.

---

## Task Protocol

### 0. Preliminary — Vital Capacity
Spirometric vital capacity measurement (retained from Zocco protocol). Single measure before vocal tasks begin.

---

### 1. Sustained Vowel (FRC-crossing)
- Sustained /a/ on a **subject-specific reference note** (defined from singer's features)
- Long enough to produce at least a FRC crossing
- Purpose: FRC-alignment analysis; direct carryover of M2 methodology
- lyrical and non lyrical voicing takes ( **Voce girata** vs **Voce non-girata**).
---

### 2. Five-Note Scale Progressions (8 takes)
Ascending + descending 5-note pattern; each take raised by one **semitone**:

| Take | Notes (Mezzo-Soprano) |
|---|---|
| 1 | C5 – D5 – E5 – F5 – G5 – F5 – E5 – D5 – C5 |
| 2 | C#5 – D#5 – F5 – G5 – G#5 – G5 – F5 – D#5 – C#5 |
| 3 | D5 – E5 – F#5 – G5 – A5 – G5 – F#5 – E5 – D5 |
| 4 | D#5 – F5 – G5 – G#5 – A#5 – G#5 – G5 – F5 – D#5 |
| 5 | E5 – F#5 – G#5 – A5 – B5 – A5 – G#5 – F#5 – E5 |
| 6 | F5 – G5 – A5 – A#5 – C6 – A#5 – A5 – G5 – F5 |
| 7 | F#5 – G#5 – A#5 – B5 – C#6 – B5 – A#5 – G#5 – F#5 |
| 8 | G5 – A5 – B5 – C6 – D6 – C6 – B5 – A5 – G5 |

---

### 3. Three-Note Scale Progressions (Thirds)
Ascending triads by major third + fifth; starting pitch steps up by one tone per take:

**Pattern:** Root – Major 3rd – 5th (e.g. C5 – E5 – G5)  
**Range (Mezzo-Soprano):** C5 → G6 (i.e. until G6 – B6 – D7, approximately)

Each take is the same rhythmic pattern; only the starting pitch changes.

---

### 4. Ascending + Descending Scale (Step-wise with thirds)
A single combined take: ascending run followed by its mirror descending run.

**Ascending (Mezzo-Soprano example):**  
C5 – E5 – D5 – F5 – E5 – G5 – F5 – A5 – G5 – B5 – A5 – C6 – B5 – D6

**Descending:** same pattern in reverse.

---

### 5. Sung Strofa — *Scarborough Fair* (2 takes)
The **first strofa** of *Scarborough Fair*, sung twice:

| Take | Phonation mode |
|---|---|
| `scarb_girata` | **Voce girata** (covered, supported operatic production — *appoggio*) |
| `scarb_non_girata` | **Voce non-girata** (straight, uncovered reference) |

 **Voce girata** vs **Voce non-girata**

---

### 6. Aria Fragments (3 takes each, sung lyrically)
Three repetitions of the assigned excerpt, sung in **lyrical** style only.

| Voice type | Opera | Aria |
|---|---|---|
| Mezzo-Soprano | Bizet, *Carmen* | *Seguidilla* ("Près des remparts de Séville") |
| Soprano | Verdi, *Aida* | *Ritorna vincitor* |
| Tenore | Leoncavallo, *Pagliacci* | *Ridi, Pagliaccio* (Vesti la giubba) |
| Baritone | Verdi, *Don Carlo* | *Dio, che nell'alma infondere* |

---

## Task Label Summary

| Label | Description | Takes |
|---|---|---|
| `vc` | Vital capacity | 1 |
| `vowel_lyr` | Sustained vowel in operatic voicing | 1 |
| `vowel_nolyr` | Sustained vowel in normal voicing | 1 |
| `scale5_t{1..8}` | 5-note scale, semitone-shifted takes | 8 |
| `scale3_t{n}` | 3-note thirds progression, starting pitch n | variable |
| `scale_asc_desc` | Full ascending + descending scale | 1 |
| `scarb_girata` | Scarborough Fair — voce girata | 1 |
| `scarb_non_girata` | Scarborough Fair — voce non-girata | 1 |
| `aria_t{1,2,3}` | Aria excerpt — lyrical | 3 |

---

## Protocol Status

> Tasks above reflect the **pre-cohort validation protocol** (2 subjects).  
> Final task set will be confirmed Thursday based on literature review.  
> Modifications may include: adding/removing scale takes, adjusting aria selection, or refining the voce girata / non-girata contrast methodology.
