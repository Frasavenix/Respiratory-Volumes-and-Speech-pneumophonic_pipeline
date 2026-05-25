"""
Singer-specific acoustic features for the pneumophonic_analysis pipeline.
========================================================================

Adds four literature-anchored acoustic features to the existing audio feature
extraction, motivated by the project's critical-reading pass:

1. Singer's-formant cluster energy ratio (Cabrera, Davis & Connolly, J. Voice
   2011) — energy in the 2-4 kHz band divided by total band energy. Captures
   the operatic projection band that is most directional in trained singers.

2. LTAS spectrum slope (Sundberg, Gu, Huang & Huang, J. Voice 2012) — linear
   regression slope of the long-term average spectrum on a log-frequency axis,
   computed in the 700-6000 Hz band, reported in dB/octave. Tracks the
   relationship between loudness and high-frequency content.

3. Vibrato rate and depth (Sundberg et al. 2012) — peak frequency and modulation
   depth extracted from the F0 contour's own spectrum after detrending.
   Vibrato is mechanically coupled to subglottal pressure oscillation, so it
   is a strong candidate for OEP-flow correlation.

4. Formant-aware band HNR (Ikuma, McWhorter, Oral & Kunduk, J. Voice 2025) —
   Harmonics-to-noise ratio computed within F1 and F2 formant bands, computed
   via Praat through Parselmouth. Avoids the spectral averaging that weakens
   whole-spectrum HNR.

Designed to operate on the audio segments already extracted by the existing
pipeline. Vectorised per-segment computation; no frame-level dependency on
audio_processing.py is required.


Author: Mateo Vitalone, thesis continuation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import linregress

# numpy 2.x renamed trapz -> trapezoid; alias for backward compatibility
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Parselmouth (Praat) is optional; we degrade gracefully if not installed.
try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    _HAS_PARSELMOUTH = True
except ImportError:  # pragma: no cover
    _HAS_PARSELMOUTH = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SingerAcousticFeatures:
    """
    Container for the four singer-oriented acoustic features.

    All values are per-segment scalars (one number per subject-task segment).
    NaN is returned when a feature cannot be computed (e.g. segment too short
    for vibrato, or pYIN failed entirely).
    """

    # Cabrera 2011 — singer's-formant cluster band
    singers_formant_ratio: float = np.nan         # E(2-4 kHz) / E(total)
    singers_formant_db: float = np.nan            # 10*log10(ratio) for log scale

    # Sundberg 2012 — LTAS slope
    ltas_slope_db_per_octave: float = np.nan      # slope, 700-6000 Hz
    ltas_slope_r2: float = np.nan                 # quality of linear fit

    # Sundberg 2012 — vibrato
    vibrato_rate_hz: float = np.nan               # F0-contour modulation freq
    vibrato_extent_cents: float = np.nan          # F0 peak-to-peak in cents
    vibrato_regularity: float = np.nan            # spectral peak prominence

    # Ikuma 2025 — formant-aware HNR
    hnr_f1_band_db: float = np.nan
    hnr_f2_band_db: float = np.nan
    hnr_full_db: float = np.nan                   # for comparison

    # Bookkeeping
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper: power spectral density on a uniform frequency grid
# ---------------------------------------------------------------------------

def _welch_psd(
    audio: np.ndarray,
    sr: int,
    nperseg: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch PSD with sensible defaults for singing-voice analysis.

    A 4096-sample window at 48 kHz gives ~12 Hz resolution, fine enough
    for formants and broad enough not to be perturbed by individual
    harmonics on slow vibrato.
    """
    if len(audio) < nperseg:
        # Fall back to FFT-bin-as-window when segment is short
        nperseg = max(256, len(audio) // 2)

    freqs, psd = sp_signal.welch(
        audio,
        fs=sr,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window='hann',
        scaling='density',
    )
    return freqs, psd


# ---------------------------------------------------------------------------
# 1. Singer's-formant cluster energy ratio (Cabrera 2011)
# ---------------------------------------------------------------------------

def compute_singers_formant_ratio(
    audio: np.ndarray,
    sr: int,
    band_lo: float = 2000.0,
    band_hi: float = 4000.0,
    total_lo: float = 80.0,
    total_hi: float = 8000.0,
) -> Tuple[float, float]:
    """
    Energy in the singer's-formant cluster band divided by total band energy.

    Cabrera, Davis & Connolly (J. Voice 2011) identify 2-4 kHz as the
    operatic projection band. The denominator is restricted to 80-8000 Hz
    rather than full Nyquist to avoid biasing by very low-frequency room
    noise or very high-frequency hiss.

    Returns
    -------
    ratio : float in (0, 1)
        Linear energy ratio.
    ratio_db : float
        10 * log10(ratio); more useful for visualisation across orders of
        magnitude.
    """
    if len(audio) < 1024:
        return np.nan, np.nan

    freqs, psd = _welch_psd(audio, sr)

    band_mask = (freqs >= band_lo) & (freqs <= band_hi)
    total_mask = (freqs >= total_lo) & (freqs <= total_hi)

    band_energy = _trapz(psd[band_mask], freqs[band_mask])
    total_energy = _trapz(psd[total_mask], freqs[total_mask])

    if total_energy <= 0:
        return np.nan, np.nan

    ratio = float(band_energy / total_energy)
    ratio_db = 10.0 * np.log10(max(ratio, 1e-12))
    return ratio, ratio_db


# ---------------------------------------------------------------------------
# 2. LTAS spectrum slope (Sundberg 2012)
# ---------------------------------------------------------------------------

def compute_ltas_slope(
    audio: np.ndarray,
    sr: int,
    lo: float = 700.0,
    hi: float = 6000.0,
) -> Tuple[float, float]:
    """
    Long-term average spectrum slope in dB/octave between `lo` and `hi`.

    Sundberg et al. (J. Voice 2012, p. 137 of issue) report that this slope
    decreases by ~0.2 dB/octave per dB of equivalent sound level in classical
    Peking opera singers. The same regression is one of the standard ways to
    quantify spectral tilt in voice analysis and should track loudness in
    your lyrical singer cohort.

    Returns
    -------
    slope : float
        dB per octave (negative under normal conditions).
    r2 : float
        Coefficient of determination of the linear fit. Use this to filter
        unreliable estimates (e.g. r2 < 0.5).
    """
    if len(audio) < 2048:
        return np.nan, np.nan

    freqs, psd = _welch_psd(audio, sr)

    mask = (freqs >= lo) & (freqs <= hi) & (psd > 0)
    if mask.sum() < 10:
        return np.nan, np.nan

    log2_freq = np.log2(freqs[mask])
    db = 10.0 * np.log10(psd[mask])

    result = linregress(log2_freq, db)
    slope_db_per_octave = float(result.slope)
    r2 = float(result.rvalue ** 2)
    return slope_db_per_octave, r2


# ---------------------------------------------------------------------------
# 3. Vibrato rate and extent (Sundberg 2012)
# ---------------------------------------------------------------------------

def compute_vibrato(
    f0_hz: np.ndarray,
    frame_rate_hz: float,
    min_rate: float = 3.0,
    max_rate: float = 8.0,
    min_voiced_fraction: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Extract vibrato rate, extent (in cents), and regularity from an F0 contour.

    The F0 contour is detrended (to remove the carrier pitch and any
    slow drift), converted to cents relative to its local mean, then the
    power spectrum of the resulting modulation signal is computed. The
    peak in the 3-8 Hz band is the vibrato rate; the peak-to-peak amplitude
    of the modulation in cents is the extent; the spectral peak prominence
    is a proxy for regularity.

    Western classical singers typically vibrate at 5-6 Hz; Peking opera
    singers at ~3.5 Hz (Sundberg 2012). The 3-8 Hz search band covers
    both, plus some slack for trill-like fast modulation.

    Parameters
    ----------
    f0_hz : np.ndarray
        F0 in Hz; NaN where unvoiced.
    frame_rate_hz : float
        Frame rate of the F0 contour (in your pipeline ~66 Hz given
        hop_length=720 at fs=48 kHz).
    min_rate, max_rate : float
        Vibrato rate search bounds in Hz.
    min_voiced_fraction : float
        If fewer than this fraction of frames are voiced, return NaN.

    Returns
    -------
    rate : float
        Vibrato rate in Hz.
    extent_cents : float
        Vibrato extent in cents, measured peak-to-peak on a bandpass-
        filtered F0 contour around the detected rate. Sundberg-convention
        peak-to-peak; divide by 2 if you want amplitude. Western classical
        singing is typically 50-150 cents PP.
    regularity : float in (0, 1)
        Spectral peak prominence; 1.0 means a perfectly clean periodic
        modulation, 0.0 means no peak above the background.
    """
    voiced = ~np.isnan(f0_hz) & (f0_hz > 0)
    if voiced.sum() < frame_rate_hz * 1.0:  # need at least 1 s of voicing
        return np.nan, np.nan, np.nan
    if voiced.mean() < min_voiced_fraction:
        return np.nan, np.nan, np.nan

    f0 = f0_hz.copy()
    f0[~voiced] = np.nan

    # Interpolate small gaps so we have a uniform signal for spectral analysis
    idx = np.arange(len(f0))
    valid = ~np.isnan(f0)
    if valid.sum() < 4:
        return np.nan, np.nan, np.nan
    f0_interp = np.interp(idx, idx[valid], f0[valid])

    # Convert to cents around the local mean
    f0_mean = np.mean(f0_interp[valid])
    if f0_mean <= 0:
        return np.nan, np.nan, np.nan
    cents = 1200.0 * np.log2(f0_interp / f0_mean)

    # Detrend to remove slow glide / aria-level pitch drift
    cents = sp_signal.detrend(cents)

    # Welch PSD of the modulation signal
    nperseg = min(len(cents), int(frame_rate_hz * 4))  # 4-s windows max
    if nperseg < 16:
        return np.nan, np.nan, np.nan

    freqs, psd = sp_signal.welch(
        cents,
        fs=frame_rate_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window='hann',
    )

    band = (freqs >= min_rate) & (freqs <= max_rate)
    if band.sum() < 2 or psd[band].sum() <= 0:
        return np.nan, np.nan, np.nan

    peak_idx_in_band = np.argmax(psd[band])
    rate = float(freqs[band][peak_idx_in_band])

    # Extent: peak-to-peak of the cents signal after bandpass to the
    # detected vibrato band.
    nyq = frame_rate_hz / 2.0
    lo = max(0.1, rate - 1.0) / nyq
    hi = min(0.99, (rate + 1.0) / nyq)
    if lo >= hi:
        extent = float(np.ptp(cents))  # fall back to raw range
    else:
        b, a = sp_signal.butter(4, [lo, hi], btype='band')
        bandpassed = sp_signal.filtfilt(b, a, cents)
        extent = float(np.ptp(bandpassed))

    # Regularity: peak-in-band power as fraction of total band power
    peak_power = psd[band][peak_idx_in_band]
    total_power = psd[band].sum()
    regularity = float(peak_power / total_power) if total_power > 0 else np.nan

    return rate, extent, regularity


# ---------------------------------------------------------------------------
# 4. Formant-aware band HNR (Ikuma 2025)
# ---------------------------------------------------------------------------

def compute_formant_band_hnr(
    audio: np.ndarray,
    sr: int,
    f1_band: Tuple[float, float] = (300.0, 1000.0),
    f2_band: Tuple[float, float] = (800.0, 2500.0),
    full_band: Tuple[float, float] = (80.0, 5000.0),
) -> Tuple[float, float, float]:
    """
    Band-limited harmonics-to-noise ratio via Praat (Parselmouth).

    Ikuma et al. (J. Voice 2025) demonstrate that HNR computed within
    formant-aware frequency bands explains substantially more variance in
    perceptual breathiness than whole-spectrum HNR. We adopt this approach.

    Bands are deliberately broad enough to cover the F1 and F2 ranges across
    voice types (sopranos have higher formant frequencies than basses;
    no per-vowel tuning here, but you can pass narrower bands if you do
    per-vowel-token analysis).

    Returns NaN values silently if Parselmouth is unavailable.

    Returns
    -------
    hnr_f1_db, hnr_f2_db, hnr_full_db : float
        HNR in dB for each band. Higher = more harmonic, less noisy.
    """
    if not _HAS_PARSELMOUTH:
        logger.warning("Parselmouth not installed; formant-band HNR skipped.")
        return np.nan, np.nan, np.nan
    if len(audio) < sr * 0.2:  # need >= 0.2 s
        return np.nan, np.nan, np.nan

    def _band_hnr(audio_band: np.ndarray) -> float:
        snd = parselmouth.Sound(audio_band, sampling_frequency=sr)
        try:
            harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75.0)
            # Average over voiced frames only (Praat marks unvoiced as -200 dB)
            values = harm.values[harm.values > -100]
            return float(np.mean(values)) if values.size else np.nan
        except Exception as e:
            logger.debug(f"Praat HNR failed: {e}")
            return np.nan

    def _bandpass(audio_in: np.ndarray, lo: float, hi: float) -> np.ndarray:
        nyq = sr / 2.0
        lo_n = max(0.001, lo / nyq)
        hi_n = min(0.999, hi / nyq)
        if lo_n >= hi_n:
            return audio_in
        b, a = sp_signal.butter(4, [lo_n, hi_n], btype='band')
        return sp_signal.filtfilt(b, a, audio_in)

    hnr_f1 = _band_hnr(_bandpass(audio, *f1_band))
    hnr_f2 = _band_hnr(_bandpass(audio, *f2_band))
    hnr_full = _band_hnr(_bandpass(audio, *full_band))

    return hnr_f1, hnr_f2, hnr_full


# ---------------------------------------------------------------------------
# Top-level convenience entry point
# ---------------------------------------------------------------------------

def compute_singer_features(
    audio: np.ndarray,
    sr: int,
    f0_hz: Optional[np.ndarray] = None,
    f0_frame_rate_hz: Optional[float] = None,
) -> SingerAcousticFeatures:
    """
    Compute all four singer-specific acoustic features on one audio segment.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio signal for one task segment (typically 1-30 s of
        sustained phonation or sung phrase).
    sr : int
        Audio sample rate (your pipeline: 48000).
    f0_hz : np.ndarray, optional
        Pre-computed F0 contour at frame rate `f0_frame_rate_hz`. Pass the
        F0 your existing pYIN-based pipeline produces. If None, vibrato
        cannot be computed.
    f0_frame_rate_hz : float, optional
        Frame rate of the F0 contour. Your pipeline: ~66.67 Hz given
        hop_length=720 @ 48 kHz.

    Returns
    -------
    SingerAcousticFeatures
        Filled with NaN for any feature that could not be computed.
    """
    feats = SingerAcousticFeatures()

    ratio, ratio_db = compute_singers_formant_ratio(audio, sr)
    feats.singers_formant_ratio = ratio
    feats.singers_formant_db = ratio_db

    slope, r2 = compute_ltas_slope(audio, sr)
    feats.ltas_slope_db_per_octave = slope
    feats.ltas_slope_r2 = r2

    if f0_hz is not None and f0_frame_rate_hz is not None:
        rate, extent, regularity = compute_vibrato(f0_hz, f0_frame_rate_hz)
        feats.vibrato_rate_hz = rate
        feats.vibrato_extent_cents = extent
        feats.vibrato_regularity = regularity

    hnr_f1, hnr_f2, hnr_full = compute_formant_band_hnr(audio, sr)
    feats.hnr_f1_band_db = hnr_f1
    feats.hnr_f2_band_db = hnr_f2
    feats.hnr_full_db = hnr_full

    feats.meta = {
        'sr': int(sr),
        'n_samples': int(len(audio)),
        'duration_s': float(len(audio) / sr),
        'parselmouth_available': bool(_HAS_PARSELMOUTH),
    }

    return feats


# ---------------------------------------------------------------------------
# Sanity test on synthetic signals
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic sustained vowel: 440 Hz fundamental with 5.5 Hz vibrato at
    # ~60 cents extent, formant emphasis around 700 Hz and 1200 Hz,
    # plus a singer's-formant bump at 3 kHz.
    sr = 48000
    duration = 3.0
    t = np.arange(int(sr * duration)) / sr

    # Vibrato: 5.5 Hz modulation, 60 cents = 0.6 semitones = 3.5% in Hz
    vib = 0.035 * np.sin(2 * np.pi * 5.5 * t)
    inst_freq = 440.0 * (1 + vib)
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr

    # Harmonic stack with formant-shaped amplitudes
    audio = np.zeros_like(t)
    for k in range(1, 30):
        f_k = k * 440.0
        # Crude formant emphasis: peaks at 700, 1200, 3000 Hz
        amp = (np.exp(-((f_k - 700) / 200) ** 2)
               + 0.7 * np.exp(-((f_k - 1200) / 250) ** 2)
               + 0.5 * np.exp(-((f_k - 3000) / 400) ** 2))
        audio = audio + amp * np.sin(k * phase) / (k ** 0.5)

    # Add a touch of noise
    rng = np.random.default_rng(42)
    audio = audio + 0.01 * rng.standard_normal(len(audio))
    audio = audio / np.max(np.abs(audio))

    # Synthetic F0 contour matching the modulation
    frame_rate = 66.67
    n_frames = int(duration * frame_rate)
    t_f = np.arange(n_frames) / frame_rate
    f0_contour = 440.0 * (1 + 0.035 * np.sin(2 * np.pi * 5.5 * t_f))

    feats = compute_singer_features(audio, sr, f0_contour, frame_rate)
    print("=" * 70)
    print("Sanity test on synthetic singing-like signal")
    print("=" * 70)
    print(f"Singer's-formant ratio:  {feats.singers_formant_ratio:.4f}  "
          f"({feats.singers_formant_db:+.2f} dB)")
    print(f"LTAS slope:              {feats.ltas_slope_db_per_octave:+.2f} "
          f"dB/octave  (R^2 = {feats.ltas_slope_r2:.3f})")
    print(f"Vibrato rate:            {feats.vibrato_rate_hz:.2f} Hz  "
          f"(expected 5.5 Hz)")
    print(f"Vibrato extent:          {feats.vibrato_extent_cents:.1f} cents  "
          f"(expected ~60 cents)")
    print(f"Vibrato regularity:      {feats.vibrato_regularity:.3f}")
    print(f"HNR F1 band:             {feats.hnr_f1_band_db:.2f} dB")
    print(f"HNR F2 band:             {feats.hnr_f2_band_db:.2f} dB")
    print(f"HNR full band:           {feats.hnr_full_db:.2f} dB")
    print(f"Meta:                    {feats.meta}")