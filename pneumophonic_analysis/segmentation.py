"""
Segmentation module for pneumophonic analysis.
Implement different segmentation methods:
- Crossing FRC (Above/Below Functional Residual Capacity)
- Novelty function for the glissando (P1/P2)
- Modulation frequency for the roll R
- Silence detection and active intervals

RReference: Zocco 2025 Thesis, Sections 3.7.2-3.7.4
"""

import numpy as np
import pandas as pd
from scipy import signal
import librosa
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass

from .config import PipelineConfig, get_config


@dataclass
class FRCSegment:
    """Segment relatif à la FRC."""
    
    # Indexes (audio samples)
    start_sample: int
    end_sample: int
    frc_cross_sample: int
    
    # audio segments
    above_frc: np.ndarray
    below_frc: np.ndarray
    
    # Metadata
    duration_above: float  # seconds
    duration_below: float  # seconds
    sample_rate: int
    
    @property
    def above_frc_duration_percent(self) -> float:
        """Percentage of time spent above the FRC."""
        total = self.duration_above + self.duration_below
        return (self.duration_above / total * 100) if total > 0 else 0


@dataclass
class GlideSegment:
    """Glissando segment in P1 (low frequencies) and P2 (high frequencies)."""
    
    # Indices
    start_sample: int
    peak_sample: int  # Point of maximum spectral transition
    end_sample: int
    
    # audio segments
    part1: np.ndarray  # Portion low frequencies
    part2: np.ndarray  # Portion high frequencies
    
    # Times
    peak_time: float
    duration_p1: float
    duration_p2: float
    
    # Value of the novelty function at the peak
    peak_novelty_value: float


@dataclass
class ModulationResult:
    """RResult of the modulation frequency analysis (roll R)."""
    
    frequency_full: float      # Frequency over the entire segment
    frequency_above_frc: float # Frequency above the FRC
    frequency_below_frc: float # Frequency below the FRC
    
    # Enveloppe and spectrum for debugging/analysis
    rms_envelope: Optional[np.ndarray] = None
    modulation_spectrum: Optional[np.ndarray] = None


class FRCSegmenter:
    """
    Segments the phonation according to the FRC threshold.
    
    The FRC (Functional Residual Capacity) is the pulmonary volume
    at the end of a passive expiration. Phonation above
    the FRC uses elastic recoil, that below requires
    active muscle effort.
    
    Example usage  :
    ```python
    segmenter = FRCSegmenter()
    
    # With OEP data
    segment = segmenter.segment_by_frc(
        audio=audio,
        oep_volume=volume_trace,
        frc_level=frc_value,
        sr_audio=48000,
        sr_oep=50
    )
    
    # Separate analysis of the two parts
    result_above = analyze(segment.above_frc)
    result_below = analyze(segment.below_frc)
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_config()
    
    def find_frc_crossing(
        self,
        volume: np.ndarray,
        frc_level: float
    ) -> int:
        """
        Finds the instant where the volume crosses the FRC (descending).
        
        Args:
            volume: Volume trace (Vcw)
            frc_level: FRC level
            
        Returns:
            Index of the crossing
        """
        # Find where the volume passes below the FRC
        below_frc = volume < frc_level
        
        # Find the first crossing (descending)
        crossings = np.where(np.diff(below_frc.astype(int)) == 1)[0]
        
        if len(crossings) == 0:
            # No crossing found, return the middle
            return len(volume) // 2
        
        return crossings[0]
    
    def segment_by_frc(
        self,
        audio: np.ndarray,
        oep_volume: np.ndarray,
        frc_level: float,
        sr_audio: int,
        sr_oep: int,
        phonation_start: int,
        phonation_end: int
    ) -> FRCSegment:
        """
        Segments the audio according to the FRC threshold.
        
        Args:
            audio: Audio signal
            oep_volume: Chest wall volume (Vcw)
            frc_level: FRC level
            sr_audio: Audio sample rate
            sr_oep: OEP sample rate
            phonation_start: Start of phonation (audio samples)
            phonation_end: End of phonation (audio samples)
            
        Returns:
            FRCSegment with the two parts
        """
        # Extract the portion of volume corresponding to the phonation
        oep_start = int(phonation_start / sr_audio * sr_oep)
        oep_end = int(phonation_end / sr_audio * sr_oep)
        volume_segment = oep_volume[oep_start:oep_end]
        
        # Find the FRC crossing in the volume
        crossing_oep = self.find_frc_crossing(volume_segment, frc_level)
        
        # Convert to audio samples
        crossing_audio = int(crossing_oep / sr_oep * sr_audio)
        frc_cross_sample = phonation_start + crossing_audio
        
        # Segment the audio
        audio_segment = audio[phonation_start:phonation_end]
        above_frc = audio_segment[:crossing_audio]
        below_frc = audio_segment[crossing_audio:]
        
        return FRCSegment(
            start_sample=phonation_start,
            end_sample=phonation_end,
            frc_cross_sample=frc_cross_sample,
            above_frc=above_frc,
            below_frc=below_frc,
            duration_above=len(above_frc) / sr_audio,
            duration_below=len(below_frc) / sr_audio,
            sample_rate=sr_audio
        )
    
    def segment_by_time(
        self,
        audio: np.ndarray,
        cross_time: float,
        start_time: float,
        end_time: float,
        sr: int
    ) -> FRCSegment:
        """
        Segments the audio with a provided crossing time.
        
        Used when the FRC crossing time is already known
        (e.g., from an Excel file).
        
        Args:
            audio: Signal audio
            cross_time: Time of the crossing (seconds)
            start_time: Start of phonation (seconds)
            end_time: End of phonation (seconds)
            sr: Sample rate
            
        Returns:
            FRCSegment
        """
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        cross_sample = int(cross_time * sr)
        
        # Ensure that cross is between start and end
        cross_sample = max(start_sample, min(cross_sample, end_sample))
        
        above = audio[start_sample:cross_sample]
        below = audio[cross_sample:end_sample]
        
        return FRCSegment(
            start_sample=start_sample,
            end_sample=end_sample,
            frc_cross_sample=cross_sample,
            above_frc=above,
            below_frc=below,
            duration_above=len(above) / sr,
            duration_below=len(below) / sr,
            sample_rate=sr
        )


class GlideSegmenter:
    """
    Segments the vocal glide into P1 (low frequencies) and P2 (high frequencies).
    
    Uses the novelty function based on the spectral centroid
    to detect the maximum transition point.
    
    Example usage  :
    ```python
    segmenter = GlideSegmenter()
    segment = segmenter.segment_glide(audio, sr)
    
    # Analyze the two parts
    f0_p1 = analyze(segment.part1)  # Low frequencies
    f0_p2 = analyze(segment.part2)  # High frequencies
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_config()
    
    def compute_novelty_function(
        self,
        audio: np.ndarray,
        sr: int,
        downsample_factor: int = 32,
        gamma: float = 10.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the novelty function based on the spectral centroid.
        
        The novelty function detects rapid changes in
        the spectrum. For a glide, it will have a peak at the moment
        of the maximum frequency transition.
        
        Args:
            audio: Signal audio
            sr: Sample rate
            downsample_factor: Downsample factor
            gamma: Logarithmic compression factor
            frame_length: STFT window size
            hop_length: STFT hop length
            
        Returns:
            Tuple (novelty_function, time_axis)
        """
        # Compute the spectral centroid
        S = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        
        # Downsampling and logarithmic compression
        centroid_down = centroid[::downsample_factor]
        centroid_log = np.log(1 + gamma * centroid_down)
        
        # Derivative (frequency change)
        diff = np.diff(centroid_log, prepend=centroid_log[0])
        
        # Half-wave rectification (we keep only the increases)
        diff[diff < 0] = 0
        
        # Subtraction of the local mean to enhance peaks
        window_size = min(len(diff) // 4, 51)
        if window_size > 5:
            local_mean = signal.savgol_filter(diff, window_size | 1, 3)
            diff = diff - local_mean
            diff[diff < 0] = 0
        
        # Time axis for the novelty function
        fps = sr / (hop_length * downsample_factor)
        time_axis = np.arange(len(diff)) / fps
        
        return diff, time_axis
    
    def find_peak_auto(
        self,
        novelty: np.ndarray,
        margin_percent: int = 10
    ) -> int:
        """
        Finds the peak automatically in the novelty function.
        
        Ignores the margins (start/end) to avoid false positives
        related to onset/offset transitions.
        
        Args:
            novelty: Novelty function
            margin_percent: Percentage of margin to ignore
            
        Returns:
            Index of the peak
        """
        margin = int(len(novelty) * margin_percent / 100)
        
        if margin >= len(novelty) // 2:
            margin = len(novelty) // 4
        
        cropped = novelty[margin:-margin] if margin > 0 else novelty
        peak_idx = np.argmax(cropped) + margin
        
        return peak_idx
    
    def segment_glide(
        self,
        audio: np.ndarray,
        sr: int,
        phonation_start: Optional[int] = None,
        phonation_end: Optional[int] = None,
        peak_override: Optional[int] = None
    ) -> GlideSegment:
        """
        Segments the glissando into P1 and P2.
        
        Args:
            audio: Signal audio
            sr: Sample rate
            phonation_start: Start of phonation (samples), auto if None
            phonation_end: End of phonation (samples), auto if None
            peak_override: Position of the manual peak (samples)
            
        Returns:
            GlideSegment with the two parts
        """
        # Automatic detection of boundaries if not provided
        if phonation_start is None:
            onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='samples')
            phonation_start = onsets[0] if len(onsets) > 0 else 0
        
        if phonation_end is None:
            intervals = librosa.effects.split(audio, top_db=45)
            phonation_end = intervals[-1, 1] if len(intervals) > 0 else len(audio)
        
        # Extract the phonation segment
        audio_segment = audio[phonation_start:phonation_end]
        
        # Compute the novelty function
        novelty, time_axis = self.compute_novelty_function(audio_segment, sr)
        
        # Find the peak in the novelty function
        if peak_override is not None:
            # Convert the absolute peak_override to a relative position
            peak_relative = peak_override - phonation_start
            # Convert to a novelty index
            fps = sr / (512 * 32)  # Default values for hop_length and downsample_factor
            peak_novelty_idx = int(peak_relative / sr * fps)
            peak_novelty_idx = min(max(0, peak_novelty_idx), len(novelty) - 1)
        else:
            peak_novelty_idx = self.find_peak_auto(novelty)
        
        # Convert the novelty index to an audio sample
        fps = sr / (512 * 32)
        peak_time = peak_novelty_idx / fps
        peak_sample = phonation_start + int(peak_time * sr)
        
        # Split the audio into two parts at the peak
        part1 = audio[phonation_start:peak_sample]
        part2 = audio[peak_sample:phonation_end]
        
        return GlideSegment(
            start_sample=phonation_start,
            peak_sample=peak_sample,
            end_sample=phonation_end,
            part1=part1,
            part2=part2,
            peak_time=peak_time,
            duration_p1=len(part1) / sr,
            duration_p2=len(part2) / sr,
            peak_novelty_value=novelty[peak_novelty_idx] if peak_novelty_idx < len(novelty) else 0
        )


class ModulationAnalyzer:
    """
    Analyze the frequency of modulation for the rolled R.
    
    The rolled (alveolar trill) produces a periodic modulation
    of the acoustic envelope corresponding to the vibration of the
    tip of the tongue (typically 20-30 Hz).
    
    Example usage  :
    ```python
    analyzer = ModulationAnalyzer()
    result = analyzer.analyze_modulation(audio, sr)
    print(f"Fréquence de modulation: {result.frequency_full:.1f} Hz")
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_config()
    
    def compute_modulation_frequency(
        self,
        audio: np.ndarray,
        sr: int
    ) -> float:
        """
        Compute the modulation frequency of an audio segment.
        
        MMethod:
        1. Extract the RMS envelope
        2. Remove the slow trend (Savitzky-Golay)
        3. FFT of the envelope
        4. Find the peak in the modulation band
        
        Args:
            audio: Signal audio
            sr: Sample rate
            
        Returns:
            Modulation frequency in Hz
        """
        cfg = self.config.modulation
        
        if audio is None or len(audio) < cfg.rms_frame_length:
            return np.nan
        
        # 1. Extraction envelope RMS
        rms = librosa.feature.rms(
            y=audio,
            frame_length=cfg.rms_frame_length,
            hop_length=cfg.rms_hop_length
        )[0]
        
        if len(rms) < 15:
            return np.nan
        
        # 2. Suppression of slow trend
        window_len = min(len(rms) // 2 * 2 - 1, cfg.savgol_window_max)
        if window_len < 5:
            window_len = 5
        if window_len % 2 == 0:
            window_len -= 1
        
        if window_len > 5:
            trend = signal.savgol_filter(rms, window_len, cfg.savgol_order)
            rms_clean = rms - trend
        else:
            rms_clean = rms - np.mean(rms)
        
        # 3. FFT of the envelope
        n_fft = len(rms_clean)
        spectrum = np.fft.rfft(rms_clean)
        freqs = np.fft.rfftfreq(n_fft, d=cfg.rms_hop_length / sr)
        
        # 4. Find the peak in the modulation band
        mask = (freqs >= cfg.mod_freq_min) & (freqs <= cfg.mod_freq_max)
        
        if not np.any(mask):
            # Fallback with wider band
            mask = (freqs >= cfg.mod_freq_fallback_min) & (freqs <= cfg.mod_freq_fallback_max)
        
        if not np.any(mask):
            return np.nan
        
        peak_idx = np.argmax(np.abs(spectrum[mask]))
        return freqs[mask][peak_idx]
    
    def analyze_with_frc(
        self,
        audio: np.ndarray,
        sr: int,
        frc_segment: Optional[FRCSegment] = None,
        cross_time: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> ModulationResult:
        """
        Analyze the modulation with FRC segmentation.
        
        Args:
            audio: Signal audio
            sr: Sample rate
            frc_segment: FRC SEGMENT pre-computed (optional)
            cross_time: Time of FRC crossing (alternative)
            start_time: Start time of phonation
            end_time: End time of phonation
            
        Returns:
            ModulationResult with full, above and below FRC frequencies
        """
        if frc_segment is None and cross_time is not None:
            segmenter = FRCSegmenter(self.config)
            frc_segment = segmenter.segment_by_time(
                audio, cross_time, start_time or 0, end_time or len(audio)/sr, sr
            )
        
        # Frequency on the entire segment
        if frc_segment is not None:
            full_audio = np.concatenate([frc_segment.above_frc, frc_segment.below_frc])
            above_audio = frc_segment.above_frc
            below_audio = frc_segment.below_frc
        else:
            full_audio = audio
            above_audio = None
            below_audio = None
        
        freq_full = self.compute_modulation_frequency(full_audio, sr)
        freq_above = self.compute_modulation_frequency(above_audio, sr) if above_audio is not None else np.nan
        freq_below = self.compute_modulation_frequency(below_audio, sr) if below_audio is not None else np.nan
        
        return ModulationResult(
            frequency_full=freq_full,
            frequency_above_frc=freq_above,
            frequency_below_frc=freq_below
        )


def detect_non_silent_intervals(
    audio: np.ndarray,
    sr: int,
    top_db: int = 45
) -> np.ndarray:
    """
    Detects non-silent intervals.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        top_db: Threshold in dB below the maximum to consider as silence
        
    Returns:
        Array of shape (n_intervals, 2) with [start, end] in samples
    """
    return librosa.effects.split(audio, top_db=top_db)


def detect_phonation_bounds(
    audio: np.ndarray,
    sr: int,
    top_db: int = 45,
    onset_delta: float = 0.05,
    highpass_hz: float = 0.0,
    min_phonation_sec: float = 0.0,
) -> Tuple[int, int]:
    """
    DDetects the start and end of phonation.

    Args:
        audio: Signal audio
        sr: Sample rate
        top_db: Threshold in dB below the maximum to consider as silence
        onset_delta: Sensitivity of onset detection (lower = more sensitive)
        highpass_hz: If > 0, the start/end are detected on a high-pass-filtered
            copy of the audio (cutoff in Hz). This suppresses a loud low-frequency
            or DC event — such as a leaked synchronization pulse — so it cannot
            be picked as the phonation onset. The returned indices are on the
            ORIGINAL time base; the filtered signal is used for detection only.
        min_phonation_sec: If > 0, non-silent intervals shorter than this are
            ignored when choosing the start/end. Together with `highpass_hz`
            this also discards the brief broadband *clicks* at the edges of a
            sync pulse (which survive the high-pass), so the onset lands on
            sustained phonation rather than the pulse.

    Returns:
        Tuple (start_sample, end_sample)
    """
    detect_sig = audio
    # High-pass only for detection, so a DC/low-frequency sync pulse cannot
    # anchor the onset. filtfilt needs a minimum length; skip very short clips.
    if highpass_hz and highpass_hz > 0 and len(audio) > 33:
        wn = highpass_hz / (0.5 * sr)
        if 0.0 < wn < 1.0:
            try:
                b, a = signal.butter(4, wn, btype='high')
                detect_sig = signal.filtfilt(b, a, audio)
            except Exception:
                detect_sig = audio

    # Non silent intervals
    intervals = detect_non_silent_intervals(detect_sig, sr, top_db)

    if len(intervals) == 0:
        return 0, len(audio)

    # Drop intervals too short to be phonation (e.g. sync-pulse edge clicks).
    if min_phonation_sec and min_phonation_sec > 0:
        min_len = int(min_phonation_sec * sr)
        long_enough = intervals[(intervals[:, 1] - intervals[:, 0]) >= min_len]
        if len(long_enough) > 0:
            intervals = long_enough

    # Take the first significant interval for the start
    start = intervals[0, 0]

    # Take the end of the last interval for the end
    end = intervals[-1, 1]

    return start, end
