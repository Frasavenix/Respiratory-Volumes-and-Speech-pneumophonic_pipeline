"""
Paired Feature Extraction Module (M1)
======================================

Produces time-aligned [audio | OEP] feature matrices for each subject × task.

This module bridges the audio and respiratory pipelines by:
1. Loading audio + OEP data for a given subject/task
2. Synchronizing them via the falling-edge sync pulse
3. Extracting frame-level audio features (MFCCs, F0, energy, spectral)
4. Extracting frame-level OEP features (volumes, flows, compartmental ratios)
5. Interpolating OEP onto the audio feature time grid (50 Hz → ~66 Hz)
6. Producing a unified NumPy array / DataFrame and exporting to HDF5

Integration:
    Drop this file into pneumophonic_analysis/ and add imports to __init__.py.
    Uses existing DataLoader, Synchronizer, AudioProcessor, config.

Column mapping (verified from actual .dat files):
    A = Vrcp (pulmonary rib cage)
    B = Vrca (abdominal rib cage)
    C = Vab  (abdomen)
    tot_vol = Vcw (chest wall = A + B + C)
    Two-compartment model (Zocco thesis): Vrc = A + B, Vab = C

Author: Thesis continuation — OEP–Audio correlation study
"""

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field
import logging
import h5py

from .config import PipelineConfig, get_config
from .io_utils import DataLoader
from .sync import Synchronizer, SyncResult
from .audio_processing import AudioProcessor, AudioFeatures

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OEPFrameFeatures:
    """
    OEP features resampled to the audio frame time grid.

    All arrays have shape (n_frames,) matching the audio feature frames.
    """

    time: np.ndarray            # Time axis in seconds (audio reference frame)

    # Volumes (litres) — two-compartment model
    vcw: np.ndarray             # Chest wall volume (tot_vol)
    vrc: np.ndarray             # Rib cage volume (Vrcp + Vrca = A + B)
    vab: np.ndarray             # Abdominal volume (C)

    # Sub-compartments (litres) — three-compartment detail
    vrcp: np.ndarray            # Pulmonary rib cage (A)
    vrca: np.ndarray            # Abdominal rib cage (B)

    # Flows (litres/s) — numerical derivative of filtered volumes
    flow_cw: np.ndarray
    flow_rc: np.ndarray
    flow_ab: np.ndarray

    # Compartmental contributions (dimensionless, 0–1)
    pct_rc: np.ndarray          # Vrc / Vcw instantaneous ratio
    pct_ab: np.ndarray          # Vab / Vcw instantaneous ratio

    # Volume relative to segment start (delta from onset)
    delta_vcw: np.ndarray
    delta_vrc: np.ndarray
    delta_vab: np.ndarray


@dataclass
class PairedFrame:
    """
    Single aligned multimodal dataset for one subject × task.

    Attributes:
        subject_id:   Subject identifier (e.g. "GaBa")
        task_name:    Task label (e.g. "a", "r", "text")
        audio_features: AudioFeatures from the existing pipeline
        oep_features:   OEP features interpolated to audio frames
        dataframe:      Unified DataFrame (n_frames × n_total_features)
        metadata:       Dict with sync info, durations, sample rates, etc.
    """

    subject_id: str
    task_name: str
    audio_features: AudioFeatures
    oep_features: OEPFrameFeatures
    dataframe: pd.DataFrame
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OEP feature computation helpers
# ---------------------------------------------------------------------------

def _butterworth_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    4th-order Butterworth low-pass filter (zero-phase).

    Matches the thesis methodology (Section 3.5.2):
    "filtered using a fourth-order low-pass Butterworth filter
     with a cutoff frequency of 10 Hz"

    Falls back to unfiltered data if the segment is too short for filtfilt.
    """
    nyq = 0.5 * fs
    b, a = sp_signal.butter(order, cutoff / nyq, btype='low')

    # filtfilt requires len(data) > padlen = 3 * max(len(a), len(b)) - 1
    min_length = 3 * max(len(a), len(b))
    if len(data) < min_length:
        logger.warning(
            f"  OEP segment too short for Butterworth filter "
            f"({len(data)} < {min_length} samples) — using unfiltered data"
        )
        return data.copy()

    return sp_signal.filtfilt(b, a, data)


def compute_oep_features_native(
    oep_df: pd.DataFrame,
    fs_oep: int = 50,
    flow_cutoff_hz: float = 10.0,
    calibration_k: float = 0.916,
    col_vcw: str = 'tot_vol',
    col_vrcp: str = 'A',
    col_vrca: str = 'B',
    col_vab: str = 'C',
) -> pd.DataFrame:
    """
    Compute OEP-derived features at the native OEP sample rate.

    This follows the Zocco thesis methodology:
    - Volumes: Vcw (tot_vol), Vrc = Vrcp + Vrca (A + B), Vab (C)
    - Flows: dV/dt after 4th-order Butterworth LP at 10 Hz
    - Calibration factor k=0.916 applied to flows (Section 4.1.3)

    Args:
        oep_df:         Raw OEP DataFrame (from DataLoader.load_oep_data)
        fs_oep:         OEP sampling frequency
        flow_cutoff_hz: LP cutoff for pre-differentiation filtering
        calibration_k:  Global calibration factor for OEP-derived flow
        col_vcw:        Column name for chest wall volume
        col_vrcp:       Column name for pulmonary rib cage volume (A)
        col_vrca:       Column name for abdominal rib cage volume (B)
        col_vab:        Column name for abdominal volume (C)

    Returns:
        DataFrame with columns:
        [time_oep, vcw, vrc, vab, vrcp, vrca, flow_cw, flow_rc, flow_ab,
         pct_rc, pct_ab, delta_vcw, delta_vrc, delta_vab]
    """
    time_oep = oep_df['time'].values
    vcw = oep_df[col_vcw].values.astype(float)
    vrcp = oep_df[col_vrcp].values.astype(float)
    vrca = oep_df[col_vrca].values.astype(float)
    vab = oep_df[col_vab].values.astype(float)

    # Two-compartment model: Vrc = Vrcp + Vrca
    vrc = vrcp + vrca

    # --- Low-pass filter before differentiation ---
    vcw_filt = _butterworth_lowpass(vcw, flow_cutoff_hz, fs_oep)
    vrc_filt = _butterworth_lowpass(vrc, flow_cutoff_hz, fs_oep)
    vab_filt = _butterworth_lowpass(vab, flow_cutoff_hz, fs_oep)

    # --- Flows via numerical differentiation ---
    dt = 1.0 / fs_oep
    flow_cw = np.gradient(vcw_filt, dt) * calibration_k
    flow_rc = np.gradient(vrc_filt, dt) * calibration_k
    flow_ab = np.gradient(vab_filt, dt) * calibration_k

    # --- Compartmental contributions ---
    eps = 1e-9
    pct_rc = vrc / (np.abs(vcw) + eps)
    pct_ab = vab / (np.abs(vcw) + eps)

    # --- Delta volumes (relative to segment start) ---
    delta_vcw = vcw - vcw[0]
    delta_vrc = vrc - vrc[0]
    delta_vab = vab - vab[0]

    return pd.DataFrame({
        'time_oep': time_oep,
        'vcw': vcw,
        'vrc': vrc,
        'vab': vab,
        'vrcp': vrcp,
        'vrca': vrca,
        'flow_cw': flow_cw,
        'flow_rc': flow_rc,
        'flow_ab': flow_ab,
        'pct_rc': pct_rc,
        'pct_ab': pct_ab,
        'delta_vcw': delta_vcw,
        'delta_vrc': delta_vrc,
        'delta_vab': delta_vab,
    })


# ---------------------------------------------------------------------------
# Interpolation: OEP native grid → audio feature grid
# ---------------------------------------------------------------------------

def interpolate_oep_to_audio_frames(
    oep_feat_df: pd.DataFrame,
    audio_frame_times: np.ndarray,
    method: str = 'linear',
) -> OEPFrameFeatures:
    """
    Resample OEP features onto the audio feature time axis.

    OEP is at 50 Hz (~20 ms spacing), audio features at ~66 Hz
    (hop_length=720 @ 48 kHz → ~15 ms). Since OEP signals are
    physiologically band-limited (<10 Hz after LP filtering),
    linear interpolation introduces no aliasing.

    Args:
        oep_feat_df:       Output of compute_oep_features_native()
        audio_frame_times: Time axis of audio features (seconds)
        method:            Interpolation kind ('linear', 'cubic', etc.)

    Returns:
        OEPFrameFeatures aligned to audio_frame_times
    """
    t_oep = oep_feat_df['time_oep'].values

    columns_to_interp = [
        'vcw', 'vrc', 'vab', 'vrcp', 'vrca',
        'flow_cw', 'flow_rc', 'flow_ab',
        'pct_rc', 'pct_ab',
        'delta_vcw', 'delta_vrc', 'delta_vab',
    ]

    interpolated = {}
    for col in columns_to_interp:
        f_interp = interp1d(
            t_oep, oep_feat_df[col].values,
            kind=method,
            bounds_error=False,
            fill_value=(oep_feat_df[col].values[0], oep_feat_df[col].values[-1]),
        )
        interpolated[col] = f_interp(audio_frame_times)

    return OEPFrameFeatures(
        time=audio_frame_times,
        vcw=interpolated['vcw'],
        vrc=interpolated['vrc'],
        vab=interpolated['vab'],
        vrcp=interpolated['vrcp'],
        vrca=interpolated['vrca'],
        flow_cw=interpolated['flow_cw'],
        flow_rc=interpolated['flow_rc'],
        flow_ab=interpolated['flow_ab'],
        pct_rc=interpolated['pct_rc'],
        pct_ab=interpolated['pct_ab'],
        delta_vcw=interpolated['delta_vcw'],
        delta_vrc=interpolated['delta_vrc'],
        delta_vab=interpolated['delta_vab'],
    )


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class PairedFeatureExtractor:
    """
    Extracts time-aligned audio + OEP features for a subject × task.

    Workflow:
        1. Load audio (renders/<task>.wav) and OEP (csv/<subject>Vocali.csv)
        2. Load sync signal and run Synchronizer.synchronize()
        3. Use Synchronizer.extract_oep_segment() to get the OEP window
        4. Run AudioProcessor.process_full_pipeline() on the audio
        5. Compute OEP features at native rate, then interpolate to audio frames
        6. Merge into a single DataFrame / PairedFrame

    Column mapping (from actual .dat files):
        A = Vrcp, B = Vrca, C = Vab, tot_vol = Vcw
        Vrc = A + B (two-compartment model)

    Example:
        ```python
        extractor = PairedFeatureExtractor(config)
        paired = extractor.extract(
            subject_folder="data/20260218_GaBa",
            task_name="a",
            audio_filename="a.wav",
            oep_csv_path="csv/GaBaVocali.csv",
            take_number=2,
        )
        # paired.dataframe is the aligned matrix
        extractor.save_hdf5(paired, "output/paired/GaBa_a.h5")
        ```
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_config()
        self.synchronizer = Synchronizer(self.config)
        self.audio_processor = AudioProcessor(self.config)

    def extract(
        self,
        subject_folder: Union[str, Path],
        task_name: str,
        audio_filename: str,
        oep_csv_path: str,
        take_number: int = 1,
        audio_onset_index: int = 1,
        oep_col_vcw: str = 'tot_vol',
        oep_col_vrcp: str = 'A',
        oep_col_vrca: str = 'B',
        oep_col_vab: str = 'C',
        calibration_k: float = 0.916,
        flow_cutoff_hz: float = 10.0,
        audio_start_sec: Optional[float] = None,
        audio_end_sec: Optional[float] = None,
        auto_trim_phonation: bool = True,
        trim_top_db: int = 30,
        trim_highpass_hz: float = 90.0,
        trim_min_phonation_sec: float = 0.08,
        oep_falling_edge_sec: Optional[float] = None,
    ) -> PairedFrame:
        """
        Full extraction pipeline for one subject × task.

        Args:
            subject_folder:    Path to subject directory
            task_name:         Task identifier (e.g. "a", "r", "text")
            audio_filename:    Audio file in renders/ (e.g. "a.wav")
            oep_csv_path:      Relative path to OEP CSV within subject folder
            take_number:       Which take in the OEP sync to align to
                               (ignored if oep_falling_edge_sec is provided)
            audio_onset_index: Which onset in the audio sync (0=rising, 1=falling)
            oep_col_vcw:       Column for chest wall volume
            oep_col_vrcp:      Column for pulmonary rib cage (A)
            oep_col_vrca:      Column for abdominal rib cage (B)
            oep_col_vab:       Column for abdomen (C)
            calibration_k:     OEP flow calibration factor (thesis: 0.916)
            flow_cutoff_hz:    LP cutoff before differentiation
            audio_start_sec:   Optional start trim (audio time, seconds)
            audio_end_sec:     Optional end trim (audio time, seconds)
            auto_trim_phonation: If True, refine start/end to actual phonation
                                 onset/offset within the Excel window using
                                 detect_phonation_bounds (default: True)
            trim_top_db:       Silence threshold in dB for auto-trim (default: 30)
            trim_highpass_hz:  High-pass cutoff (Hz) applied to the signal used
                               for auto-trim onset/offset detection only. Stops
                               a loud low-frequency / DC synchronization pulse
                               from anchoring the phonation onset. Set 0 to
                               disable (default: 90)
            trim_min_phonation_sec: Auto-trim ignores non-silent runs shorter
                               than this (seconds) — discards the brief edge
                               clicks of a sync pulse so the onset lands on
                               sustained phonation (default: 0.08)
            oep_falling_edge_sec: If provided, use this OEP time (seconds) as
                                  the falling edge instead of auto-detecting via
                                  take_number. This comes from the 'falling edge'
                                  column of the subject's Excel timing file and
                                  is the most reliable sync method.

        Returns:
            PairedFrame with unified DataFrame and component objects
        """
        loader = DataLoader(subject_folder, self.config)
        fs_oep = self.config.oep.fs_kinematic

        # ---- 1. Load signals ----
        audio, sr = loader.load_audio(audio_filename)
        sync_audio, sr_sync = loader.load_sync_signal()
        oep_df = loader.load_oep_data(oep_csv_path)

        # Auto-detect the OEP sample rate from the CSV 'time' column.
        # Acquisitions are not uniform: some are 50 Hz, some 100 Hz. Using the
        # wrong rate time-stretches the OEP, mis-places the sync falling edge,
        # scales the flows, and desyncs OEP from audio. Trust the data.
        if 'time' in oep_df.columns and len(oep_df) > 1:
            dt = float(np.median(np.diff(oep_df['time'].values)))
            if dt > 0:
                detected = int(round(1.0 / dt))
                if detected > 0 and detected != fs_oep:
                    logger.warning(
                        f"  OEP rate detected = {detected} Hz from CSV time column "
                        f"(config default {fs_oep} Hz) — using {detected} Hz"
                    )
                    fs_oep = detected

        logger.info(
            f"[{loader.subject_id}/{task_name}] "
            f"audio={len(audio)/sr:.1f}s  OEP={len(oep_df)/fs_oep:.1f}s"
        )

        # ---- 2. Synchronize ----
        if oep_falling_edge_sec is not None:
            # Direct sync: use the known OEP falling edge from Excel
            oep_falling_sample = int(oep_falling_edge_sec * fs_oep)

            # Detect audio falling edge as usual
            audio_falling_sample = self.synchronizer.detect_falling_edge_audio(
                sync_audio, sr_sync, audio_onset_index
            )
            audio_falling_time = audio_falling_sample / sr_sync

            time_offset = audio_falling_time - oep_falling_edge_sec

            sync_result = SyncResult(
                sync_falling_edge_sample=audio_falling_sample,
                oep_falling_edge_sample=oep_falling_sample,
                time_offset_sec=time_offset,
                sync_onsets_samples=np.array([audio_falling_sample]),
                oep_sync_onsets_samples=np.array([oep_falling_sample]),
            )
            logger.info(
                f"  sync (Excel falling edge) = {oep_falling_edge_sec:.2f}s  "
                f"audio_edge={audio_falling_sample}  "
                f"oep_edge={oep_falling_sample}  "
                f"offset={time_offset:.4f}s"
            )
        else:
            # Fallback: auto-detect via take_number
            sync_result = self.synchronizer.synchronize(
                sync_audio=sync_audio,
                sr_audio=sr_sync,
                oep_df=oep_df,
                take_number=take_number,
                fs_oep=fs_oep,
                audio_onset_index=audio_onset_index,
            )
            logger.info(
                f"  sync (auto) offset = {sync_result.time_offset_sec:.4f}s  "
                f"audio_edge={sync_result.sync_falling_edge_sample}  "
                f"oep_edge={sync_result.oep_falling_edge_sample}"
            )

        # ---- 3. Determine audio segment boundaries ----
        if audio_start_sec is None:
            audio_start_sec = 0.0
        if audio_end_sec is None:
            audio_end_sec = len(audio) / sr

        # ---- 3b. Auto-trim to actual phonation within the window ----
        if auto_trim_phonation:
            from .segmentation import detect_phonation_bounds as _detect_bounds

            # Extract the coarse window first
            coarse_start_sample = int(audio_start_sec * sr)
            coarse_end_sample = int(audio_end_sec * sr)
            coarse_segment = audio[coarse_start_sample:coarse_end_sample]

            # Detect phonation onset/offset within that window. The high-pass
            # keeps a loud low-frequency / DC sync pulse from anchoring the
            # onset (see the sync-leak fix); detection only, audio is untouched.
            onset_sample, offset_sample = _detect_bounds(
                coarse_segment, sr, top_db=trim_top_db,
                highpass_hz=trim_highpass_hz,
                min_phonation_sec=trim_min_phonation_sec,
            )

            # Convert back to absolute times
            trimmed_start = audio_start_sec + onset_sample / sr
            trimmed_end = audio_start_sec + offset_sample / sr

            logger.info(
                f"  auto-trim: {audio_start_sec:.2f}–{audio_end_sec:.2f}s "
                f"→ {trimmed_start:.2f}–{trimmed_end:.2f}s "
                f"(removed {(trimmed_start - audio_start_sec):.2f}s head, "
                f"{(audio_end_sec - trimmed_end):.2f}s tail)"
            )

            audio_start_sec = trimmed_start
            audio_end_sec = trimmed_end

        # ---- 4. Extract matching OEP segment ----
        oep_segment = self.synchronizer.extract_oep_segment(
            oep_df=oep_df,
            audio_start_sec=audio_start_sec,
            audio_end_sec=audio_end_sec,
            sync_result=sync_result,
            fs_oep=fs_oep,
        )
        logger.info(f"  OEP segment: {len(oep_segment)} samples ({len(oep_segment)/fs_oep:.2f}s)")

        # ---- 5. Audio processing + feature extraction ----
        audio_segment = audio[int(audio_start_sec * sr):int(audio_end_sec * sr)]
        audio_clean, audio_feats = self.audio_processor.process_full_pipeline(
            audio_segment, sr
        )

        # Audio feature time axis (in seconds, relative to segment start)
        n_audio_frames = audio_feats.f0.shape[0]
        hop = self.config.audio.hop_length_samples
        audio_frame_times = np.arange(n_audio_frames) * hop / sr

        # True raw-amplitude envelope on the audio frame grid — for intuitive
        # visualization. The 'energy' feature is computed on the pre-emphasised
        # + L2-normalised + denoised signal, which annihilates the steady
        # low-frequency vowel and collapses sustained phonation to a single
        # onset spike. rms_audio is the RMS of the *raw* segment and preserves
        # the real loudness contour (so plots show a sustained vowel as sustained).
        import librosa as _librosa
        rms_audio = _librosa.feature.rms(
            y=audio_segment, frame_length=hop * 2, hop_length=hop, center=True
        )[0]
        if len(rms_audio) >= n_audio_frames:
            rms_audio = rms_audio[:n_audio_frames]
        else:
            rms_audio = np.pad(rms_audio, (0, n_audio_frames - len(rms_audio)), mode='edge')

        # ---- 6. OEP feature computation at native rate ----
        oep_segment = oep_segment.copy()
        oep_segment['time'] = np.arange(len(oep_segment)) / fs_oep

        oep_native = compute_oep_features_native(
            oep_segment,
            fs_oep=fs_oep,
            flow_cutoff_hz=flow_cutoff_hz,
            calibration_k=calibration_k,
            col_vcw=oep_col_vcw,
            col_vrcp=oep_col_vrcp,
            col_vrca=oep_col_vrca,
            col_vab=oep_col_vab,
        )

        # ---- 7. Interpolate OEP → audio frame grid ----
        oep_interp = interpolate_oep_to_audio_frames(
            oep_native, audio_frame_times, method='linear'
        )

        # ---- 8. Build unified DataFrame ----
        df = self._build_dataframe(audio_feats, oep_interp, audio_frame_times)
        df['rms_audio'] = rms_audio

        # ---- 8b. Genuine FRC reference from quiet-breathing baseline ----
        # FRC (functional residual capacity) = resting end-expiratory lung
        # volume. We estimate it as the median of the expiratory troughs of the
        # subject's baseline (quiet-breathing) recording, on the same absolute
        # tot_vol scale as the phonation vcw. delta_vcw_frc = vcw - FRC then
        # crosses zero at the *true* FRC level (vs. the within-phonation midpoint
        # proxy used when no baseline reference is available).
        frc_volume = float('nan')
        try:
            _bl = loader.load_oep_data(f"csv/{loader.subject_id}_baseline.csv")
            _tv = _bl[oep_col_vcw].values.astype(float)
            _bdt = float(np.median(np.diff(_bl['time'].values)))
            _bfs = (1.0 / _bdt) if _bdt > 0 else fs_oep
            from scipy.signal import find_peaks as _find_peaks
            _troughs, _ = _find_peaks(-_tv, distance=max(1, int(2.0 * _bfs)))
            frc_volume = (float(np.median(_tv[_troughs])) if len(_troughs) >= 3
                          else float(np.percentile(_tv, 30)))
        except Exception as _e:
            logger.warning(f"  no baseline FRC reference ({_e}); delta_vcw_frc omitted")
        if np.isfinite(frc_volume) and 'vcw' in df.columns:
            df['delta_vcw_frc'] = df['vcw'] - frc_volume

        metadata = {
            'subject_id': loader.subject_id,
            'task_name': task_name,
            'sr_audio': sr,
            'fs_oep': fs_oep,
            'hop_length': hop,
            'n_frames': n_audio_frames,
            'audio_duration_sec': audio_end_sec - audio_start_sec,
            'oep_segment_samples': len(oep_segment),
            'sync_time_offset_sec': sync_result.time_offset_sec,
            'calibration_k': calibration_k,
            'take_number': take_number,
            'frc_volume': frc_volume,
        }

        paired = PairedFrame(
            subject_id=loader.subject_id,
            task_name=task_name,
            audio_features=audio_feats,
            oep_features=oep_interp,
            dataframe=df,
            metadata=metadata,
        )

        logger.info(f"  Paired matrix: {df.shape[0]} frames × {df.shape[1]} features")
        return paired

    # ------------------------------------------------------------------
    # F0 cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def clean_f0(
        f0: np.ndarray,
        voiced: np.ndarray,
        f0_min: float = 60.0,
        f0_max: float = 350.0,
        median_kernel: int = 5,
    ) -> np.ndarray:
        """
        Clean F0 track by removing octave-doubling artifacts.

        Two-step process:
        1. Physiological clamp: any F0 outside [f0_min, f0_max] is set to NaN
           and the frame is marked unvoiced. This catches gross octave jumps
           (e.g. 400-500 Hz spikes in male connected speech).
        2. Median filter: a short sliding window (default 5 frames ≈ 75ms)
           smooths remaining isolated outliers without blurring real prosody.

        Args:
            f0:            Raw F0 array (NaN for unvoiced frames)
            voiced:        Voiced flag array (1.0 = voiced)
            f0_min:        Minimum plausible F0 in Hz (default: 60)
            f0_max:        Maximum plausible F0 in Hz (default: 350)
            median_kernel: Median filter kernel size in frames (default: 5)

        Returns:
            Cleaned F0 array (NaN for unvoiced/rejected frames)
        """
        from scipy.ndimage import median_filter

        f0_clean = f0.copy()

        # Step 1: clamp to physiological range
        out_of_range = (f0_clean < f0_min) | (f0_clean > f0_max)
        f0_clean[out_of_range] = np.nan

        # Step 2: median filter on voiced segments only
        # Replace NaN temporarily for filtering, then restore
        valid_mask = ~np.isnan(f0_clean)
        if np.sum(valid_mask) > median_kernel:
            # Fill NaN with interpolation for median filter, then mask back
            f0_filled = f0_clean.copy()
            f0_filled[~valid_mask] = np.interp(
                np.where(~valid_mask)[0],
                np.where(valid_mask)[0],
                f0_clean[valid_mask],
            )
            f0_filtered = median_filter(f0_filled, size=median_kernel)
            # Only keep filtered values where original was valid
            f0_clean[valid_mask] = f0_filtered[valid_mask]

        return f0_clean

    # ------------------------------------------------------------------
    # DataFrame assembly
    # ------------------------------------------------------------------

    def _build_dataframe(
        self,
        audio_feats: AudioFeatures,
        oep_feats: OEPFrameFeatures,
        time_axis: np.ndarray,
    ) -> pd.DataFrame:
        """
        Merge audio and OEP features into a single DataFrame.

        Audio columns:
            time, f0, f0_raw, voiced, voicing_prob, energy, spectral_centroid,
            mfcc_0 … mfcc_12

        OEP columns:
            vcw, vrc, vab, vrcp, vrca, flow_cw, flow_rc, flow_ab,
            pct_rc, pct_ab, delta_vcw, delta_vrc, delta_vab
        """
        n_frames = len(time_axis)

        data = {'time': time_axis}

        # --- Audio: scalar per-frame features ---
        f0_raw = audio_feats.f0[:n_frames]
        voiced_flag = audio_feats.voiced_flag[:n_frames].astype(float)

        # Clean F0 (remove octave doublings)
        f0_clean = self.clean_f0(f0_raw, voiced_flag)

        # Update voiced flag: frames where F0 was clamped are now unvoiced
        voiced_clean = voiced_flag.copy()
        voiced_clean[np.isnan(f0_clean) & (voiced_flag == 1.0)] = 0.0

        n_rejected = int(np.sum(np.isnan(f0_clean) & ~np.isnan(f0_raw)))
        if n_rejected > 0:
            logger.info(f"  F0 cleanup: {n_rejected} frames rejected "
                        f"({n_rejected / n_frames * 100:.1f}%)")

        data['f0'] = f0_clean
        data['f0_raw'] = f0_raw
        data['voiced'] = voiced_clean
        data['voicing_prob'] = audio_feats.voiced_probs[:n_frames]

        # Energy (RMS from STFT)
        stft = audio_feats.stft  # shape: (n_freq, n_frames)
        energy = np.sqrt(np.mean(stft ** 2, axis=0))
        data['energy'] = energy[:n_frames]

        # Spectral centroid from STFT
        freqs = np.arange(stft.shape[0])
        spectral_centroid = np.sum(freqs[:, None] * stft, axis=0) / (np.sum(stft, axis=0) + 1e-9)
        data['spectral_centroid'] = spectral_centroid[:n_frames]

        # MFCCs — each coefficient becomes a column
        mfcc = audio_feats.mfcc  # shape: (n_mfcc, n_frames)
        n_mfcc = mfcc.shape[0]
        for i in range(n_mfcc):
            data[f'mfcc_{i}'] = mfcc[i, :n_frames]

        # --- OEP features (already interpolated) ---
        data['vcw'] = oep_feats.vcw[:n_frames]
        data['vrc'] = oep_feats.vrc[:n_frames]
        data['vab'] = oep_feats.vab[:n_frames]
        data['vrcp'] = oep_feats.vrcp[:n_frames]
        data['vrca'] = oep_feats.vrca[:n_frames]
        data['flow_cw'] = oep_feats.flow_cw[:n_frames]
        data['flow_rc'] = oep_feats.flow_rc[:n_frames]
        data['flow_ab'] = oep_feats.flow_ab[:n_frames]
        data['pct_rc'] = oep_feats.pct_rc[:n_frames]
        data['pct_ab'] = oep_feats.pct_ab[:n_frames]
        data['delta_vcw'] = oep_feats.delta_vcw[:n_frames]
        data['delta_vrc'] = oep_feats.delta_vrc[:n_frames]
        data['delta_vab'] = oep_feats.delta_vab[:n_frames]

        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def save_hdf5(paired: PairedFrame, path: Union[str, Path]) -> Path:
        """
        Save a PairedFrame to HDF5.

        Structure:
            /aligned/      — each column of the aligned DataFrame
            /mfcc          — full MFCC matrix (n_mfcc × n_frames)
            /mel           — mel spectrogram (n_mels × n_frames)
            /stft          — power spectrogram
            /attrs         — metadata as HDF5 attributes

        Args:
            paired: PairedFrame to save
            path:   Output .h5 file path

        Returns:
            Path to the saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(path), 'w') as f:
            # Main aligned DataFrame → stored as individual datasets
            grp = f.create_group('aligned')
            for col in paired.dataframe.columns:
                arr = paired.dataframe[col].values
                grp.create_dataset(col, data=arr, compression='gzip')

            # Full spectral representations (for future model input)
            f.create_dataset(
                'mfcc', data=paired.audio_features.mfcc, compression='gzip'
            )
            f.create_dataset(
                'mel', data=paired.audio_features.mel_spectrogram, compression='gzip'
            )
            f.create_dataset(
                'stft', data=paired.audio_features.stft, compression='gzip'
            )

            # Metadata
            for k, v in paired.metadata.items():
                f.attrs[k] = v

        logger.info(f"  Saved → {path}  ({path.stat().st_size / 1024:.0f} KB)")
        return path

    @staticmethod
    def load_hdf5(path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict]:
        """
        Load a paired dataset from HDF5.

        Returns:
            Tuple (DataFrame with aligned features, metadata dict)
        """
        path = Path(path)
        with h5py.File(str(path), 'r') as f:
            grp = f['aligned']
            data = {col: grp[col][:] for col in grp.keys()}
            df = pd.DataFrame(data)
            meta = dict(f.attrs)

        return df, meta

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        subject_task_list: List[Dict],
        output_dir: Union[str, Path],
        skip_existing: bool = True,
    ) -> pd.DataFrame:
        """
        Run paired extraction on multiple subject × task entries.

        Args:
            subject_task_list: List of dicts, each with keys:
                - subject_folder (str/Path)
                - task_name (str)
                - audio_filename (str)
                - oep_csv_path (str)
                - take_number (int)
                - (optional) audio_start_sec, audio_end_sec
            output_dir: Directory for HDF5 outputs
            skip_existing: Skip if .h5 already exists

        Returns:
            Summary DataFrame with one row per extraction
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []

        for entry in subject_task_list:
            subject_folder = entry['subject_folder']
            task_name = entry['task_name']
            loader = DataLoader(subject_folder, self.config)
            sid = loader.subject_id

            h5_path = output_dir / f"{sid}_{task_name}.h5"

            if skip_existing and h5_path.exists():
                logger.info(f"[SKIP] {h5_path.name} already exists")
                summary_rows.append({
                    'subject_id': sid, 'task': task_name,
                    'status': 'skipped', 'n_frames': None,
                })
                continue

            try:
                paired = self.extract(
                    subject_folder=subject_folder,
                    task_name=task_name,
                    audio_filename=entry['audio_filename'],
                    oep_csv_path=entry['oep_csv_path'],
                    take_number=entry.get('take_number', 1),
                    audio_start_sec=entry.get('audio_start_sec'),
                    audio_end_sec=entry.get('audio_end_sec'),
                )
                self.save_hdf5(paired, h5_path)
                summary_rows.append({
                    'subject_id': sid, 'task': task_name,
                    'status': 'ok', 'n_frames': paired.dataframe.shape[0],
                    'n_features': paired.dataframe.shape[1],
                    'duration_sec': paired.metadata['audio_duration_sec'],
                })

            except Exception as e:
                logger.error(f"[FAIL] {sid}/{task_name}: {e}")
                summary_rows.append({
                    'subject_id': sid, 'task': task_name,
                    'status': f'error: {e}', 'n_frames': None,
                })

        return pd.DataFrame(summary_rows)