"""
Pneumophonic Analysis Pipeline
==============================

Package Python for integrated analysis of respiratory-phonatory functions.

Based on the thesis work:
"Integrated Analysis of Respiratory–Phonatory Functions: Normative Patterns Across Sex and Age"
Bianca Zocco, Politecnico di Milano, 2024-2025

Main modules
- config: Global pipeline configuration
- io_utils: File reading/writing (OEP, audio, Excel)
- sync: Synchronization between OEP and audio signals
- audio_processing: Audio signal processing (noise reduction, features)
- acoustic_features: Extraction of acoustic parameters via Praat
- segmentation: Signal segmentation (FRC, novelty, modulation)
- task_analyzers: Task-specific vocal analyzers
- visualization: Visualization of signals and results
- pipeline: Orchestration of the full pipeline


"""

__version__ = "1.0.0"
__author__ = "Pipeline based on work by Bianca Zocco - DEIB Thesis aa 2025/2026"
__email__ = ""

# Configuration
from .config import (
    PipelineConfig,
    AudioConfig,
    PitchConfig,
    OEPConfig,
    get_config,
    create_config,
    DEFAULT_CONFIG,
)

# I/O
from .io_utils import (
    DataLoader,
    ResultsWriter,
    discover_subjects,
    load_master_excel,
    save_audio,
)

# Synchronisation
from .sync import (
    Synchronizer,
    SyncResult,
    compute_relative_timing,
    detect_onset_in_phonation,
    detect_end_of_phonation,
)

# Audio processing
from .audio_processing import (
    AudioProcessor,
    AudioFeatures,
    to_db,
    compute_spectral_centroid,
    compute_rms_envelope,
)

# Acoustic features (Praat)
from .acoustic_features import (
    PraatAnalyzer,
    AcousticAnalysisResult,
    PitchMetrics,
    PerturbationMetrics,
    FormantMetrics,
    VoiceQualityMetrics,
    quick_analysis,
)

# Segmentation
from .segmentation import (
    FRCSegmenter,
    FRCSegment,
    GlideSegmenter,
    GlideSegment,
    ModulationAnalyzer,
    ModulationResult,
    detect_non_silent_intervals,
    detect_phonation_bounds,
)

# Task analyzers
from .task_analyzers import (
    TaskResult,
    BaseTaskAnalyzer,
    VowelAnalyzer,
    PhraseAnalyzer,
    TrillAnalyzer,
    GlideAnalyzer,
    get_analyzer_for_task,
)

# Visualization
from .visualization import (
    Visualizer,
    quick_plot_audio,
    quick_plot_spectrogram,
)

# Pipeline
from .pipeline import (
    PneumophonicPipeline,
    SubjectAnalysis,
    BatchAnalysis,
    run_pipeline,
)


# API (public)
__all__ = [
    # Version
    "__version__",
    
    # Configuration
    "PipelineConfig",
    "AudioConfig",
    "PitchConfig",
    "OEPConfig",
    "get_config",
    "create_config",
    "DEFAULT_CONFIG",
    
    # I/O
    "DataLoader",
    "ResultsWriter",
    "discover_subjects",
    "load_master_excel",
    "save_audio",
    
    # Sync
    "Synchronizer",
    "SyncResult",
    
    # Audio
    "AudioProcessor",
    "AudioFeatures",
    
    # Praat
    "PraatAnalyzer",
    "AcousticAnalysisResult",
    "quick_analysis",
    
    # Segmentation
    "FRCSegmenter",
    "FRCSegment",
    "GlideSegmenter",
    "GlideSegment",
    "ModulationAnalyzer",
    "ModulationResult",
    
    # Analyzers
    "TaskResult",
    "VowelAnalyzer",
    "PhraseAnalyzer",
    "TrillAnalyzer",
    "GlideAnalyzer",
    "get_analyzer_for_task",
    
    # Visualization
    "Visualizer",
    "quick_plot_audio",
    "quick_plot_spectrogram",
    
    # Pipeline
    "PneumophonicPipeline",
    "SubjectAnalysis",
    "BatchAnalysis",
    "run_pipeline",

    "PairedEffectSize",
    "compute_paired_effect_size",
    "median_absolute_deviation",
    "interpret_cohen_d",
]

# Paired features
from .paired_features import (
    PairedFeatureExtractor,
    PairedFrame,
    OEPFrameFeatures,
)

from .effect_size import (
    PairedEffectSize,
    compute_paired_effect_size,
    median_absolute_deviation,
    interpret_cohen_d,
)