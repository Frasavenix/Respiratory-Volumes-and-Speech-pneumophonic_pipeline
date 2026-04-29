"""
Main Pipeline for orchestrating pneumophonic analysis.

This module coordinates the complete analysis:
- Subject discovery
- Data loading
- OEP/Audio synchronization
- Task-based analysis
- Results export

Reference: Bianca Zocco MSc Thesis - Analysis Protocol
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from tqdm import tqdm

from .config import PipelineConfig, get_config, create_config
from .io_utils import DataLoader, ResultsWriter, discover_subjects, load_master_excel
from .sync import Synchronizer, SyncResult
from .audio_processing import AudioProcessor
from .acoustic_features import PraatAnalyzer
from .task_analyzers import (
    TaskResult, BaseTaskAnalyzer,
    VowelAnalyzer, PhraseAnalyzer, TrillAnalyzer, GlideAnalyzer,
    get_analyzer_for_task
)
from .visualization import Visualizer


#  Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SubjectAnalysis:
    """RResult of the complete analysis of a subject."""
    
    subject_id: str
    subject_folder: Path
    
    # Results by task
    results: Dict[str, TaskResult] = field(default_factory=dict)
    
    # Synchronization result (if applicable)
    sync_result: Optional[SyncResult] = None
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Combines all results into a single DataFrame."""
        dfs = []
        for task_name, result in self.results.items():
            df = result.to_dataframe()
            dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()


@dataclass
class BatchAnalysis:
    """Result of a batch analysis on multiple subjects."""
    
    subjects: Dict[str, SubjectAnalysis] = field(default_factory=dict)
    
    @property
    def n_subjects(self) -> int:
        return len(self.subjects)
    
    @property
    def n_successful(self) -> int:
        return sum(1 for s in self.subjects.values() if s.success)
    
    @property
    def n_failed(self) -> int:
        return self.n_subjects - self.n_successful
    
    def to_dataframe(self) -> pd.DataFrame:
        """Combines all results into a single DataFrame."""
        dfs = []
        for subject_id, analysis in self.subjects.items():
            df = analysis.to_dataframe()
            if not df.empty:
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def get_errors_summary(self) -> pd.DataFrame:
        """Returns a summary of all errors and warnings."""
        data = []
        for subject_id, analysis in self.subjects.items():
            for error in analysis.errors:
                data.append({
                    'subject_id': subject_id,
                    'type': 'error',
                    'message': error
                })
            for warning in analysis.warnings:
                data.append({
                    'subject_id': subject_id,
                    'type': 'warning',
                    'message': warning
                })
        return pd.DataFrame(data)


class PneumophonicPipeline:
    """
    Principal pipeline for pneumophonic analysis.
    
    Coordinates the complete analysis of one or more subjects,
    orchestrating the different modules of the package.
    
    Example usage:
    ```python
    # Configuration
    config = create_config(
        data_root=Path("/data/subjects"),
        output_root=Path("/results")
    )
    
    # Pipeline
    pipeline = PneumophonicPipeline(config)
    
    # Analyse d'un sujet
    result = pipeline.analyze_subject("20260218_GaBa")
    
    # Analyse batch
    batch_result = pipeline.analyze_all()
    
    # Export
    pipeline.export_results(batch_result, "all_results.xlsx")
    ```
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        data_root: Optional[Path] = None,
        output_root: Optional[Path] = None
    ):
        """
        Initializes the pipeline.
        
        Args:
            config: Configuration of the pipeline
            data_root: Root directory for the data (override config)
            output_root: Output directory (override config)
        """
        self.config = config or get_config()
        
        if data_root:
            self.config.data_root = Path(data_root)
        if output_root:
            self.config.output_root = Path(output_root)
        
        # Composants
        self.audio_processor = AudioProcessor(self.config)
        self.praat_analyzer = PraatAnalyzer(self.config)
        self.synchronizer = Synchronizer(self.config)
        self.visualizer = Visualizer(self.config)
        
        # Analyseurs par tâche
        self.analyzers: Dict[str, BaseTaskAnalyzer] = {
            'vowel': VowelAnalyzer(self.config),
            'phrase': PhraseAnalyzer(self.config),
            'trill': TrillAnalyzer(self.config),
            'glide': GlideAnalyzer(self.config),
        }
        
        logger.info(f"Pipeline initialized with configuration:")
        logger.info(f"  Data root: {self.config.data_root}")
        logger.info(f"  Output root: {self.config.output_root}")
    
    def discover_subjects(self) -> List[Path]:
        """
        Discovers available subjects.
        
        Returns:
            List of paths to subject folders
        """
        if self.config.data_root is None:
            raise ValueError("data_root not configured")
        
        subjects = discover_subjects(self.config.data_root)
        logger.info(f"Discovered {len(subjects)} subjects")
        
        return subjects
    
    def analyze_subject(
        self,
        subject_folder: Union[str, Path],
        tasks: Optional[List[str]] = None,
        timing_excel: Optional[Path] = None,
        save_figures: bool = False
    ) -> SubjectAnalysis:
        """
        Full analysis of a subject.
        
        Args:
            subject_folder: Path to the subject's folder
            tasks: List of tasks to analyze (None = all)
            timing_excel: Excel file with the timings
            save_figures: Save the figures
            
        Returns:
            SubjectAnalysis with all the results
        """
        subject_folder = Path(subject_folder)
        loader = DataLoader(subject_folder, self.config)
        
        analysis = SubjectAnalysis(
            subject_id=loader.subject_id,
            subject_folder=subject_folder
        )
        
        logger.info(f"Analyzing subject {loader.subject_id}")
        
        # Load the timings if available
        timing_data = None
        if timing_excel is not None:
            try:
                timing_data = pd.read_excel(timing_excel)
            except Exception as e:
                analysis.warnings.append(f"Impossible de charger les timings: {e}")
        
        # Synchronization (if sync_signal.wav exists)
        try:
            sync_audio, sr_sync = loader.load_sync_signal()
            # Note: the full sync requires also the OEP data
            logger.debug(f"Sync signal loaded: {len(sync_audio)} samples")
        except Exception as e:
            analysis.warnings.append(f"Sync signal error: {e}")
            sync_audio = None
        
        # Analyze each task
        tasks_to_analyze = tasks or ['vowel', 'phrase', 'trill', 'glide']
        
        for task_name in tasks_to_analyze:
            try:
                result = self._analyze_task(
                    loader, task_name, timing_data
                )
                if result:
                    analysis.results[task_name] = result
                    
                    if save_figures and self.config.output_root:
                        self._save_task_figure(result, loader.subject_id)
                        
            except Exception as e:
                error_msg = f"Error on task {task_name}: {str(e)}"
                analysis.errors.append(error_msg)
                logger.error(error_msg)
        
        return analysis
    
    def _analyze_task(
        self,
        loader: DataLoader,
        task_name: str,
        timing_data: Optional[pd.DataFrame] = None
    ) -> Optional[TaskResult]:
        """Analyze a specific task."""
        
        # Mapping of audio files by task (to be adapted based on actual data structure)
        task_files = {
            'vowel': ['a.wav', 'vocali_a.wav', 'AL.wav'],
            'phrase': ['phrase_1.wav', 'phrase_5.wav', 'frase_1.wav'],
            'trill': ['r.wav', 'R.wav', 'trill.wav'],
            'glide': ['glide.wav', 'AG.wav', 'phonema_a_7.wav'],
        }
        
        # Find the audio file for this task
        audio_file = None
        for candidate in task_files.get(task_name, []):
            try:
                audio, sr = loader.load_audio(candidate)
                audio_file = candidate
                break
            except FileNotFoundError:
                continue
        
        if audio_file is None:
            logger.warning(f"Audio file not found for {task_name}")
            return None
        
        # Get the analyzer
        analyzer = self.analyzers.get(task_name)
        if analyzer is None:
            analyzer = get_analyzer_for_task(task_name, self.config)
        
        # Additional parameters from timings
        kwargs = {}
        if timing_data is not None:
            # Look for the timings for this task
            # (Structure depends on the Excel file format)
            pass
        
        # Analyze
        result = analyzer.analyze(
            audio=audio,
            sr=sr,
            subject_id=loader.subject_id,
            **kwargs
        )
        
        logger.info(f"  {task_name}: {result.duration_sec:.2f}s")
        
        return result
    
    def _save_task_figure(self, result: TaskResult, subject_id: str):
        """Save the figure of a result."""
        if self.config.output_root is None:
            return
        
        fig = self.visualizer.plot_task_result(result)
        
        output_path = (
            self.config.output_root / 
            "figures" / 
            subject_id / 
            f"{result.task_name}.png"
        )
        
        self.visualizer.save_figure(fig, output_path)
    
    def analyze_batch(
        self,
        subjects: Optional[List[Path]] = None,
        tasks: Optional[List[str]] = None,
        progress: bool = True,
        stop_on_error: bool = False
    ) -> BatchAnalysis:
        """
        Analyze a batch of subjects.
        
        Args:
            subjects: List of subjects (None = all)
            tasks: Tasks to analyze
            progress: Display progress bar
            stop_on_error: Arrêter en cas d'erreur
            
        Returns:
            BatchAnalysis with all the results
        """
        if subjects is None:
            subjects = self.discover_subjects()
        
        batch = BatchAnalysis()
        
        iterator = tqdm(subjects, desc="Analyzing") if progress else subjects
        
        for subject_folder in iterator:
            try:
                analysis = self.analyze_subject(
                    subject_folder,
                    tasks=tasks
                )
                batch.subjects[analysis.subject_id] = analysis
                
            except Exception as e:
                import traceback
                # old : logger.error(f"Erreur sujet {subject_folder}: {e}")
                logger.error(f"Error with subject {subject_folder}:\n{traceback.format_exc()}")
                
                failed = SubjectAnalysis(
                    subject_id=subject_folder.name,
                    subject_folder=subject_folder,
                    errors=[str(e)]
                )
                batch.subjects[subject_folder.name] = failed
                if stop_on_error:
                    raise
        
        logger.info(f"Batch completed: {batch.n_successful}/{batch.n_subjects} successful")
        
        return batch
    
    def export_results(
        self,
        results: Union[SubjectAnalysis, BatchAnalysis],
        output_path: Union[str, Path],
        include_errors: bool = True
    ):
        """Exporte les résultats vers Excel."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_main = results.to_dataframe()
        df_errors = None
        if include_errors and isinstance(results, BatchAnalysis):
            df_errors = results.get_errors_summary()
        
        has_results = not df_main.empty
        has_errors = df_errors is not None and not df_errors.empty
        
        if not has_results and not has_errors:
            logger.warning("No results to export")
            df_summary = pd.DataFrame({
                'Status': ['No results'],
                'Message': ['All analyses failed or no subjects found']
            })
            df_summary.to_excel(output_path, sheet_name='Summary', index=False)
            return
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            if has_results:
                df_main.to_excel(writer, sheet_name='Results', index=False)
            if has_errors:
                df_errors.to_excel(writer, sheet_name='Errors', index=False)
        
        logger.info(f"Results exported: {output_path}")
    """
    def export_results(
        self,
        results: Union[SubjectAnalysis, BatchAnalysis],
        output_path: Union[str, Path],
        include_errors: bool = True
    ):
        
        #Export the results to Excel.
        
        #Args:
        #    results: RResults to export
        #    output_path: Path to the output file
        #    include_errors: Include the errors sheet
        #
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data
            df_main = results.to_dataframe()
            if not df_main.empty:
                df_main.to_excel(writer, sheet_name='Results', index=False)
            
            # Errors
            if include_errors and isinstance(results, BatchAnalysis):
                df_errors = results.get_errors_summary()
                if not df_errors.empty:
                    df_errors.to_excel(writer, sheet_name='Errors', index=False)
        
        logger.info(f"RResults exported: {output_path}")
    """
    def generate_report(
        self,
        results: BatchAnalysis,
        output_folder: Union[str, Path],
        include_figures: bool = True
    ):
        """
        Generate a complete report with figures.
        
        Args:
            results: RResults of the batch analysis
            output_folder: Output folder
            include_figures: Include the figures
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Export Excel
        self.export_results(results, output_folder / "results.xlsx")
        
        # Figures 
        if include_figures:
            df = results.to_dataframe()
            
            if not df.empty and 'pitch_mean_f0' in df.columns:
                fig = self.visualizer.plot_metrics_comparison(
                    df, 'pitch_mean_f0', 'subject_id', 'Mean F0 by Subject'
                )
                self.visualizer.save_figure(
                    fig, output_folder / "figures" / "mean_f0_comparison.png"
                )
        
        logger.info(f"Report generated: {output_folder}")


def run_pipeline(
    data_root: Union[str, Path],
    output_root: Union[str, Path],
    tasks: Optional[List[str]] = None,
    subjects: Optional[List[str]] = None
) -> BatchAnalysis:
    """
    Convenience function to run the complete pipeline with minimal code.
    
    Args:
        data_root: Root directory of the data
        output_root: Root directory of the output
        tasks: Tasks to analyze
        subjects: Specific subjects (None = all)
        
    Returns:
        BatchAnalysis
    
    Example usage  :
    ```python
    results = run_pipeline(
        data_root="/data/subjects",
        output_root="/results",
        tasks=['vowel', 'trill']
    )
    print(f"Analyzed {results.n_subjects} subjects")
    ```
    """
    config = create_config(
        data_root=Path(data_root),
        output_root=Path(output_root)
    )
    
    pipeline = PneumophonicPipeline(config)
    
    # Filter subjects if specified
    if subjects:
        subject_folders = [
            Path(data_root) / s for s in subjects
        ]
    else:
        subject_folders = None
    
    # Analyzer
    batch = pipeline.analyze_batch(subjects=subject_folders, tasks=tasks)
    
    # Export and report
    pipeline.generate_report(batch, output_root)
    
    return batch
