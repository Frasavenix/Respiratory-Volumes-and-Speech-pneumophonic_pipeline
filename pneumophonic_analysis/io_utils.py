"""
Input/Output module for the pneumophonic pipeline.

Handles loading of:
- .dat : OEP data (volumes, sync)
- .wav : Audio signals
- .xlsx : Excel files for results and timing

Author: Pipeline based on the work of Bianca Zocco (2025)
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

from .config import PipelineConfig, get_config


class DataLoader:
    """
    Class for loading subject data.
    
    Example usage:
    ```python
    loader = DataLoader(subject_folder="20260218_GaBa")
    audio, sr = loader.load_audio("a.wav")
    sync, sr_sync = loader.load_sync_signal()
    oep_data = loader.load_oep_data("csv/GaBaVocali.csv")
    ```
    """
    
    def __init__(
        self,
        subject_folder: Union[str, Path],
        config: Optional[PipelineConfig] = None,
        renders_subfolder: str = "renders"
    ):
        """
        Initializes the loader for a subject.
        
        Args:
            subject_folder: Path to the subject folder (e.g., "20260218_GaBa")
            config: Pipeline configuration (uses DEFAULT_CONFIG if None)
            renders_subfolder: Name of the subfolder containing rendered audio files
        """
        self.config = config or get_config()
        self.subject_folder = Path(subject_folder)
        self.renders_folder = self.subject_folder / renders_subfolder
        
        # Extract subject ID from folder name
        # Expected format: "YYYYMMDD_SubjectID"
        folder_name = self.subject_folder.name
        parts = folder_name.split('_')
        self.subject_id = parts[1] if len(parts) > 1 else folder_name
        self.date = parts[0] if len(parts) > 1 else None
        
    def load_audio(
        self,
        filename: str,
        sr: Optional[int] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Loads an audio file.
        
        Args:
            filename: File name (searched in renders_folder then subject_folder)
            sr: Target sampling rate (None = original, default = config)
            normalize: Normalize amplitude between -1 and 1
            
        Returns:
            Tuple (audio signal, sample rate)
            
        Raises:
            FileNotFoundError: If the file does not exist
        """
        sr = sr or self.config.audio.sample_rate
        
        # Search for the file
        audio_path = self._find_file(filename, [self.renders_folder, self.subject_folder])
        
        # Load with librosa
        audio, sample_rate = librosa.load(str(audio_path), sr=sr)
        
        if normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
                
        return audio, sample_rate
    
    def load_sync_signal(
        self,
        filename: str = "sync_signal.wav",
        sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Loads the synchronization signal.
        
        Args:
            filename: File name of the sync signal (default: "sync_signal.wav")
            sr: Target sampling rate (None = original, default = config)
            
        Returns:
            Tuple (signal sync, sample rate)
        """
        return self.load_audio(filename, sr=sr, normalize=True)
    
    def load_oep_data(
        self,
        csv_path: str,
        fs_dat: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Loads OEP data from a .csv/.dat file.
        
        Args:
            csv_path: Relative path to the CSV file
            fs_dat: Sampling frequency of the data (for verification)
            
        Returns:
            DataFrame with the standard OEP columns
        """
        full_path = self.subject_folder / csv_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"File OEP not found: {full_path}")
        
        # Load with the standard column names
        df = pd.read_csv(
            full_path,
            sep=' ',
            names=self.config.oep.dat_columns,
            index_col=False
        )
        
        # Verify temporal consistency if fs_dat is provided
        if fs_dat is not None and len(df) > 1:
            dt_expected = 1 / fs_dat
            dt_actual = df['time'].iloc[1] - df['time'].iloc[0]
            if not np.isclose(dt_actual, dt_expected, rtol=0.1):
                print(f"⚠️  Attention: fs_dat expected={fs_dat}Hz, "
                      f"computed ={1/dt_actual:.1f}Hz")
        
        return df
    
    def load_timing_excel(
        self,
        sheet_name: str = "Timing"
    ) -> pd.DataFrame:
        """
        Loads timing data from the subject's Excel file.
        
        Args:
            sheet_name: Name of the sheet containing the timings
            
        Returns:
            DataFrame with the timings per task
        """
        excel_path = self.subject_folder / f"{self.subject_id}_audio.xlsx"
        
        if not excel_path.exists():
            raise FileNotFoundError(f"File timing not found: {excel_path}")
        
        return pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    
    def get_task_timing(
        self,
        task_name: str,
        timing_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Retrieves the timings for a specific task.
        
        Args:
            task_name: Name of the task (e.g., "a", "r", "phrase_1")
            timing_df: DataFrame of the timings (automatically loaded if None)
        
        Returns:
            Dict with 't_start', 't_stop', and other available timings
        """
        if timing_df is None:
            timing_df = self.load_timing_excel()
        
        # Search for the corresponding row
        mask = timing_df[0].astype(str).str.lower() == task_name.lower()
        row = timing_df[mask]
        
        if row.empty:
            raise ValueError(f"Task '{task_name}' not found in the timings")
        
        row = row.iloc[0]
        
        result = {
            't_start': float(row.iloc[1]) if pd.notna(row.iloc[1]) else None,
            't_stop': float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else None,
        }
        
        # Add other columns if present
        for i in range(3, len(row)):
            if pd.notna(row.iloc[i]):
                result[f'col_{i}'] = row.iloc[i]
        
        return result
    
    def list_audio_files(self, pattern: str = "*.wav") -> List[Path]:
        """List all audio files in the renders folder."""
        return list(self.renders_folder.glob(pattern))
    
    def _find_file(self, filename: str, search_paths: List[Path]) -> Path:
        """Search for a file in multiple paths."""
        for path in search_paths:
            full_path = path / filename
            if full_path.exists():
                return full_path
        
        # If the filename is already an absolute or relative path that exists
        if Path(filename).exists():
            return Path(filename)
        
        raise FileNotFoundError(
            f"File '{filename}' not found in: {[str(p) for p in search_paths]}"
        )


class ResultsWriter:
    """
    Class for saving analysis results.
    
    Example usage:
    ```python
    writer = ResultsWriter("results/subject_analysis.xlsx")
    writer.write_metrics(df_metrics, sheet="Metrics")
    writer.write_formants(df_formants, sheet="Formants")
    writer.close()
    ```
    """
    
    def __init__(
        self,
        output_path: Union[str, Path],
        config: Optional[PipelineConfig] = None
    ):
        """
        Initialize the writer.
        
        Args:
            output_path: Path to the output file (.xlsx)
            config: Pipeline configuration
        """
        self.config = config or get_config()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = None
        
    def write_dataframe(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        index: bool = True,
        startrow: int = 0,
        header: bool = True
    ):
        """
        Writes a DataFrame to an Excel sheet.
        
        Args:
            df: DataFrame to write
            sheet_name: Name of the sheet
            index: Include the index
            startrow: Starting row
            header: Include headers
        """
        mode = 'a' if self.output_path.exists() else 'w'
        
        with pd.ExcelWriter(
            self.output_path,
            engine=self.config.output.excel_engine,
            mode=mode,
            if_sheet_exists='overlay' if mode == 'a' else None
        ) as writer:
            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=index,
                startrow=startrow,
                header=header
            )
    
    def append_row(
        self,
        df_row: pd.DataFrame,
        sheet_name: str,
        row_index: int
    ):
        """
        Adds a row to an existing sheet.
        
        Args:
            df_row: DataFrame with a single row
            sheet_name: Name of the sheet
            row_index: Index of the row (1-based, like Excel)
        """
        self.write_dataframe(
            df_row,
            sheet_name=sheet_name,
            index=False,
            startrow=row_index - 1,  # Conversion to 0-based
            header=False
        )


def save_audio(
    audio: np.ndarray,
    path: Union[str, Path],
    sr: int,
    normalize: bool = True
):
    """
    Saves an audio signal to a WAV file.
    
    Args:
        audio: Audio signal (1D numpy array)
        path: Output path
        sr: Sampling rate
        normalize: Normalize before saving
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if normalize:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    
    sf.write(str(path), audio, sr)


def discover_subjects(
    data_root: Union[str, Path],
    pattern: str = "*_*"
) -> List[Path]:
    """
    Discovers all subject folders in a directory.
    
    Args:
        data_root: Root directory containing subject folders
        pattern: Glob pattern for folders (default: "DATE_ID" format)
        
    Returns:
        List of paths to subject folders
    """
    data_root = Path(data_root)
    
    subjects = []
    for folder in sorted(data_root.glob(pattern)):
        if folder.is_dir():
            # Verify that this is a subject folder (contains renders/)
            if (folder / "renders").exists() or (folder / "sync_signal.wav").exists():
                subjects.append(folder)
    
    return subjects


def load_master_excel(
    path: Union[str, Path],
    sheet_name: str = "cross"
) -> pd.DataFrame:
    """
    Loads the master Excel file containing data for all subjects.

    Args:
        path: Path to the master Excel file
        sheet_name: Name of the sheet to load

    Returns:
        DataFrame with the master data
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------------------------
# Interactive folder selection (GUI file browser)
# ---------------------------------------------------------------------------

def _looks_like_subject_folder(folder: Path) -> bool:
    """True if the folder has the layout the extractor expects (renders/ or csv/)."""
    return (folder / "renders").exists() or (folder / "csv").exists()


def _select_folders_console(title: str) -> List[Path]:
    """Fallback used when no GUI is available: type/paste folder paths."""
    print(f"\n[No file browser available] {title}")
    print("Paste one folder path per line (blank line to finish):")
    out: List[Path] = []
    while True:
        try:
            line = input("  folder> ").strip().strip('"').strip("'")
        except EOFError:
            break
        if not line:
            break
        p = Path(line)
        if p.exists() and p.is_dir():
            out.append(p)
        else:
            print(f"    not a directory: {p}")
    return out


def select_folders_gui(
    title: str = "Select folder(s)",
    initialdir: Optional[Union[str, Path]] = None,
    multiple: bool = True,
) -> List[Path]:
    """
    Open a native folder browser and return the chosen folder(s).

    The raw datasets no longer need to live inside ``data_root/``: this lets the
    user point the pipeline at wherever the subject folders are stored on disk.

    Behaviour:
        * ``multiple=True``  -> the dialog reopens after each pick so several
          folders can be chosen; press **Cancel** to finish.
        * ``multiple=False`` -> a single pick is returned.

    Falls back to a console path prompt if Tkinter / a display is unavailable.

    Args:
        title:      Dialog window title.
        initialdir: Folder the browser opens in (defaults to the user home).
        multiple:   Allow selecting more than one folder.

    Returns:
        List of selected folder paths (empty list if the user cancels immediately).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return _select_folders_console(title)

    if initialdir is not None:
        initialdir = str(initialdir)

    folders: List[Path] = []
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        while True:
            n = len(folders) + 1
            t = f"{title}  (#{n} — Cancel when done)" if multiple else title
            sel = filedialog.askdirectory(
                title=t,
                initialdir=initialdir,
                mustexist=True,
            )
            if not sel:          # user pressed Cancel / closed the dialog
                break
            folders.append(Path(sel))
            # Next dialog opens where the last pick was
            initialdir = str(Path(sel).parent)
            if not multiple:
                break
        root.destroy()
    except Exception as e:
        if not folders:
            print(f"  (file browser failed: {e})")
            return _select_folders_console(title)

    return folders


def select_subject_folders_gui(
    title: str = "Select subject folder(s) to extract",
    initialdir: Optional[Union[str, Path]] = None,
    multiple: bool = True,
    warn_non_subject: bool = True,
) -> List[Path]:
    """
    Folder browser tuned for picking subject folders for paired extraction.

    Same as :func:`select_folders_gui` but emits a warning for any chosen folder
    that does not look like a subject folder (missing ``renders/`` and ``csv/``).
    """
    folders = select_folders_gui(title=title, initialdir=initialdir, multiple=multiple)
    if warn_non_subject:
        for f in folders:
            if not _looks_like_subject_folder(f):
                print(f"  ⚠️  {f.name}: no renders/ or csv/ subfolder — may not be a subject folder")
    return folders


def select_file_gui(
    title: str = "Select a file",
    filetypes: Optional[List[Tuple[str, str]]] = None,
    initialdir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    Open a native file browser and return the chosen file (or None if cancelled).

    Falls back to a console path prompt if Tkinter / a display is unavailable.

    Args:
        title:      Dialog window title.
        filetypes:  List of (label, pattern) tuples, e.g. [("WAV", "*.wav")].
        initialdir: Folder the browser opens in.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        print(f"\n[No file browser available] {title}")
        try:
            line = input("  file path> ").strip().strip('"').strip("'")
        except EOFError:
            return None
        p = Path(line)
        return p if (line and p.exists() and p.is_file()) else None

    if initialdir is not None:
        initialdir = str(initialdir)
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        sel = filedialog.askopenfilename(
            title=title,
            initialdir=initialdir,
            filetypes=filetypes or [("All files", "*.*")],
        )
        root.destroy()
    except Exception as e:
        print(f"  (file browser failed: {e})")
        return None
    return Path(sel) if sel else None
