#!/usr/bin/env python3
"""
Robust PCA-CCA analysis pipeline.

Main features:
- Separate static and dynamic analysis modes.
- Optional speech sensitivity analysis: dynamic vs absolute.
- Out-of-sample canonical correlation via time-series cross-validation.
- Phase-randomized surrogate testing for dynamic tasks.
- Fixed 55-year age split with YM / EM / YF / EF groups.
- Descriptive feature-weight export, not causal interpretation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import hilbert
from scipy.stats import ttest_ind, ttest_rel
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Ensure this import matches your local project structure.
from pneumophonic_analysis.paired_features import PairedFeatureExtractor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


@dataclass(frozen=True)
class Config:
    project_root: Path
    data_target: Path
    metadata_path: Path
    batches: Tuple[str, ...] = ("healthy_subjects", "pathological_subjects")
    age_threshold: int = 55
    min_frames: int = 200
    frame_rate_approx: float = 66.7
    n_surrogates: int = 100
    requested_cv_splits: int = 3
    pca_variance_threshold: float = 0.95
    max_cca_components: int = 2
    random_state: int = 42
    sensitivity_analysis: bool = True


TASK_CATEGORIES = {
    "SUSTAINED": {"a", "a_2", "a_3", "a_7", "r"},
    "SPEECH": {"f_1", "f_2", "f_3", "f_4", "f_5", "testo"},
    "VOWEL": {"a", "e", "i", "o", "u"},
}

PRIMARY_ANALYSIS_MODE = {
    "SUSTAINED": "absolute",
    "VOWEL": "absolute",
    "SPEECH": "dynamic",
}

DEMOGRAPHIC_ORDER = ["YM", "EM", "YF", "EF"]


def build_config() -> Config:
    project_root = Path(__file__).resolve().parent.parent
    return Config(
        project_root=project_root,
        data_target=project_root / "data_target",
        metadata_path=project_root / "personal_informations.xlsx",
    )


def save_table_as_pdf(df: pd.DataFrame, title: str, filepath: Path) -> None:
    """
    Render a dataframe as a clean PDF table with dynamic sizing to prevent overflow.
    """
    n_cols = len(df.columns)
    n_rows = len(df)

    # 1. Dynamic Figure Size Allocation
    # Base width is 12 inches, but allocates ~1.8 inches per column if the table is wide
    fig_w = max(12.0, n_cols * 1.8)
    # Give vertical breathing room, accounting for header and title padding
    fig_h = max(2.5, 0.6 * (n_rows + 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # 2. Render Table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )

    # Disable automatic font sizing to maintain strict control over readability
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # 3. Dynamic Column Auto-sizing
    # This forces Matplotlib to read the longest string in each column 
    # and adjust the cell width to strictly contain the text.
    table.auto_set_column_width(col=list(range(n_cols)))

    # 4. Unidirectional Scaling
    # Scale ONLY the y-axis (height) by 1.8 for vertical readability.
    # Leaving the x-axis scale at 1.0 preserves the auto_set_column_width calculation.
    table.scale(1.0, 1.8)

    # 5. Aesthetic Formatting
    for (row, col), cell in table.get_celld().items():
        # Add slight internal padding to prevent text from touching cell borders
        cell.PAD = 0.05 
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#4C72B0")
        else:
            cell.set_edgecolor("#DDDDDD")

    # Adjust title padding to prevent overlap with the header row
    plt.title(title, weight="bold", size=14, pad=20)
    
    # Use tight_layout to snap the bounding box precisely to the rendered elements
    plt.tight_layout()
    
    # Save with high DPI to ensure crisp vector rendering
    fig.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close(fig)

def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction implemented without external dependencies."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return out

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    restored = np.empty_like(adj)
    restored[order] = adj
    out[valid] = restored
    return out


def load_metadata(path: Path) -> pd.DataFrame:
    """Load metadata and normalize column names."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        meta = pd.read_excel(path)
    else:
        meta = pd.read_csv(path, sep=",", engine="python", encoding="utf-8-sig")

    meta.columns = [str(c).strip().lower() for c in meta.columns]
    rename_map = {"id": "subject_id", "gender": "sex"}
    meta = meta.rename(columns={k: v for k, v in rename_map.items() if k in meta.columns})

    required = {"subject_id", "sex", "age"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata file is missing required columns: {sorted(missing)}")

    meta["subject_id"] = meta["subject_id"].astype(str).str.strip()
    meta["sex"] = meta["sex"].astype(str).str.upper().str.strip().str[0]
    meta["sex"] = meta["sex"].replace({"W": "F", "D": "F"})
    meta["age"] = pd.to_numeric(meta["age"], errors="coerce")

    meta = meta.dropna(subset=["subject_id", "sex", "age"]).reset_index(drop=True)
    meta = meta[meta["sex"].isin(["M", "F"])].copy()
    return meta


def assign_demographic_group(df: pd.DataFrame, age_threshold: int) -> pd.DataFrame:
    """Assign coarse demographic labels for descriptive summaries."""
    df = df.copy()
    df["age_group"] = np.where(df["age"] >= age_threshold, "Elder", "Young")
    label_map = {
        ("M", "Young"): "YM",
        ("M", "Elder"): "EM",
        ("F", "Young"): "YF",
        ("F", "Elder"): "EF",
    }
    df["demographic"] = [label_map.get((s, g), "Unknown") for s, g in zip(df["sex"], df["age_group"])]
    return df


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Select acoustic and respiratory feature columns that are present in the dataframe."""
    acoustic_candidates = ["f0", "energy", "spectral_centroid"] + [f"mfcc_{i}" for i in range(1, 14)]
    respiratory_candidates = ["flow_cw", "pct_rc", "phase_angle"]
    acoustic_cols = [c for c in acoustic_candidates if c in df.columns]
    respiratory_cols = [c for c in respiratory_candidates if c in df.columns]
    return acoustic_cols, respiratory_cols


def add_phase_angle(df: pd.DataFrame) -> pd.DataFrame:
    """Add the phase-angle feature if the required respiratory variables are available."""
    df = df.copy()
    if "pct_rc" in df.columns and "pct_ab" in df.columns:
        rc_centered = df["pct_rc"] - df["pct_rc"].mean()
        ab_centered = df["pct_ab"] - df["pct_ab"].mean()
        analytic_rc = hilbert(rc_centered.to_numpy(dtype=float))
        analytic_ab = hilbert(ab_centered.to_numpy(dtype=float))
        df["phase_angle"] = np.unwrap(np.angle(analytic_rc) - np.angle(analytic_ab))
    return df


def prepare_analysis_matrix(
    df: pd.DataFrame,
    analysis_mode: str,
    min_frames: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], List[str]]]:
    """Build X and Y matrices for one subject, preserving signal continuity."""
    
    # 1. DSP Corrected: calculate phase angle before filtering voiced frames
    df_continuous = add_phase_angle(df)

    acoustic_cols, respiratory_cols = get_feature_columns(df_continuous)
    if not acoustic_cols or not respiratory_cols:
        return None

    # 2. DSP Corrected: calculate derivatives on the continuous signal
    if analysis_mode == "dynamic":
        df_continuous[acoustic_cols] = df_continuous[acoustic_cols].diff()
        df_continuous[respiratory_cols] = df_continuous[respiratory_cols].diff()

    # 3. Only now apply the 'voiced' mask and discard the generated NaNs
    df_analysis = df_continuous[df_continuous["voiced"] == 1.0].copy()
    df_analysis = df_analysis.dropna(subset=acoustic_cols + respiratory_cols)

    if len(df_analysis) < min_frames:
        return None

    X = df_analysis[acoustic_cols].to_numpy(dtype=float)
    Y = df_analysis[respiratory_cols].to_numpy(dtype=float)

    # 4. Semantically correct: rename the extracted features
    if analysis_mode == "dynamic":
        acoustic_cols = [f"delta_{c}" for c in acoustic_cols]
        respiratory_cols = [f"delta_{c}" for c in respiratory_cols]

    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[mask]
    Y = Y[mask]
    if len(X) < max(20, min_frames // 4):
        return None

    return X, Y, acoustic_cols, respiratory_cols

def phase_randomize_matrix(Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Phase-randomize each column independently.

    This preserves the power spectrum while breaking the precise temporal alignment
    with the acoustic predictors.
    """
    Y = np.asarray(Y, dtype=float)
    n_samples, n_features = Y.shape
    surrogate = np.empty_like(Y)

    for j in range(n_features):
        x = Y[:, j] - np.mean(Y[:, j])
        fft = np.fft.rfft(x)
        mag = np.abs(fft)

        phases = rng.uniform(0.0, 2.0 * np.pi, size=len(fft))
        phases[0] = 0.0
        if n_samples % 2 == 0:
            phases[-1] = 0.0

        new_fft = mag * np.exp(1j * phases)
        new_fft[0] = fft[0].real + 0.0j
        if n_samples % 2 == 0:
            new_fft[-1] = fft[-1].real + 0.0j

        surrogate[:, j] = np.fft.irfft(new_fft, n=n_samples) + np.mean(Y[:, j])

    return surrogate


def get_time_series_splits(n_samples: int, requested_splits: int) -> Optional[TimeSeriesSplit]:
    """Return a safe TimeSeriesSplit object or None when the series is too short."""
    if n_samples < 30:
        return None

    n_splits = min(requested_splits, max(2, n_samples // 40))
    n_splits = min(n_splits, n_samples - 2)
    if n_splits < 2:
        return None
    return TimeSeriesSplit(n_splits=n_splits)


def pca_cca_time_series_cv(
    X: np.ndarray,
    Y: np.ndarray,
    requested_splits: int,
    variance_threshold: float,
    max_cca_components: int,
) -> Dict[str, object]:
    """Fit PCA and CCA inside each time-series fold and evaluate the held-out correlation."""
    splitter = get_time_series_splits(len(X), requested_splits)
    if splitter is None:
        return {"mean_test_corr": np.nan, "fold_test_corrs": [], "fold_models": []}

    fold_test_corrs: List[float] = []
    fold_models: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        if len(X_train) < 30 or len(X_test) < 20:
            continue

        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        Y_train_s = scaler_Y.fit_transform(Y_train)
        X_test_s = scaler_X.transform(X_test)
        Y_test_s = scaler_Y.transform(Y_test)

        pca_X = PCA(n_components=variance_threshold, svd_solver="full")
        pca_Y = PCA(n_components=variance_threshold, svd_solver="full")
        X_train_p = pca_X.fit_transform(X_train_s)
        Y_train_p = pca_Y.fit_transform(Y_train_s)
        X_test_p = pca_X.transform(X_test_s)
        Y_test_p = pca_Y.transform(Y_test_s)

        n_cca = min(X_train_p.shape[1], Y_train_p.shape[1], max_cca_components)
        if n_cca < 1:
            continue

        cca = CCA(n_components=n_cca, max_iter=2000)
        cca.fit(X_train_p, Y_train_p)
        X_test_c, Y_test_c = cca.transform(X_test_p, Y_test_p)
        if X_test_c.shape[1] == 0 or np.std(X_test_c[:, 0]) == 0 or np.std(Y_test_c[:, 0]) == 0:
            continue

        test_corr = float(np.corrcoef(X_test_c[:, 0], Y_test_c[:, 0])[0, 1])
        if np.isfinite(test_corr):
            fold_test_corrs.append(test_corr)

        fold_models.append(
            {
                "fold": fold_idx,
                "scaler_X": scaler_X,
                "scaler_Y": scaler_Y,
                "pca_X": pca_X,
                "pca_Y": pca_Y,
                "cca": cca,
                "test_corr": test_corr,
            }
        )

    mean_corr = float(np.mean(fold_test_corrs)) if fold_test_corrs else np.nan
    std_corr = float(np.std(fold_test_corrs, ddof=1)) if len(fold_test_corrs) > 1 else np.nan
    ci_low = float(np.percentile(fold_test_corrs, 2.5)) if len(fold_test_corrs) else np.nan
    ci_high = float(np.percentile(fold_test_corrs, 97.5)) if len(fold_test_corrs) else np.nan

    return {
        "mean_test_corr": mean_corr,
        "std_test_corr": std_corr,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "fold_test_corrs": fold_test_corrs,
        "fold_models": fold_models,
    }


def fit_full_pca_cca(
    X: np.ndarray,
    Y: np.ndarray,
    variance_threshold: float,
    max_cca_components: int,
) -> Optional[Dict[str, object]]:
    """Fit a final PCA-CCA model on the full dataset for interpretability."""
    if len(X) < 20:
        return None

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    Y_s = scaler_Y.fit_transform(Y)

    pca_X = PCA(n_components=variance_threshold, svd_solver="full")
    pca_Y = PCA(n_components=variance_threshold, svd_solver="full")
    X_p = pca_X.fit_transform(X_s)
    Y_p = pca_Y.fit_transform(Y_s)

    n_cca = min(X_p.shape[1], Y_p.shape[1], max_cca_components)
    if n_cca < 1:
        return None

    cca = CCA(n_components=n_cca, max_iter=2000)
    cca.fit(X_p, Y_p)
    X_c, Y_c = cca.transform(X_p, Y_p)
    full_corr = float(np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1])

    return {
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "pca_X": pca_X,
        "pca_Y": pca_Y,
        "cca": cca,
        "full_corr": full_corr,
        "X_scores": X_c,
        "Y_scores": Y_c,
    }


def backproject_feature_weights(
    pca: PCA,
    cca_weights: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, object]]:
    """Back-project latent weights to the original feature space."""
    latent_weights = cca_weights[:, 0] if cca_weights.ndim == 2 else cca_weights
    raw_weights = np.abs(pca.components_.T @ latent_weights)
    if np.sum(raw_weights) <= 0:
        raw_weights = np.ones_like(raw_weights, dtype=float)

    norm_weights = raw_weights / np.sum(raw_weights)
    return [{"feature": f, "weight": float(w)} for f, w in zip(feature_names, norm_weights)]


def empirical_p_value(real_value: float, surrogate_values: List[float]) -> float:
    """One-sided empirical p-value against a surrogate distribution."""
    surrogate_values = np.asarray(surrogate_values, dtype=float)
    surrogate_values = surrogate_values[np.isfinite(surrogate_values)]
    if len(surrogate_values) == 0 or not np.isfinite(real_value):
        return np.nan
    return float((np.sum(surrogate_values >= real_value) + 1.0) / (len(surrogate_values) + 1.0))


def cohen_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for two independent samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def cohen_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for paired samples."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def summarize_group_labels(df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    """Generate descriptive statistics by coarse demographic labels."""
    if "demographic" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("demographic")[outcome_col]
        .agg(["count", "mean", "std"])
        .reset_index()
        .sort_values("demographic")
    )


def compare_demographic_groups(df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    """Run the four requested Welch comparisons on the coarse demographic groups."""
    comparisons = [
        ("YM", "EM", "Age Effect (Males)"),
        ("YF", "EF", "Age Effect (Females)"),
        ("YM", "YF", "Sex Effect (Young)"),
        ("EM", "EF", "Sex Effect (Elder)"),
    ]

    rows: List[Dict[str, object]] = []
    for group1, group2, label in comparisons:
        data1 = df[df["demographic"] == group1][outcome_col].dropna().to_numpy(dtype=float)
        data2 = df[df["demographic"] == group2][outcome_col].dropna().to_numpy(dtype=float)
        n1, n2 = len(data1), len(data2)

        if n1 > 1 and n2 > 1:
            t_stat, p_val = ttest_ind(data1, data2, equal_var=False)
            d_val = cohen_d_independent(data1, data2)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            d_mag = "Large" if abs(d_val) >= 0.8 else "Medium" if abs(d_val) >= 0.5 else "Small"
            rows.append(
                {
                    "Comparison": label,
                    "Groups": f"{group1} ({n1}) vs {group2} ({n2})",
                    "Group 1": group1,
                    "Group 2": group2,
                    "Mean G1": round(float(np.mean(data1)), 3),
                    "Mean G2": round(float(np.mean(data2)), 3),
                    "t-stat": float(t_stat),
                    "p-value": float(p_val),
                    "Sig.": sig,
                    "Cohen's d": round(float(d_val), 2),
                    "Effect Size": d_mag,
                }
            )

    return pd.DataFrame(rows)


def apply_fdr_to_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Benjamini-Hochberg correction to the p-values."""
    if df.empty:
        return df
    work = df.copy()
    work["p_fdr"] = bh_fdr(work["p-value"].to_numpy(dtype=float))
    work["significant_fdr"] = work["p_fdr"] < 0.05
    return work


def compare_speech_modes(results_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compare dynamic vs absolute modes on speech subjects only."""
    speech = results_df[results_df["category_name"] == "SPEECH"].copy()
    if speech.empty:
        return pd.DataFrame()

    pivot = speech.pivot_table(
        index=["batch_name", "subject_id"],
        columns="analysis_mode",
        values="cca_r_cv_mean",
        aggfunc="mean",
    ).reset_index()

    if "dynamic" not in pivot.columns or "absolute" not in pivot.columns:
        return pd.DataFrame()

    paired = pivot.dropna(subset=["dynamic", "absolute"]).copy()
    if len(paired) < 3:
        return pd.DataFrame()

    t_stat, p_val = ttest_rel(paired["dynamic"], paired["absolute"])
    d_val = cohen_d_paired(paired["dynamic"].to_numpy(), paired["absolute"].to_numpy())

    summary = pd.DataFrame(
        [
            {
                "n_subjects": int(len(paired)),
                "mean_dynamic": float(paired["dynamic"].mean()),
                "mean_absolute": float(paired["absolute"].mean()),
                "mean_difference": float((paired["dynamic"] - paired["absolute"]).mean()),
                "paired_t_stat": float(t_stat),
                "p_value": float(p_val),
                "cohen_d_paired": float(d_val),
            }
        ]
    )
    summary.to_csv(output_dir / "Speech_Mode_Comparison.csv", index=False, sep=";")

    long_df = paired.melt(
        id_vars=["batch_name", "subject_id"],
        value_vars=["dynamic", "absolute"],
        var_name="analysis_mode",
        value_name="cca_r_cv_mean",
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=long_df, x="analysis_mode", y="cca_r_cv_mean", ax=ax)
    sns.stripplot(data=long_df, x="analysis_mode", y="cca_r_cv_mean", ax=ax, color="black", alpha=0.5)
    ax.set_title("Speech sensitivity analysis: dynamic vs absolute")
    ax.set_xlabel("Analysis mode")
    ax.set_ylabel("Mean CV canonical correlation")
    plt.tight_layout()
    fig.savefig(output_dir / "Speech_Mode_Comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    return summary


def main() -> None:
    config = build_config()
    metadata_df = assign_demographic_group(load_metadata(config.metadata_path), config.age_threshold)
    metadata_lookup = metadata_df.set_index("subject_id").to_dict(orient="index")

    for batch_name in config.batches:
        paired_dir = config.data_target / batch_name / "paired"
        if not paired_dir.exists():
            logger.warning("Skipping missing folder: %s", paired_dir)
            continue

        h5_files = sorted(paired_dir.glob("*.h5"))
        fig_dir = config.data_target / batch_name / "figures" / "multivariate" / "pca_cca"
        fig_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing batch %s (%d files)", batch_name, len(h5_files))

        subject_rows: List[Dict[str, object]] = []
        weight_rows: List[Dict[str, object]] = []

        for category_name, task_set in TASK_CATEGORIES.items():
            logger.info("Category: %s", category_name)
            primary_mode = PRIMARY_ANALYSIS_MODE[category_name]
            modes_to_run = [primary_mode]
            if config.sensitivity_analysis and category_name == "SPEECH":
                modes_to_run = ["dynamic", "absolute"]

            for h5_path in h5_files:
                try:
                    df, meta = PairedFeatureExtractor.load_hdf5(h5_path)
                except Exception as exc:
                    logger.warning("Could not load %s: %s", h5_path.name, exc)
                    continue

                if meta.get("task_name") not in task_set:
                    continue

                subject_id = str(meta.get("subject_id", h5_path.stem.split("_")[0])).strip()
                meta_row = metadata_lookup.get(subject_id)
                if not meta_row:
                    continue

                age = float(meta_row.get("age", np.nan))
                sex = str(meta_row.get("sex", "")).upper().strip()
                demographic = str(meta_row.get("demographic", "Unknown"))

                for analysis_mode in modes_to_run:
                    matrix = prepare_analysis_matrix(df=df, analysis_mode=analysis_mode, min_frames=config.min_frames)
                    if matrix is None:
                        continue

                    X, Y, acoustic_cols, respiratory_cols = matrix
                    if len(X) < 30:
                        continue

                    cv_res = pca_cca_time_series_cv(
                        X=X,
                        Y=Y,
                        requested_splits=config.requested_cv_splits,
                        variance_threshold=config.pca_variance_threshold,
                        max_cca_components=config.max_cca_components,
                    )
                    if not np.isfinite(cv_res["mean_test_corr"]):
                        continue

                    full_model = fit_full_pca_cca(
                        X=X,
                        Y=Y,
                        variance_threshold=config.pca_variance_threshold,
                        max_cca_components=config.max_cca_components,
                    )
                    full_corr = full_model["full_corr"] if full_model is not None else np.nan

                    surrogate_mean = np.nan
                    surrogate_ci_low = np.nan
                    surrogate_ci_high = np.nan
                    surrogate_p = np.nan

                    if analysis_mode == "dynamic" and primary_mode == "dynamic":
                        rng = np.random.default_rng(config.random_state)
                        surrogate_values: List[float] = []
                        for s_idx in range(config.n_surrogates):
                            if s_idx % 10 == 0:
                                logger.info("Surrogate %d/%d for %s / %s / %s", s_idx + 1, config.n_surrogates, batch_name, category_name, subject_id)
                            Y_surr = phase_randomize_matrix(Y, rng)
                            surr_cv = pca_cca_time_series_cv(
                                X=X,
                                Y=Y_surr,
                                requested_splits=config.requested_cv_splits,
                                variance_threshold=config.pca_variance_threshold,
                                max_cca_components=config.max_cca_components,
                            )
                            surrogate_values.append(float(surr_cv["mean_test_corr"]))

                        surrogate_values_np = np.asarray(surrogate_values, dtype=float)
                        surrogate_values_np = surrogate_values_np[np.isfinite(surrogate_values_np)]
                        if len(surrogate_values_np):
                            surrogate_mean = float(np.mean(surrogate_values_np))
                            surrogate_ci_low = float(np.percentile(surrogate_values_np, 2.5))
                            surrogate_ci_high = float(np.percentile(surrogate_values_np, 97.5))
                            surrogate_p = empirical_p_value(cv_res["mean_test_corr"], surrogate_values_np.tolist())

                    subject_rows.append(
                        {
                            "batch_name": batch_name,
                            "category_name": category_name,
                            "analysis_mode": analysis_mode,
                            "subject_id": subject_id,
                            "age": age,
                            "sex": sex,
                            "demographic": demographic,
                            "n_frames": int(len(X)),
                            "n_acoustic_features": int(len(acoustic_cols)),
                            "n_respiratory_features": int(len(respiratory_cols)),
                            "cca_r_cv_mean": float(cv_res["mean_test_corr"]),
                            "cca_r_cv_std": float(cv_res["std_test_corr"]) if np.isfinite(cv_res["std_test_corr"]) else np.nan,
                            "cca_r_cv_ci_low": float(cv_res["ci_low"]) if np.isfinite(cv_res["ci_low"]) else np.nan,
                            "cca_r_cv_ci_high": float(cv_res["ci_high"]) if np.isfinite(cv_res["ci_high"]) else np.nan,
                            "cca_r_full": float(full_corr) if np.isfinite(full_corr) else np.nan,
                            "surrogate_mean": surrogate_mean,
                            "surrogate_ci_low": surrogate_ci_low,
                            "surrogate_ci_high": surrogate_ci_high,
                            "surrogate_p": surrogate_p,
                        }
                    )

                    if analysis_mode == primary_mode and full_model is not None:
                        weight_x = backproject_feature_weights(
                            pca=full_model["pca_X"],
                            cca_weights=full_model["cca"].x_weights_,
                            feature_names=acoustic_cols,
                        )
                        weight_y = backproject_feature_weights(
                            pca=full_model["pca_Y"],
                            cca_weights=full_model["cca"].y_weights_,
                            feature_names=respiratory_cols,
                        )
                        for row in weight_x:
                            weight_rows.append(
                                {
                                    "batch_name": batch_name,
                                    "category_name": category_name,
                                    "analysis_mode": analysis_mode,
                                    "subject_id": subject_id,
                                    "domain": "Acoustic",
                                    "feature": row["feature"],
                                    "weight": row["weight"],
                                }
                            )
                        for row in weight_y:
                            weight_rows.append(
                                {
                                    "batch_name": batch_name,
                                    "category_name": category_name,
                                    "analysis_mode": analysis_mode,
                                    "subject_id": subject_id,
                                    "domain": "Respiratory",
                                    "feature": row["feature"],
                                    "weight": row["weight"],
                                }
                            )

        if not subject_rows:
            logger.warning("No valid PCA-CCA results for batch %s", batch_name)
            continue

        results_df = pd.DataFrame(subject_rows)
        weights_df = pd.DataFrame(weight_rows)

        results_df.to_csv(fig_dir / f"PCA_CCA_Subject_Results_{batch_name}.csv", index=False, sep=";")
        if not weights_df.empty:
            weights_df.to_csv(fig_dir / f"PCA_CCA_Feature_Weights_{batch_name}.csv", index=False, sep=";")

        # Safe filter with apply to avoid silently dropping rows due to map/indexing issues
        primary_mask = results_df.apply(
            lambda row: row["analysis_mode"] == PRIMARY_ANALYSIS_MODE.get(row["category_name"]), axis=1
        )
        primary_df = results_df[primary_mask].copy()

        for category_name in primary_df["category_name"].dropna().unique():
            sub = primary_df[primary_df["category_name"] == category_name].copy()
            if len(sub) < 8:
                continue

            descriptive = summarize_group_labels(sub, outcome_col="cca_r_full")
            if not descriptive.empty:
                descriptive.to_csv(
                    fig_dir / f"Descriptive_Group_Summary_{batch_name}_{category_name}.csv",
                    index=False,
                    sep=";",
                )
                # Integration of PDF Render for descriptive summaries
                save_table_as_pdf(
                    descriptive,
                    f"Descriptive Summary - {category_name}",
                    fig_dir / f"Descriptive_Group_Summary_{batch_name}_{category_name}.pdf"
                )

            comparison_table = compare_demographic_groups(sub, outcome_col="cca_r_full")
            comparison_table = apply_fdr_to_comparisons(comparison_table)
            if not comparison_table.empty:
                comparison_table.to_csv(
                    fig_dir / f"Demographic_Comparisons_{batch_name}_{category_name}.csv",
                    index=False,
                    sep=";",
                )
                # Integration of PDF Render for demographic comparisons
                save_table_as_pdf(
                    comparison_table,
                    f"Demographic Comparisons - {category_name}",
                    fig_dir / f"Demographic_Comparisons_{batch_name}_{category_name}.pdf"
                )

            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            sns.boxplot(data=sub, x="demographic", y="cca_r_full", order=DEMOGRAPHIC_ORDER, ax=ax)
            sns.stripplot(
                data=sub,
                x="demographic",
                y="cca_r_full",
                order=DEMOGRAPHIC_ORDER,
                ax=ax,
                color="black",
                alpha=0.35,
            )
            ax.set_title(f"Demographic groups - {batch_name} / {category_name}")
            ax.set_xlabel("Demographic group")
            ax.set_ylabel("Mean CV canonical correlation")
            plt.tight_layout()
            fig.savefig(fig_dir / f"Demographic_Groups_{batch_name}_{category_name}.pdf", bbox_inches="tight")
            plt.close(fig)

        if config.sensitivity_analysis:
            speech_summary = compare_speech_modes(results_df, fig_dir)
            if not speech_summary.empty:
                logger.info(
                    "Speech sensitivity summary for %s: mean dynamic=%.3f, mean absolute=%.3f",
                    batch_name,
                    speech_summary["mean_dynamic"].iloc[0],
                    speech_summary["mean_absolute"].iloc[0],
                )

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=results_df, x="category_name", y="cca_r_full", hue="analysis_mode", ax=ax)
        sns.stripplot(
            data=results_df,
            x="category_name",
            y="cca_r_full",
            hue="analysis_mode",
            dodge=True,
            ax=ax,
            color="black",
            alpha=0.25,
        )
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), title="Analysis mode", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_title(f"PCA-CCA cross-validated correlation - {batch_name}")
        ax.set_xlabel("Category")
        ax.set_ylabel("Mean test canonical correlation")
        plt.tight_layout()
        fig.savefig(fig_dir / f"PCA_CCA_Results_{batch_name}.pdf", bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved batch outputs to %s", fig_dir)

    logger.info("PCA-CCA processing complete.")


if __name__ == "__main__":
    main()
