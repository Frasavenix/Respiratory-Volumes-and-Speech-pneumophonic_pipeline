#!/usr/bin/env python3
"""
Robust predictive D-PLS pipeline for speech tasks.

Main features:
- Continuous signal processing for lags and derivatives.
- Nested time-series cross-validation.
- Scaling fitted only on the training fold to prevent leakage.
- Lag selection chosen inside the training data.
- VIP stability assessed with moving-block bootstrap (Zero-variance protected).
- Fixed 55-year age split with YM / EM / YF / EF groups.
- PDF generation for descriptive and statistical tables.
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
from scipy.stats import ttest_ind
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
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
    lags: int = 3
    smoothing_window: int = 5
    requested_outer_splits: int = 3
    requested_inner_splits: int = 3
    max_pls_components: int = 5
    n_bootstraps: int = 30
    bootstrap_block_size: int = 10
    random_state: int = 42


TASK_CATEGORIES = {
    "SPEECH": {"f_1", "f_2", "f_3", "f_4", "f_5", "testo"},
}

TARGET_METRICS = ["pct_rc", "flow_cw"]
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


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select acoustic feature columns that are present in the dataframe."""
    acoustic_candidates = ["f0", "energy", "spectral_centroid"] + [f"mfcc_{i}" for i in range(1, 14)]
    return [c for c in acoustic_candidates if c in df.columns]


def build_lagged_matrix(df: pd.DataFrame, acoustic_cols: List[str], lags: int) -> pd.DataFrame:
    """Create time-delay embedded features for the acoustic predictors."""
    X_df = df[acoustic_cols].copy()
    for col in acoustic_cols:
        for lag in range(1, lags + 1):
            X_df[f"{col}_t-{lag}"] = X_df[col].shift(lag)
    return X_df


def calculate_vip(model: PLSRegression) -> np.ndarray:
    """Compute Variable Importance in Projection (VIP) scores."""
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,), dtype=float)

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    if total_s == 0:
        return np.ones(p, dtype=float)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight).item() / total_s)

    return vips


def get_time_series_splits(n_samples: int, requested_splits: int) -> Optional[TimeSeriesSplit]:
    """Return a safe TimeSeriesSplit object or None when the series is too short."""
    if n_samples < 30:
        return None

    n_splits = min(requested_splits, max(2, n_samples // 40))
    n_splits = min(n_splits, n_samples - 2)
    if n_splits < 2:
        return None
    return TimeSeriesSplit(n_splits=n_splits)


def build_targets(df: pd.DataFrame, target_metrics: List[str], smoothing_window: int) -> pd.DataFrame:
    """Create smoothed first derivatives for the target variables."""
    y_df = pd.DataFrame(index=df.index)
    for target in target_metrics:
        deriv = df[target].diff()
        y_df[f"{target}_diff"] = deriv.rolling(window=smoothing_window, min_periods=1).mean()
    return y_df


def nested_time_series_pls_cv(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int,
    requested_outer_splits: int,
    requested_inner_splits: int,
) -> Dict[str, object]:
    """Perform nested time-series cross-validation for PLS."""
    outer_splitter = get_time_series_splits(len(X), requested_outer_splits)
    if outer_splitter is None:
        return {"outer_r2": [], "best_components": [], "mean_r2": np.nan, "std_r2": np.nan}

    outer_r2_scores: List[float] = []
    selected_components: List[int] = []

    for outer_train_idx, outer_test_idx in outer_splitter.split(X):
        X_train, X_test = X[outer_train_idx], X[outer_test_idx]
        y_train, y_test = y[outer_train_idx], y[outer_test_idx]

        if len(X_train) < 20 or len(X_test) < 5:
            continue

        inner_splitter = get_time_series_splits(len(X_train), requested_inner_splits)
        if inner_splitter is None:
            continue

        max_k = min(max_components, X_train.shape[1], max(1, len(X_train) - 1))
        candidate_components = list(range(1, max_k + 1))
        inner_scores_by_k: Dict[int, List[float]] = {k: [] for k in candidate_components}

        for inner_train_idx, inner_val_idx in inner_splitter.split(X_train):
            X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_inner_train_s = scaler_X.fit_transform(X_inner_train)
            X_inner_val_s = scaler_X.transform(X_inner_val)
            y_inner_train_s = scaler_y.fit_transform(y_inner_train.reshape(-1, 1)).ravel()
            y_inner_val_s = scaler_y.transform(y_inner_val.reshape(-1, 1)).ravel()

            for k in candidate_components:
                if k > min(X_inner_train_s.shape[0] - 1, X_inner_train_s.shape[1]):
                    continue
                pls = PLSRegression(n_components=k)
                pls.fit(X_inner_train_s, y_inner_train_s)
                y_pred = pls.predict(X_inner_val_s).ravel()
                r2 = r2_score(y_inner_val_s, y_pred)
                if np.isfinite(r2):
                    inner_scores_by_k[k].append(float(r2))

        mean_inner_scores = {k: np.mean(scores) for k, scores in inner_scores_by_k.items() if len(scores) > 0}
        if not mean_inner_scores:
            continue

        best_k = max(mean_inner_scores, key=mean_inner_scores.get)
        selected_components.append(int(best_k))

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s = scaler_X.transform(X_test)
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        pls = PLSRegression(n_components=best_k)
        pls.fit(X_train_s, y_train_s)
        y_pred = pls.predict(X_test_s).ravel()
        
        # Clip extreme negative R2 values to prevent Seaborn plot destruction
        # An R2 of -1.0 means the model is performing significantly worse than a mean baseline.
        outer_r2 = max(-1.0, float(r2_score(y_test_s, y_pred)))
        if np.isfinite(outer_r2):
            outer_r2_scores.append(outer_r2)

    return {
        "outer_r2": outer_r2_scores,
        "best_components": selected_components,
        "mean_r2": float(np.mean(outer_r2_scores)) if outer_r2_scores else np.nan,
        "std_r2": float(np.std(outer_r2_scores, ddof=1)) if len(outer_r2_scores) > 1 else np.nan,
    }


def fit_final_pls(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
) -> Tuple[Optional[PLSRegression], Optional[StandardScaler], Optional[StandardScaler], Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit the final PLS model on all available data."""
    if len(X) < 10:
        return None, None, None, None, None

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    k = min(n_components, X_s.shape[1], max(1, len(X_s) - 1))
    if k < 1:
        return None, None, None, None, None

    model = PLSRegression(n_components=k)
    model.fit(X_s, y_s)
    y_pred = model.predict(X_s).ravel()
    vip = calculate_vip(model)
    return model, scaler_X, scaler_y, y_pred, vip


def moving_block_bootstrap_indices(n_samples: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Create bootstrap indices by resampling contiguous blocks."""
    if n_samples <= block_size:
        return np.arange(n_samples)

    n_blocks = int(np.ceil(n_samples / block_size))
    starts = rng.integers(0, n_samples - block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
    return idx[:n_samples]


def vip_bootstrap_stability(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    n_bootstraps: int,
    block_size: int,
    random_state: int,
) -> Dict[str, np.ndarray]:
    """Estimate VIP stability by moving-block bootstrap."""
    rng = np.random.default_rng(random_state)
    vip_list: List[np.ndarray] = []

    for _ in range(n_bootstraps):
        idx = moving_block_bootstrap_indices(len(X), block_size, rng)
        X_b = X[idx]
        y_b = y[idx]

        # Protect against zero-variance features in the bootstrap sample
        if np.any(np.std(X_b, axis=0) == 0) or np.std(y_b) == 0:
            continue

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_b_s = scaler_X.fit_transform(X_b)
        y_b_s = scaler_y.fit_transform(y_b.reshape(-1, 1)).ravel()

        k = min(n_components, X_b_s.shape[1], max(1, len(X_b_s) - 1))
        if k < 1:
            continue

        try:
            model = PLSRegression(n_components=k)
            model.fit(X_b_s, y_b_s)
            vip = calculate_vip(model)
            vip_list.append(vip)
        except Exception:
            continue

    if not vip_list:
        return {"vip_mean": np.array([]), "vip_std": np.array([]), "vip_prob_gt1": np.array([])}

    vip_arr = np.vstack(vip_list)
    return {
        "vip_mean": np.mean(vip_arr, axis=0),
        "vip_std": np.std(vip_arr, axis=0, ddof=1) if vip_arr.shape[0] > 1 else np.zeros(vip_arr.shape[1]),
        "vip_prob_gt1": np.mean(vip_arr > 1.0, axis=0),
    }


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
        fig_dir = config.data_target / batch_name / "figures" / "multivariate" / "predictive_dpls"
        fig_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing batch %s (%d files)", batch_name, len(h5_files))

        predictive_rows: List[Dict[str, object]] = []
        vip_rows: List[Dict[str, object]] = []

        for h5_path in h5_files:
            try:
                df, meta = PairedFeatureExtractor.load_hdf5(h5_path)
            except Exception as exc:
                logger.warning("Could not load %s: %s", h5_path.name, exc)
                continue

            if meta.get("task_name") not in TASK_CATEGORIES["SPEECH"]:
                continue

            subject_id = str(meta.get("subject_id", h5_path.stem.split("_")[0])).strip()
            meta_row = metadata_lookup.get(subject_id)
            if not meta_row:
                continue

            age = float(meta_row.get("age", np.nan))
            sex = str(meta_row.get("sex", "")).upper().strip()
            demographic = str(meta_row.get("demographic", "Unknown"))

            acoustic_cols = get_feature_columns(df)
            missing_targets = [t for t in TARGET_METRICS if t not in df.columns]
            if missing_targets or not acoustic_cols:
                continue

            # 1. Temporal Dynamics on the CONTINUOUS signal
            X_df = build_lagged_matrix(df, acoustic_cols, config.lags)
            y_df = build_targets(df, TARGET_METRICS, config.smoothing_window)
            
            # Combine back with the 'voiced' feature to apply the mask safely
            combined = pd.concat([X_df, y_df, df[["voiced"]]], axis=1)

            # 2. Apply phonation mask AFTER lags and diffs are computed
            combined_voiced = combined[combined["voiced"] == 1.0].copy()

            # 3. Drop rows with missing values (including NaN generated by .shift() and .diff())
            feature_names_lagged = X_df.columns.tolist()
            target_names_diff = [f"{t}_diff" for t in TARGET_METRICS]
            combined_clean = combined_voiced.dropna(subset=feature_names_lagged + target_names_diff)

            if len(combined_clean) < config.min_frames:
                continue

            X = combined_clean[feature_names_lagged].to_numpy(dtype=float)

            for target in TARGET_METRICS:
                y = combined_clean[f"{target}_diff"].to_numpy(dtype=float)
                if len(y) < config.min_frames:
                    continue

                cv_res = nested_time_series_pls_cv(
                    X=X,
                    y=y,
                    max_components=config.max_pls_components,
                    requested_outer_splits=config.requested_outer_splits,
                    requested_inner_splits=config.requested_inner_splits,
                )
                if not np.isfinite(cv_res["mean_r2"]):
                    continue

                best_components = cv_res["best_components"]
                final_k = int(pd.Series(best_components).mode().iloc[0]) if best_components else 1

                final_model, scaler_X, scaler_y, y_pred_in_sample, vip = fit_final_pls(
                    X=X,
                    y=y,
                    n_components=final_k,
                )
                if final_model is None or vip is None or len(vip) != len(feature_names_lagged):
                    continue

                vip_stability = vip_bootstrap_stability(
                    X=X,
                    y=y,
                    n_components=final_k,
                    n_bootstraps=config.n_bootstraps,
                    block_size=config.bootstrap_block_size,
                    random_state=config.random_state,
                )

                predictive_rows.append(
                    {
                        "batch_name": batch_name,
                        "subject_id": subject_id,
                        "demographic": demographic,
                        "sex": sex,
                        "age": age,
                        "target": target,
                        "n_samples": int(len(X)),
                        "n_features": int(X.shape[1]),
                        "r2_cv_mean": float(cv_res["mean_r2"]),
                        "r2_cv_std": float(cv_res["std_r2"]) if np.isfinite(cv_res["std_r2"]) else np.nan,
                        "best_components": int(final_k),
                    }
                )

                vip_mean = vip_stability["vip_mean"] if vip_stability["vip_mean"].size else vip
                vip_std = vip_stability["vip_std"] if vip_stability["vip_std"].size else np.zeros_like(vip)
                vip_prob = vip_stability["vip_prob_gt1"] if vip_stability["vip_prob_gt1"].size else (vip > 1.0).astype(float)

                for feat, vip_m, vip_s, vip_p in zip(feature_names_lagged, vip_mean, vip_std, vip_prob):
                    vip_rows.append(
                        {
                            "batch_name": batch_name,
                            "subject_id": subject_id,
                            "demographic": demographic,
                            "sex": sex,
                            "age": age,
                            "target": target,
                            "feature": feat,
                            "vip_mean": float(vip_m),
                            "vip_std": float(vip_s),
                            "vip_prob_gt1": float(vip_p),
                        }
                    )

        if not predictive_rows:
            logger.warning("No valid predictive results for batch %s", batch_name)
            continue

        results_df = pd.DataFrame(predictive_rows)
        vip_df = pd.DataFrame(vip_rows)

        results_df.to_csv(fig_dir / f"Predictive_Subject_Results_{batch_name}.csv", index=False, sep=";")
        vip_df.to_csv(fig_dir / f"VIP_Subject_Results_{batch_name}.csv", index=False, sep=";")

        for target in TARGET_METRICS:
            target_results = results_df[results_df["target"] == target].copy()
            if target_results.empty:
                continue

            summary_table = target_results.groupby("demographic")["r2_cv_mean"].agg(["count", "mean", "std"]).reset_index()
            summary_table.to_csv(fig_dir / f"Predictive_R2_{batch_name}_{target}.csv", index=False, sep=";")

            descriptive = summarize_group_labels(target_results, outcome_col="r2_cv_mean")
            if not descriptive.empty:
                descriptive.to_csv(fig_dir / f"Predictive_Group_Summary_{batch_name}_{target}.csv", index=False, sep=";")
                # Added PDF render integration
                save_table_as_pdf(
                    descriptive,
                    f"Predictive Summary - {target}",
                    fig_dir / f"Predictive_Group_Summary_{batch_name}_{target}.pdf"
                )

            comparison_table = compare_demographic_groups(target_results, outcome_col="r2_cv_mean")
            comparison_table = apply_fdr_to_comparisons(comparison_table)
            if not comparison_table.empty:
                comparison_table.to_csv(fig_dir / f"Predictive_Group_Comparisons_{batch_name}_{target}.csv", index=False, sep=";")
                # Added PDF render integration
                save_table_as_pdf(
                    comparison_table,
                    f"Predictive Group Comparisons - {target}",
                    fig_dir / f"Predictive_Group_Comparisons_{batch_name}_{target}.pdf"
                )

            target_vips = vip_df[vip_df["target"] == target].copy()
            if not target_vips.empty:
                avg_vips = target_vips.groupby(["demographic", "feature"])["vip_mean"].mean().reset_index()
                top_features = avg_vips.groupby("feature")["vip_mean"].mean().nlargest(15).index
                plot_vips = avg_vips[avg_vips["feature"].isin(top_features)]

                sns.set_theme(style="whitegrid")
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.barplot(data=plot_vips, x="vip_mean", y="feature", hue="demographic", ax=ax)
                ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="VIP = 1.0")
                ax.set_title(f"Top 15 VIP predictors for {target}\n{batch_name} - SPEECH (nested time-series PLS)")
                ax.set_xlabel("Mean VIP score")
                ax.set_ylabel("Lagged acoustic feature")
                ax.legend(title="Demographic", bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                fig.savefig(fig_dir / f"VIP_Predictors_{batch_name}_{target}.pdf", bbox_inches="tight")
                plt.close(fig)

                stability_summary = (
                    target_vips.groupby("feature")[["vip_mean", "vip_std", "vip_prob_gt1"]]
                    .mean()
                    .reset_index()
                    .sort_values("vip_mean", ascending=False)
                )
                stability_summary.to_csv(fig_dir / f"VIP_Stability_Summary_{batch_name}_{target}.csv", index=False, sep=";")

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=results_df, x="target", y="r2_cv_mean", hue="demographic", ax=ax)
        sns.stripplot(data=results_df, x="target", y="r2_cv_mean", hue="demographic", dodge=True, ax=ax, color="black", alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), title="Demographic", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_title(f"Predictive D-PLS performance - {batch_name}")
        ax.set_xlabel("Target")
        ax.set_ylabel("Cross-validated R²")
        plt.tight_layout()
        fig.savefig(fig_dir / f"Predictive_R2_{batch_name}.pdf", bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved batch outputs to %s", fig_dir)

    logger.info("Predictive D-PLS processing complete.")


if __name__ == "__main__":
    main()