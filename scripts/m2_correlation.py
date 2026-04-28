"""
M2 — Exploratory Correlation Analysis (Extended)
==================================================

Seven levels of analysis on the paired HDF5 corpus:

1. GLOBAL: per-segment summary statistics correlated across subjects
2. TIME-RESOLVED: sliding-window cross-correlation within recordings
3. EVENT-ALIGNED: above-FRC vs below-FRC comparison (sustained tasks only)
4. SEX-STRATIFIED: global correlations split by sex + partial correlations
5. CROSS-CORRELATION WITH LAGS: temporal precedence between respiratory and acoustic events
6. BREATH-GROUP: per-breath-group analysis for connected speech
7. MFCC-RESPIRATORY: spectral shape correlations with respiratory state

Outputs:
    data_target/<batch>/m2_correlation/
    ├── global_*.pdf                  — Level 1 outputs
    ├── time_resolved/                — Level 2 outputs
    ├── frc_*.pdf                     — Level 3 outputs
    ├── sex_*.pdf                     — Level 4 outputs
    ├── lag_*.pdf                     — Level 5 outputs
    ├── breath_group_*.pdf            — Level 6 outputs
    ├── mfcc_*.pdf                    — Level 7 outputs
    └── m2_report.txt                 — text summary

Usage:
    python m2_correlation.py
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import correlate
from pneumophonic_analysis.paired_features import PairedFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data_root"
DATA_TARGET  = PROJECT_ROOT / "data_target"

BATCHES = ["healthy_subjects", "pathological_subjects"]

# ---- Batch selection ----
def select_batch():
    print("\nAvailable batches:")
    for idx, name in enumerate(BATCHES):
        print(f"  [{idx}] {name}")
    while True:
        sel = input("Select batch by number: ")
        if sel.isdigit() and 0 <= int(sel) < len(BATCHES):
            return BATCHES[int(sel)]
        print("Invalid selection. Try again.")

# ---- Task categories ----
SUSTAINED_TASKS = {'a', 'a_2', 'a_3', 'a_7', 'r'}
SPEECH_TASKS = {'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'testo'}
VOWEL_TASKS = {'a', 'e', 'i', 'o', 'u'}

# ---- Metadata loading ----
def load_subject_metadata(batch_name):
    """Load subjects_metadata.csv with sex/age info."""
    meta_path = DATA_ROOT / batch_name / "subjects_metadata.csv"
    if not meta_path.exists():
        logger.warning(f"  No subjects_metadata.csv found at {meta_path}")
        return None
    df = pd.read_csv(meta_path, sep=';')
    # Clean column names (remove trailing spaces)
    df.columns = df.columns.str.strip()
    # Standardize gender: W -> F for consistency
    if 'Gender' in df.columns:
        df['sex'] = df['Gender'].str.strip().map({'M': 'M', 'W': 'F'})
    if 'ID' in df.columns:
        df['subject_id'] = df['ID'].str.strip()
    return df[['subject_id', 'sex']].dropna()


# =====================================================================
# LEVEL 1: GLOBAL (per-segment) correlations
# =====================================================================

def compute_segment_summary(df, meta):
    """Compute summary statistics for one paired segment."""
    voiced = df[df['voiced'] == 1.0]
    n_voiced = len(voiced)
    n_total = len(df)

    summary = {
        'subject_id': meta.get('subject_id', ''),
        'task': meta.get('task_name', ''),
        'duration_sec': meta.get('audio_duration_sec', n_total * 0.015),
        'n_frames': n_total,
        'n_voiced': n_voiced,
        'voiced_ratio': n_voiced / n_total if n_total > 0 else 0,
    }

    # --- Audio features (voiced frames only) ---
    if n_voiced > 10:
        summary['f0_mean'] = np.nanmean(voiced['f0'])
        summary['f0_std'] = np.nanstd(voiced['f0'])
        summary['f0_range'] = np.nanmax(voiced['f0']) - np.nanmin(voiced['f0'])
        summary['energy_mean'] = voiced['energy'].mean()
        summary['energy_std'] = voiced['energy'].std()
        summary['spectral_centroid_mean'] = voiced['spectral_centroid'].mean()

        # MFCCs — mean of first 5
        for i in range(min(5, sum(1 for c in voiced.columns if c.startswith('mfcc_')))):
            summary[f'mfcc_{i}_mean'] = voiced[f'mfcc_{i}'].mean()
    else:
        summary['f0_mean'] = np.nan
        summary['f0_std'] = np.nan
        summary['f0_range'] = np.nan
        summary['energy_mean'] = np.nan
        summary['energy_std'] = np.nan
        summary['spectral_centroid_mean'] = np.nan

    # --- OEP features (all frames) ---
    summary['vcw_mean'] = df['vcw'].mean()
    summary['delta_vcw_range'] = df['delta_vcw'].max() - df['delta_vcw'].min()
    summary['flow_cw_mean'] = df['flow_cw'].mean()
    summary['flow_cw_std'] = df['flow_cw'].std()
    summary['flow_rc_mean'] = df['flow_rc'].mean()
    summary['flow_ab_mean'] = df['flow_ab'].mean()
    summary['pct_rc_mean'] = df['pct_rc'].mean()
    summary['pct_rc_std'] = df['pct_rc'].std()
    summary['pct_ab_mean'] = df['pct_ab'].mean()

    # --- Within-segment correlations (voiced only) ---
    if n_voiced > 20:
        r, p = stats.pearsonr(voiced['energy'], voiced['delta_vcw'])
        summary['corr_energy_deltavcw'] = r
        summary['pval_energy_deltavcw'] = p

        f0_valid = voiced.dropna(subset=['f0'])
        if len(f0_valid) > 20:
            r, p = stats.pearsonr(f0_valid['f0'], f0_valid['flow_cw'])
            summary['corr_f0_flowcw'] = r
            summary['pval_f0_flowcw'] = p
        else:
            summary['corr_f0_flowcw'] = np.nan
            summary['pval_f0_flowcw'] = np.nan

        r, p = stats.pearsonr(voiced['energy'], voiced['flow_cw'])
        summary['corr_energy_flowcw'] = r
        summary['pval_energy_flowcw'] = p
    else:
        summary['corr_energy_deltavcw'] = np.nan
        summary['pval_energy_deltavcw'] = np.nan
        summary['corr_f0_flowcw'] = np.nan
        summary['pval_f0_flowcw'] = np.nan
        summary['corr_energy_flowcw'] = np.nan
        summary['pval_energy_flowcw'] = np.nan

    return summary


def run_global_analysis(paired_dir, output_dir):
    """Level 1: compute summary stats and cross-subject correlations."""
    print("\n" + "="*60)
    print("LEVEL 1: Global per-segment correlations")
    print("="*60)

    h5_files = sorted(paired_dir.glob("*.h5"))
    print(f"  Loading {len(h5_files)} paired datasets...")

    summaries = []
    for h5 in h5_files:
        try:
            df, meta = PairedFeatureExtractor.load_hdf5(h5)
            summaries.append(compute_segment_summary(df, meta))
        except Exception as e:
            logger.warning(f"  Failed to load {h5.name}: {e}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "global_summary.csv", index=False)
    print(f"  Summary: {len(summary_df)} segments from "
          f"{summary_df['subject_id'].nunique()} subjects")

    # ---- Correlation heatmap ----
    audio_cols = ['f0_mean', 'f0_std', 'energy_mean', 'energy_std', 'spectral_centroid_mean']
    oep_cols = ['delta_vcw_range', 'flow_cw_mean', 'flow_cw_std', 'pct_rc_mean', 'pct_rc_std']
    all_corr_cols = audio_cols + oep_cols
    available = [c for c in all_corr_cols if c in summary_df.columns]
    corr_data = summary_df[available].dropna()

    if len(corr_data) > 10:
        corr_matrix = corr_data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True, ax=ax, linewidths=0.5)
        ax.set_title('Global Correlation: Audio vs OEP Features\n(per-segment summaries across subjects)')
        plt.tight_layout()
        fig.savefig(output_dir / "global_correlation_matrix.pdf", bbox_inches='tight')
        plt.close(fig)
        print("  Saved: global_correlation_matrix.pdf")

    # ---- By task type ----
    for task_group_name, task_set in [("sustained", SUSTAINED_TASKS), ("speech", SPEECH_TASKS)]:
        subset = summary_df[summary_df['task'].isin(task_set)]
        sub_corr = subset[available].dropna()
        if len(sub_corr) > 10:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(sub_corr.corr(), annot=True, fmt='.2f', cmap='RdBu_r',
                        center=0, vmin=-1, vmax=1, square=True, ax=ax, linewidths=0.5)
            ax.set_title(f'Global Correlation - {task_group_name.upper()} tasks only')
            plt.tight_layout()
            fig.savefig(output_dir / f"global_correlation_{task_group_name}.pdf", bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: global_correlation_{task_group_name}.pdf")

    # ---- Key scatter plots ----
    scatter_pairs = [
        ('delta_vcw_range', 'energy_mean', 'Volume Excursion vs Mean Energy'),
        ('flow_cw_mean', 'f0_mean', 'Mean Flow vs Mean F0'),
        ('flow_cw_std', 'energy_std', 'Flow Variability vs Energy Variability'),
        ('pct_rc_mean', 'f0_mean', 'Mean %RC vs Mean F0'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, (x_col, y_col, title) in enumerate(scatter_pairs):
        ax = axes.flatten()[idx]
        if x_col in summary_df.columns and y_col in summary_df.columns:
            valid = summary_df[[x_col, y_col, 'task']].dropna()
            colors = ['steelblue' if t in SUSTAINED_TASKS else 'coral' if t in SPEECH_TASKS else 'gray'
                      for t in valid['task']]
            ax.scatter(valid[x_col], valid[y_col], c=colors, alpha=0.6, s=30)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(title)
            r, p = stats.pearsonr(valid[x_col], valid[y_col])
            ax.annotate(f'r={r:.3f}, p={p:.3e}', xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', label='Sustained', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='coral', label='Speech', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Vowels', markersize=8),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_dir / "global_scatter_plots.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved: global_scatter_plots.pdf")

    # ---- Within-segment correlation distributions ----
    corr_cols_within = ['corr_energy_deltavcw', 'corr_f0_flowcw', 'corr_energy_flowcw']
    available_within = [c for c in corr_cols_within if c in summary_df.columns]
    if available_within:
        fig, axes = plt.subplots(1, len(available_within), figsize=(6 * len(available_within), 5))
        if len(available_within) == 1:
            axes = [axes]
        for ax, col in zip(axes, available_within):
            data = summary_df[col].dropna()
            ax.hist(data, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(data.median(), color='red', linestyle='--', linewidth=2,
                      label=f'median={data.median():.3f}')
            ax.set_xlabel('Pearson r')
            ax.set_ylabel('Count')
            ax.set_title(col.replace('corr_', '').replace('_', ' vs '))
            ax.legend()
            ax.set_xlim(-1, 1)
        plt.suptitle('Distribution of Within-Segment Correlations Across All Subjects x Tasks')
        plt.tight_layout()
        fig.savefig(output_dir / "within_segment_correlation_distributions.pdf", bbox_inches='tight')
        plt.close(fig)
        print("  Saved: within_segment_correlation_distributions.pdf")

    return summary_df


# =====================================================================
# LEVEL 2: TIME-RESOLVED cross-correlations
# =====================================================================

def compute_sliding_correlation(x, y, window_frames=33):
    """Sliding-window Pearson correlation. window_frames=33 ~ 0.5s at ~66 fps."""
    n = len(x)
    if n < window_frames:
        return np.full(n, np.nan)
    half_w = window_frames // 2
    r_values = np.full(n, np.nan)
    for i in range(half_w, n - half_w):
        x_win = x[i - half_w:i + half_w + 1]
        y_win = y[i - half_w:i + half_w + 1]
        valid = ~(np.isnan(x_win) | np.isnan(y_win))
        if np.sum(valid) > 10:
            r, _ = stats.pearsonr(x_win[valid], y_win[valid])
            r_values[i] = r
    return r_values


def run_time_resolved(paired_dir, output_dir, max_subjects=10):
    """Level 2: sliding-window cross-correlation within recordings."""
    print("\n" + "="*60)
    print("LEVEL 2: Time-resolved cross-correlations")
    print("="*60)

    tr_dir = output_dir / "time_resolved"
    tr_dir.mkdir(exist_ok=True)

    h5_files = sorted(paired_dir.glob("*.h5"))
    processed = 0
    all_energy_vcw_corrs = []

    for h5 in h5_files:
        try:
            df, meta = PairedFeatureExtractor.load_hdf5(h5)
            task = meta.get('task_name', '')
            sid = meta.get('subject_id', '')
            if task not in SUSTAINED_TASKS or len(df) < 100:
                continue

            r_energy_vcw = compute_sliding_correlation(
                df['energy'].values, df['delta_vcw'].values, window_frames=33)
            all_energy_vcw_corrs.append({
                'subject': sid, 'task': task, 'r_values': r_energy_vcw, 'time': df['time'].values})

            if processed < max_subjects:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
                ax1.plot(df['time'], df['energy'], color='steelblue', alpha=0.7, label='Energy')
                ax1b = ax1.twinx()
                ax1b.plot(df['time'], df['delta_vcw'], color='coral', alpha=0.7, label='dVcw')
                ax1.set_ylabel('Energy', color='steelblue')
                ax1b.set_ylabel('dVcw (L)', color='coral')
                ax1.set_title(f'{sid} - {task}: Time-Resolved Correlation')
                ax1.legend(loc='upper left')
                ax1b.legend(loc='upper right')
                ax2.plot(df['time'], r_energy_vcw, color='purple', linewidth=1.5)
                ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax2.fill_between(df['time'], r_energy_vcw, 0,
                               where=~np.isnan(r_energy_vcw), alpha=0.3, color='purple')
                ax2.set_ylabel('Pearson r (0.5s window)')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylim(-1, 1)
                ax2.set_title('Sliding Energy vs dVcw Correlation')
                plt.tight_layout()
                fig.savefig(tr_dir / f"{sid}_{task}_time_resolved.pdf", bbox_inches='tight')
                plt.close(fig)
            processed += 1
        except Exception as e:
            logger.warning(f"  Time-resolved failed for {h5.name}: {e}")

    print(f"  Processed {processed} sustained segments")
    print(f"  Saved {min(processed, max_subjects)} individual plots in time_resolved/")

    if all_energy_vcw_corrs:
        mean_rs = [np.nanmean(e['r_values']) for e in all_energy_vcw_corrs]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(mean_rs, bins=15, edgecolor='black', alpha=0.7, color='purple')
        ax.axvline(np.median(mean_rs), color='red', linestyle='--',
                   label=f'median={np.median(mean_rs):.3f}')
        ax.set_xlabel('Mean sliding r (Energy vs dVcw)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Time-Resolved Energy-Volume Coupling\n(sustained tasks, 0.5s window)')
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_dir / "time_resolved_aggregate.pdf", bbox_inches='tight')
        plt.close(fig)
        print("  Saved: time_resolved_aggregate.pdf")

    return all_energy_vcw_corrs


# =====================================================================
# LEVEL 3: EVENT-ALIGNED (FRC crossing)
# =====================================================================

def run_frc_analysis(paired_dir, output_dir):
    """Level 3: compare audio features above vs below FRC."""
    print("\n" + "="*60)
    print("LEVEL 3: Event-aligned FRC analysis")
    print("="*60)

    frc_tasks = {'a_2', 'a_3', 'a_7'}
    h5_files = sorted(paired_dir.glob("*.h5"))
    frc_results = []

    for h5 in h5_files:
        try:
            df, meta = PairedFeatureExtractor.load_hdf5(h5)
            task = meta.get('task_name', '')
            sid = meta.get('subject_id', '')
            if task not in frc_tasks:
                continue

            dcw = df['delta_vcw'].values
            peak_idx = np.argmax(dcw)
            post_peak = dcw[peak_idx:]
            descending_crossings = np.where((post_peak[:-1] > 0) & (post_peak[1:] <= 0))[0]

            if len(descending_crossings) > 0:
                cross_idx = peak_idx + descending_crossings[0]
            else:
                cross_idx = peak_idx + len(post_peak) // 2
                if cross_idx >= len(df) - 20:
                    continue

            above = df.iloc[:cross_idx]
            below = df.iloc[cross_idx:]
            if len(above) < 20 or len(below) < 20:
                continue

            above_v = above[above['voiced'] == 1.0]
            below_v = below[below['voiced'] == 1.0]
            if len(above_v) < 10 or len(below_v) < 10:
                continue

            result = {
                'subject_id': sid, 'task': task,
                'frc_cross_time': df['time'].iloc[cross_idx],
                'duration_above': len(above) * 0.015,
                'duration_below': len(below) * 0.015,
                'f0_above': np.nanmean(above_v['f0']),
                'energy_above': above_v['energy'].mean(),
                'spectral_centroid_above': above_v['spectral_centroid'].mean(),
                'f0_below': np.nanmean(below_v['f0']),
                'energy_below': below_v['energy'].mean(),
                'spectral_centroid_below': below_v['spectral_centroid'].mean(),
                'flow_cw_above': above['flow_cw'].mean(),
                'flow_cw_below': below['flow_cw'].mean(),
                'pct_rc_above': above['pct_rc'].mean(),
                'pct_rc_below': below['pct_rc'].mean(),
            }
            result['f0_shift'] = result['f0_below'] - result['f0_above']
            result['energy_shift'] = result['energy_below'] - result['energy_above']
            result['pct_rc_shift'] = result['pct_rc_below'] - result['pct_rc_above']
            result['flow_shift'] = result['flow_cw_below'] - result['flow_cw_above']
            frc_results.append(result)
        except Exception as e:
            logger.warning(f"  FRC analysis failed for {h5.name}: {e}")

    if not frc_results:
        print("  No valid FRC segments found.")
        return pd.DataFrame()

    frc_df = pd.DataFrame(frc_results)
    frc_df.to_csv(output_dir / "frc_analysis.csv", index=False)
    print(f"  Analyzed {len(frc_df)} segments from {frc_df['subject_id'].nunique()} subjects")

    # ---- Shift histograms ----
    shift_cols = [
        ('f0_shift', 'F0 shift (Hz)', 'Below-Above FRC'),
        ('energy_shift', 'Energy shift', 'Below-Above FRC'),
        ('pct_rc_shift', '%RC shift', 'Below-Above FRC'),
        ('flow_shift', 'Flow shift (L/s)', 'Below-Above FRC'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, (col, ylabel, xlabel) in enumerate(shift_cols):
        ax = axes.flatten()[idx]
        data = frc_df[col].dropna()
        ax.hist(data, bins=15, edgecolor='black', alpha=0.7, color='teal')
        ax.axvline(0, color='gray', linestyle='--')
        ax.axvline(data.median(), color='red', linestyle='--', label=f'median={data.median():.4f}')
        ax.set_xlabel(ylabel)
        ax.set_ylabel('Count')
        ax.set_title(f'{xlabel}: {ylabel}')
        ax.legend(fontsize=8)
        if len(data) > 5:
            try:
                stat, p = stats.wilcoxon(data)
                ax.annotate(f'Wilcoxon p={p:.3e}', xy=(0.05, 0.85), xycoords='axes fraction',
                           fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except Exception:
                pass
    plt.suptitle('Effect of FRC Crossing on Audio and Respiratory Features', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "frc_shifts.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved: frc_shifts.pdf")

    # ---- Paired scatter ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (a_col, b_col, label) in zip(axes, [
        ('f0_above', 'f0_below', 'F0 (Hz)'),
        ('energy_above', 'energy_below', 'Energy'),
        ('pct_rc_above', 'pct_rc_below', '%RC'),
    ]):
        valid = frc_df[[a_col, b_col]].dropna()
        ax.scatter(valid[a_col], valid[b_col], alpha=0.6, s=30, color='teal')
        lims = [min(valid[a_col].min(), valid[b_col].min()),
                max(valid[a_col].max(), valid[b_col].max())]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='identity')
        ax.set_xlabel(f'{label} - Above FRC')
        ax.set_ylabel(f'{label} - Below FRC')
        ax.set_title(f'{label}: Above vs Below FRC')
        ax.legend()
    plt.suptitle('Paired Comparison: Above vs Below FRC', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "frc_paired_scatter.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved: frc_paired_scatter.pdf")

    return frc_df


# =====================================================================
# LEVEL 4: SEX-STRATIFIED correlations
# =====================================================================

def run_sex_stratified(summary_df, meta_df, output_dir):
    """Level 4: split correlations by sex + partial correlations."""
    print("\n" + "="*60)
    print("LEVEL 4: Sex-stratified correlations")
    print("="*60)

    if meta_df is None or 'sex' not in meta_df.columns:
        print("  No sex metadata available - skipping")
        return

    # Merge sex info into summary
    merged = summary_df.merge(meta_df, on='subject_id', how='left')
    merged = merged.dropna(subset=['sex'])
    n_m = (merged['sex'] == 'M').sum()
    n_f = (merged['sex'] == 'F').sum()
    print(f"  Segments with sex info: {len(merged)} (M={n_m}, F={n_f})")

    if n_m < 10 or n_f < 10:
        print("  Not enough data in each group - skipping")
        return

    audio_cols = ['f0_mean', 'f0_std', 'energy_mean', 'energy_std', 'spectral_centroid_mean']
    oep_cols = ['delta_vcw_range', 'flow_cw_mean', 'flow_cw_std', 'pct_rc_mean', 'pct_rc_std']
    all_cols = audio_cols + oep_cols
    available = [c for c in all_cols if c in merged.columns]

    # ---- Side-by-side heatmaps ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    males = merged[merged['sex'] == 'M'][available].dropna()
    females = merged[merged['sex'] == 'F'][available].dropna()

    if len(males) > 10:
        sns.heatmap(males.corr(), annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True, ax=ax1, linewidths=0.5)
        ax1.set_title(f'Male subjects (n={len(males)} segments)')

    if len(females) > 10:
        sns.heatmap(females.corr(), annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True, ax=ax2, linewidths=0.5)
        ax2.set_title(f'Female subjects (n={len(females)} segments)')

    plt.suptitle('Sex-Stratified Correlation Matrices', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "sex_stratified_heatmaps.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved: sex_stratified_heatmaps.pdf")

    # ---- Key scatter plots colored by sex ----
    scatter_pairs = [
        ('flow_cw_mean', 'f0_mean', 'Mean Flow vs Mean F0'),
        ('delta_vcw_range', 'energy_mean', 'Volume Excursion vs Mean Energy'),
        ('pct_rc_mean', 'f0_mean', 'Mean %RC vs Mean F0'),
        ('flow_cw_std', 'f0_std', 'Flow Variability vs F0 Variability'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, (x_col, y_col, title) in enumerate(scatter_pairs):
        ax = axes.flatten()[idx]
        if x_col in merged.columns and y_col in merged.columns:
            for sex, color, marker in [('M', 'steelblue', 'o'), ('F', 'coral', 's')]:
                sub = merged[(merged['sex'] == sex) & merged[x_col].notna() & merged[y_col].notna()]
                ax.scatter(sub[x_col], sub[y_col], c=color, marker=marker,
                          alpha=0.6, s=30, label=f'{sex} (n={len(sub)})')
                # Per-sex regression
                if len(sub) > 10:
                    r, p = stats.pearsonr(sub[x_col], sub[y_col])
                    ax.annotate(f'{sex}: r={r:.3f}, p={p:.2e}',
                               xy=(0.05, 0.95 - 0.08 * (0 if sex == 'M' else 1)),
                               xycoords='axes fraction', fontsize=8, color=color,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(title)
            ax.legend(fontsize=8)

    plt.suptitle('Sex-Stratified Scatter Plots', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "sex_stratified_scatter.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved: sex_stratified_scatter.pdf")

    # ---- Partial correlation: F0 vs flow controlling for sex ----
    from sklearn.linear_model import LinearRegression
    print("\n  Partial correlations (controlling for sex):")
    sex_numeric = (merged['sex'] == 'F').astype(float).values.reshape(-1, 1)

    partial_pairs = [
        ('f0_mean', 'flow_cw_mean'),
        ('f0_mean', 'pct_rc_mean'),
        ('energy_mean', 'delta_vcw_range'),
    ]

    partial_results = []
    for col_a, col_b in partial_pairs:
        valid = merged[[col_a, col_b, 'sex']].dropna()
        if len(valid) < 20:
            continue
        sex_num = (valid['sex'] == 'F').astype(float).values.reshape(-1, 1)
        a = valid[col_a].values
        b = valid[col_b].values

        # Residualize both variables on sex
        res_a = a - LinearRegression().fit(sex_num, a).predict(sex_num)
        res_b = b - LinearRegression().fit(sex_num, b).predict(sex_num)

        r_raw, p_raw = stats.pearsonr(a, b)
        r_partial, p_partial = stats.pearsonr(res_a, res_b)

        print(f"    {col_a} vs {col_b}:")
        print(f"      raw:     r={r_raw:.3f}, p={p_raw:.3e}")
        print(f"      partial: r={r_partial:.3f}, p={p_partial:.3e}")

        partial_results.append({
            'pair': f'{col_a} vs {col_b}',
            'r_raw': r_raw, 'p_raw': p_raw,
            'r_partial': r_partial, 'p_partial': p_partial,
        })

    if partial_results:
        pd.DataFrame(partial_results).to_csv(
            output_dir / "sex_partial_correlations.csv", index=False)
        print("  Saved: sex_partial_correlations.csv")


# =====================================================================
# LEVEL 5: CROSS-CORRELATION WITH LAGS
# =====================================================================

def compute_xcorr_lag(x, y, max_lag_frames=33, fps=66.67):
    """
    Compute normalized cross-correlation and find the peak lag.
    max_lag_frames=33 ~ 0.5s at ~66 fps.
    Returns: (lags_in_seconds, xcorr_values, peak_lag_sec, peak_r)
    """
    x_clean = x - np.nanmean(x)
    y_clean = y - np.nanmean(y)
    # Replace NaN with 0 for correlation
    x_clean = np.nan_to_num(x_clean, 0)
    y_clean = np.nan_to_num(y_clean, 0)

    norm = np.sqrt(np.sum(x_clean**2) * np.sum(y_clean**2))
    if norm < 1e-12:
        return None, None, 0, 0

    full_xcorr = correlate(x_clean, y_clean, mode='full')
    full_xcorr /= norm

    mid = len(x_clean) - 1
    xcorr = full_xcorr[mid - max_lag_frames:mid + max_lag_frames + 1]
    lags_frames = np.arange(-max_lag_frames, max_lag_frames + 1)
    lags_sec = lags_frames / fps

    peak_idx = np.argmax(np.abs(xcorr))
    peak_lag_sec = lags_sec[peak_idx]
    peak_r = xcorr[peak_idx]

    return lags_sec, xcorr, peak_lag_sec, peak_r


def run_lag_analysis(paired_dir, output_dir):
    """Level 5: cross-correlation with temporal lags."""
    print("\n" + "="*60)
    print("LEVEL 5: Cross-correlation with time lags")
    print("="*60)

    h5_files = sorted(paired_dir.glob("*.h5"))
    fps = 66.67  # approximate frame rate
    max_lag_frames = 33  # ~0.5s

    lag_results = []
    all_xcorr_flow_energy = []
    all_xcorr_flow_f0 = []

    for h5 in h5_files:
        try:
            df, meta = PairedFeatureExtractor.load_hdf5(h5)
            task = meta.get('task_name', '')
            sid = meta.get('subject_id', '')

            if task not in SUSTAINED_TASKS or len(df) < 100:
                continue

            voiced = df[df['voiced'] == 1.0]
            if len(voiced) < 50:
                continue

            # Flow -> Energy lag
            lags_s, xcorr_fe, peak_lag_fe, peak_r_fe = compute_xcorr_lag(
                voiced['flow_cw'].values, voiced['energy'].values,
                max_lag_frames=max_lag_frames, fps=fps)

            # Flow -> F0 lag
            f0_valid = voiced.dropna(subset=['f0'])
            if len(f0_valid) > 50:
                lags_s2, xcorr_ff, peak_lag_ff, peak_r_ff = compute_xcorr_lag(
                    f0_valid['flow_cw'].values, f0_valid['f0'].values,
                    max_lag_frames=max_lag_frames, fps=fps)
            else:
                peak_lag_ff, peak_r_ff = np.nan, np.nan
                xcorr_ff = None

            lag_results.append({
                'subject_id': sid, 'task': task,
                'lag_flow_energy_sec': peak_lag_fe, 'r_flow_energy': peak_r_fe,
                'lag_flow_f0_sec': peak_lag_ff, 'r_flow_f0': peak_r_ff,
            })

            if xcorr_fe is not None and lags_s is not None:
                all_xcorr_flow_energy.append(xcorr_fe)
            if xcorr_ff is not None:
                all_xcorr_flow_f0.append(xcorr_ff)

        except Exception as e:
            logger.warning(f"  Lag analysis failed for {h5.name}: {e}")

    if not lag_results:
        print("  No valid segments for lag analysis.")
        return pd.DataFrame()

    lag_df = pd.DataFrame(lag_results)
    lag_df.to_csv(output_dir / "lag_analysis.csv", index=False)
    print(f"  Analyzed {len(lag_df)} segments")

    # ---- Mean cross-correlation curves ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    lags_sec = np.arange(-max_lag_frames, max_lag_frames + 1) / fps

    if all_xcorr_flow_energy:
        mean_xcorr = np.mean(all_xcorr_flow_energy, axis=0)
        std_xcorr = np.std(all_xcorr_flow_energy, axis=0)
        ax1.plot(lags_sec, mean_xcorr, color='steelblue', linewidth=2)
        ax1.fill_between(lags_sec, mean_xcorr - std_xcorr, mean_xcorr + std_xcorr,
                        alpha=0.2, color='steelblue')
        ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
        peak_idx = np.argmax(np.abs(mean_xcorr))
        ax1.axvline(lags_sec[peak_idx], color='red', linestyle='--',
                   label=f'peak lag = {lags_sec[peak_idx]*1000:.0f} ms')
        ax1.set_xlabel('Lag (s) [negative = flow leads]')
        ax1.set_ylabel('Cross-correlation')
        ax1.set_title('Flow CW vs Energy')
        ax1.legend()

    if all_xcorr_flow_f0:
        mean_xcorr = np.mean(all_xcorr_flow_f0, axis=0)
        std_xcorr = np.std(all_xcorr_flow_f0, axis=0)
        ax2.plot(lags_sec, mean_xcorr, color='purple', linewidth=2)
        ax2.fill_between(lags_sec, mean_xcorr - std_xcorr, mean_xcorr + std_xcorr,
                        alpha=0.2, color='purple')
        ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
        peak_idx = np.argmax(np.abs(mean_xcorr))
        ax2.axvline(lags_sec[peak_idx], color='red', linestyle='--',
                   label=f'peak lag = {lags_sec[peak_idx]*1000:.0f} ms')
        ax2.set_xlabel('Lag (s) [negative = flow leads]')
        ax2.set_ylabel('Cross-correlation')
        ax2.set_title('Flow CW vs F0')
        ax2.legend()

    plt.suptitle('Mean Cross-Correlation with Time Lags (sustained tasks)', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "lag_cross_correlation.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved: lag_cross_correlation.pdf")

    # ---- Peak lag distribution ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    data_fe = lag_df['lag_flow_energy_sec'].dropna() * 1000  # to ms
    ax1.hist(data_fe, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(data_fe.median(), color='red', linestyle='--',
               label=f'median={data_fe.median():.0f} ms')
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Peak lag (ms)')
    ax1.set_ylabel('Count')
    ax1.set_title('Flow -> Energy: Peak Lag Distribution')
    ax1.legend()

    data_ff = lag_df['lag_flow_f0_sec'].dropna() * 1000
    ax2.hist(data_ff, bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax2.axvline(data_ff.median(), color='red', linestyle='--',
               label=f'median={data_ff.median():.0f} ms')
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Peak lag (ms)')
    ax2.set_ylabel('Count')
    ax2.set_title('Flow -> F0: Peak Lag Distribution')
    ax2.legend()

    plt.suptitle('Distribution of Peak Cross-Correlation Lags', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "lag_distributions.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved: lag_distributions.pdf")

    return lag_df


# =====================================================================
# LEVEL 6: BREATH-GROUP analysis (connected speech)
# =====================================================================

def run_breath_group_analysis(paired_dir, output_dir):
    """Level 6: segment connected speech into breath groups and analyze each."""
    print("\n" + "="*60)
    print("LEVEL 6: Breath-group analysis (connected speech)")
    print("="*60)

    h5_files = sorted(paired_dir.glob("*.h5"))
    bg_results = []

    for h5 in h5_files:
        try:
            df, meta = PairedFeatureExtractor.load_hdf5(h5)
            task = meta.get('task_name', '')
            sid = meta.get('subject_id', '')

            if task != 'testo' or len(df) < 200:
                continue

            # Detect inspiratory peaks in delta_vcw (breath boundaries)
            dcw = df['delta_vcw'].values
            # Smooth for peak detection
            from scipy.ndimage import uniform_filter1d
            dcw_smooth = uniform_filter1d(dcw, size=20)

            # Find local maxima (inspiratory peaks)
            from scipy.signal import find_peaks
            peaks, props = find_peaks(dcw_smooth, distance=100, prominence=0.1)

            if len(peaks) < 2:
                continue

            # Each breath group: from one peak to the next
            for bg_idx in range(len(peaks) - 1):
                start = peaks[bg_idx]
                end = peaks[bg_idx + 1]
                bg = df.iloc[start:end]

                if len(bg) < 30:
                    continue

                voiced_bg = bg[bg['voiced'] == 1.0]
                if len(voiced_bg) < 10:
                    continue

                bg_results.append({
                    'subject_id': sid,
                    'breath_group': bg_idx + 1,
                    'n_breath_groups': len(peaks) - 1,
                    'bg_position': (bg_idx + 1) / (len(peaks) - 1),  # 0..1 normalized
                    'duration_sec': (end - start) * 0.015,
                    'n_voiced': len(voiced_bg),
                    'f0_mean': np.nanmean(voiced_bg['f0']),
                    'energy_mean': voiced_bg['energy'].mean(),
                    'delta_vcw_used': bg['delta_vcw'].iloc[0] - bg['delta_vcw'].iloc[-1],
                    'flow_cw_mean': bg['flow_cw'].mean(),
                    'pct_rc_mean': bg['pct_rc'].mean(),
                })

        except Exception as e:
            logger.warning(f"  Breath-group failed for {h5.name}: {e}")

    if not bg_results:
        print("  No valid breath groups found.")
        return pd.DataFrame()

    bg_df = pd.DataFrame(bg_results)
    bg_df.to_csv(output_dir / "breath_group_analysis.csv", index=False)
    print(f"  Analyzed {len(bg_df)} breath groups from {bg_df['subject_id'].nunique()} subjects")

    # ---- Evolution across breath groups ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_cols = [
        ('bg_position', 'f0_mean', 'F0 (Hz)', 'F0 Evolution Across Reading'),
        ('bg_position', 'energy_mean', 'Energy', 'Energy Evolution Across Reading'),
        ('bg_position', 'delta_vcw_used', 'Volume Used (L)', 'Volume Per Breath Group'),
        ('bg_position', 'pct_rc_mean', '%RC', '%RC Evolution Across Reading'),
    ]

    for idx, (x_col, y_col, ylabel, title) in enumerate(plot_cols):
        ax = axes.flatten()[idx]
        ax.scatter(bg_df[x_col], bg_df[y_col], alpha=0.3, s=15, color='teal')

        # Binned means
        bins = np.linspace(0, 1, 8)
        bg_df['pos_bin'] = pd.cut(bg_df[x_col], bins, labels=False)
        binned = bg_df.groupby('pos_bin')[y_col].agg(['mean', 'std']).dropna()
        bin_centers = (bins[:-1] + bins[1:]) / 2
        if len(binned) > 2:
            ax.errorbar(bin_centers[:len(binned)], binned['mean'], yerr=binned['std'],
                       color='red', linewidth=2, capsize=3, label='binned mean +/- SD')

        # Trend
        valid = bg_df[[x_col, y_col]].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid[x_col], valid[y_col])
            ax.annotate(f'r={r:.3f}, p={p:.3e}', xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Position in text (normalized)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle('Breath-Group Analysis During Text Reading', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "breath_group_evolution.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved: breath_group_evolution.pdf")

    # ---- Within breath-group correlation ----
    bg_corr_cols = ['f0_mean', 'energy_mean', 'delta_vcw_used', 'flow_cw_mean', 'pct_rc_mean', 'duration_sec']
    bg_available = [c for c in bg_corr_cols if c in bg_df.columns]
    bg_corr_data = bg_df[bg_available].dropna()

    if len(bg_corr_data) > 15:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(bg_corr_data.corr(), annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True, ax=ax, linewidths=0.5)
        ax.set_title('Breath-Group Feature Correlations (text reading)')
        plt.tight_layout()
        fig.savefig(output_dir / "breath_group_correlations.pdf", bbox_inches='tight')
        plt.close(fig)
        print("  Saved: breath_group_correlations.pdf")

    # Clean up temp column
    if 'pos_bin' in bg_df.columns:
        bg_df.drop(columns=['pos_bin'], inplace=True)

    return bg_df


# =====================================================================
# LEVEL 7: MFCC-RESPIRATORY correlations
# =====================================================================

def run_mfcc_analysis(paired_dir, output_dir):
    """Level 7: correlate MFCC spectral features with respiratory state."""
    print("\n" + "="*60)
    print("LEVEL 7: MFCC-Respiratory correlations")
    print("="*60)

    h5_files = sorted(paired_dir.glob("*.h5"))

    # Collect per-segment MFCC-OEP correlations
    mfcc_corr_all = []

    for h5 in h5_files:
        try:
            df, meta = PairedFeatureExtractor.load_hdf5(h5)
            task = meta.get('task_name', '')
            sid = meta.get('subject_id', '')

            voiced = df[df['voiced'] == 1.0]
            if len(voiced) < 30:
                continue

            # Find MFCC columns
            mfcc_cols = [c for c in voiced.columns if c.startswith('mfcc_')]
            oep_targets = ['flow_cw', 'delta_vcw', 'pct_rc']

            for oep_col in oep_targets:
                for mfcc_col in mfcc_cols[:8]:  # first 8 MFCCs
                    valid = voiced[[mfcc_col, oep_col]].dropna()
                    if len(valid) > 20:
                        r, p = stats.pearsonr(valid[mfcc_col], valid[oep_col])
                        mfcc_corr_all.append({
                            'subject_id': sid, 'task': task,
                            'mfcc': mfcc_col, 'oep_feature': oep_col,
                            'r': r, 'p': p,
                            'task_type': 'sustained' if task in SUSTAINED_TASKS else 'speech',
                        })

        except Exception as e:
            logger.warning(f"  MFCC analysis failed for {h5.name}: {e}")

    if not mfcc_corr_all:
        print("  No valid MFCC correlations computed.")
        return pd.DataFrame()

    mfcc_df = pd.DataFrame(mfcc_corr_all)
    mfcc_df.to_csv(output_dir / "mfcc_respiratory_correlations.csv", index=False)
    print(f"  Computed {len(mfcc_df)} MFCC-OEP correlations")

    # ---- Mean correlation heatmap: MFCC x OEP feature ----
    oep_targets = ['flow_cw', 'delta_vcw', 'pct_rc']
    mfcc_labels = sorted(mfcc_df['mfcc'].unique(), key=lambda x: int(x.split('_')[1]))[:8]

    for task_type in ['sustained', 'speech', 'all']:
        if task_type == 'all':
            subset = mfcc_df
        else:
            subset = mfcc_df[mfcc_df['task_type'] == task_type]

        if len(subset) < 10:
            continue

        # Pivot to mean r
        pivot = subset.groupby(['mfcc', 'oep_feature'])['r'].mean().unstack(fill_value=0)
        # Reorder
        pivot = pivot.reindex(index=[m for m in mfcc_labels if m in pivot.index],
                             columns=[c for c in oep_targets if c in pivot.columns])

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r',
                    center=0, vmin=-0.3, vmax=0.3, ax=ax, linewidths=0.5)
        label = task_type.upper() if task_type != 'all' else 'ALL'
        ax.set_title(f'Mean MFCC vs Respiratory Correlation ({label} tasks)')
        ax.set_xlabel('Respiratory Feature')
        ax.set_ylabel('MFCC Coefficient')
        plt.tight_layout()
        fig.savefig(output_dir / f"mfcc_oep_heatmap_{task_type}.pdf", bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: mfcc_oep_heatmap_{task_type}.pdf")

    # ---- Distribution of r-values per MFCC ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, oep_col in zip(axes, oep_targets):
        sub = mfcc_df[mfcc_df['oep_feature'] == oep_col]
        data_pivot = []
        labels = []
        for mfcc in mfcc_labels:
            vals = sub[sub['mfcc'] == mfcc]['r'].values
            if len(vals) > 0:
                data_pivot.append(vals)
                labels.append(mfcc.replace('mfcc_', 'C'))
        if data_pivot:
            ax.boxplot(data_pivot, labels=labels)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel('Pearson r')
            ax.set_title(f'MFCC vs {oep_col}')
            ax.set_xlabel('MFCC coefficient')
    plt.suptitle('Distribution of MFCC-Respiratory Correlations', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "mfcc_oep_boxplots.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved: mfcc_oep_boxplots.pdf")

    return mfcc_df


# =====================================================================
# TEXT REPORT
# =====================================================================

def write_report(summary_df, frc_df, lag_df, bg_df, mfcc_df, meta_df, output_dir):
    """Write a comprehensive text summary of all M2 findings."""
    report_path = output_dir / "m2_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("M2 - EXPLORATORY CORRELATION ANALYSIS REPORT (Extended)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total segments analyzed: {len(summary_df)}\n")
        f.write(f"Unique subjects: {summary_df['subject_id'].nunique()}\n")
        f.write(f"Tasks: {sorted(summary_df['task'].unique())}\n\n")

        # Within-segment correlations
        f.write("WITHIN-SEGMENT CORRELATIONS (per recording)\n")
        f.write("-" * 40 + "\n")
        for col, label in [
            ('corr_energy_deltavcw', 'Energy vs dVcw'),
            ('corr_f0_flowcw', 'F0 vs Flow CW'),
            ('corr_energy_flowcw', 'Energy vs Flow CW'),
        ]:
            if col in summary_df.columns:
                data = summary_df[col].dropna()
                f.write(f"  {label}:\n")
                f.write(f"    median r = {data.median():.3f}\n")
                f.write(f"    mean r   = {data.mean():.3f} +/- {data.std():.3f}\n")
                f.write(f"    range    = [{data.min():.3f}, {data.max():.3f}]\n")
                f.write(f"    n        = {len(data)}\n\n")

        # FRC analysis
        if frc_df is not None and len(frc_df) > 0:
            f.write("\nFRC CROSSING ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Segments with FRC crossing: {len(frc_df)}\n\n")
            for col, label in [
                ('f0_shift', 'F0 shift (Below-Above)'),
                ('energy_shift', 'Energy shift'),
                ('pct_rc_shift', '%RC shift'),
            ]:
                data = frc_df[col].dropna()
                f.write(f"  {label}:\n")
                f.write(f"    median = {data.median():.4f}\n")
                f.write(f"    mean   = {data.mean():.4f} +/- {data.std():.4f}\n")
                if len(data) > 5:
                    try:
                        stat, p = stats.wilcoxon(data)
                        f.write(f"    Wilcoxon p = {p:.3e}\n")
                    except Exception:
                        pass
                f.write("\n")

        # Sex-stratified
        if meta_df is not None and len(meta_df) > 0:
            f.write("\nSEX-STRATIFIED ANALYSIS\n")
            f.write("-" * 40 + "\n")
            merged = summary_df.merge(meta_df, on='subject_id', how='left').dropna(subset=['sex'])
            n_m = (merged['sex'] == 'M').sum()
            n_f = (merged['sex'] == 'F').sum()
            f.write(f"  Male segments: {n_m}\n")
            f.write(f"  Female segments: {n_f}\n")
            f.write("  See: sex_partial_correlations.csv for details\n\n")

        # Lag analysis
        if lag_df is not None and len(lag_df) > 0:
            f.write("\nCROSS-CORRELATION LAG ANALYSIS\n")
            f.write("-" * 40 + "\n")
            for col, label in [
                ('lag_flow_energy_sec', 'Flow -> Energy peak lag'),
                ('lag_flow_f0_sec', 'Flow -> F0 peak lag'),
            ]:
                data = lag_df[col].dropna() * 1000  # to ms
                f.write(f"  {label}:\n")
                f.write(f"    median = {data.median():.1f} ms\n")
                f.write(f"    mean   = {data.mean():.1f} +/- {data.std():.1f} ms\n")
                f.write(f"    n      = {len(data)}\n\n")

        # Breath-group
        if bg_df is not None and len(bg_df) > 0:
            f.write("\nBREATH-GROUP ANALYSIS (text reading)\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total breath groups: {len(bg_df)}\n")
            f.write(f"  Subjects: {bg_df['subject_id'].nunique()}\n")
            f.write(f"  Mean breath groups per subject: "
                    f"{bg_df.groupby('subject_id').size().mean():.1f}\n")
            f.write(f"  Mean duration: {bg_df['duration_sec'].mean():.2f} +/- "
                    f"{bg_df['duration_sec'].std():.2f} s\n\n")

        # MFCC
        if mfcc_df is not None and len(mfcc_df) > 0:
            f.write("\nMFCC-RESPIRATORY CORRELATIONS\n")
            f.write("-" * 40 + "\n")
            # Find strongest MFCC-OEP pairs
            mean_abs_r = mfcc_df.groupby(['mfcc', 'oep_feature'])['r'].apply(
                lambda x: np.mean(np.abs(x))).reset_index()
            mean_abs_r.columns = ['mfcc', 'oep_feature', 'mean_abs_r']
            top = mean_abs_r.nlargest(5, 'mean_abs_r')
            f.write("  Strongest mean |r| pairs:\n")
            for _, row in top.iterrows():
                f.write(f"    {row['mfcc']} vs {row['oep_feature']}: "
                        f"mean |r| = {row['mean_abs_r']:.3f}\n")
            f.write("\n")

    print(f"\n  Saved: m2_report.txt")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    batch_name = select_batch()
    paired_dir = DATA_TARGET / batch_name / "paired"
    output_dir = DATA_TARGET / batch_name / "m2_correlation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta_df = load_subject_metadata(batch_name)

    # Level 1: Global
    summary_df = run_global_analysis(paired_dir, output_dir)

    # Level 2: Time-resolved
    time_corrs = run_time_resolved(paired_dir, output_dir, max_subjects=10)

    # Level 3: FRC
    frc_df = run_frc_analysis(paired_dir, output_dir)

    # Level 4: Sex-stratified
    run_sex_stratified(summary_df, meta_df, output_dir)

    # Level 5: Lag analysis
    lag_df = run_lag_analysis(paired_dir, output_dir)

    # Level 6: Breath-group
    bg_df = run_breath_group_analysis(paired_dir, output_dir)

    # Level 7: MFCC-respiratory
    mfcc_df = run_mfcc_analysis(paired_dir, output_dir)

    # Report
    write_report(summary_df, frc_df, lag_df, bg_df, mfcc_df, meta_df, output_dir)

    print("\n" + "="*60)
    print("M2 COMPLETE (Extended)")
    print(f"All outputs in: {output_dir.relative_to(PROJECT_ROOT)}")
    print("="*60)