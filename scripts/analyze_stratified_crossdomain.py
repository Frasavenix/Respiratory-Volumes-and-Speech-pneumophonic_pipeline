#!/usr/bin/env python3
"""
Stratified, per-task cross-domain correlations (audio <-> OEP, incl. %RC/%AB).
=============================================================================

For every paired recording we compute the *within-segment* Pearson r between
each acoustic feature and each OEP feature over voiced frames, then aggregate
by (task x demographic stratum) to find which acoustic feature and which task
give a **consistent** audio<->physiology coupling.

Consistency = high |mean r| AND high sign-consistency (fraction of recordings
whose r shares the group's median sign) over enough recordings.

Note: pct_rc + pct_ab = 1, so r(x, pct_ab) = -r(x, pct_rc) exactly — pct_ab is
reported as the mirror of pct_rc for completeness.

Outputs (under --output-dir):
  crossdomain_per_recording.csv     long table, one row per (recording, pair)
  crossdomain_by_task_stratum.csv   aggregated mean r / sign-consistency / n
  heatmap_<stratum>_task_by_pair.pdf mean r, task x feature-pair, per stratum
  heatmap_pair_by_stratum_sustained.pdf mean r over sustained tasks, pair x stratum
  top_consistent_couplings.csv      ranked shortlist

Usage:
  python scripts/analyze_stratified_crossdomain.py \
      --paired-dir data_target/healthy_subjects/paired \
      --metadata   data_root/healthy_subjects/subjects_metadata.csv \
      --output-dir data_target/healthy_subjects/M2_crossdomain
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pneumophonic_analysis.paired_features import PairedFeatureExtractor
from analyze_l3_stratified import load_metadata, assign_demographic_group

ACOUSTIC = ['f0', 'energy', 'spectral_centroid']
OEP = ['delta_vcw', 'flow_cw', 'pct_rc', 'pct_ab']
SUSTAINED = ['a', 'e', 'i', 'o', 'u', 'a_2', 'a_3', 'a_7', 'r']
SPEECH = ['f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'testo']
STRATA_ORDER = ['All', 'Male', 'Female', 'Young', 'Elder', 'YM', 'YF', 'EM', 'EF']
MIN_FRAMES = 15


def per_recording_corr(df: pd.DataFrame) -> dict:
    """Within-recording Pearson r for each acoustic x OEP pair, voiced frames."""
    v = df[df['voiced'] == 1.0]
    out = {}
    if len(v) < MIN_FRAMES:
        return out
    for a in ACOUSTIC:
        if a not in v.columns:
            continue
        for o in OEP:
            if o not in v.columns:
                continue
            x = v[a].values.astype(float)
            y = v[o].values.astype(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= MIN_FRAMES and np.std(x[m]) > 1e-9 and np.std(y[m]) > 1e-9:
                out[f"{a}~{o}"] = float(pearsonr(x[m], y[m])[0])
    return out


def strata_of(sex, ageg, demo):
    s = ['All', demo]
    s.append('Male' if sex == 'M' else 'Female')
    s.append(ageg)  # 'Young' / 'Elder'
    return s


def agg(group: pd.DataFrame) -> pd.Series:
    r = group['r'].values
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return pd.Series({'mean_r': np.nan, 'sign_consistency': np.nan, 'n': 0})
    med_sign = np.sign(np.median(r)) or 1
    sign_consistency = float(np.mean(np.sign(r) == med_sign))
    return pd.Series({'mean_r': float(np.mean(r)),
                      'sign_consistency': sign_consistency,
                      'n': int(len(r))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paired-dir', type=Path, required=True)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = assign_demographic_group(load_metadata(args.metadata))
    mm = {r['subject_id']: (r['sex'], r['age_group'], r['demographic'])
          for _, r in meta.iterrows()}

    rows = []
    for h5 in sorted(args.paired_dir.glob('*.h5')):
        try:
            df, attrs = PairedFeatureExtractor.load_hdf5(h5)
        except Exception:
            continue
        sid = attrs.get('subject_id') or h5.stem.split('_', 1)[0]
        task = attrs.get('task_name') or h5.stem.split('_', 1)[1]
        if sid not in mm:
            continue
        sex, ageg, demo = mm[sid]
        for pair, r in per_recording_corr(df).items():
            rows.append(dict(subject=sid, task=task, sex=sex, age_group=ageg,
                             demographic=demo, pair=pair, r=r))
    L = pd.DataFrame(rows)
    L.to_csv(args.output_dir / 'crossdomain_per_recording.csv', index=False)
    print(f"per-recording correlations: {len(L)} rows, "
          f"{L['subject'].nunique()} subjects, {L['task'].nunique()} tasks")

    # ---- aggregate by (pair, task, stratum) ----
    out = []
    for stratum in STRATA_ORDER:
        if stratum == 'All':
            sub = L
        elif stratum in ('Male', 'Female'):
            sub = L[L['sex'] == ('M' if stratum == 'Male' else 'F')]
        elif stratum in ('Young', 'Elder'):
            sub = L[L['age_group'] == stratum]
        else:
            sub = L[L['demographic'] == stratum]
        for (pair, task), g in sub.groupby(['pair', 'task']):
            a = agg(g)
            out.append(dict(stratum=stratum, pair=pair, task=task,
                            mean_r=a['mean_r'], sign_consistency=a['sign_consistency'],
                            n=int(a['n'])))
        # sustained / speech scope aggregates
        for scope, tasks in [('SUSTAINED', SUSTAINED), ('SPEECH', SPEECH)]:
            ss = sub[sub['task'].isin(tasks)]
            for pair, g in ss.groupby('pair'):
                a = agg(g)
                out.append(dict(stratum=stratum, pair=pair, task=scope,
                                mean_r=a['mean_r'], sign_consistency=a['sign_consistency'],
                                n=int(a['n'])))
    S = pd.DataFrame(out)
    S.to_csv(args.output_dir / 'crossdomain_by_task_stratum.csv', index=False)

    # ---- heatmap: pair x stratum, over SUSTAINED scope ----
    piv = (S[S['task'] == 'SUSTAINED']
           .pivot_table(index='pair', columns='stratum', values='mean_r')
           .reindex(columns=[s for s in STRATA_ORDER]))
    pair_order = [f"{a}~{o}" for o in OEP for a in ACOUSTIC]
    piv = piv.reindex([p for p in pair_order if p in piv.index])
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(piv.values, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=7, color='black' if abs(val) < 0.32 else 'white')
    ax.set_title('Cross-domain mean r — SUSTAINED phonation  (acoustic~OEP x stratum)')
    fig.colorbar(im, ax=ax, fraction=0.025, label='mean Pearson r')
    plt.tight_layout(); fig.savefig(args.output_dir / 'heatmap_pair_by_stratum_sustained.pdf', bbox_inches='tight')
    plt.close(fig)

    # ---- per-stratum task x pair heatmaps (key strata) ----
    for stratum in ['All', 'Young', 'Elder', 'Male', 'Female']:
        sub = S[(S['stratum'] == stratum) & (S['task'].isin(SUSTAINED + SPEECH))]
        piv = sub.pivot_table(index='task', columns='pair', values='mean_r')
        piv = piv.reindex(index=[t for t in SUSTAINED + SPEECH if t in piv.index],
                          columns=[p for p in pair_order if p in piv.columns])
        if piv.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 6))
        im = ax.imshow(piv.values, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6,
                            color='black' if abs(val) < 0.32 else 'white')
        ax.set_title(f'Cross-domain mean r — stratum={stratum}  (task x acoustic~OEP)')
        fig.colorbar(im, ax=ax, fraction=0.025, label='mean Pearson r')
        plt.tight_layout(); fig.savefig(args.output_dir / f'heatmap_{stratum}_task_by_pair.pdf', bbox_inches='tight')
        plt.close(fig)

    # ---- ranked shortlist: strong + consistent couplings ----
    cand = S[(S['n'] >= 6) & (S['task'].isin(SUSTAINED + ['SUSTAINED']))].copy()
    cand['score'] = cand['mean_r'].abs() * cand['sign_consistency']
    top = cand.sort_values('score', ascending=False).head(30)
    top.to_csv(args.output_dir / 'top_consistent_couplings.csv', index=False)
    print("\n=== TOP consistent cross-domain couplings (sustained) ===")
    print(top[['stratum', 'task', 'pair', 'mean_r', 'sign_consistency', 'n']]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == '__main__':
    main()
