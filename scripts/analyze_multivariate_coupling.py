#!/usr/bin/env python3
"""Multivariate audio <-> OEP coupling: CCA (symmetric) + PLS (audio -> %RC).
=============================================================================

Supervisors' ask 5: move beyond pairwise Pearson to *subsets* of features, and
ask 3 (inverse direction). We analyse **within-subject z-scored voiced frames**
from sustained phonation (the glissando ``a_7`` is excluded), per demographic
stratum, with everything cross-validated by subject so the numbers do not
overfit.

Two complementary tools:

* **CCA** finds the linear combination of the audio block and the linear
  combination of the OEP block whose correlation is maximal. We report the
  first canonical correlation (a) in-sample and (b) under subject-held-out
  cross-validation, plus the audio/OEP feature loadings on the first canonical
  variate (which features carry the shared variance). This is the symmetric,
  *both-directions* coupling.
* **PLS** regresses %RC on the audio block (the multivariate version of the
  per-feature Ridge in notebook n7), reported as leave-subjects-out CV R^2 with
  the first-component audio weights (which features drive the prediction).

Audio block (16): f0, energy, spectral_centroid, mfcc_0..12.
OEP block (3):    delta_vcw, flow_cw, pct_rc   (pct_ab = 1 - pct_rc, dropped).

Outputs (under --output-dir):
  multivariate_cca_by_stratum.csv     first canonical r (in-sample + CV), n
  multivariate_pls_by_stratum.csv     audio -> %RC CV R^2, n
  multivariate_audio_loadings.csv     audio-feature loadings (CCA variate) x stratum
  cca_canonical_r_by_stratum.pdf
  pls_r2_by_stratum.pdf
  audio_loadings_heatmap.pdf

Usage:
  python scripts/analyze_multivariate_coupling.py \
      --paired-dir data_target/healthy_subjects/paired \
      --metadata   data_root/healthy_subjects/subjects_metadata.csv \
      --output-dir data_target/healthy_subjects/M3_multivariate
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pneumophonic_analysis.paired_features import PairedFeatureExtractor
from analyze_l3_stratified import load_metadata, assign_demographic_group, STRATA_ORDER

AUDIO = ['f0', 'energy', 'spectral_centroid'] + [f'mfcc_{i}' for i in range(13)]
OEP = ['delta_vcw', 'flow_cw', 'pct_rc']
# Sustained phonation only; the glissando a_7 (A-GLIDE) is excluded — its
# volitional pitch sweep injects a trivial f0<->volume coupling.
SUSTAINED = ['a', 'e', 'i', 'o', 'u', 'a_2', 'a_3', 'r']
MIN_FRAMES_SUBJ = 50      # drop subjects with fewer voiced frames
MIN_FRAMES_STRAT = 200    # skip a stratum with fewer frames
MIN_SUBJ_STRAT = 4        # ...or fewer subjects


def build_frames(paired_dir: Path, metadata: pd.DataFrame, tasks=SUSTAINED) -> pd.DataFrame:
    """Voiced frames from sustained tasks, z-scored *within subject*, with demographics."""
    valid = set(metadata['subject_id'])
    chunks = []
    for h5 in sorted(Path(paired_dir).glob('*.h5')):
        try:
            df, attrs = PairedFeatureExtractor.load_hdf5(h5)
        except Exception:
            continue
        sid, task = attrs.get('subject_id'), attrs.get('task_name')
        if task not in tasks or sid not in valid:
            continue
        v = df[df['voiced'] == 1.0]
        cols = [c for c in AUDIO + OEP if c in v.columns]
        if len(cols) < len(AUDIO + OEP):
            continue
        sub = v[cols].copy()
        sub['subject_id'] = sid
        chunks.append(sub)
    F = pd.concat(chunks, ignore_index=True).dropna(subset=AUDIO + OEP)

    # z-score each feature within subject (remove between-subject baselines:
    # this isolates the *within-subject* coupling, the calibration-free signal)
    for c in AUDIO + OEP:
        g = F.groupby('subject_id')[c]
        F[c] = (F[c] - g.transform('mean')) / (g.transform('std').replace(0, np.nan) + 1e-9)
    F = F.dropna(subset=AUDIO + OEP)

    vc = F['subject_id'].value_counts()
    F = F[F['subject_id'].isin(vc[vc >= MIN_FRAMES_SUBJ].index)]

    meta = assign_demographic_group(metadata.dropna(subset=['sex', 'age']), 55)
    dmap = {r['subject_id']: (r['sex'], r['age_group'], r['demographic'])
            for _, r in meta.iterrows()}
    F = F[F['subject_id'].isin(dmap)].copy()
    F['sex'] = F['subject_id'].map(lambda s: dmap[s][0])
    F['age_group'] = F['subject_id'].map(lambda s: dmap[s][1])
    F['demographic'] = F['subject_id'].map(lambda s: dmap[s][2])
    return F


def stratum_index(F: pd.DataFrame, stratum: str):
    if stratum == 'All':
        return F.index
    if stratum in ('Male', 'Female'):
        return F.index[F['sex'] == ('M' if stratum == 'Male' else 'F')]
    if stratum in ('Young', 'Elder'):
        return F.index[F['age_group'] == stratum]
    return F.index[F['demographic'] == stratum]


def _n_splits(n_subj):
    return int(min(5, n_subj))


def cca_eval(F: pd.DataFrame, idx) -> dict | None:
    """First canonical correlation (in-sample + subject-CV) and feature loadings."""
    sub = F.loc[idx]
    subs = sub['subject_id'].unique()
    if len(subs) < MIN_SUBJ_STRAT or len(sub) < MIN_FRAMES_STRAT:
        return None
    X, Y, g = sub[AUDIO].values, sub[OEP].values, sub['subject_id'].values

    # subject-held-out canonical correlation (guards against CCA overfitting)
    gkf = GroupKFold(n_splits=_n_splits(len(subs)))
    cv_rs = []
    for tr, te in gkf.split(X, Y, g):
        try:
            cca = CCA(n_components=1, max_iter=1000).fit(X[tr], Y[tr])
            xc, yc = cca.transform(X[te], Y[te])
            if np.std(xc[:, 0]) > 1e-9 and np.std(yc[:, 0]) > 1e-9:
                cv_rs.append(abs(pearsonr(xc[:, 0], yc[:, 0])[0]))
        except Exception:
            continue
    r_cv = float(np.mean(cv_rs)) if cv_rs else np.nan

    # in-sample fit for the loadings (corr of each feature with its canonical variate)
    cca = CCA(n_components=1, max_iter=1000).fit(X, Y)
    xc, yc = cca.transform(X, Y)
    r_in = abs(pearsonr(xc[:, 0], yc[:, 0])[0])
    # sign-align so the audio variate's dominant loading is positive (readability)
    a_load = {a: float(pearsonr(sub[a].values, xc[:, 0])[0]) for a in AUDIO}
    o_load = {o: float(pearsonr(sub[o].values, yc[:, 0])[0]) for o in OEP}
    if sum(v for v in a_load.values()) < 0:
        a_load = {k: -v for k, v in a_load.items()}
        o_load = {k: -v for k, v in o_load.items()}
    return dict(r_in=r_in, r_cv=r_cv, n=len(sub), n_subj=len(subs),
                audio_loadings=a_load, oep_loadings=o_load)


def pls_cv(F: pd.DataFrame, idx, target='pct_rc', n_comp=3) -> dict | None:
    """Leave-subjects-out CV R^2 for audio -> target, plus first-component weights."""
    sub = F.loc[idx]
    subs = sub['subject_id'].unique()
    if len(subs) < 5 or len(sub) < MIN_FRAMES_STRAT:
        return None
    X, y, g = sub[AUDIO].values, sub[target].values, sub['subject_id'].values
    gkf = GroupKFold(n_splits=_n_splits(len(subs)))
    yhat = np.full_like(y, np.nan, dtype=float)
    for tr, te in gkf.split(X, y, g):
        pls = PLSRegression(n_components=min(n_comp, X.shape[1])).fit(X[tr], y[tr])
        yhat[te] = pls.predict(X[te]).ravel()
    m = np.isfinite(yhat)
    r2 = r2_score(y[m], yhat[m]) if m.sum() > 10 else np.nan
    pls = PLSRegression(n_components=min(n_comp, X.shape[1])).fit(X, y)
    w = pls.x_weights_[:, 0]
    weights = {a: float(w[i]) for i, a in enumerate(AUDIO)}
    return dict(r2_cv=float(r2), n=len(sub), n_subj=len(subs), weights=weights)


def run(paired_dir: Path, metadata_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(metadata_path)
    F = build_frames(paired_dir, meta)
    print(f"frames: {len(F):,} voiced (sustained, a_7 excluded) | "
          f"{F['subject_id'].nunique()} subjects")

    cca_rows, pls_rows, load_rows = [], [], []
    for stratum in STRATA_ORDER:
        idx = stratum_index(F, stratum)
        c = cca_eval(F, idx)
        if c:
            cca_rows.append(dict(stratum=stratum, r_in=c['r_in'], r_cv=c['r_cv'],
                                 n=c['n'], n_subj=c['n_subj']))
            for a, v in c['audio_loadings'].items():
                load_rows.append(dict(stratum=stratum, feature=a, loading=v))
        p = pls_cv(F, idx)
        if p:
            pls_rows.append(dict(stratum=stratum, r2_cv=p['r2_cv'],
                                 n=p['n'], n_subj=p['n_subj']))
    C = pd.DataFrame(cca_rows)
    P = pd.DataFrame(pls_rows)
    L = pd.DataFrame(load_rows)
    C.to_csv(output_dir / 'multivariate_cca_by_stratum.csv', index=False)
    P.to_csv(output_dir / 'multivariate_pls_by_stratum.csv', index=False)
    L.to_csv(output_dir / 'multivariate_audio_loadings.csv', index=False)

    # ---- plots ----
    if not C.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(C)); w = 0.4
        ax.bar(x - w/2, C['r_in'], w, label='in-sample', color='#bdbdbd')
        ax.bar(x + w/2, C['r_cv'], w, label='subject-CV', color='#4c72b0')
        ax.set_xticks(x); ax.set_xticklabels(C['stratum'])
        ax.set_ylabel('First canonical correlation'); ax.set_ylim(0, 1)
        ax.set_title('Multivariate audio<->OEP coupling (CCA) by stratum')
        ax.legend(); plt.tight_layout()
        fig.savefig(output_dir / 'cca_canonical_r_by_stratum.pdf', bbox_inches='tight')
        plt.close(fig)
    if not P.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(P)), P['r2_cv'], color='#55a868')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(range(len(P))); ax.set_xticklabels(P['stratum'])
        ax.set_ylabel('CV R² (audio -> %RC, PLS)')
        ax.set_title('Multivariate audio -> %RC prediction (PLS) by stratum')
        plt.tight_layout()
        fig.savefig(output_dir / 'pls_r2_by_stratum.pdf', bbox_inches='tight')
        plt.close(fig)
    if not L.empty:
        piv = L.pivot_table(index='feature', columns='stratum', values='loading').reindex(
            index=AUDIO, columns=[s for s in STRATA_ORDER if s in set(L['stratum'])])
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(piv.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_title('Audio-feature loadings on the first canonical variate')
        fig.colorbar(im, ax=ax, fraction=0.03, label='loading (corr with audio variate)')
        plt.tight_layout()
        fig.savefig(output_dir / 'audio_loadings_heatmap.pdf', bbox_inches='tight')
        plt.close(fig)

    print("\n=== CCA (first canonical correlation) ===")
    print(C.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== PLS (audio -> %RC, CV R²) ===")
    print(P.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote outputs to {output_dir}")
    return F, C, P, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paired-dir', type=Path, required=True)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    run(args.paired_dir, args.metadata, args.output_dir)


if __name__ == '__main__':
    main()
