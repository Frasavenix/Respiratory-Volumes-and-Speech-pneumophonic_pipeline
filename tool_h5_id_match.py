from pathlib import Path
from pneumophonic_analysis.paired_features import PairedFeatureExtractor
import numpy as np

paired_dir = Path('data_target/healthy_subjects/paired')

sample_files = [h5 for h5 in paired_dir.rglob('*.h5')
                if any(t in h5.stem for t in ['_a_2', '_a_3', '_a_7'])][:8]

for h5 in sample_files:
    df, _ = PairedFeatureExtractor.load_hdf5(h5)
    dv = df['delta_vcw'].values
    print(f'{h5.stem:40s}  n={len(dv):4d}  '
          f'dv[0]={dv[0]:+.3f}  '
          f'max={dv.max():+.3f}@{dv.argmax():4d}  '
          f'min={dv.min():+.3f}  '
          f'monotonic_down={bool(np.all(np.diff(dv) <= 0))}')