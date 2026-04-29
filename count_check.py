import pandas as pd
from pathlib import Path

# Cohort
meta = pd.read_csv('data_root/healthy_subjects/subjects_metadata.csv', sep=';')
meta.columns = meta.columns.str.strip()
meta['Gender'] = meta['Gender'].str.strip()

m = meta[meta['Gender'] == 'M']
f = meta[meta['Gender'] == 'W']
print(f'=== COHORT ===')
print(f'Total: {len(meta)}  |  Male: {len(m)}  |  Female: {len(f)}')
print(f'Age M: {m["Age"].mean():.1f} +/- {m["Age"].std():.1f}')
print(f'Age F: {f["Age"].mean():.1f} +/- {f["Age"].std():.1f}')

# Extraction
summary = pd.read_csv('data_target/healthy_subjects/paired/extraction_summary.csv')
ok = (summary['status'] == 'ok').sum()
skip = (summary['status'] == 'skipped').sum()
fail = len(summary) - ok - skip
subj = summary[summary['status'] == 'ok']['subject'].nunique()
print(f'\n=== EXTRACTION ===')
print(f'Total tasks: {len(summary)}')
print(f'Successful: {ok}  |  Skipped: {skip}  |  Failed: {fail}')
print(f'Subjects extracted: {subj}')

# FRC
frc_path = Path('data_target/healthy_subjects/m2_correlation/frc_analysis.csv')
if frc_path.exists():
    frc = pd.read_csv(frc_path)
    print(f'\n=== FRC ===')
    print(f'Segments: {len(frc)}  |  Subjects: {frc["subject_id"].nunique()}')

# Breath groups
bg_path = Path('data_target/healthy_subjects/m2_correlation/breath_group_analysis.csv')
if bg_path.exists():
    bg = pd.read_csv(bg_path)
    print(f'\n=== BREATH GROUPS ===')
    print(f'Total: {len(bg)}  |  Subjects: {bg["subject_id"].nunique()}')

# HDF5
n_h5 = len(list(Path('data_target/healthy_subjects/paired').glob('*.h5')))
print(f'\n=== HDF5 FILES ===')
print(f'Total: {n_h5}')

print(f'\n==============================================')

frc = pd.read_csv('data_target/healthy_subjects/m2_correlation/frc_analysis.csv')
print(f'FRC subjects: {frc["subject_id"].nunique()}')