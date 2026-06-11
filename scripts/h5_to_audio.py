#!/usr/bin/env python3
"""
Reconstruct listenable audio from a paired HDF5 file's stored STFT.
==================================================================

The paired `.h5` files do not store raw audio, but they DO store the STFT power
spectrogram that every audio feature was built from. This tool inverts it with
Griffin-Lim so you can *hear exactly what was extracted* into a segment — useful
for confirming that the real phonation/phrase is captured and the sync pulse is
(or isn't) inside the window.

Notes / caveats:
  * The stored STFT is of the pre-emphasised, noise-reduced, normalised audio.
    A de-emphasis filter is applied by default to restore a natural spectral
    balance (and to make a low-frequency sync pulse audible again).
  * Griffin-Lim reconstructs phase, so the result is intelligible but slightly
    "phasey"/robotic — fine for verification, not for publication.

Usage:
    # one folder of .h5 -> .wav (all files)
    python scripts/h5_to_audio.py --in-dir data_target/healthy_subjects/paired \
        --out-dir data_target/healthy_subjects/audio_check

    # only specific files, tagged with a suffix
    python scripts/h5_to_audio.py --in-dir data_target/healthy_subjects/paired \
        --out-dir data_target/healthy_subjects/audio_check \
        --files FrRo_f_1 FrRo_i AnGu_r --suffix repaired
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py
import librosa
import soundfile as sf
from scipy.signal import lfilter


def h5_to_wav(h5_path: Path, out_path: Path, n_iter: int = 48,
              deemphasis: bool = True) -> float:
    """Reconstruct a wav from one paired .h5 via Griffin-Lim. Returns duration (s)."""
    with h5py.File(str(h5_path), 'r') as f:
        S_power = f['stft'][:]                      # |STFT|^2 (power)
        sr = int(f.attrs.get('sr_audio', 48000))
        hop = int(f.attrs.get('hop_length', 720))
    n_fft = (S_power.shape[0] - 1) * 2              # 721 -> 1440
    mag = np.sqrt(np.maximum(S_power, 0.0))         # power -> magnitude
    y = librosa.griffinlim(mag, n_iter=n_iter, hop_length=hop,
                           win_length=n_fft, n_fft=n_fft)
    if deemphasis:
        # inverse of pre-emphasis y[n] = x[n] - 0.97 x[n-1]
        y = lfilter([1.0], [1.0, -0.97], y)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = 0.95 * y / peak
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y.astype(np.float32), sr)
    return len(y) / sr


def main():
    ap = argparse.ArgumentParser(description="Reconstruct audio from paired HDF5 STFT")
    ap.add_argument('--in-dir', type=Path, required=True, help='Folder of paired *.h5')
    ap.add_argument('--out-dir', type=Path, required=True, help='Where to write *.wav')
    ap.add_argument('--files', nargs='*', default=None,
                    help='Specific file stems (e.g. FrRo_f_1). Default: all *.h5')
    ap.add_argument('--suffix', type=str, default='',
                    help='Append "__<suffix>" to each output name (e.g. repaired)')
    ap.add_argument('--n-iter', type=int, default=48, help='Griffin-Lim iterations')
    ap.add_argument('--no-deemphasis', action='store_true',
                    help='Do not apply de-emphasis (leave pre-emphasised/bright)')
    args = ap.parse_args()

    if not args.in_dir.exists():
        sys.exit(f"in-dir not found: {args.in_dir}")

    if args.files:
        paths = [args.in_dir / (s if s.endswith('.h5') else s + '.h5') for s in args.files]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(args.in_dir.glob('*.h5'))
    if not paths:
        sys.exit("No matching .h5 files found.")

    tag = f"__{args.suffix}" if args.suffix else ""
    print(f"Reconstructing {len(paths)} file(s) -> {args.out_dir}")
    for p in paths:
        out = args.out_dir / f"{p.stem}{tag}.wav"
        try:
            dur = h5_to_wav(p, out, n_iter=args.n_iter,
                            deemphasis=not args.no_deemphasis)
            print(f"  ✓ {out.name}   ({dur:.2f}s)")
        except Exception as e:
            print(f"  ✗ {p.name}: {e}")
    print(f"\nDone. Open {args.out_dir} and listen.")


if __name__ == '__main__':
    main()
