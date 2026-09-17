"""
Dataset Loader — loads RadioML 2016.10A and 2016.04C datasets for ML training.

Supports:
  - RML2016.10a_dict.pkl       (Python 3 pickle, latin1)
  - 2016.04C.multisnr.tar.bz2  (Python 2 pickle, latin1, inside tar.bz2)
  - RML2016.10a.tar.bz2        (same as .pkl, compressed)

Output: (X, y) where:
  X: complex64 numpy array, shape (N, 1024) — upsampled from 128 to 1024
  y: int32 numpy array, shape (N,)

Class mapping aligns to your existing MODULATION_LABELS in training_data.py.
"""
import os
import pickle
import tarfile
import numpy as np
from typing import Tuple, List, Optional

# Map RadioML 2016 modulation names → your project's labels
# RadioML 2016 uses 11 classes; map to the nearest label in your 17-class set
RADIOML_TO_PROJECT = {
    '8PSK':   '8PSK',
    'AM-DSB': 'AM-DSB',
    'AM-SSB': 'AM-SSB',
    'BPSK':   'BPSK',
    'CPFSK':  'CPFSK',
    'GFSK':   'GFSK',
    'PAM4':   'PAM4',
    'QAM16':  'QAM16',
    'QAM64':  'QAM64',
    'QPSK':   'QPSK',
    'WBFM':   'WBFM',
}


def load_radioml_pkl(pkl_path: str,
                     snr_min: int = -20,
                     snr_max: int = 18,
                     upsample_to: int = 1024,
                     verbose: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a RadioML 2016 pickle file directly.

    Parameters
    ----------
    pkl_path    : path to .pkl file (latin1 encoding for Py2 pickles)
    snr_min/max : filter SNR range (dB)
    upsample_to : resample each 128-point sample to this length for the classifier
    verbose     : print progress

    Returns
    -------
    X          : complex64 array (N, upsample_to)
    y          : int32 label array (N,)
    label_names: list of class names indexed by y
    """
    if verbose:
        print(f"  Loading: {os.path.basename(pkl_path)}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    return _parse_radioml_dict(data, snr_min, snr_max, upsample_to, verbose)


def load_radioml_tar(tar_path: str,
                     snr_min: int = -20,
                     snr_max: int = 18,
                     upsample_to: int = 1024,
                     verbose: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a RadioML 2016 dataset from a .tar.bz2 archive.
    Works with both Python 2 (latin1) and Python 3 pickles inside.

    Parameters: same as load_radioml_pkl
    """
    if verbose:
        print(f"  Loading from tar: {os.path.basename(tar_path)}")

    with tarfile.open(tar_path, "r:bz2") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".pkl") and m.size > 1000:
                if verbose:
                    print(f"    Extracting: {m.name} ({m.size/1024/1024:.1f} MB)")
                fobj = tar.extractfile(m)
                raw = fobj.read()
                # Try latin1 first (Python 2 pickle), then default
                for enc in ("latin1", "bytes", "utf-8"):
                    try:
                        data = pickle.loads(raw, encoding=enc)
                        if isinstance(data, dict) and len(data) > 0:
                            # Fix bytes keys from encoding='bytes'
                            if enc == "bytes":
                                data = {
                                    (k[0].decode("latin1") if isinstance(k[0], bytes) else k[0],
                                     k[1]): v
                                    for k, v in data.items()
                                }
                            break
                    except Exception:
                        continue
                else:
                    raise ValueError(f"Cannot decode pickle in {tar_path}")
                return _parse_radioml_dict(data, snr_min, snr_max,
                                           upsample_to, verbose)

    raise ValueError(f"No .pkl file found in {tar_path}")


def _parse_radioml_dict(data: dict,
                         snr_min: int,
                         snr_max: int,
                         upsample_to: int,
                         verbose: bool
                         ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert RadioML dict → (X complex64, y int32, label_names)."""
    keys  = list(data.keys())
    mods  = sorted(set(k[0] for k in keys))
    snrs  = sorted(set(k[1] for k in keys))

    if verbose:
        print(f"    Modulations ({len(mods)}): {mods}")
        print(f"    SNR levels  ({len(snrs)}): {snrs}")

    # Build label list (only mods we recognise in the project)
    label_names = sorted(set(
        RADIOML_TO_PROJECT.get(m, m) for m in mods
    ))
    label_to_idx = {l: i for i, l in enumerate(label_names)}

    X_list, y_list = [], []
    for (mod, snr), samples in data.items():
        if snr < snr_min or snr > snr_max:
            continue
        proj_label = RADIOML_TO_PROJECT.get(mod, mod)
        if proj_label not in label_to_idx:
            continue

        # samples: (N, 2, 128)  →  complex: (N, 128)
        N = samples.shape[0]
        iq = samples[:, 0, :] + 1j * samples[:, 1, :]  # (N, 128)

        # Upsample from 128 → upsample_to using linear interp (fast, no scipy needed)
        if upsample_to != 128:
            iq_up = np.zeros((N, upsample_to), dtype=np.complex64)
            t_src = np.linspace(0, 1, 128)
            t_dst = np.linspace(0, 1, upsample_to)
            for i in range(N):
                iq_up[i] = (np.interp(t_dst, t_src, iq[i].real) +
                             1j * np.interp(t_dst, t_src, iq[i].imag))
            iq = iq_up

        X_list.append(iq.astype(np.complex64))
        y_list.extend([label_to_idx[proj_label]] * N)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)

    if verbose:
        print(f"    Loaded: {X.shape[0]:,} samples, {len(label_names)} classes")

    return X, y, label_names


def load_all_available(base_dir: str,
                       snr_min: int = -6,
                       snr_max: int = 18,
                       upsample_to: int = 1024,
                       verbose: bool = True
                       ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Automatically discovers and loads all available RadioML datasets
    in base_dir, combining them into a single (X, y) dataset.

    SNR filtering: default -6 to +18 dB (avoids very low SNR noise)

    Returns
    -------
    X          : combined complex64 array (N_total, upsample_to)
    y          : combined int32 labels (N_total,)
    label_names: class names indexed by y
    """
    candidates = [
        ("RML2016.10a_dict.pkl",       "pkl"),   # primary — direct pickle
        # ("RML2016.10a.tar.bz2",      "tar"),   # SKIP — duplicate of above
        ("2016.04C.multisnr.tar.bz2",  "tar"),   # unique dataset (different channel model)
    ]

    all_X, all_y = [], []
    combined_labels = None

    for fname, ftype in candidates:
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            continue

        if verbose:
            print(f"\nLoading {fname}...")
        try:
            if ftype == "pkl":
                X, y, labels = load_radioml_pkl(
                    path, snr_min, snr_max, upsample_to, verbose)
            else:
                X, y, labels = load_radioml_tar(
                    path, snr_min, snr_max, upsample_to, verbose)

            # Align labels across datasets
            if combined_labels is None:
                combined_labels = labels
                all_X.append(X)
                all_y.append(y)
            else:
                # Remap y to combined label space
                label_to_combined = {}
                for i, lbl in enumerate(labels):
                    if lbl not in combined_labels:
                        combined_labels.append(lbl)
                    label_to_combined[i] = combined_labels.index(lbl)

                y_remapped = np.array(
                    [label_to_combined[yi] for yi in y], dtype=np.int32)
                all_X.append(X)
                all_y.append(y_remapped)

        except Exception as e:
            print(f"  WARNING: Could not load {fname}: {e}")

    if not all_X:
        raise RuntimeError(f"No RadioML datasets found in {base_dir}")

    X_combined = np.vstack(all_X)
    y_combined  = np.concatenate(all_y).astype(np.int32)

    # Shuffle
    rng  = np.random.RandomState(42)
    perm = rng.permutation(len(X_combined))
    X_combined = X_combined[perm]
    y_combined  = y_combined[perm]

    if verbose:
        print(f"\nCombined dataset:")
        print(f"  Total samples:  {len(X_combined):,}")
        print(f"  Classes ({len(combined_labels)}): {combined_labels}")
        for i, lbl in enumerate(combined_labels):
            cnt = int(np.sum(y_combined == i))
            print(f"    [{i:2d}] {lbl:12s}: {cnt:,}")

    return X_combined, y_combined, combined_labels
