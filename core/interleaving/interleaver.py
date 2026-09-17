"""
Interleaver — block, convolutional (Forney), diagonal, pseudo-random.

Created for CATEGORY 1 verification: enables roundtrip interleave→deinterleave testing.
Matches deinterleaver.py algorithms exactly.
"""
import numpy as np
from typing import Optional


def interleave(bits: np.ndarray, method: str,
               rows: int = 8, cols: int = None,
               seed: int = 42) -> np.ndarray:
    """
    Interleave a bit array.
    
    Parameters
    ----------
    bits   : input bit array (numpy uint8)
    method : 'block' | 'convolutional' | 'diagonal' | 'pseudorandom'
    rows   : block rows / interleaver depth
    cols   : block columns (auto = len(bits)//rows if None)
    seed   : seed for pseudo-random interleaving
    
    Returns
    -------
    interleaved : rearranged bit array (same length as input)
    """
    method = method.lower().replace('-', '').replace('_', '').replace(' ', '')
    
    if method == 'block':
        return _interleave_block(bits, rows, cols)
    elif method in ('convolutional', 'conv'):
        return _interleave_convolutional(bits, rows)
    elif method == 'diagonal':
        return _interleave_diagonal(bits, rows, cols)
    elif method in ('pseudorandom', 'pseudo', 'random'):
        return _interleave_pseudorandom(bits, seed)
    else:
        raise ValueError(f"Unknown interleaving method: {method}")


def _interleave_block(bits: np.ndarray, rows: int,
                      cols: Optional[int]) -> np.ndarray:
    """
    Block interleaver.
    
    Algorithm (content rephrased for compliance with licensing restrictions):
    - Fill matrix row-by-row (rows × cols)
    - Read column-by-column
    
    This spreads burst errors across multiple codewords.
    Always returns exactly len(bits) elements.
    
    Reference: [Block interleaver concept](https://researchgate.net/figure/Convolutional-Interleaving_fig4_280697951)
    """
    n = len(bits)
    if cols is None:
        cols = max(1, n // rows)
    
    total = rows * cols
    
    # Pad to complete block, process, trim to original length
    if n < total:
        padded = np.pad(bits, (0, total - n))
    elif n > total:
        cols = int(np.ceil(n / rows))
        total = rows * cols
        padded = np.pad(bits, (0, total - n))
    else:
        padded = bits.copy()
    
    # Fill matrix row-by-row
    matrix = padded.reshape(rows, cols)
    # Read column-by-column: transpose then flatten
    interleaved = matrix.T.flatten()
    
    return interleaved[:n].astype(np.uint8)


def _interleave_convolutional(bits: np.ndarray, depth: int) -> np.ndarray:
    """
    Convolutional interleaver (Forney-style) — correct finite-block implementation.

    The Forney interleaver introduces different delays on each of D branches.
    Branch i (0 ≤ i < D) delays the bits assigned to it by i×D positions.

    For a finite block of N bits:
      1. Assign bit k to branch (k mod D)
      2. Within branch i, delay each bit by i×D positions (shift forward)
      3. Collect output in branch order

    This produces a permutation distinct from diagonal interleaving.
    The matching de-interleaver _deinterleave_convolutional reverses it exactly.
    Always returns exactly len(bits) elements.
    """
    n = len(bits)
    D = depth
    output = np.zeros(n, dtype=np.uint8)

    # Collect bits for each branch: branch i gets bits at positions i, i+D, i+2D, ...
    for i in range(D):
        branch_positions = list(range(i, n, D))
        if not branch_positions:
            continue
        branch_bits = bits[branch_positions]
        delay = i * D  # branch i has delay i*D

        # Apply circular delay within the branch
        branch_len = len(branch_bits)
        if delay > 0 and branch_len > 0:
            delay_eff = delay % branch_len
            branch_bits = np.roll(branch_bits, delay_eff)

        # Write back to output at same positions
        for out_idx, pos in enumerate(branch_positions):
            output[pos] = branch_bits[out_idx]

    return output.astype(np.uint8)


def _interleave_diagonal(bits: np.ndarray, rows: int,
                         cols: Optional[int]) -> np.ndarray:
    """
    Diagonal (helical) interleaver.
    
    Algorithm (content rephrased for compliance with licensing restrictions):
    - Fill matrix row-by-row sequentially
    - Read diagonally with wraparound
    
    This provides better decorrelation than simple block interleaving.
    Always returns exactly len(bits) elements.
    
    Reference: Diagonal interleaving pattern from wireless communications
    """
    n = len(bits)
    if cols is None:
        cols = max(1, n // rows)
    
    total = rows * cols
    if n > total:
        cols = int(np.ceil(n / rows))
        total = rows * cols
    
    padded = np.pad(bits, (0, total - n)) if n < total else bits[:total]
    
    # Fill matrix row-by-row
    matrix = padded.reshape(rows, cols)
    
    # Read diagonally: for output index i, read from position (r, c) where:
    # r = i % rows, c = (i // rows + r) % cols
    interleaved = np.zeros(total, dtype=np.uint8)
    for idx in range(total):
        r = idx % rows
        c = (idx // rows + r) % cols
        interleaved[idx] = matrix[r, c]
    
    return interleaved[:n].astype(np.uint8)


def _interleave_pseudorandom(bits: np.ndarray, seed: int) -> np.ndarray:
    """
    Pseudo-random interleaver.
    
    Algorithm (content rephrased for compliance with licensing restrictions):
    - Generate random permutation using fixed seed
    - Apply permutation to input bits
    - Provides maximum symbol separation
    
    Deterministic (same seed → same permutation) for matching deinterleaver.
    
    Reference: Pseudo-random interleaving for burst error mitigation
    """
    n = len(bits)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    return bits[perm]
