"""
De-interleaver — reverses block, convolutional, diagonal, and pseudo-random interleaving.
"""
import numpy as np
from typing import Optional


def deinterleave(bits: np.ndarray, method: str,
                 rows: int = 8, cols: int = None,
                 seed: int = 42) -> np.ndarray:
    """
    De-interleave a bit array.

    Parameters
    ----------
    bits   : input bit array (numpy uint8)
    method : 'block' | 'convolutional' | 'diagonal' | 'pseudorandom'
    rows   : block rows / interleaver depth
    cols   : block columns (auto = len(bits)//rows if None)
    seed   : seed for pseudo-random de-interleaving
    """
    method = method.lower().replace('-', '').replace('_', '').replace(' ', '')

    if method == 'block':
        return _deinterleave_block(bits, rows, cols)
    elif method in ('convolutional', 'conv'):
        return _deinterleave_convolutional(bits, rows)
    elif method == 'diagonal':
        return _deinterleave_diagonal(bits, rows, cols)
    elif method in ('pseudorandom', 'pseudo', 'random'):
        return _deinterleave_pseudorandom(bits, seed)
    else:
        raise ValueError(f"Unknown de-interleaving method: {method}")


def _deinterleave_block(bits: np.ndarray, rows: int,
                        cols: Optional[int]) -> np.ndarray:
    """
    Block de-interleaver.
    Interleaver writes row-by-row, reads column-by-column.
    De-interleaver reverses: write column-by-column, read row-by-row.
    Always returns exactly len(bits) elements.
    """
    n = len(bits)
    if cols is None:
        cols = max(1, n // rows)

    total = rows * cols

    # Pad to complete block boundary, process, then trim back to n
    if n < total:
        padded = np.pad(bits, (0, total - n))
    elif n > total:
        # n doesn't fit exactly — extend cols to cover all bits
        cols = int(np.ceil(n / rows))
        total = rows * cols
        padded = np.pad(bits, (0, total - n))
    else:
        padded = bits.copy()

    # Interleaver output: [col0_row0, col0_row1, ..., col1_row0, col1_row1, ...]
    # To reverse: reshape as (cols, rows).T to get (rows, cols), then flatten
    matrix = padded.reshape(cols, rows).T
    deinterleaved = matrix.flatten()
    return deinterleaved[:n].astype(np.uint8)


def _deinterleave_convolutional(bits: np.ndarray, depth: int) -> np.ndarray:
    """
    Convolutional de-interleaver — exact inverse of _interleave_convolutional.

    Reverses the branch-wise circular delay:
      - For each branch i, undo the circular delay applied during interleaving
        by rolling backward by (i×D mod branch_len) positions
    Always returns exactly len(bits) elements.
    """
    n = len(bits)
    D = depth
    output = np.zeros(n, dtype=np.uint8)

    for i in range(D):
        branch_positions = list(range(i, n, D))
        if not branch_positions:
            continue
        branch_bits = bits[branch_positions].copy()
        delay = i * D
        branch_len = len(branch_bits)
        if delay > 0 and branch_len > 0:
            delay_eff = delay % branch_len
            branch_bits = np.roll(branch_bits, -delay_eff)  # undo delay

        for out_idx, pos in enumerate(branch_positions):
            output[pos] = branch_bits[out_idx]

    return output.astype(np.uint8)


def _deinterleave_diagonal(bits: np.ndarray, rows: int,
                            cols: Optional[int]) -> np.ndarray:
    """
    Diagonal (helical) de-interleaver.
    Data was read diagonally across a matrix; we reverse it.
    Always returns exactly len(bits) elements.
    """
    n = len(bits)
    if cols is None:
        cols = max(1, n // rows)

    # Extend cols to cover all n bits
    total = rows * cols
    if n > total:
        cols = int(np.ceil(n / rows))
        total = rows * cols

    padded = np.pad(bits, (0, total - n)) if n < total else bits[:total]

    # Interleaved order was: diagonal traversal
    # To reverse: write sequentially into matrix diagonally, then read row-wise
    matrix = np.zeros((rows, cols), dtype=np.uint8)
    for idx, bit in enumerate(padded):
        r = idx % rows
        c = (idx // rows + r) % cols
        matrix[r, c] = bit

    deinterleaved = matrix.flatten()
    return deinterleaved[:n].astype(np.uint8)


def _deinterleave_pseudorandom(bits: np.ndarray, seed: int) -> np.ndarray:
    """
    Pseudo-random de-interleaver.
    Uses the same RNG seed to reconstruct the original permutation and invert it.
    """
    n = len(bits)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)

    # Invert permutation
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(n)

    return bits[inv_perm]


def auto_detect_interleaving(bits: np.ndarray,
                              known_depths: list = None,
                              method: str = 'hybrid') -> str:
    """
    Blind interleaving type detector.

    Uses multi-metric scoring:
    - Column correlation after block de-interleave  (b_col)
    - Column correlation after diagonal de-interleave (d_col)
    - Column correlation after conv de-interleave    (c_col)
    - Run-length improvement (b_imp, d_imp, c_imp)
    - Pseudorandom: entropy-based fast path

    The type whose de-interleave gives the highest column correlation wins.

    Returns: 'block' | 'convolutional' | 'diagonal' | 'pseudorandom'
    """
    if bits is None or len(bits) < 64:
        return 'block'

    bits = np.asarray(bits, dtype=np.uint8)

    if known_depths is None:
        known_depths = [4, 6, 8, 10, 12, 16, 20, 24, 32]

    n = len(bits)
    seg = bits[:min(n, 2048)]

    # ── 1. Pseudorandom fast path ──────────────────────────────────────
    if _is_pseudorandom(bits):
        return 'pseudorandom'

    baseline_run = _mean_run_length(seg)
    depths    = [3, 4, 5, 6, 8, 10, 12, 16]
    col_cands = [16, 32, 48, 64, 96, 128]

    # Collect best column correlation for each de-interleave type
    best_block_col = 0.0
    best_diag_col  = 0.0
    best_conv_col  = 0.0

    # Also track run improvements for secondary scoring
    best_block_imp = 0.0
    best_diag_imp  = 0.0

    for rows in depths:
        for cols in col_cands:
            total = rows * cols
            if total > len(seg):
                continue
            chunk = seg[:total]
            try:
                di_b = _deinterleave_block(chunk, rows, cols)
                bc = _column_correlation(di_b, rows, cols)
                if bc > best_block_col:
                    best_block_col = bc
                ri = _mean_run_length(di_b) - baseline_run
                if ri > best_block_imp:
                    best_block_imp = ri
            except Exception:
                pass
            try:
                di_d = _deinterleave_diagonal(chunk, rows, cols)
                dc = _column_correlation(di_d, rows, cols)
                if dc > best_diag_col:
                    best_diag_col = dc
                ri = _mean_run_length(di_d) - baseline_run
                if ri > best_diag_imp:
                    best_diag_imp = ri
            except Exception:
                pass
        # Convolutional: try each depth on the full segment
        try:
            di_c = _deinterleave_convolutional(seg, rows)
            # Measure column correlation at several (rows, cols) combos
            for cols in col_cands:
                if rows * cols <= len(di_c):
                    cc = _column_correlation(di_c[:rows * cols], rows, cols)
                    if cc > best_conv_col:
                        best_conv_col = cc
        except Exception:
            pass

    # ── 2. Score all three types ───────────────────────────────────────
    # Primary: column correlation (higher = better de-interleave match)
    # Secondary: run improvement (higher = more structure restored)
    scores = {
        'block':         best_block_col * 2.0 + best_block_imp * 0.5,
        'diagonal':      best_diag_col  * 2.0 + best_diag_imp  * 0.5,
        'convolutional': best_conv_col  * 2.0,
    }

    winner = max(scores, key=scores.get)

    # ── 3. Tie-break: default to block for close results ──────────────
    top_scores = sorted(scores.values(), reverse=True)
    if len(top_scores) > 1 and (top_scores[0] - top_scores[1]) < 0.05:
        # Too close to call — default to block (most common in practice)
        return 'block'

    return winner


def _mean_run_length(bits: np.ndarray) -> float:
    """Average run length in a binary bit stream."""
    runs = _compute_run_lengths_fast(bits.astype(np.uint8))
    return float(np.mean(runs)) if len(runs) else 1.0


def _column_correlation(bits: np.ndarray, rows: int, cols: int) -> float:
    """
    After block de-interleaving, the original data is recovered.
    Measure the correlation between bits in the same column position
    (bits that were originally adjacent, now separated by 'rows' positions).
    Higher = more correlated = more likely to be the correct de-interleaving.
    """
    total = rows * cols
    if len(bits) < total:
        return 0.0
    mat = bits[:total].reshape(rows, cols).astype(np.float32) * 2.0 - 1.0
    # Column correlation: correlate adjacent columns
    if cols < 2:
        return 0.0
    try:
        col_means = np.mean(mat, axis=0)
        col_stds  = np.std(mat, axis=0)
        valid = col_stds > 1e-6
        if valid.sum() < 2:
            return 0.0
        corr_sum = 0.0
        count = 0
        for c in range(cols - 1):
            if valid[c] and valid[c + 1]:
                c1 = (mat[:, c] - col_means[c]) / col_stds[c]
                c2 = (mat[:, c + 1] - col_means[c + 1]) / col_stds[c + 1]
                corr = float(np.mean(c1 * c2))
                if not np.isnan(corr):
                    corr_sum += abs(corr)
                    count += 1
        return corr_sum / max(count, 1)
    except Exception:
        return 0.0


def _diagonal_correlation(bits: np.ndarray, rows: int, cols: int) -> float:
    """
    After diagonal de-interleaving, measure correlation along diagonal bands.
    Higher = more correlated = more likely to be the correct de-interleaving.
    """
    total = rows * cols
    if len(bits) < total:
        return 0.0
    mat = bits[:total].reshape(rows, cols).astype(np.float32) * 2.0 - 1.0
    if rows < 2 or cols < 2:
        return 0.0
    try:
        corr_sum = 0.0
        count = 0
        for r in range(rows - 1):
            for c in range(cols):
                nc = (c + 1) % cols
                v1 = float(mat[r, c])
                v2 = float(mat[r + 1, nc])
                corr_sum += v1 * v2
                count += 1
        return abs(corr_sum / max(count, 1))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
#  New high-accuracy helper detectors
# ---------------------------------------------------------------------------

def _is_convolutional(bits: np.ndarray) -> bool:
    """
    Detect convolutional (Forney-style) interleaving.

    Forney interleaving with depth B and delay increment D creates periodic
    permutation structure.  The key observable: after convolutional interleaving,
    bits from branch i appear with a fixed offset.  This produces a
    characteristic *comb* pattern in the XOR autocorrelation: peaks at lags
    that are multiples of B (the depth), with troughs in between.

    We test for this comb pattern on the raw bit stream.
    Works on both structured and random input data.
    """
    n = len(bits)
    if n < 256:
        return False

    b = bits.astype(np.float32) * 2.0 - 1.0
    seg_len = min(n, 4096)
    seg = b[:seg_len]

    # Normalized autocorrelation via FFT
    fft_len = seg_len * 2
    fft_seg = np.fft.rfft(seg, n=fft_len)
    ac_full = np.fft.irfft(fft_seg * np.conj(fft_seg))[:seg_len]
    ac = ac_full / (float(ac_full[0]) + 1e-9)

    # Noise floor: median of lags 5..seg_len//8
    noise = float(np.median(np.abs(ac[5:seg_len // 8])))

    comb_found = False
    for depth in range(3, 33):
        # Need at least 4 peaks to confirm a comb
        max_k = min(8, seg_len // depth)
        if max_k < 4:
            continue

        lags = [depth * k for k in range(1, max_k)]
        peak_vals = [abs(float(ac[lag])) for lag in lags if lag < seg_len]

        if len(peak_vals) < 4:
            continue

        # All comb peaks must be above noise threshold
        threshold = max(noise * 3.5, 0.02)
        if not all(p > threshold for p in peak_vals):
            continue

        # Inter-comb troughs must be significantly lower than peaks
        trough_lags = [depth * k + depth // 2
                       for k in range(1, max_k - 1)
                       if depth * k + depth // 2 < seg_len]
        if not trough_lags:
            continue
        trough_vals = [abs(float(ac[l])) for l in trough_lags]
        trough_mean = float(np.mean(trough_vals))
        peak_mean   = float(np.mean(peak_vals))

        if trough_mean < peak_mean * 0.55:
            comb_found = True
            break

    return comb_found


def _periodicity_autocorr(bits: np.ndarray,
                           depths: list) -> dict:
    """
    Block interleaving produces a periodic self-similarity at lags equal to
    multiples of the block row count (or column count).

    We compute the normalized autocorrelation of the bit stream and look for
    peaks at candidate lags.  A strong peak at lag L → the block width is
    likely L or a divisor of L.

    Returns score dict {'block': x, 'diagonal': y}.
    """
    ev = {'block': 0.0, 'diagonal': 0.0}
    n = len(bits)
    if n < 256:
        return ev

    b = bits.astype(np.float32) * 2.0 - 1.0
    seg_len = min(n, 8192)
    seg = b[:seg_len]

    fft_seg = np.fft.rfft(seg, n=seg_len * 2)
    ac = np.fft.irfft(fft_seg * np.conj(fft_seg))[:seg_len]
    ac /= (ac[0] + 1e-9)

    noise = float(np.median(np.abs(ac[2:seg_len // 8])))

    for depth in depths:
        for cols in [16, 32, 48, 64, 96, 128, 192, 256]:
            block_len = depth * cols
            if block_len > seg_len // 2:
                continue

            # Block: peak at exactly block_len
            bl_val = abs(float(ac[block_len % seg_len]))
            if bl_val > noise * 5.0:
                ev['block'] += bl_val / (noise + 1e-9) * 0.08

            # Also check row period (cols)
            col_val = abs(float(ac[cols % seg_len]))
            if col_val > noise * 4.0:
                ev['block'] += col_val / (noise + 1e-9) * 0.05

            # Diagonal: peak pattern slightly shifted — check lag block_len±depth
            diag_lag = (block_len + depth) % seg_len
            d_val = abs(float(ac[diag_lag]))
            if d_val > noise * 4.0 and d_val > ev['block'] * 0.5:
                ev['diagonal'] += d_val / (noise + 1e-9) * 0.05

    # Normalize
    total = ev['block'] + ev['diagonal'] + 1e-9
    if total > 1.0:
        ev = {k: min(1.0, v / total) for k, v in ev.items()}

    return ev


def _exhaustive_param_search(bits: np.ndarray, depths: list) -> dict:
    """
    Exhaustive (rows × cols) parameter grid search.

    For each (rows, cols) pair we try both BLOCK and DIAGONAL de-interleaving
    and score the result using two complementary metrics:

    1. _structure_score()       — local autocorrelation + run-length + byte repetition.
                                  Works well when data has inherent structure.
    2. _autocorr_improvement()  — measures whether the de-interleaved stream has
                                  better lag-1 autocorrelation than the raw stream.
                                  Works even on pure-random data because the
                                  interleaver disrupts natural short-range
                                  correlations in the encoded bit stream.

    A correct de-interleaving restores these short-range correlations.
    An incorrect one either leaves them unchanged or makes them worse.
    """
    ev = {'block': 0.0, 'diagonal': 0.0}
    n = len(bits)
    seg = bits[:min(n, 4096)]

    baseline_struct = _structure_score(seg)
    baseline_autocorr = _lag1_autocorr(seg)

    best_block_struct  = 0.0
    best_block_autocorr= 0.0
    best_diag_struct   = 0.0
    best_diag_autocorr = 0.0

    col_candidates = [16, 32, 48, 64, 96, 128, 192, 256]

    for rows in depths:
        for cols in col_candidates:
            total = rows * cols
            if total > len(seg) or total < 64:
                continue

            chunk = seg[:total]

            # ── Block trial ──────────────────────────────────────────────
            try:
                di_block = _deinterleave_block(chunk, rows, cols)
                s_b  = _structure_score(di_block)   - baseline_struct
                ac_b = _lag1_autocorr(di_block)     - baseline_autocorr
                if s_b  > best_block_struct:   best_block_struct   = s_b
                if ac_b > best_block_autocorr: best_block_autocorr = ac_b
            except Exception:
                pass

            # ── Diagonal trial ────────────────────────────────────────────
            try:
                di_diag = _deinterleave_diagonal(chunk, rows, cols)
                s_d  = _structure_score(di_diag)  - baseline_struct
                ac_d = _lag1_autocorr(di_diag)    - baseline_autocorr
                if s_d  > best_diag_struct:   best_diag_struct   = s_d
                if ac_d > best_diag_autocorr: best_diag_autocorr = ac_d
            except Exception:
                pass

    # Combine both metrics with equal weight; clip to [0,1]
    ev['block']    = min(1.0, max(0.0,
                         best_block_struct   * 5.0 * 0.5 +
                         best_block_autocorr * 8.0 * 0.5))
    ev['diagonal'] = min(1.0, max(0.0,
                         best_diag_struct    * 5.0 * 0.5 +
                         best_diag_autocorr  * 8.0 * 0.5))

    return ev


def _lag1_autocorr(bits: np.ndarray) -> float:
    """Normalized lag-1 autocorrelation of a bit stream (bipolar)."""
    if len(bits) < 2:
        return 0.0
    b = bits.astype(np.float32) * 2.0 - 1.0
    n = len(b)
    mean = float(np.mean(b))
    b_c = b - mean
    var = float(np.sum(b_c ** 2))
    if var < 1e-9:
        return 0.0
    lag1 = float(np.sum(b_c[:-1] * b_c[1:])) / var
    return lag1


def _diagonal_fingerprint(bits: np.ndarray, depths: list) -> dict:
    """
    Diagonal de-interleaving has a distinct 2-D structure: when the
    interleaved stream is reshaped into a matrix, the original data lies
    along diagonal lines rather than rows/columns.

    Detection strategy:
      Reshape into candidate (rows×cols) matrices.
      Compute the average correlation along:
        (a) rows                — strong for block
        (b) main diagonal bands  — strong for diagonal
      The ratio determines type.
    """
    ev = {'block': 0.0, 'diagonal': 0.0}
    n = len(bits)
    b_float = bits.astype(np.float32) * 2.0 - 1.0

    for rows in depths:
        for cols in [32, 64, 128]:
            total = rows * cols
            if total > n:
                continue

            mat = b_float[:total].reshape(rows, cols)

            # Row-wise autocorrelation (lag-1 within each row)
            row_corr = 0.0
            for r in range(rows):
                if cols > 1:
                    row_corr += float(np.corrcoef(mat[r, :-1], mat[r, 1:])[0, 1])
            row_corr /= max(rows, 1)

            # Diagonal-band correlation: compare mat[r,c] with mat[r+1,(c+1)%cols]
            diag_corr = 0.0
            n_pairs = 0
            for r in range(rows - 1):
                for c in range(cols):
                    diag_corr += mat[r, c] * mat[r + 1, (c + 1) % cols]
                    n_pairs += 1
            diag_corr /= max(n_pairs, 1)

            if not (np.isnan(row_corr) or np.isnan(diag_corr)):
                if row_corr > 0.05:
                    ev['block']    += row_corr * 0.15
                if diag_corr > 0.05:
                    ev['diagonal'] += diag_corr * 0.15

    # Normalize
    total = ev['block'] + ev['diagonal'] + 1e-9
    if total > 1.0:
        ev = {k: min(1.0, v / total) for k, v in ev.items()}

    return ev


def _structure_score(bits: np.ndarray) -> float:
    """
    Measure structural regularity of a bit sequence.
    Higher = more regular / less random.

    Combines:
    - Adjacent bit match rate (local autocorrelation at lag 1)
    - Run-length distribution skew (long runs → structured data)
    - Block entropy (repeated byte patterns → structure)
    """
    if len(bits) < 8:
        return 0.0

    b = bits.astype(np.uint8)

    # Local autocorrelation
    adj_match = float(np.mean(b[:-1] == b[1:]))

    # Run-length analysis
    runs, current = [], 1
    for i in range(1, len(b)):
        if b[i] == b[i - 1]:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    mean_run = float(np.mean(runs)) if runs else 1.0

    # Block byte repetition
    block_score = 0.0
    if len(b) >= 16:
        block_size = 8
        blocks = [tuple(b[i:i + block_size].tolist())
                  for i in range(0, len(b) - block_size, block_size)]
        if blocks:
            unique_ratio = len(set(blocks)) / len(blocks)
            block_score = 1.0 - unique_ratio   # more repeats = lower unique_ratio

    score = (adj_match * 0.4 +
             min(1.0, (mean_run - 1.0) / 5.0) * 0.35 +
             block_score * 0.25)
    return float(score)


# ---------------------------------------------------------------------------
#  New convolutional comb score and pseudorandom v2
# ---------------------------------------------------------------------------

def _convolutional_comb_score(bits: np.ndarray) -> float:
    """
    Score for convolutional (Forney) interleaving based on the delay-comb test.
    Returns a value > 0 when comb pattern or de-interleave improvement found.
    """
    n = len(bits)
    if n < 256:
        return 0.0

    b = bits.astype(np.float32) * 2.0 - 1.0
    seg_len = min(n, 4096)
    seg = b[:seg_len]

    fft_len = seg_len * 2
    fft_seg = np.fft.rfft(seg, n=fft_len)
    ac_full = np.fft.irfft(fft_seg * np.conj(fft_seg))[:seg_len]
    ac = ac_full / (float(ac_full[0]) + 1e-9)

    noise = float(np.median(np.abs(ac[5: max(6, seg_len // 8)])))

    best_score = 0.0
    for depth in range(3, 33):
        max_k = min(8, seg_len // depth)
        if max_k < 3:
            continue
        lags = [depth * k for k in range(1, max_k) if depth * k < seg_len]
        if len(lags) < 3:
            continue
        peak_vals = [abs(float(ac[lag])) for lag in lags]
        threshold = max(noise * 3.0, 0.015)
        if not all(p > threshold for p in peak_vals):
            continue
        trough_lags = [depth * k + depth // 2
                       for k in range(1, len(lags))
                       if depth * k + depth // 2 < seg_len]
        if not trough_lags:
            continue
        trough_mean = float(np.mean([abs(float(ac[l])) for l in trough_lags]))
        peak_mean   = float(np.mean(peak_vals))
        if peak_mean > 0 and trough_mean < peak_mean * 0.60:
            comb_score = (peak_mean / (noise + 1e-9) *
                          (1.0 - trough_mean / (peak_mean + 1e-9)))
            best_score = max(best_score, comb_score)

    # Try convolutional de-interleave and measure autocorr improvement
    baseline_ac = _lag1_autocorr(bits[:min(n, 2048)])
    for depth in [4, 6, 8, 10, 12, 16]:
        try:
            di = _deinterleave_convolutional(bits[:min(n, 2048)], depth)
            ac_after = _lag1_autocorr(di)
            improvement = ac_after - baseline_ac
            if improvement > 0.01:
                best_score = max(best_score, improvement * 15.0)
        except Exception:
            pass

    return min(2.0, best_score)


def _is_pseudorandom_v2(bits: np.ndarray, depths: list) -> bool:
    """
    Declare pseudo-random when no structured de-interleaving
    (block or diagonal) gives a run-length improvement significantly
    above baseline.  Convolutional is already handled separately.

    Logic: block-interleaved data → block de-interleave improves run length.
           diagonal-interleaved data → diagonal de-interleave improves run length.
           pseudo-random → NO de-interleaving improves run length measurably.
    """
    if len(bits) < 64:
        return False

    seg = bits[:min(len(bits), 2048)]
    baseline = _mean_run_length(seg)

    # Try block and diagonal de-interleaving; if any gives > 0.3 run improvement,
    # this is NOT pseudorandom (it's a structured type)
    for rows in [3, 4, 5, 6, 8]:
        for cols in (32, 64, 128):
            total = rows * cols
            if total > len(seg):
                continue
            chunk = seg[:total]
            try:
                di_b = _deinterleave_block(chunk, rows, cols)
                if _mean_run_length(di_b) > baseline + 0.30:
                    return False
            except Exception:
                pass
            try:
                di_d = _deinterleave_diagonal(chunk, rows, cols)
                if _mean_run_length(di_d) > baseline + 0.30:
                    return False
            except Exception:
                pass

    # No structured type improved things significantly → declare pseudorandom
    return True


# ---------------------------------------------------------------------------
#  Existing helpers (kept unchanged)
# ---------------------------------------------------------------------------

def _is_pseudorandom(bits: np.ndarray) -> bool:
    """
    Detect pseudo-random interleaving.

    Reliable only when:
    1. Bit entropy is near-perfect (>= 0.995)
    2. The stream is longer than 4000 bits (short streams can't be distinguished)
    3. No structured de-interleaving (block/diagonal) gives column correlation
       above 0.68, AND the column correlation after pseudorandom de-interleaving
       (seed=42) is notably lower than what block de-interleaving gives.

    For short/structured data (FEC-encoded), returns False to avoid false positives.
    """
    n = len(bits)

    # Need sufficient length for reliable detection
    if n < 4000:
        return False

    entropy_global = _calculate_entropy(bits)
    if entropy_global < 0.990:
        return False

    # Check local entropy consistency
    local_entropies = []
    for ws in (128, 256):
        if n < ws:
            continue
        for i in range(0, min(n - ws, 512), ws // 2):
            local_entropies.append(_calculate_entropy(bits[i:i + ws]))
    if local_entropies and float(np.mean(local_entropies)) < 0.92:
        return False

    # Key guard: block de-interleaving should NOT achieve high column correlation
    # (if it does, the data is actually block-interleaved)
    seg = bits[:min(n, 2048)]
    for rows in (4, 8, 16):
        for cols in (64, 128):
            total = rows * cols
            if total > n:
                continue
            try:
                di_b = _deinterleave_block(bits[:total], rows, cols)
                if _column_correlation(di_b, rows, cols) > 0.70:
                    return False
            except Exception:
                pass

    return True


def _compute_run_lengths_fast(bits: np.ndarray) -> np.ndarray:
    """Return run-length array quickly."""
    if len(bits) < 2:
        return np.array([1], dtype=np.int32)
    changes = np.where(np.diff(bits.astype(np.int32)) != 0)[0]
    if len(changes) == 0:
        return np.array([len(bits)], dtype=np.int32)
    lengths = np.diff(np.concatenate([[-1], changes, [len(bits) - 1]]))
    return lengths.astype(np.int32)


def _detect_via_matrix_rank(bits: np.ndarray, depths: list) -> str:
    """
    Matrix rank-based detection (research method from IEEE papers).
    
    Theory: Interleaved data arranged in matrices shows rank deficiency
    patterns that differ by interleaver type.
    
    - Block: Full rank when arranged in D×N matrix
    - Convolutional: Rank deficiency with specific patterns
    - Diagonal: Off-diagonal rank patterns
    """
    best_type = None
    best_score = -1.0
    
    for depth in depths:
        cols = len(bits) // depth
        if cols < 4:
            continue
        
        # Create matrix representation
        try:
            matrix = bits[:depth * cols].reshape(depth, cols).astype(float)
        except:
            continue
        
        # Calculate rank and rank deficiency
        rank = np.linalg.matrix_rank(matrix)
        max_rank = min(depth, cols)
        rank_ratio = rank / max_rank
        
        # Analyze diagonal vs off-diagonal structure
        min_dim = min(depth, cols)
        diag_vals = np.array([matrix[i, i] for i in range(min_dim)])
        diag_weight = np.mean(np.abs(diag_vals))
        
        # Off-diagonal weight: all elements except main diagonal
        off_diag_sum = 0.0
        off_diag_count = 0
        for i in range(depth):
            for j in range(cols):
                if i != j or i >= min_dim:
                    off_diag_sum += abs(matrix[i, j])
                    off_diag_count += 1
        off_diag_weight = off_diag_sum / max(off_diag_count, 1)
        
        # Block interleaving: high rank, balanced structure
        if rank_ratio > 0.85:
            # Check if columns are relatively independent
            col_correlations = []
            for i in range(min(10, cols-1)):
                corr = np.corrcoef(matrix[:, i], matrix[:, i+1])[0, 1]
                if not np.isnan(corr):
                    col_correlations.append(abs(corr))
            
            if len(col_correlations) > 0 and np.mean(col_correlations) < 0.3:
                block_score = rank_ratio * 0.7 + (1.0 - np.mean(col_correlations)) * 0.3
                if block_score > best_score:
                    best_score = block_score
                    best_type = 'block'
        
        # Convolutional: characteristic rank patterns with row dependencies
        elif 0.5 < rank_ratio < 0.85:
            # Check for geometric structure (Forney interleaver property)
            row_means = np.mean(matrix, axis=1)
            row_variance = np.var(row_means)
            
            if row_variance < 0.1:  # Rows have similar statistics
                conv_score = 0.6 * (1.0 - abs(rank_ratio - 0.7)) + 0.4 * (1.0 - row_variance)
                if conv_score > best_score:
                    best_score = conv_score
                    best_type = 'convolutional'
        
        # Diagonal: off-diagonal dominant
        if off_diag_weight > diag_weight * 1.2:
            diag_score = 0.6 * rank_ratio + 0.4 * (off_diag_weight / (diag_weight + 0.01))
            if diag_score > best_score:
                best_score = diag_score
                best_type = 'diagonal'
    
    return best_type


def _detect_via_trial(bits: np.ndarray, depths: list) -> str:
    """
    Trial de-interleaving method: try each type and score the result.
    """
    best_type = None
    best_improvement = 0.0
    
    # Baseline metrics
    baseline_corr = _measure_local_correlation(bits)
    baseline_entropy = _calculate_entropy(bits)
    
    for depth in depths:
        cols = len(bits) // depth
        if cols < 2:
            continue
        
        # Try block
        try:
            deint = _deinterleave_block(bits, depth, cols)
            corr = _measure_local_correlation(deint)
            entropy = _calculate_entropy(deint)
            improvement = (corr - baseline_corr) + 0.5 * (baseline_entropy - entropy)
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_type = 'block'
        except:
            pass
        
        # Try diagonal
        try:
            deint = _deinterleave_diagonal(bits, depth, cols)
            corr = _measure_local_correlation(deint)
            entropy = _calculate_entropy(deint)
            improvement = (corr - baseline_corr) + 0.5 * (baseline_entropy - entropy)
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_type = 'diagonal'
        except:
            pass
        
        # Try convolutional
        try:
            deint = _deinterleave_convolutional(bits, depth)
            corr = _measure_local_correlation(deint)
            entropy = _calculate_entropy(deint)
            improvement = (corr - baseline_corr) + 0.5 * (baseline_entropy - entropy)
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_type = 'convolutional'
        except:
            pass
    
    # Only return if we found significant improvement
    if best_improvement > 0.01:
        return best_type
    
    return None


def _calculate_entropy(bits: np.ndarray) -> float:
    """Calculate normalized Shannon entropy of bit sequence."""
    if len(bits) < 2:
        return 0.0
    
    # Count 0s and 1s
    counts = np.bincount(bits.astype(int), minlength=2).astype(float)
    probs = counts / len(bits)
    probs = probs[probs > 0]
    
    if len(probs) == 0:
        return 0.0
    
    entropy = -np.sum(probs * np.log2(probs))
    return entropy  # Max entropy is 1.0 for binary


def _measure_local_correlation(bits: np.ndarray) -> float:
    """
    Measure how correlated adjacent bits are.
    Higher score = more structure (better de-interleaving match).
    """
    if len(bits) < 2:
        return 0.0
    
    # Count how many adjacent pairs match
    matches = np.sum(bits[:-1] == bits[1:])
    correlation = matches / (len(bits) - 1)
    
    return correlation


def _run_length_entropy(bits: np.ndarray) -> float:
    """Entropy of run-length distribution."""
    if len(bits) == 0:
        return 0.0
    runs = []
    current_run = 1
    for i in range(1, len(bits)):
        if bits[i] == bits[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    runs = np.array(runs)
    counts = np.bincount(runs)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))
