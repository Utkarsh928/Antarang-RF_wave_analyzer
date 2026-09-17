"""
Timing Recovery — Gardner and Mueller-Müller symbol timing recovery algorithms.
Corrects timing offset between transmitter and receiver clocks.
This is the last major DSP gap for real-world signal accuracy.
"""
import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple


def gardner_timing_recovery(samples: np.ndarray,
                              sps: int,
                              loop_bw: float = 0.01,
                              damping: float = 0.707) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gardner timing error detector with 2nd-order loop filter.
    Works for any modulation — does not require known modulation type.

    Parameters
    ----------
    samples  : complex IQ samples (at sps samples/symbol)
    sps      : nominal samples per symbol
    loop_bw  : loop bandwidth (normalized, 0.01 = 1% of symbol rate)
    damping  : loop damping factor (0.707 = critically damped)

    Returns
    -------
    (symbols, timing_error) — downsampled symbols + error signal for debugging
    """
    if not np.iscomplexobj(samples):
        samples = samples.astype(np.float32) + 0j

    n = len(samples)
    if n < sps * 4:
        # Too short — just downsample
        return samples[::sps], np.zeros(n // sps)

    # Loop filter coefficients (2nd order)
    K1, K2 = _loop_filter_coeffs(loop_bw, damping, sps)

    # Output buffers
    max_syms = n // sps + 1
    out_syms = np.zeros(max_syms, dtype=np.complex64)
    out_err = np.zeros(max_syms, dtype=np.float32)

    mu = 0.0          # fractional timing offset [0, 1)
    vi = 0.0          # integrator state
    sym_count = 0
    k = sps           # sample index (start at first full symbol)

    prev_sample = samples[0]
    prev_sym = samples[0]

    while k < n and sym_count < max_syms:
        # Interpolate sample at current timing estimate
        k_int = int(k)
        if k_int + 1 >= n:
            break

        # Linear interpolation
        frac = k - k_int
        curr = (1 - frac) * samples[k_int] + frac * samples[min(k_int + 1, n - 1)]

        # Mid-point sample (half symbol ago)
        k_mid = k - sps / 2
        k_mid_int = int(k_mid)
        if k_mid_int < 0 or k_mid_int + 1 >= n:
            k += sps
            continue
        frac_mid = k_mid - k_mid_int
        mid = ((1 - frac_mid) * samples[k_mid_int] +
                frac_mid * samples[min(k_mid_int + 1, n - 1)])

        # Gardner timing error detector
        # e = Re{(curr - prev_sym) * conj(mid)}
        ted = float(np.real((curr - prev_sym) * np.conj(mid)))

        # Loop filter
        vi += K2 * ted
        vp = K1 * ted + vi
        mu += vp

        # Wrap mu to [0, 1)
        if mu >= 1.0:
            mu -= 1.0
        if mu < 0.0:
            mu += 1.0

        # Store symbol
        out_syms[sym_count] = curr
        out_err[sym_count] = ted
        sym_count += 1

        prev_sym = curr
        k += sps + mu * sps

    return out_syms[:sym_count], out_err[:sym_count]


def mueller_muller_timing(samples: np.ndarray,
                           sps: int,
                           loop_bw: float = 0.01) -> np.ndarray:
    """
    Mueller-Müller timing recovery. Works well for PSK signals.
    Simpler than Gardner but requires roughly correct sps.

    Returns downsampled symbols.
    """
    if not np.iscomplexobj(samples):
        samples = samples.astype(np.float32) + 0j

    n = len(samples)
    if n < sps * 4:
        return samples[::sps]

    K = loop_bw
    mu = 0.0
    sym_count = 0
    max_syms = n // sps + 1
    out = np.zeros(max_syms, dtype=np.complex64)

    prev = samples[0]
    k = sps

    while k < n and sym_count < max_syms:
        k_int = int(k)
        if k_int + 1 >= n:
            break
        frac = k - k_int
        curr = ((1 - frac) * samples[k_int] +
                 frac * samples[min(k_int + 1, n - 1)])

        # M&M timing error
        ted = float(np.real(curr) * float(np.imag(prev)) -
                    float(np.real(prev)) * float(np.imag(curr)))

        mu += K * ted
        if mu >= 1.0:
            mu -= 1.0
        if mu < 0.0:
            mu += 1.0

        out[sym_count] = curr
        sym_count += 1
        prev = curr
        k += sps * (1 + mu * 0.1)

    return out[:sym_count]


def apply_timing_recovery(samples: np.ndarray,
                            sample_rate: float,
                            symbol_rate: float,
                            method: str = 'gardner') -> np.ndarray:
    """
    Apply timing recovery to get clean symbol-rate samples.
    Call this before symbol decision in the demodulator.

    Parameters
    ----------
    samples     : IQ samples at sample_rate
    sample_rate : samples per second
    symbol_rate : symbols per second
    method      : 'gardner' or 'mueller'

    Returns
    -------
    Symbol-rate complex samples with corrected timing.
    """
    sps = max(2, int(round(sample_rate / symbol_rate)))

    if sps < 2:
        return samples

    if method == 'gardner':
        syms, _ = gardner_timing_recovery(samples, sps)
    else:
        syms = mueller_muller_timing(samples, sps)

    return syms


def _loop_filter_coeffs(bw: float, zeta: float,
                         sps: int) -> Tuple[float, float]:
    """
    Compute 2nd-order loop filter gains K1, K2.
    Based on Gardner loop filter design.
    """
    Ts = 1.0 / sps
    theta = bw / (zeta + 1.0 / (4.0 * zeta))
    d = 1.0 + 2.0 * zeta * theta * Ts + theta ** 2 * Ts ** 2

    K1 = (4.0 * zeta * theta * Ts) / d
    K2 = (4.0 * theta ** 2 * Ts ** 2) / d

    return float(K1), float(K2)
