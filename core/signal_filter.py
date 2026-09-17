"""
Signal Filtering — carrier recovery, bandpass filtering, decimation.
Applied before demodulation to improve real-world signal quality.

GNU Radio enhancement (automatic when installed):
  auto_preprocess() uses GnuRadioConditioner for professional-grade
  AGC + LPF + FLL + Costas + timing recovery instead of basic SciPy.
  Falls back to pure SciPy transparently when GNU Radio is absent.
"""
import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, Optional


def bandpass_filter(samples: np.ndarray, sample_rate: float,
                    center_freq: float, bandwidth: float,
                    order: int = 63) -> np.ndarray:
    """
    Apply a bandpass FIR filter centered at center_freq with given bandwidth.
    Handles both real and complex signals.
    Returns filtered samples.
    """
    if sample_rate <= 0 or bandwidth <= 0:
        return samples

    nyq = sample_rate / 2.0
    low = max(1.0, center_freq - bandwidth / 2.0)
    high = min(nyq - 1.0, center_freq + bandwidth / 2.0)

    if low >= high:
        return samples

    if np.iscomplexobj(samples):
        # Filter I and Q separately
        taps = scipy_signal.firwin(order, [low / nyq, high / nyq],
                                    pass_zero=False)
        i_filt = scipy_signal.lfilter(taps, 1.0, samples.real.astype(np.float32))
        q_filt = scipy_signal.lfilter(taps, 1.0, samples.imag.astype(np.float32))
        return (i_filt + 1j * q_filt).astype(np.complex64)
    else:
        taps = scipy_signal.firwin(order, [low / nyq, high / nyq],
                                    pass_zero=False)
        return scipy_signal.lfilter(taps, 1.0,
                                     samples.astype(np.float32)).astype(np.float32)


def lowpass_filter(samples: np.ndarray, sample_rate: float,
                   cutoff_hz: float, order: int = 63) -> np.ndarray:
    """
    Apply a lowpass FIR filter. Useful for baseband signal cleanup.
    """
    nyq = sample_rate / 2.0
    cutoff_norm = min(0.99, cutoff_hz / nyq)
    taps = scipy_signal.firwin(order, cutoff_norm)

    if np.iscomplexobj(samples):
        i_filt = scipy_signal.lfilter(taps, 1.0, samples.real.astype(np.float32))
        q_filt = scipy_signal.lfilter(taps, 1.0, samples.imag.astype(np.float32))
        return (i_filt + 1j * q_filt).astype(np.complex64)
    else:
        return scipy_signal.lfilter(taps, 1.0,
                                     samples.astype(np.float32)).astype(np.float32)


def recover_carrier(samples: np.ndarray,
                    sample_rate: float,
                    estimated_offset_hz: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Carrier frequency recovery — removes frequency offset from complex signal.
    Uses the M-th power algorithm to estimate residual carrier offset.
    Returns (corrected_samples, actual_offset_hz).
    """
    if not np.iscomplexobj(samples):
        return samples, 0.0

    n = len(samples)
    if n < 64:
        return samples, 0.0

    # Coarse correction: remove known offset
    if abs(estimated_offset_hz) > 1.0:
        t = np.arange(n) / sample_rate
        correction = np.exp(-1j * 2 * np.pi * estimated_offset_hz * t)
        samples = (samples * correction).astype(np.complex64)

    # Fine correction using 4th-power method (works for QPSK/BPSK)
    try:
        n_fft = min(n, 16384)
        n_fft = int(2 ** np.floor(np.log2(n_fft)))
        n_fft = max(n_fft, 512)

        # 4th power removes modulation for PSK
        powered = samples[:n_fft] ** 4
        spectrum = np.abs(np.fft.fft(powered, n_fft))
        freqs = np.fft.fftfreq(n_fft, d=1.0 / sample_rate)

        peak_idx = np.argmax(spectrum)
        residual_offset = freqs[peak_idx] / 4.0  # divide by M=4

        if abs(residual_offset) < sample_rate / 4:
            t = np.arange(n) / sample_rate
            fine_correction = np.exp(-1j * 2 * np.pi * residual_offset * t)
            samples = (samples * fine_correction).astype(np.complex64)
            actual_offset = estimated_offset_hz + residual_offset
        else:
            actual_offset = estimated_offset_hz

    except Exception:
        actual_offset = estimated_offset_hz

    return samples, float(actual_offset)


def decimate(samples: np.ndarray, sample_rate: float,
             target_rate: float) -> Tuple[np.ndarray, float]:
    """
    Decimate signal to target sample rate.
    Anti-aliasing filter applied automatically.
    Returns (decimated_samples, actual_new_rate).
    """
    if target_rate >= sample_rate:
        return samples, sample_rate

    factor = int(sample_rate / target_rate)
    if factor <= 1:
        return samples, sample_rate

    factor = min(factor, 64)  # safety cap

    if np.iscomplexobj(samples):
        # Decimate I and Q separately
        i_dec = scipy_signal.decimate(samples.real.astype(np.float64),
                                       factor, ftype='fir', zero_phase=True)
        q_dec = scipy_signal.decimate(samples.imag.astype(np.float64),
                                       factor, ftype='fir', zero_phase=True)
        result = (i_dec + 1j * q_dec).astype(np.complex64)
    else:
        result = scipy_signal.decimate(samples.astype(np.float64),
                                        factor, ftype='fir',
                                        zero_phase=True).astype(np.float32)

    new_rate = sample_rate / factor
    return result, new_rate


def normalize_power(samples: np.ndarray,
                    target_power: float = 1.0) -> np.ndarray:
    """Normalize signal to target RMS power."""
    pwr = np.mean(np.abs(samples) ** 2)
    if pwr < 1e-15:
        return samples
    scale = np.sqrt(target_power / pwr)
    return (samples * scale).astype(samples.dtype)


def remove_dc(samples: np.ndarray) -> np.ndarray:
    """Remove DC offset from signal."""
    if np.iscomplexobj(samples):
        return (samples - np.mean(samples)).astype(np.complex64)
    return (samples - np.mean(samples)).astype(np.float32)


def auto_preprocess(samples: np.ndarray, sample_rate: float,
                    center_freq_hz: float = 0.0,
                    bandwidth_hz: float = 0.0) -> np.ndarray:
    """
    Full preprocessing pipeline for real-world signals.

    When GNU Radio is installed:
        Uses GnuRadioConditioner — AGC + LPF + FLL + Costas loop.
        This is professional-grade signal conditioning identical to
        what GQRX, SDR# and hardware demodulators use.

    When GNU Radio is NOT installed (fallback):
        1. Remove DC offset
        2. Coarse carrier frequency offset correction
        3. Butterworth low-pass filter to signal bandwidth
        4. Normalise power to unit RMS

    Both paths return complex64 with the same shape as input.
    """
    # ── Try GNU Radio conditioner first ───────────────────────────────────
    try:
        from core.gnu_radio.conditioner import condition_signal
        from core.gnu_radio.availability import GR_AVAILABLE, GR_MODULES
        if GR_AVAILABLE and GR_MODULES.get("analog") and GR_MODULES.get("filter"):
            return condition_signal(
                samples, sample_rate,
                bandwidth=bandwidth_hz if bandwidth_hz > 0 else None,
            )
    except Exception:
        pass   # fall through to SciPy

    # ── SciPy fallback ────────────────────────────────────────────────────
    samples = remove_dc(samples)

    if abs(center_freq_hz) > sample_rate * 0.01:
        samples, _ = recover_carrier(samples, sample_rate, center_freq_hz)

    if bandwidth_hz > 0 and bandwidth_hz < sample_rate / 2:
        cutoff = min(bandwidth_hz * 0.6, sample_rate / 2.5)
        samples = lowpass_filter(samples, sample_rate, cutoff)

    samples = normalize_power(samples)
    return samples


def costas_loop(samples: np.ndarray,
                modulation_order: int = 2,
                loop_bw: float = 0.005) -> np.ndarray:
    """
    Costas Loop carrier phase recovery.

    Removes residual frequency offset and phase rotation left after
    coarse CFO correction. Essential for correct PSK decisions on
    real hardware signals.

    Works for:
        BPSK   (order=2)  — corrects up to ±45 degrees
        QPSK   (order=4)  — corrects up to ±22.5 degrees
        8PSK   (order=8)  — corrects up to ±11.25 degrees
        QAM    (order=2)  — use order=2, applies phase correction

    Parameters
    ----------
    samples          : complex IQ samples
    modulation_order : M for M-PSK (2, 4, 8)
    loop_bw          : normalised loop bandwidth (0.001-0.01 recommended)

    Returns
    -------
    Phase-corrected complex64 samples (same length as input).
    """
    if not np.iscomplexobj(samples):
        return samples

    n = len(samples)
    if n < 32:
        return samples

    M = modulation_order

    # 2nd-order loop filter gains (based on loop bandwidth)
    alpha = loop_bw            # proportional gain (phase error)
    beta  = (loop_bw ** 2) / 4.0  # integral gain  (frequency error)

    phase = 0.0
    freq  = 0.0
    out   = np.zeros(n, dtype=np.complex64)

    for i in range(n):
        # Remove current phase estimate from sample
        corrected = samples[i] * np.exp(-1j * phase)
        out[i] = corrected

        # Phase Error Detector: M-th power law (Viterbi-Viterbi)
        # Raises symbol to M-th power to remove modulation,
        # leaving only the M-times-amplified carrier phase error.
        powered = corrected ** M
        # Instantaneous phase error (divided back by M)
        error = float(np.angle(powered)) / M

        # 2nd-order loop filter update
        freq  += beta  * error
        phase += alpha * error + freq

        # Wrap phase to [-pi, pi) to prevent numeric overflow
        if phase > np.pi:
            phase -= 2 * np.pi
        elif phase < -np.pi:
            phase += 2 * np.pi

    return out
