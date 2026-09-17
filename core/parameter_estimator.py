"""
Parameter Estimator — estimates SNR, bandwidth, center frequency from raw samples.
"""
import numpy as np
from scipy import signal as scipy_signal
from core.signal_info import SignalInfo


def estimate_parameters(info: SignalInfo) -> SignalInfo:
    """
    Run all parameter estimations on the loaded signal.
    Updates info in-place and returns it.
    """
    if not info.is_loaded():
        return info

    samples = info.samples
    sr = info.sample_rate

    # Get real-valued signal for processing
    if np.iscomplexobj(samples):
        analytic = samples
    else:
        # Create analytic signal from real samples
        analytic = scipy_signal.hilbert(samples)

    # --- FFT ---
    n_fft = min(len(analytic), 65536)
    # Use a power-of-2 length for speed
    n_fft = int(2 ** np.floor(np.log2(n_fft)))
    n_fft = max(n_fft, 512)

    window = np.hanning(n_fft)
    segment = analytic[:n_fft]
    spectrum = np.fft.fftshift(np.fft.fft(segment * window, n_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sr))
    power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

    # --- Center frequency ---
    peak_idx = np.argmax(power_db)
    info.center_freq_hz = float(freqs[peak_idx])

    # --- Noise floor (median of lower 20% power values) ---
    sorted_power = np.sort(power_db)
    noise_floor = float(np.median(sorted_power[:len(sorted_power) // 5]))
    info.noise_floor_db = noise_floor

    # --- Signal power (mean of top 5% power values) ---
    signal_power = float(np.mean(sorted_power[int(0.95 * len(sorted_power)):]))

    # --- SNR ---
    info.snr_db = float(signal_power - noise_floor)

    # --- Bandwidth: 3dB bandwidth around center ---
    threshold = signal_power - 3.0
    above_thresh = power_db > threshold
    if above_thresh.any():
        indices = np.where(above_thresh)[0]
        bw_low = freqs[indices[0]]
        bw_high = freqs[indices[-1]]
        info.bandwidth_hz = float(abs(bw_high - bw_low))
    else:
        info.bandwidth_hz = sr / 4.0

    # --- Band classification (HF / VHF / UHF / SHF) ---
    cf_abs = abs(info.center_freq_hz)
    if cf_abs < 30e6:
        info.band = 'HF'           # 3–30 MHz (and baseband < 30 MHz)
    elif cf_abs < 300e6:
        info.band = 'VHF'          # 30–300 MHz
    elif cf_abs < 3e9:
        info.band = 'UHF'          # 300 MHz – 3 GHz
    elif cf_abs < 30e9:
        info.band = 'SHF'          # 3–30 GHz (microwave)
    else:
        info.band = 'EHF'          # >30 GHz

    return info


def compute_fft(samples: np.ndarray, sample_rate: float,
                n_fft: int = 4096) -> tuple:
    """
    Compute FFT for display.
    Returns (frequencies_hz, power_db) arrays.
    """
    if np.iscomplexobj(samples):
        seg = samples[:n_fft] if len(samples) >= n_fft else np.pad(
            samples, (0, n_fft - len(samples)))
    else:
        analytic = scipy_signal.hilbert(
            samples[:n_fft] if len(samples) >= n_fft
            else np.pad(samples, (0, n_fft - len(samples))))
        seg = analytic

    window = np.hanning(len(seg))
    spectrum = np.fft.fftshift(np.fft.fft(seg * window))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(seg), d=1.0 / sample_rate))
    power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
    return freqs, power_db


def compute_spectrogram(samples: np.ndarray, sample_rate: float,
                        nperseg: int = 256,
                        noverlap: int = None) -> tuple:
    """
    Compute short-time Fourier transform for waterfall display.
    Returns (frequencies_hz, times_sec, power_db_2d).
    """
    if noverlap is None:
        noverlap = nperseg // 2

    if np.iscomplexobj(samples):
        real_sig = samples.real
    else:
        real_sig = samples.astype(np.float32)

    freqs, times, Sxx = scipy_signal.spectrogram(
        real_sig,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window='hann',
        scaling='spectrum'
    )
    power_db = 10 * np.log10(Sxx + 1e-12)
    return freqs, times, power_db
