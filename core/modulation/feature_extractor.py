"""
Feature Extractor for ML Modulation Classifier — Enhanced v2.
Extracts a rich 72-dimensional feature vector from IQ samples.

v2 additions (+16 features over v1's 56):
  56-59 : Phase constellation cluster count + spread (BPSK/QPSK/8PSK discrimination)
  60-63 : Amplitude level quantization features (QAM16 vs QAM64)
  64-67 : Frequency transition rate + smoothness (GFSK vs CPFSK vs FSK2)
  68-69 : 8th-order cumulants (C80, C84) for dense PSK discrimination
  70    : Spectral asymmetry ratio (USB/SSB/RTTY detection)
  71    : Instantaneous freq histogram entropy (FSK M-ary discrimination)

When C++ extension DLL is available (core/signal_features.dll), the first
16 features are computed by the compiled C++ implementation (~30x faster).
"""
import numpy as np
from scipy import signal as scipy_signal
from typing import Optional

# Try C++ feature extraction (optional, falls back to NumPy)
try:
    from core.cpp_extensions import extract_features as _cpp_extract_features, cpp_available as _cpp_avail
    _USE_CPP_FEATURES = _cpp_avail().get('signal_features', False)
except ImportError:
    _USE_CPP_FEATURES = False


def extract_features(samples: np.ndarray,
                     n_seg: int = 1024) -> np.ndarray:
    """
    Extract 72-dimensional feature vector from IQ samples.
    Works on complex or real input.
    Returns float32 array of shape (72,).
    """
    if not np.iscomplexobj(samples):
        samples = samples.astype(np.float32) + 0j

    # Use middle segment for stability
    n = len(samples)
    if n > n_seg:
        start = (n - n_seg) // 2
        seg = samples[start:start + n_seg].copy()
    else:
        seg = samples.copy()

    # Normalize power to 1.0
    pwr = np.mean(np.abs(seg) ** 2)
    if pwr > 1e-15:
        seg = seg / np.sqrt(pwr)

    feats = []

    # ── C++ accelerated features (16 higher-order cumulants) ─────────────
    if _USE_CPP_FEATURES:
        cpp_feats = _cpp_extract_features(seg)   # float32 array of 16 values
        feats.extend(cpp_feats.tolist())
    else:
        # NumPy fallback — compute same 16 features
        _feats_16 = _numpy_hoc_features(seg)
        feats.extend(_feats_16)

    # ── 1. Higher-order cumulants (7 features) ──────────────────────
    feats.extend(_compute_hoc(seg))

    # ── 2. Amplitude statistics (8 features) ────────────────────────
    feats.extend(_compute_amplitude_stats(seg))

    # ── 3. Phase statistics (6 features) ────────────────────────────
    feats.extend(_compute_phase_stats(seg))

    # ── 4. Instantaneous frequency stats (5 features) ───────────────
    feats.extend(_compute_freq_stats(seg))

    # ── 5. Spectral features (8 features) ───────────────────────────
    feats.extend(_compute_spectral_features(seg))

    # ── 6. Cyclostationary / pattern features (6 features) ──────────
    feats.extend(_compute_pattern_features(seg))

    # ── v2: 7. Phase constellation cluster features (4 features) ────
    feats.extend(_compute_constellation_cluster_features(seg))

    # ── v2: 8. Amplitude quantization level features (4 features) ───
    feats.extend(_compute_amplitude_quantization_features(seg))

    # ── v2: 9. Frequency transition & smoothness (4 features) ───────
    feats.extend(_compute_freq_transition_features(seg))

    # ── v2: 10. 8th-order cumulants (2 features) ────────────────────
    feats.extend(_compute_8th_order_cumulants(seg))

    # ── v2: 11. Spectral asymmetry ratio (1 feature) ────────────────
    feats.append(_compute_spectral_asymmetry(seg))

    # ── v2: 12. Inst. freq histogram entropy (1 feature) ────────────
    feats.append(_compute_freq_histogram_entropy(seg))

    arr = np.array(feats, dtype=np.float32)

    # Replace NaN/inf with 0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip extreme values
    arr = np.clip(arr, -1e6, 1e6)

    return arr


# =============================================================================
# v2 NEW FEATURE GROUPS
# =============================================================================

def _compute_constellation_cluster_features(seg: np.ndarray) -> list:
    """Phase constellation cluster count + spread — 4 features.
    BPSK=2 clusters, QPSK=4, 8PSK=8. Uses phase histogram peak counting.
    Derotates coarse carrier offset so clusters emerge cleanly even with CFO.
    Uses multi-hypothesis non-linear carrier estimation (2nd power for BPSK,
    4th power for QPSK, 8th power for 8PSK, with robust median fallback)."""
    n = len(seg)
    if n > 16:
        freqs = np.fft.fftfreq(n)
        dphi = np.diff(np.unwrap(np.angle(seg)))
        fo_base = float(np.median(dphi)) / (2.0 * np.pi)

        # Hypothesis 2: Squaring (BPSK)
        s2 = seg ** 2
        f2 = np.abs(np.fft.fft(s2))
        p2_max = np.max(f2)
        p2_mean = np.mean(f2) + 1e-12
        c2 = float(p2_max / p2_mean)
        fo_2 = float(freqs[np.argmax(f2)]) / 2.0

        # Hypothesis 4: 4th-power (QPSK)
        s4 = seg ** 4
        f4 = np.abs(np.fft.fft(s4))
        p4_max = np.max(f4)
        p4_mean = np.mean(f4) + 1e-12
        c4 = float(p4_max / p4_mean)
        fo_4 = float(freqs[np.argmax(f4)]) / 4.0

        # Hypothesis 8: 8th-power (8PSK)
        s8 = seg ** 8
        f8 = np.abs(np.fft.fft(s8))
        p8_max = np.max(f8)
        p8_mean = np.mean(f8) + 1e-12
        c8 = float(p8_max / p8_mean)
        fo_8 = float(freqs[np.argmax(f8)]) / 8.0

        # Select by confidence / peak prominence
        if c2 >= 22.0 and c2 > 1.2 * c4:
            est_fo = fo_2
        elif c4 >= 18.0 and c4 > 1.1 * c8:
            est_fo = fo_4
        elif c8 >= 18.0:
            est_fo = fo_8
        else:
            est_fo = fo_base

        t = np.arange(n)
        seg_derot = seg * np.exp(-1j * 2.0 * np.pi * est_fo * t)

        # Verification: ensure derotation did not increase circular variance
        cv_derot = 1.0 - float(np.abs(np.mean(np.exp(1j * np.angle(seg_derot)))))
        cv_base = 1.0 - float(np.abs(np.mean(np.exp(1j * np.angle(seg * np.exp(-1j * 2.0 * np.pi * fo_base * t))))))
        if cv_derot > cv_base + 0.1:
            seg_derot = seg * np.exp(-1j * 2.0 * np.pi * fo_base * t)

        phases = np.angle(seg_derot)
    elif n > 8:
        dphi = np.diff(np.unwrap(np.angle(seg)))
        est_fo = float(np.median(dphi)) / (2.0 * np.pi)
        t = np.arange(n)
        seg_derot = seg * np.exp(-1j * 2.0 * np.pi * est_fo * t)
        phases = np.angle(seg_derot)
    else:
        phases = np.angle(seg)

    hist, _ = np.histogram(phases, bins=36, range=(-np.pi, np.pi))
    hist_norm = hist.astype(float) / (hist.sum() + 1e-12)
    from scipy.signal import find_peaks as _find_peaks
    peaks, _ = _find_peaks(hist_norm, height=np.max(hist_norm) * 0.2, distance=2)
    n_phase_clusters = float(min(len(peaks), 16))
    phase_hist_entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-12)))
    if len(peaks) >= 2:
        peak_phases = np.sort(phases[::max(1, len(phases) // 100)])
        cluster_spread = float(np.std(peak_phases))
    else:
        cluster_spread = float(np.std(phases))
    circ_var = float(1.0 - np.abs(np.mean(np.exp(1j * phases))))
    return [n_phase_clusters, phase_hist_entropy, cluster_spread, circ_var]


def _compute_amplitude_quantization_features(seg: np.ndarray) -> list:
    """Amplitude quantization level features — 4 features.
    QAM16: 3 amp levels, QAM64: 7, OOK: 2, PSK: 1."""
    amp = np.abs(seg)
    amp_norm = amp / (np.max(amp) + 1e-12)
    hist, _ = np.histogram(amp_norm, bins=32, range=(0, 1))
    hist_norm = hist.astype(float) / (hist.sum() + 1e-12)
    from scipy.signal import find_peaks as _find_peaks
    peaks_a, _ = _find_peaks(hist_norm, height=np.max(hist_norm) * 0.15, distance=2)
    n_amp_levels = float(min(len(peaks_a), 12))
    amp_hist_entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-12)))
    q25, q75 = float(np.percentile(amp, 25)), float(np.percentile(amp, 75))
    amp_iqr = q75 - q25
    zero_ratio = float(np.mean(amp < 0.1))
    return [n_amp_levels, amp_hist_entropy, amp_iqr, zero_ratio]


def _compute_freq_transition_features(seg: np.ndarray) -> list:
    """Frequency transition rate and smoothness — 4 features.
    GFSK: smooth transitions, FSK2: sharp jumps, CPFSK: continuous."""
    phase = np.unwrap(np.angle(seg))
    inst_freq = np.diff(phase)
    if len(inst_freq) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    sign_changes = np.sum(np.diff(np.sign(inst_freq)) != 0)
    transition_rate = float(sign_changes / len(inst_freq))
    freq_accel = np.diff(inst_freq)
    freq_smoothness = float(np.mean(np.abs(freq_accel)))
    freq_median = float(np.median(np.abs(inst_freq)))
    freq_deviation_ratio = float(np.std(inst_freq) / (freq_median + 1e-6))
    if len(inst_freq) > 20:
        lag = max(1, len(inst_freq) // 20)
        ac = float(np.corrcoef(inst_freq[:-lag], inst_freq[lag:])[0, 1])
        freq_autocorr = 0.0 if np.isnan(ac) else ac
    else:
        freq_autocorr = 0.0
    return [transition_rate, freq_smoothness, freq_deviation_ratio, freq_autocorr]


def _compute_8th_order_cumulants(seg: np.ndarray) -> list:
    """8th-order cumulants — 2 features. High-order PSK/APSK discrimination."""
    x = seg
    c80 = float(min(np.abs(np.mean(x ** 8)), 1e6))
    c84 = float(min(np.mean(np.abs(x) ** 8), 1e6))
    return [c80, c84]


def _compute_spectral_asymmetry(seg: np.ndarray) -> float:
    """Spectral asymmetry ratio — 1 feature.
    USB/SSB: ratio != 0, symmetric signals (FSK/PSK/QAM): ~0.
    Range: [-1, 1], 0=perfectly symmetric."""
    n = len(seg)
    spec = np.abs(np.fft.fft(seg)) ** 2
    pos_power = float(np.sum(spec[1:n//2]))
    neg_power = float(np.sum(spec[n//2+1:]))
    total = pos_power + neg_power + 1e-12
    return float(np.clip((pos_power - neg_power) / total, -1.0, 1.0))


def _compute_freq_histogram_entropy(seg: np.ndarray) -> float:
    """Instantaneous frequency histogram entropy — 1 feature.
    FSK2: ~1 bit, FSK4: ~2 bits, analog FM: high entropy."""
    phase = np.unwrap(np.angle(seg))
    inst_freq = np.diff(phase)
    if len(inst_freq) < 4:
        return 0.0
    inst_freq = np.clip(inst_freq, -np.pi, np.pi)
    hist, _ = np.histogram(inst_freq, bins=32, range=(-np.pi, np.pi))
    hist_norm = hist.astype(float) / (hist.sum() + 1e-12)
    return float(-np.sum(hist_norm * np.log2(hist_norm + 1e-12)))


def _compute_hoc(seg: np.ndarray) -> list:
    """Higher-order cumulants — 7 features.
    Uses CFO-invariant cyclic spectral cumulant peaks combined with time averages."""
    n = len(seg)
    x = seg
    pwr = float(np.mean(np.abs(x) ** 2)) + 1e-12

    # 2nd order (CFO-invariant cyclic peak combined with time average)
    c20_raw = float(np.abs(np.mean(x ** 2))) / pwr
    f2 = np.abs(np.fft.fft(x ** 2))
    c20_inv = float(np.max(f2) / (n * pwr))
    c20 = max(c20_raw, c20_inv)
    c21 = 1.0

    # 4th order (CFO-invariant cyclic peak combined with time average)
    m40_raw = np.mean(x ** 4)
    m20_raw = np.mean(x ** 2)
    c40_raw = float(np.abs(m40_raw - 3.0 * m20_raw ** 2)) / (pwr ** 2)
    f4 = np.abs(np.fft.fft(x ** 4))
    c40_inv = float(np.max(f4) / (n * pwr ** 2))
    c40 = max(c40_raw, c40_inv)

    c41 = float(np.abs(np.mean(np.abs(x) ** 2 * x ** 2) - 2 * pwr * np.mean(x ** 2))) / (pwr ** 2)
    c42 = float(np.mean(np.abs(x) ** 4) - 2 * pwr ** 2 - abs(np.mean(x ** 2)) ** 2) / (pwr ** 2)

    # 6th order (simplified)
    c60 = float(np.abs(np.mean(x ** 6))) / (pwr ** 3)
    c63 = float(np.mean(np.abs(x) ** 6)) / (pwr ** 3)

    return [c20, c21, c40, c41, abs(c42), c60, c63]


def _compute_amplitude_stats(seg: np.ndarray) -> list:
    """Amplitude envelope statistics — 8 features."""
    amp = np.abs(seg)

    mean_a = float(np.mean(amp))
    var_a = float(np.var(amp))
    skew_a = float(_skewness(amp))
    kurt_a = float(_kurtosis(amp))
    max_a = float(np.max(amp))
    min_a = float(np.min(amp))
    # Amplitude fluctuation ratio
    afr = float(np.std(amp) / (np.mean(amp) + 1e-12))
    # Peak-to-average ratio (PAR)
    par = float(np.max(amp ** 2) / (np.mean(amp ** 2) + 1e-12))

    return [mean_a, var_a, skew_a, kurt_a, max_a, min_a, afr, par]


def _compute_phase_stats(seg: np.ndarray) -> list:
    """Phase statistics — 6 features."""
    phase = np.unwrap(np.angle(seg))
    phase_diff = np.diff(phase)

    var_p = float(np.var(phase_diff))
    kurt_p = float(_kurtosis(phase_diff))
    mean_abs_p = float(np.mean(np.abs(phase_diff)))
    std_p = float(np.std(phase_diff))
    # Non-linear component
    phase_nl = float(np.var(np.abs(phase_diff) - np.mean(np.abs(phase_diff))))
    # Phase symmetry (abs skewness)
    skew_p = float(abs(_skewness(phase_diff)))

    return [var_p, kurt_p, mean_abs_p, std_p, phase_nl, skew_p]


def _compute_freq_stats(seg: np.ndarray) -> list:
    """Instantaneous frequency statistics — 5 features."""
    phase = np.unwrap(np.angle(seg))
    inst_freq = np.diff(phase) / (2 * np.pi)

    if len(inst_freq) == 0:
        return [0.0] * 5

    mean_f = float(np.mean(inst_freq))
    std_f = float(np.std(inst_freq))
    kurt_f = float(_kurtosis(inst_freq))
    # Frequency deviation (std of |freq - mean|)
    freq_dev = float(np.std(np.abs(inst_freq - mean_f)))
    # Bimodality coefficient (for FSK detection)
    bmc = float(_bimodality_coefficient(inst_freq))

    return [mean_f, std_f, kurt_f, freq_dev, bmc]


def _compute_spectral_features(seg: np.ndarray) -> list:
    """Spectral features — 8 features."""
    n = len(seg)
    n_fft = min(n, 512)

    # Power spectrum
    window = np.hanning(n_fft)
    spec = np.abs(np.fft.fftshift(np.fft.fft(seg[:n_fft] * window))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft))
    spec_norm = spec / (np.sum(spec) + 1e-12)

    # Spectral entropy
    entropy = float(-np.sum(spec_norm * np.log2(spec_norm + 1e-12)))

    # Spectral flatness
    geom = float(np.exp(np.mean(np.log(spec + 1e-12))))
    arith = float(np.mean(spec) + 1e-12)
    flatness = geom / arith

    # Spectral centroid
    centroid = float(np.sum(np.abs(freqs) * spec_norm))

    # Peak-to-mean ratio in spectrum
    pmr = float(np.max(spec) / (np.mean(spec) + 1e-12))

    # Spectral spread
    spread = float(np.sqrt(np.sum((freqs - centroid) ** 2 * spec_norm) + 1e-12))

    # Number of significant spectral peaks (above 10% of max)
    threshold = 0.1 * np.max(spec)
    peaks = np.sum(
        (spec[1:-1] > spec[:-2]) & (spec[1:-1] > spec[2:]) & (spec[1:-1] > threshold)
    )
    n_peaks = float(min(peaks, 20))  # cap at 20

    # DC ratio (power at DC vs total)
    mid = n_fft // 2
    dc_ratio = float(spec[mid] / (np.sum(spec) + 1e-12))

    # Sideband symmetry (for AM detection)
    half = n_fft // 2
    left = spec[:half]
    right = spec[half:][::-1]
    min_len = min(len(left), len(right))
    sym = float(np.corrcoef(left[:min_len], right[:min_len])[0, 1]
                if min_len > 2 else 0.0)
    sym = 0.0 if np.isnan(sym) else sym

    return [entropy, flatness, centroid, pmr, spread, n_peaks, dc_ratio, sym]


def _compute_pattern_features(seg: np.ndarray) -> list:
    """Cyclostationary and pattern features — 6 features."""
    n = len(seg)

    # Cyclic power at symbol-rate multiples
    # Use autocorrelation of squared envelope
    env_sq = np.abs(seg) ** 2
    env_sq -= np.mean(env_sq)

    if len(env_sq) > 4:
        autocorr = np.correlate(env_sq, env_sq, mode='full')
        mid = len(autocorr) // 2
        autocorr = autocorr[mid:mid + min(64, len(autocorr) - mid)]
        autocorr_norm = autocorr / (autocorr[0] + 1e-12)

        # Peak of autocorrelation beyond lag 1 (symbol period)
        if len(autocorr_norm) > 4:
            cyclic_peak = float(np.max(autocorr_norm[2:]))
        else:
            cyclic_peak = 0.0

        # Autocorrelation at lag 1
        ac_lag1 = float(autocorr_norm[1]) if len(autocorr_norm) > 1 else 0.0
    else:
        cyclic_peak = 0.0
        ac_lag1 = 0.0

    # I/Q imbalance (amplitude difference between I and Q channels)
    i_pwr = float(np.mean(seg.real ** 2))
    q_pwr = float(np.mean(seg.imag ** 2))
    iq_imbalance = float(abs(i_pwr - q_pwr) / (i_pwr + q_pwr + 1e-12))

    # I/Q correlation
    iq_corr_val = float(np.corrcoef(seg.real, seg.imag)[0, 1])
    iq_corr_val = 0.0 if np.isnan(iq_corr_val) else iq_corr_val

    # Kurtosis of real part
    kurt_i = float(_kurtosis(seg.real))

    # Kurtosis of imaginary part
    kurt_q = float(_kurtosis(seg.imag))

    return [cyclic_peak, ac_lag1, iq_imbalance, iq_corr_val, kurt_i, kurt_q]


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis."""
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4) - 3.0)


def _skewness(x: np.ndarray) -> float:
    """Skewness."""
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3))


def _bimodality_coefficient(x: np.ndarray) -> float:
    """
    Physical frequency bimodality metric.
    Detects sustained two-tone frequency states (FSK2) while rejecting
    transient 1-sample derivative transition spikes (BPSK/QPSK/8PSK).
    FSK2: two sustained tones -> platykurtic (kurt < 0), symmetric (skew ~ 0) -> BC in [0.35, 0.95].
    BPSK/QPSK/8PSK: central carrier + transition spikes -> leptokurtic (kurt >> 0) -> BC in [0.01, 0.20].
    """
    n = len(x)
    if n < 8:
        return 0.0
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-8:
        return 0.0

    norm = (x - mu) / sigma
    skew = float(np.mean(norm ** 3))
    kurt = float(np.mean(norm ** 4) - 3.0)

    # Base kurtosis-based bimodality (platykurtic vs leptokurtic)
    if kurt > 1.0:
        base_bc = 1.0 / (kurt + 3.0)
    else:
        sym_factor = max(0.0, 1.0 - abs(skew) / 1.5)
        base_bc = (1.0 / max(0.5, kurt + 3.0)) * sym_factor

    # Histogram mode verification (look for 2 distinct sustained peaks)
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    span = p99 - p1
    if span > 1e-5:
        hist, _ = np.histogram(x, bins=28, range=(p1, p99))
        h_smooth = np.convolve(hist.astype(float), [0.2, 0.6, 0.2], mode='same')
        h_norm = h_smooth / (n + 1e-12)
        max_h = np.max(h_norm)
        from scipy.signal import find_peaks as _find_peaks
        peaks, _ = _find_peaks(h_norm, height=max_h * 0.25, distance=3)
        if len(peaks) >= 2:
            mode_factor = 1.2
        else:
            mode_factor = 0.5
    else:
        mode_factor = 0.5

    return float(np.clip(base_bc * mode_factor, 0.0, 1.0))


def _kurtosity_excess(x: np.ndarray) -> float:
    """Sample excess kurtosis with bias correction."""
    n = len(x)
    if n < 4:
        return 0.0
    mu = np.mean(x)
    s = np.std(x)
    if s < 1e-12:
        return 0.0
    k4 = np.mean(((x - mu) / s) ** 4)
    # Bias correction
    return float((k4 - 3) * (n + 1) * n / ((n - 1) * (n - 2)) - 6.0 / (n + 1))


def extract_batch(samples_list: list) -> np.ndarray:
    """
    Extract features from a list of sample arrays.
    Returns (N, 72) float32 array.
    """
    return np.array([extract_features(s) for s in samples_list],
                    dtype=np.float32)


def _numpy_hoc_features(seg: np.ndarray) -> list:
    """
    NumPy fallback: compute the same 16 features as signal_features.dll.
    Used when the C++ DLL is not available.
    """
    n = len(seg)
    amp  = np.abs(seg)
    phi  = np.unwrap(np.angle(seg))
    freq = np.diff(phi)

    mean_amp = float(np.mean(amp))
    if mean_amp < 1e-15:
        return [0.0] * 16

    an = amp / mean_amp - 1.0

    # 2nd-order moments (CFO-invariant cyclic peak combined with time average)
    p2 = float(np.mean(np.abs(seg) ** 2)) + 1e-12
    norm = p2 ** 2
    M20 = np.mean(seg ** 2)
    M21 = p2
    c20_raw = float(np.abs(M20)) / p2
    f2 = np.abs(np.fft.fft(seg ** 2))
    c20_inv = float(np.max(f2) / (n * p2))
    C20 = max(c20_raw, c20_inv)

    # 4th-order cumulants (CFO-invariant cyclic peak combined with time average)
    M40 = np.mean(seg ** 4)
    c40_raw = float(np.abs(M40 - 3.0 * M20 ** 2)) / norm
    f4 = np.abs(np.fft.fft(seg ** 4))
    c40_inv = float(np.max(f4) / (n * norm))
    C40 = max(c40_raw, c40_inv)

    p4 = float(np.mean(np.abs(seg) ** 4))
    C42 = (p4 - float(np.abs(M20)) ** 2 - 2.0 * norm) / norm

    sigma_aa = float(np.std(an))
    mean_abs = float(np.mean(np.abs(an)))
    skew_an  = float(_skewness(an))
    kurt_an  = float(_kurtosis(an))
    sigma_dp = float(np.std(phi))
    sigma_af = float(np.std(freq))

    fft_mag  = np.abs(np.fft.fft(seg))
    total    = float(np.sum(fft_mag)) + 1e-15
    gamma_max = float(np.max(fft_mag)) / (total / n)
    P = float(np.sum(fft_mag[:n//2])) / total

    return [
        float(C20), float(M21), float(C40), 0.0, float(C42),
        float(np.abs(M20)), float(np.angle(M20)),
        sigma_aa, sigma_dp, sigma_af, gamma_max,
        sigma_aa, P, skew_an, kurt_an, mean_abs,
    ]
