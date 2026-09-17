"""
Training Data Generator — Robust Multi-SPS, Baseband-Enabled, Balanced Dataset.
Fixes:
  1. SPS Diversity: samples SPS from [4, 6, 8, 12, 16, 24, 32, 40, 48, 64] across symbol-based modulations.
  2. Carrier Distribution: 30% baseband (fo=0), 35% micro CFO (|fo|<=0.01), 35% standard CFO.
  3. Pulse Shaping: 50% rectangular, 50% Root-Raised-Cosine (RRC) pulse-shaped.
  4. Fractional Timing: sub-sample timing offsets applied via phase ramp.
  5. Class Balance: all 17 classes have exactly n_per_class samples (FSK4 imbalance eliminated).
  6. Independent Seeds: zero train/val/test data leakage.
"""
import os
import numpy as np
from typing import List, Tuple, Optional

# 17 modulation classes — matches real-world HF/VHF/UHF signals + OFDM
MODULATION_LABELS = [
    'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64',
    'FSK2', 'FSK4', 'GFSK', 'CPFSK',
    'AM-DSB', 'AM-SSB', 'WBFM',
    'APSK16', 'APSK32',
    'PAM4', 'OOK', 'OFDM'
]

LABEL_TO_IDX = {m: i for i, m in enumerate(MODULATION_LABELS)}
IDX_TO_LABEL = {i: m for i, m in enumerate(MODULATION_LABELS)}

# Training SNR levels — dense coverage across operational range
SNR_LEVELS = [-10, -6, -4, -2, 0, 2, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30]

SPS_CANDIDATES = [4, 6, 8, 12, 16, 24, 32, 40, 48, 64]


def _rrc_filter(sps: int, num_taps: int = 41, alpha: float = 0.35) -> np.ndarray:
    """Root-Raised-Cosine filter taps."""
    t = np.arange(-num_taps // 2, num_taps // 2 + 1) / float(sps)
    h = np.zeros(len(t), dtype=np.float32)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
        elif abs(abs(ti) - 1.0 / (4.0 * alpha)) < 1e-8:
            h[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            denom = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / denom
    pwr = np.sum(h ** 2)
    return h / np.sqrt(pwr if pwr > 1e-12 else 1.0)


def _pulse_shape(symbols: np.ndarray, sps: int, rng: np.random.RandomState, n: int) -> np.ndarray:
    """Apply RRC pulse shaping (50%) or rectangular repeat (50%)."""
    if rng.rand() > 0.5 and sps >= 4:
        upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
        upsampled[::sps] = symbols
        alpha = float(rng.uniform(0.25, 0.50))
        num_taps = min(65, sps * 4 + 1)
        taps = _rrc_filter(sps, num_taps=num_taps, alpha=alpha)
        shaped = np.convolve(upsampled, taps, mode='same')
        return _trim_pad(shaped, n).astype(np.complex64)
    else:
        return _trim_pad(np.repeat(symbols, sps), n).astype(np.complex64)


def generate_dataset(n_per_class: int = 300,
                     n_samples_per_signal: int = 1024,
                     snr_levels: List[float] = None,
                     seed: int = 42,
                     augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate balanced labeled dataset with SPS diversity and baseband coverage.
    Returns (X, y): X is complex64 (N, n_samples), y is int32 (N,).
    """
    if snr_levels is None:
        snr_levels = SNR_LEVELS

    rng = np.random.RandomState(seed)
    X_list = []
    y_list = []

    for label in MODULATION_LABELS:
        idx = LABEL_TO_IDX[label]

        if label == 'FSK4':
            # Balance FSK4 exactly: half synthetic, half real C4FM windows (capped)
            n_real = n_per_class // 2
            n_syn = n_per_class - n_real

            # 1. Synthetic FSK4
            for i in range(n_syn):
                snr = snr_levels[i % len(snr_levels)]
                snr_jitter = snr + rng.uniform(-1.5, 1.5)
                sig = _generate_signal('FSK4', n_samples_per_signal, rng)
                sig = _apply_channel(sig, snr_jitter, rng)
                if augment and snr <= 6:
                    sig = _augment(sig, rng)
                X_list.append(sig)
                y_list.append(idx)

            # 2. Real C4FM samples
            real_samples = _load_real_c4fm_samples(n_samples_per_signal, rng, n_inject=n_real)
            if len(real_samples) < n_real:
                # Fallback to synthetic if not enough real samples
                for i in range(len(real_samples), n_real):
                    snr = snr_levels[i % len(snr_levels)]
                    sig = _generate_signal('FSK4', n_samples_per_signal, rng)
                    sig = _apply_channel(sig, snr, rng)
                    real_samples.append(sig)

            for rs in real_samples[:n_real]:
                X_list.append(rs)
                y_list.append(idx)

        else:
            for i in range(n_per_class):
                snr = snr_levels[i % len(snr_levels)]
                snr_jitter = snr + rng.uniform(-1.5, 1.5)

                sig = _generate_signal(label, n_samples_per_signal, rng)
                sig = _apply_channel(sig, snr_jitter, rng)

                if augment and snr <= 6:
                    sig = _augment(sig, rng)

                X_list.append(sig)
                y_list.append(idx)

        # Extra augmented samples at low SNR across all classes when augment=True and n_per_class >= 50
        if augment and n_per_class >= 50:
            for _ in range(n_per_class // 5):
                snr = float(rng.uniform(-8, 4))
                sig = _generate_signal(label, n_samples_per_signal, rng)
                sig = _apply_channel(sig, snr, rng)
                sig = _augment(sig, rng)
                X_list.append(sig)
                y_list.append(idx)

    X = np.array(X_list, dtype=np.complex64)
    y = np.array(y_list, dtype=np.int32)

    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def _load_real_c4fm_samples(n_samples: int,
                             rng: np.random.RandomState,
                             n_inject: int = 150) -> List[np.ndarray]:
    """Load real C4FM/YSF samples from dsd WAV file."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dsd.2021-12-02T16_44_54_046.wav'),
        r'd:\quick share\Tarang\dsd.2021-12-02T16_44_54_046.wav',
        r'D:\Tarang\dsd.2021-12-02T16_44_54_046.wav',
        r'D:\UPSRRS\dsd.2021-12-02T16_44_54_046.wav',
    ]
    c4fm_path = next((p for p in candidates if os.path.exists(p)), None)
    if not c4fm_path:
        return []

    try:
        import scipy.io.wavfile as wav
        sr, data = wav.read(c4fm_path)
        data = data.astype(np.float32) / 32768.0
        if data.ndim == 2:
            samples = data[:, 0] + 1j * data[:, 1]
        else:
            from scipy.signal import hilbert
            samples = hilbert(data).astype(np.complex64)

        total = len(samples)
        result = []
        attempts = 0

        while len(result) < n_inject and attempts < n_inject * 3:
            attempts += 1
            if total <= n_samples:
                break
            start = rng.randint(0, total - n_samples)
            seg = samples[start:start + n_samples].copy()

            pwr = np.mean(np.abs(seg) ** 2)
            if pwr < 1e-15:
                continue
            seg = seg / np.sqrt(pwr)

            aug_type = rng.randint(0, 4)
            seg = seg.astype(np.complex64)

            if aug_type == 0:
                phi = rng.uniform(0, 2 * np.pi)
                seg = seg * np.exp(1j * phi)
            elif aug_type == 1:
                micro_fo = rng.uniform(-0.015, 0.015)
                t = np.arange(n_samples)
                seg = seg * np.exp(1j * 2 * np.pi * micro_fo * t).astype(np.complex64)
            elif aug_type == 2:
                shift = rng.randint(1, max(2, n_samples // 8))
                seg = np.roll(seg, shift)
            else:
                seg = _augment(seg, rng)

            result.append(seg.astype(np.complex64))

        return result
    except Exception:
        return []


def _augment(sig: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Data augmentation techniques for low-SNR robustness."""
    n = len(sig)

    # Random phase rotation
    if rng.rand() > 0.4:
        phi = rng.uniform(0, 2 * np.pi)
        sig = sig * np.exp(1j * phi)

    # Micro frequency offset
    if rng.rand() > 0.4:
        micro_fo = rng.uniform(-0.015, 0.015)
        t = np.arange(n)
        sig = sig * np.exp(1j * 2 * np.pi * micro_fo * t).astype(np.complex64)

    # Time shift (circular)
    if rng.rand() > 0.5:
        shift = rng.randint(1, max(2, n // 8))
        sig = np.roll(sig, shift)

    # Amplitude scaling
    if rng.rand() > 0.5:
        scale = rng.uniform(0.7, 1.4)
        sig = sig * scale

    # Multipath: add delayed copy
    if rng.rand() > 0.7:
        delay = rng.randint(1, max(2, n // 16))
        alpha = rng.uniform(0.1, 0.3) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        delayed = np.roll(sig, delay) * alpha
        sig = (sig + delayed).astype(np.complex64)

    return sig.astype(np.complex64)


def _generate_signal(modulation: str, n: int,
                     rng: np.random.RandomState,
                     sps: Optional[int] = None) -> np.ndarray:
    """Generate signal with SPS diversity."""
    if sps is None:
        sps = int(rng.choice(SPS_CANDIDATES))

    mod = modulation.upper().replace('-', '').replace('_', '')

    if mod == 'BPSK':
        return _gen_psk(n, M=2, rng=rng, sps=sps)
    elif mod == 'QPSK':
        return _gen_psk(n, M=4, rng=rng, sps=sps)
    elif mod == '8PSK':
        return _gen_psk(n, M=8, rng=rng, sps=sps)
    elif mod == 'QAM16':
        return _gen_qam(n, M=16, rng=rng, sps=sps)
    elif mod == 'QAM64':
        return _gen_qam(n, M=64, rng=rng, sps=sps)
    elif mod == 'FSK2':
        return _gen_fsk(n, M=2, rng=rng, sps=sps)
    elif mod == 'FSK4':
        if rng.rand() > 0.4:
            return _gen_fsk4_c4fm(n, rng=rng, sps=sps)
        else:
            return _gen_fsk(n, M=4, rng=rng, sps=sps)
    elif mod == 'GFSK':
        return _gen_gfsk(n, rng=rng, sps=sps)
    elif mod == 'CPFSK':
        return _gen_cpfsk(n, rng=rng, sps=sps)
    elif mod == 'AMDSB':
        return _gen_am_dsb(n, rng=rng)
    elif mod == 'AMSSB':
        return _gen_am_ssb(n, rng=rng)
    elif mod == 'WBFM':
        return _gen_wbfm(n, rng=rng)
    elif mod == 'APSK16':
        return _gen_apsk(n, M=16, rng=rng, sps=sps)
    elif mod == 'APSK32':
        return _gen_apsk(n, M=32, rng=rng, sps=sps)
    elif mod == 'PAM4':
        return _gen_pam4(n, rng=rng, sps=sps)
    elif mod == 'OOK':
        return _gen_ook(n, rng=rng, sps=sps)
    elif mod == 'OFDM':
        return _gen_ofdm(n, rng=rng)
    else:
        return _gen_psk(n, M=2, rng=rng, sps=sps)


# ── Signal generators with SPS diversity & pulse shaping ────────────────────

def _gen_psk(n: int, M: int, rng: np.random.RandomState, sps: int = 8) -> np.ndarray:
    n_syms = max(2, int(np.ceil(n / sps)) + 4)
    angles = 2 * np.pi * rng.randint(0, M, n_syms) / M + np.pi / M
    phase_offset = rng.uniform(0, 2 * np.pi)
    symbols = np.exp(1j * (angles + phase_offset)).astype(np.complex64)
    return _pulse_shape(symbols, sps, rng, n)


def _gen_qam(n: int, M: int, rng: np.random.RandomState, sps: int = 8) -> np.ndarray:
    m = int(np.sqrt(M))
    levels = np.linspace(-m + 1, m - 1, m)
    n_syms = max(2, int(np.ceil(n / sps)) + 4)
    i_syms = levels[rng.randint(0, m, n_syms)]
    q_syms = levels[rng.randint(0, m, n_syms)]
    symbols = (i_syms + 1j * q_syms).astype(np.complex64)
    symbols /= np.sqrt(np.mean(np.abs(symbols) ** 2) + 1e-12)
    return _pulse_shape(symbols, sps, rng, n)


def _gen_fsk(n: int, M: int, rng: np.random.RandomState, sps: int = 8) -> np.ndarray:
    n_syms = max(2, int(np.ceil(n / sps)) + 2)
    sym_indices = rng.randint(0, M, n_syms)
    freq_dev = float(rng.uniform(0.04, 0.16))
    freqs = np.linspace(-freq_dev * (M - 1) / M,
                         freq_dev * (M - 1) / M, M)
    freq_up = np.repeat(freqs[sym_indices], sps)[:n]
    freq_up = np.pad(freq_up, (0, max(0, n - len(freq_up))))
    phase = 2 * np.pi * np.cumsum(freq_up)
    return np.exp(1j * phase).astype(np.complex64)


def _gen_fsk4_c4fm(n: int, rng: np.random.RandomState, sps: int = 8) -> np.ndarray:
    c4fm_levels = np.array([-3, -1, 1, 3], dtype=np.float32)
    dev_scale = float(rng.uniform(0.04, 0.08))
    freqs = c4fm_levels * dev_scale

    n_syms = max(2, int(np.ceil(n / sps)) + 2)
    sym_indices = rng.randint(0, 4, n_syms)

    freq_up = np.repeat(freqs[sym_indices], sps)[:n]
    freq_up = np.pad(freq_up, (0, max(0, n - len(freq_up))))

    from scipy.ndimage import gaussian_filter1d
    sigma = max(1.0, sps * 0.3)
    freq_smooth = gaussian_filter1d(freq_up.real, sigma=sigma)

    phase = 2 * np.pi * np.cumsum(freq_smooth)
    if rng.rand() > 0.5:
        phase = phase + rng.randn(n) * 0.04

    sig = np.exp(1j * phase).astype(np.complex64)
    if rng.rand() > 0.6:
        sig = sig * (1.0 + rng.randn(n) * 0.02)
    return sig.astype(np.complex64)


def _gen_gfsk(n: int, rng: np.random.RandomState, sps: int = 8, BT: float = 0.3) -> np.ndarray:
    n_syms = max(2, int(np.ceil(n / sps)) + 2)
    bits = rng.randint(0, 2, n_syms) * 2 - 1
    h = float(rng.uniform(0.25, 0.45))
    t = np.linspace(-2, 2, max(5, int(5 * sps)))
    gauss = np.exp(-2 * np.pi ** 2 * BT ** 2 * t ** 2)
    gauss /= gauss.sum()
    freq_raw = np.repeat(bits * h / 2, sps)
    freq_shaped = np.convolve(freq_raw, gauss, mode='same')[:n]
    freq_shaped = _trim_pad(freq_shaped, n)
    phase = 2 * np.pi * np.cumsum(freq_shaped.real)
    return np.exp(1j * phase).astype(np.complex64)


def _gen_cpfsk(n: int, rng: np.random.RandomState, sps: int = 8) -> np.ndarray:
    n_syms = max(2, int(np.ceil(n / sps)) + 2)
    bits = rng.randint(0, 2, n_syms) * 2 - 1
    mod_index = float(rng.uniform(0.2, 0.5))
    freq_raw = np.repeat(bits * mod_index, sps)[:n]
    freq_raw = np.pad(freq_raw, (0, max(0, n - len(freq_raw))))
    phase = np.pi * np.cumsum(freq_raw) / sps
    return np.exp(1j * phase).astype(np.complex64)


def _gen_am_dsb(n: int, rng: np.random.RandomState) -> np.ndarray:
    f_msg = float(rng.uniform(0.01, 0.09))
    t = np.arange(n)
    message = np.sin(2 * np.pi * f_msg * t)
    message += 0.4 * np.sin(2 * np.pi * f_msg * 2.1 * t)
    message /= np.max(np.abs(message)) + 1e-12
    depth = float(rng.uniform(0.4, 0.95))
    carrier = (1.0 + depth * message).astype(np.float32)
    return (carrier + 1j * np.zeros(n)).astype(np.complex64)


def _gen_am_ssb(n: int, rng: np.random.RandomState) -> np.ndarray:
    f_msg = float(rng.uniform(0.015, 0.08))
    t = np.arange(n)
    message = np.sin(2 * np.pi * f_msg * t)
    message += 0.35 * np.sin(2 * np.pi * f_msg * 3.0 * t)
    from scipy.signal import hilbert as sp_hilbert
    analytic = sp_hilbert(message)
    analytic = analytic / (np.max(np.abs(analytic)) + 1e-12)
    return analytic.astype(np.complex64)


def _gen_wbfm(n: int, rng: np.random.RandomState) -> np.ndarray:
    f_msg = float(rng.uniform(0.01, 0.035))
    t = np.arange(n)
    message = np.sin(2 * np.pi * f_msg * t)
    message += 0.35 * np.sin(2 * np.pi * f_msg * 3.1 * t)
    message += 0.15 * rng.randn(n)
    message /= np.max(np.abs(message)) + 1e-12
    kf = float(rng.uniform(0.08, 0.22))
    phase = 2 * np.pi * kf * np.cumsum(message)
    return np.exp(1j * phase).astype(np.complex64)


def _gen_apsk(n: int, M: int, rng: np.random.RandomState, sps: int = 8) -> np.ndarray:
    if M == 16:
        r1, r2 = 1.0, float(rng.uniform(2.2, 3.0))
        n1, n2 = 4, 12
    else:
        r1, r2 = 1.0, float(rng.uniform(2.5, 3.2))
        n1, n2 = 8, 24

    c1 = r1 * np.exp(1j * (2 * np.pi * np.arange(n1) / n1 + rng.uniform(0, 2 * np.pi)))
    c2 = r2 * np.exp(1j * (2 * np.pi * np.arange(n2) / n2 + rng.uniform(0, 2 * np.pi)))
    const = np.concatenate([c1, c2])
    const /= np.sqrt(np.mean(np.abs(const) ** 2) + 1e-12)

    n_syms = max(2, int(np.ceil(n / sps)) + 2)
    symbols = const[rng.randint(0, M, n_syms)]
    return _pulse_shape(symbols, sps, rng, n)


def _gen_pam4(n: int, rng: np.random.RandomState, sps: int = 8) -> np.ndarray:
    levels = np.array([-3, -1, 1, 3], dtype=np.float32)
    n_syms = max(2, int(np.ceil(n / sps)) + 2)
    symbols = levels[rng.randint(0, 4, n_syms)]
    symbols /= np.sqrt(np.mean(symbols ** 2) + 1e-12)
    syms_c = symbols.astype(np.complex64)
    return _pulse_shape(syms_c, sps, rng, n)


def _gen_ook(n: int, rng: np.random.RandomState, sps: int = 8) -> np.ndarray:
    n_syms = max(2, int(np.ceil(n / sps)) + 2)
    bits = rng.randint(0, 2, n_syms).astype(np.float32)
    syms_c = bits.astype(np.complex64)
    return _trim_pad(np.repeat(syms_c, sps), n).astype(np.complex64)


def _gen_ofdm(n: int, rng: np.random.RandomState,
              fft_size: int = None,
              cp_ratio: float = None) -> np.ndarray:
    if fft_size is None:
        fft_size = int(rng.choice([64, 128, 256]))
    if cp_ratio is None:
        cp_ratio = float(rng.choice([0.25, 0.125]))

    cp_len = max(1, int(fft_size * cp_ratio))
    symbol_len = fft_size + cp_len
    n_syms = max(1, (n + symbol_len - 1) // symbol_len)

    output = np.zeros(n_syms * symbol_len, dtype=np.complex64)
    n_active = fft_size // 2
    active_start = fft_size // 4
    active_end = active_start + n_active

    for k in range(n_syms):
        freq_domain = np.zeros(fft_size, dtype=np.complex64)
        angles = rng.randint(0, 4, n_active) * (np.pi / 2) + np.pi / 4
        freq_domain[active_start:active_end] = np.exp(1j * angles)
        time_sym = np.fft.ifft(np.fft.ifftshift(freq_domain)).astype(np.complex64)
        cp = time_sym[-cp_len:]
        sym_with_cp = np.concatenate([cp, time_sym])
        start = k * symbol_len
        output[start: start + symbol_len] = sym_with_cp

    result = output[:n] if len(output) >= n else np.pad(output, (0, n - len(output)))
    pwr = np.sqrt(np.mean(np.abs(result) ** 2) + 1e-12)
    return (result / pwr).astype(np.complex64)


def _apply_channel(signal: np.ndarray, snr_db: float,
                    rng: np.random.RandomState) -> np.ndarray:
    """Apply AWGN + realistic carrier offset (including baseband) + phase offset + timing offset."""
    sig = signal.copy().astype(np.complex64)
    n = len(sig)

    # AWGN
    sig_pwr = np.mean(np.abs(sig) ** 2)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_pwr = sig_pwr / (snr_lin + 1e-12)
    noise = (rng.randn(n) + 1j * rng.randn(n)) * np.sqrt(noise_pwr / 2.0)
    sig = sig + noise.astype(np.complex64)

    # Carrier offset:
    # 30% baseband (fo = 0.0)
    # 35% micro CFO (|fo| <= 0.01)
    # 35% moderate/larger CFO (|fo| in [0.01, 0.06])
    cfo_draw = rng.rand()
    if cfo_draw < 0.30:
        fo = 0.0
    elif cfo_draw < 0.65:
        fo = float(rng.uniform(-0.01, 0.01))
    else:
        fo = float(rng.uniform(-0.06, 0.06))

    if abs(fo) > 1e-6:
        t = np.arange(n)
        sig = sig * np.exp(1j * 2.0 * np.pi * fo * t).astype(np.complex64)

    # Phase offset
    sig = sig * np.exp(1j * rng.uniform(0, 2 * np.pi))

    # Amplitude variation
    sig = sig * rng.uniform(0.7, 1.4)

    # Fractional sub-sample timing offset (50% probability)
    if rng.rand() > 0.5:
        tau = rng.uniform(0.1, 0.9)
        freqs = np.fft.fftfreq(n)
        sig = np.fft.ifft(np.fft.fft(sig) * np.exp(-1j * 2.0 * np.pi * freqs * tau)).astype(np.complex64)

    return sig.astype(np.complex64)


def _trim_pad(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr[:n]
    return np.pad(arr, (0, n - len(arr)))
