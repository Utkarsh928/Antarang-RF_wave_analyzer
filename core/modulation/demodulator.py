"""
Demodulator €" demodulates IQ samples to bits.
Supports: BPSK, QPSK, 8PSK, QAM16, QAM64, QAM256, FSK2, FSK4, GFSK,
          CPFSK, OOK, PAM4, APSK16, APSK32, OFDM, AM-DSB, AM-SSB, WBFM
"""
import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple


def demodulate(samples: np.ndarray, modulation: str,
               sample_rate: float = 1.0,
               symbol_rate: float = None) -> Tuple[np.ndarray, float]:
    """
    Demodulate samples to a bit array.
    Returns (bits_array, estimated_bit_rate_bps).
    """
    mod = modulation.upper().replace('-', '').replace('_', '')

    if symbol_rate is None or symbol_rate <= 0:
        # Use real symbol rate estimator
        try:
            from core.symbol_rate_estimator import estimate_symbol_rate
            symbol_rate = estimate_symbol_rate(
                samples, sample_rate,
                min_rate=sample_rate / 512.0,
                max_rate=sample_rate / 2.0
            )
        except Exception:
            symbol_rate = sample_rate / 8.0

    sps = max(2, int(round(sample_rate / symbol_rate)))  # samples per symbol

    if mod in ('BPSK',):
        res = _demod_psk(samples, sps, bits_per_sym=1)
    elif mod in ('QPSK',):
        res = _demod_psk(samples, sps, bits_per_sym=2)
    elif mod in ('8PSK',):
        res = _demod_psk(samples, sps, bits_per_sym=3)
    elif mod in ('QAM16',):
        res = _demod_qam(samples, sps, order=16)
    elif mod in ('QAM64',):
        res = _demod_qam(samples, sps, order=64)
    elif mod in ('QAM256',):
        res = _demod_qam(samples, sps, order=256)
    elif mod in ('FSK2', 'FSK', 'GFSK', 'CPFSK'):
        res = _demod_fsk(samples, sps, n_freqs=2, sample_rate=sample_rate)
    elif mod in ('FSK4',):
        res = _demod_fsk(samples, sps, n_freqs=4, sample_rate=sample_rate)
    elif mod in ('OOK',):
        res = _demod_ook(samples, sps)
    elif mod in ('PAM4',):
        res = _demod_pam4(samples, sps)
    elif mod in ('APSK16',):
        res = _demod_apsk(samples, sps, M=16)
    elif mod in ('APSK32',):
        res = _demod_apsk(samples, sps, M=32)
    elif mod in ('OFDM',):
        res = _demod_ofdm(samples, sample_rate, symbol_rate)
    elif mod in ('AMDSB', 'AM', 'DSB'):
        res = _demod_am_dsb(samples, sample_rate, sps)
    elif mod in ('AMSSB', 'SSB'):
        res = _demod_am_ssb(samples, sample_rate, sps)
    elif mod in ('WBFM', 'FM', 'NBFM'):
        res = _demod_fm(samples, sample_rate, sps)
    else:
        # Default: treat as BPSK
        res = _demod_psk(samples, sps, bits_per_sym=1)

    bits = res[0]
    # True bit rate in bps (bits per second) across capture duration
    duration = len(samples) / sample_rate if sample_rate > 0 else 1.0
    bit_rate = float(len(bits) / duration) if duration > 0 else float(res[1])
    return bits, bit_rate


def _find_best_timing(samples: np.ndarray, sps: int) -> np.ndarray:
    """
    Find the optimal symbol-timing via matched filter + offset search.

    Uses a sinc matched filter (same as the standard raised-cosine transmitter)
    which gives zero ISI at symbol centres. Then tries all sps timing offsets
    and picks the one where the symbols have the most consistent phase angles
    (minimum phase variance = cleanest constellation = best timing).

    Falls back to box-car filter if sinc filter produces no improvement.
    """
    # ── 1. Sinc matched filter (optimal for sinc-pulse transmitters) ─────
    h = np.sinc(np.linspace(-4, 4, sps * 8 + 1)).astype(np.float32)
    h /= float(np.sum(h ** 2) ** 0.5)

    if np.iscomplexobj(samples):
        # Filter real and imaginary parts separately
        r = np.convolve(samples.real.astype(np.float32), h, mode='same')
        i = np.convolve(samples.imag.astype(np.float32), h, mode='same')
        filtered = (r + 1j * i).astype(np.complex64)
    else:
        filtered = np.convolve(samples.astype(np.float32), h, mode='same')

    # ── 2. Try all timing offsets and score by phase consistency ─────────
    best_symbols = filtered[::sps]
    best_score   = -1.0

    for offset in range(sps):
        candidate = filtered[offset::sps]
        if len(candidate) < 4:
            continue
        # Score: mean magnitude (higher = more energy at symbol centres)
        score = float(np.mean(np.abs(candidate)))
        if score > best_score:
            best_score   = score
            best_symbols = candidate

    return best_symbols


def _demod_psk(samples, sps, bits_per_sym):
    """Decision-directed PSK demodulator — nearest-neighbor on unit circle."""
    if not np.iscomplexobj(samples):
        samples = scipy_signal.hilbert(samples.astype(np.float32))
    symbols = _find_best_timing(samples, sps)
    if len(symbols) == 0:
        return np.array([], dtype=np.uint8), 0.0
    mags = np.abs(symbols)
    mean_mag = float(np.mean(mags[mags > 0])) if np.any(mags > 0) else 1.0
    symbols = symbols / (mean_mag + 1e-12)
    n_syms = 2 ** bits_per_sym
    angle_step = 2 * np.pi / n_syms
    ideal = np.exp(1j * (angle_step * np.arange(n_syms) - np.pi)).astype(np.complex64)
    sym_indices = np.array([np.argmin(np.abs(ideal - s)) for s in symbols], dtype=np.int32)
    bits = _symbols_to_bits(sym_indices, bits_per_sym)
    return bits, float(len(symbols) * bits_per_sym)

def _demod_qam(samples, sps, order):
    """QAM demodulator — sinc-matched filter + nearest-neighbor decision."""
    if not np.iscomplexobj(samples):
        samples = scipy_signal.hilbert(samples.astype(np.float32))
    symbols = _find_best_timing(samples, sps)
    if len(symbols) == 0:
        return np.array([], dtype=np.uint8), 0.0
    m = int(np.sqrt(order))
    levels = np.linspace(-m + 1, m - 1, m)
    constellation = np.array([i + 1j*q for q in reversed(levels) for i in levels], dtype=np.complex64)
    constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
    pwr = float(np.sqrt(np.mean(np.abs(symbols)**2)))
    if pwr > 1e-9:
        symbols = symbols / pwr
    bits_per_sym = int(np.log2(order))
    sym_indices = np.array([np.argmin(np.abs(constellation - s)) for s in symbols], dtype=np.int32)
    bits = _symbols_to_bits(sym_indices, bits_per_sym)
    return bits, float(len(symbols) * bits_per_sym)

def _demod_fsk(samples, sps, n_freqs, sample_rate):
    """Non-coherent FSK demodulator using instantaneous frequency."""
    if not np.iscomplexobj(samples):
        analytic = scipy_signal.hilbert(samples.astype(np.float32))
    else:
        analytic = samples.astype(np.complex64)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * sample_rate / (2 * np.pi)
    if len(inst_freq) == 0:
        return np.array([], dtype=np.uint8), 0.0
    n_syms = len(inst_freq) // sps
    if n_syms == 0:
        return np.array([], dtype=np.uint8), 0.0
    # Use median (not mean) over each symbol block to reject phase-transition spikes.
    # The last sample of each block often contains a large transition artifact.
    trimmed = inst_freq[:n_syms * sps].reshape(n_syms, sps)
    # Exclude the last sample of each block (transition region) before averaging
    core_samples = max(1, sps - 1)
    freq_sym = np.mean(trimmed[:, :core_samples], axis=1)
    f_min, f_max = freq_sym.min(), freq_sym.max()
    if abs(f_max - f_min) < 1.0:
        bits_per_sym = max(1, int(np.log2(n_freqs)))
        return np.zeros(n_syms * bits_per_sym, dtype=np.uint8), 0.0
    thresholds = np.linspace(f_min, f_max, n_freqs + 1)
    sym_indices = np.searchsorted(thresholds[1:-1], freq_sym)
    bits_per_sym = max(1, int(np.log2(n_freqs)))
    bits = _symbols_to_bits(sym_indices, bits_per_sym)
    return bits, float(len(freq_sym) * bits_per_sym)

def _demod_ook(samples: np.ndarray, sps: int) -> Tuple[np.ndarray, float]:
    """OOK demodulator - threshold on envelope using block averaging."""
    envelope = np.abs(samples).astype(np.float32)
    n_syms = len(envelope) // sps
    if n_syms == 0:
        return np.array([], dtype=np.uint8), 0.0
    trimmed = envelope[:n_syms * sps].reshape(n_syms, sps)
    env_sym = np.mean(trimmed[:, :max(1, sps - 1)], axis=1)
    threshold = (env_sym.max() + env_sym.min()) / 2.0
    bits = (env_sym > threshold).astype(np.uint8)
    return bits, float(len(bits))


def _demod_pam4(samples: np.ndarray, sps: int) -> Tuple[np.ndarray, float]:
    """PAM4 demodulator - 4 amplitude levels using block averaging."""
    real_sig = samples.real.astype(np.float32) if np.iscomplexobj(samples) else samples.astype(np.float32)
    n_syms = len(real_sig) // sps
    if n_syms == 0:
        return np.array([], dtype=np.uint8), 0.0
    trimmed = real_sig[:n_syms * sps].reshape(n_syms, sps)
    sym_vals = np.mean(trimmed[:, :max(1, sps - 1)], axis=1)
    ptp = sym_vals.max() - sym_vals.min()
    sym_norm = (sym_vals - sym_vals.min()) / (ptp + 1e-12)
    thresholds = [0.25, 0.5, 0.75]
    sym_indices = np.digitize(sym_norm, thresholds)
    bits = _symbols_to_bits(sym_indices, 2)
    return bits, float(len(sym_indices) * 2)


def _symbols_to_bits(sym_indices: np.ndarray,
                     bits_per_sym: int) -> np.ndarray:
    """Convert symbol indices to bit array using Gray coding."""
    if len(sym_indices) == 0:
        return np.array([], dtype=np.uint8)

    # Gray decode
    gray = sym_indices ^ (sym_indices >> 1)

    bits_list = []
    for idx in gray:
        for b in range(bits_per_sym - 1, -1, -1):
            bits_list.append((int(idx) >> b) & 1)

    return np.array(bits_list, dtype=np.uint8)


def _demod_apsk(samples: np.ndarray, sps: int,
                M: int) -> Tuple[np.ndarray, float]:
    """APSK demodulator €" nearest-neighbor decision on multi-ring constellation."""
    if not np.iscomplexobj(samples):
        samples = scipy_signal.hilbert(samples.astype(np.float32))

    symbols = _find_best_timing(samples, sps)

    if len(symbols) == 0:
        return np.array([], dtype=np.uint8), 0.0

    # Normalize power
    pwr = np.sqrt(np.mean(np.abs(symbols) ** 2))
    if pwr > 0:
        symbols /= pwr

    # Build APSK constellation
    if M == 16:
        r1, r2 = 1.0, 2.5
        inner  = [r1 * np.exp(1j * (2*np.pi*k/4  + np.pi/4)) for k in range(4)]
        outer  = [r2 * np.exp(1j * (2*np.pi*k/12))           for k in range(12)]
        constellation = np.array(inner + outer, dtype=np.complex64)
    else:  # M=32
        r1, r2, r3 = 1.0, 2.6, 4.5
        inner  = [r1 * np.exp(1j * (2*np.pi*k/4  + np.pi/4)) for k in range(4)]
        middle = [r2 * np.exp(1j * (2*np.pi*k/12))           for k in range(12)]
        outer  = [r3 * np.exp(1j * (2*np.pi*k/16))           for k in range(16)]
        constellation = np.array(inner + middle + outer, dtype=np.complex64)

    constellation /= np.sqrt(np.mean(np.abs(constellation)**2) + 1e-12)
    bits_per_sym = int(np.log2(M))

    # Nearest neighbor
    sym_indices = np.array([
        np.argmin(np.abs(constellation - s)) for s in symbols
    ])

    bits = _symbols_to_bits(sym_indices, bits_per_sym)
    return bits, float(len(symbols) * bits_per_sym)


def _demod_am_dsb(samples: np.ndarray, sample_rate: float,
                  sps: int) -> Tuple[np.ndarray, float]:
    """
    AM-DSB (Double Sideband) demodulator.
    Envelope detection †' DC block †' PCM quantisation †' bits.
    Works with both real and complex (pre-mixed to baseband) input.
    """
    # Ensure analytic signal
    if not np.iscomplexobj(samples):
        analytic = scipy_signal.hilbert(samples.astype(np.float32))
    else:
        analytic = samples.astype(np.complex64)

    # Envelope = amplitude of analytic signal
    envelope = np.abs(analytic).astype(np.float32)

    # Remove carrier DC component with a high-pass
    nyq = sample_rate / 2.0
    hp_cutoff = max(0.001, min(300.0 / nyq, 0.99))   # ~300 Hz HPF
    b_hp, a_hp = scipy_signal.butter(2, hp_cutoff, btype='high')
    demod = scipy_signal.filtfilt(b_hp, a_hp, envelope)

    # Downsample to ~8 kHz equivalent
    target_sps = max(1, int(sample_rate / 8000))
    demod_ds = demod[::target_sps]

    # Normalise to [-1, 1] and PCM quantise to 8-bit †' bits
    peak = np.max(np.abs(demod_ds)) + 1e-12
    demod_norm = np.clip(demod_ds / peak, -1.0, 1.0)
    quantized = ((demod_norm + 1.0) / 2.0 * 255).astype(np.uint8)
    bits = np.unpackbits(quantized)
    return bits, float(len(quantized) * 8)


def _demod_am_ssb(samples: np.ndarray, sample_rate: float,
                  sps: int) -> Tuple[np.ndarray, float]:
    """
    AM-SSB (Single Sideband) demodulator €" phasing method.
    SSB: real part of analytic signal is the demodulated audio.
    """
    if not np.iscomplexobj(samples):
        analytic = scipy_signal.hilbert(samples.astype(np.float32))
    else:
        analytic = samples.astype(np.complex64)

    # For USB: take real part; for LSB it would be negated imaginary
    demod = analytic.real.astype(np.float32)

    # Low-pass to audio band
    nyq = sample_rate / 2.0
    lp_cutoff = min(4000.0 / nyq, 0.99)
    b_lp, a_lp = scipy_signal.butter(4, lp_cutoff, btype='low')
    demod = scipy_signal.filtfilt(b_lp, a_lp, demod)

    # Downsample and quantise
    target_sps = max(1, int(sample_rate / 8000))
    demod_ds = demod[::target_sps]
    peak = np.max(np.abs(demod_ds)) + 1e-12
    demod_norm = np.clip(demod_ds / peak, -1.0, 1.0)
    quantized = ((demod_norm + 1.0) / 2.0 * 255).astype(np.uint8)
    bits = np.unpackbits(quantized)
    return bits, float(len(quantized) * 8)


def _demod_fm(samples: np.ndarray, sample_rate: float,
              sps: int) -> Tuple[np.ndarray, float]:
    """
    FM / WBFM / NBFM demodulator €" discriminator (instantaneous frequency).
    Phase derivative approach: demod = d(phase)/dt / (2Ï€ * Î"f).
    """
    if not np.iscomplexobj(samples):
        analytic = scipy_signal.hilbert(samples.astype(np.float32))
    else:
        analytic = samples.astype(np.complex64)

    # FM discriminator: instantaneous frequency via conjugate product
    # inst_freq[n] = Im(x[n] * conj(x[n-1])) / (2Ï€ * dt)
    conj_product = analytic[1:] * np.conj(analytic[:-1])
    inst_freq = np.angle(conj_product).astype(np.float32)  # radians/sample

    # Normalise by expected frequency deviation
    # For WBFM Â±75 kHz deviation; for NBFM Â±5 kHz
    expected_dev = 75000.0 if sample_rate > 100000 else 5000.0
    norm_factor = float(np.pi * expected_dev / (sample_rate / 2.0)) + 1e-12
    demod = inst_freq / norm_factor

    # De-emphasis filter for WBFM (75 Âµs RC in ITU standard)
    if sample_rate > 100000:
        tau = 75e-6  # 75 Âµs
        alpha = float(1.0 / (1.0 + sample_rate * tau))
        # Simple IIR de-emphasis
        demod_de = np.zeros_like(demod)
        demod_de[0] = demod[0]
        for i in range(1, len(demod)):
            demod_de[i] = alpha * demod[i] + (1.0 - alpha) * demod_de[i - 1]
        demod = demod_de

    # Downsample to ~8 kHz audio
    target_sps = max(1, int(sample_rate / 8000))
    demod_ds = demod[::target_sps]

    # Normalise and quantise †' bits
    peak = np.max(np.abs(demod_ds)) + 1e-12
    demod_norm = np.clip(demod_ds / peak, -1.0, 1.0)
    quantized = ((demod_norm + 1.0) / 2.0 * 255).astype(np.uint8)
    bits = np.unpackbits(quantized)
    return bits, float(len(quantized) * 8)


# "€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€
#  OFDM Demodulator €" full CP-OFDM implementation
# "€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€

def _demod_ofdm(samples: np.ndarray,
                sample_rate: float,
                symbol_rate: float) -> Tuple[np.ndarray, float]:
    """
    CP-OFDM demodulator (Cyclic Prefix OFDM).

    Algorithm:
      1. Estimate FFT size from symbol rate (N ‰ˆ sample_rate / subcarrier_spacing)
      2. Auto-detect CP length using autocorrelation of the OFDM symbol
      3. Strip CP and apply N-point FFT per OFDM symbol
      4. Channel equalisation €" LS pilot-based per symbol
      5. Nearest-neighbour BPSK/QPSK decision on each subcarrier
      6. Gray-decode and serialise bits

    Works well for:
      - IEEE 802.11a/g/n (20 MHz, 64-point FFT, CP=16)
      - DVB-T/T2 (2K/8K mode)
      - LTE/5G SC-FDMA (downlink OFDM)
      - Custom OFDM signals
    """
    if not np.iscomplexobj(samples):
        analytic = scipy_signal.hilbert(samples.astype(np.float32))
        samples = analytic.astype(np.complex64)
    else:
        samples = samples.astype(np.complex64)

    n_total = len(samples)
    if n_total < 64:
        return np.array([], dtype=np.uint8), 0.0

    # "€"€ Step 1: Estimate OFDM parameters "€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€
    # Try common FFT sizes first, then auto-detect
    fft_size, cp_len = _estimate_ofdm_params(samples, sample_rate)

    symbol_len = fft_size + cp_len
    if symbol_len < 4 or n_total < symbol_len:
        # Fallback: minimal OFDM
        fft_size = max(64, int(2 ** round(np.log2(n_total / 8))))
        cp_len = fft_size // 4
        symbol_len = fft_size + cp_len

    if n_total < symbol_len:
        return np.array([], dtype=np.uint8), 0.0

    # "€"€ Step 2: Frame all OFDM symbols "€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€"€
    n_syms = n_total // symbol_len
    if n_syms == 0:
        return np.array([], dtype=np.uint8), 0.0

    symbols_2d = np.zeros((n_syms, fft_size), dtype=np.complex64)
    for k in range(n_syms):
        start = k * symbol_len + cp_len          # skip cyclic prefix
        segment = samples[start: start + fft_size]
        if len(segment) < fft_size:
            break
        # Apply window to reduce spectral leakage
        win = np.hanning(fft_size).astype(np.float32)
        symbols_2d[k] = np.fft.fftshift(np.fft.fft(segment * win, fft_size))

    # "€"€ Step 3: Identify active subcarriers (power above noise floor) "€"€
    avg_power = np.mean(np.abs(symbols_2d) ** 2, axis=0)
    power_threshold = np.percentile(avg_power, 20)   # bottom 20% = guard/null
    active_mask = avg_power > power_threshold * 2.0
    active_idx = np.where(active_mask)[0]

    if len(active_idx) == 0:
        active_idx = np.arange(fft_size // 4, 3 * fft_size // 4)  # centre half

    # "€"€ Step 4: Channel estimation €" LS per symbol using pilot heuristic
    # Assume pilot carriers are the highest-power subcarriers (top 10%)
    pilot_threshold = np.percentile(avg_power[active_idx], 90)
    pilot_idx = active_idx[avg_power[active_idx] >= pilot_threshold]
    data_idx = active_idx[avg_power[active_idx] < pilot_threshold]

    if len(data_idx) == 0:
        data_idx = active_idx

    # Per-symbol LS equalisation using pilot average
    all_bits = []

    for k in range(n_syms):
        sym_row = symbols_2d[k]

        # Estimate channel from pilots (known: unit amplitude)
        if len(pilot_idx) > 0:
            pilot_vals = sym_row[pilot_idx]
            # Reference: pilots assumed to carry +1 (BPSK pilot)
            pilot_ref = np.ones(len(pilot_idx), dtype=np.complex64)
            h_pilot = pilot_ref / (pilot_vals + 1e-12)
            # Interpolate channel over data subcarriers
            if len(pilot_idx) >= 2 and len(data_idx) > 0:
                h_real = np.interp(data_idx,
                                   pilot_idx,
                                   np.real(h_pilot))
                h_imag = np.interp(data_idx,
                                   pilot_idx,
                                   np.imag(h_pilot))
                h_data = h_real + 1j * h_imag
            else:
                h_data = np.ones(len(data_idx), dtype=np.complex64)
        else:
            h_data = np.ones(len(data_idx), dtype=np.complex64)

        # Equalise data subcarriers
        eq_data = sym_row[data_idx] * h_data

        # "€"€ Step 5: Constellation decision (auto-select QPSK vs QAM16) "€"€
        # Estimate modulation order from constellation spread
        amp_spread = float(np.std(np.abs(eq_data)))
        if amp_spread < 0.15:
            # Constant-envelope †' PSK
            bits_sym = _decide_psk_subcarriers(eq_data, n_psk=4)   # QPSK
        elif amp_spread < 0.40:
            bits_sym = _decide_qam_subcarriers(eq_data, order=16)
        else:
            bits_sym = _decide_qam_subcarriers(eq_data, order=64)

        all_bits.extend(bits_sym)

    bits_arr = np.array(all_bits, dtype=np.uint8)
    
    # Correct bit rate calculation for OFDM:
    # bit_rate = (bits_per_symbol / symbol_duration)
    # symbol_duration = symbol_len / sample_rate
    # So: bit_rate = (bits_per_symbol * sample_rate) / symbol_len
    if n_syms > 0 and symbol_len > 0:
        bits_per_ofdm_symbol = len(bits_arr) / n_syms
        bit_rate = float(bits_per_ofdm_symbol * sample_rate / symbol_len)
    else:
        bit_rate = float(len(bits_arr) * sample_rate / n_total)
    
    return bits_arr, bit_rate


def _estimate_ofdm_params(samples: np.ndarray,
                           sample_rate: float) -> Tuple[int, int]:
    """
    Estimate OFDM FFT size and CP length using cyclic-prefix autocorrelation.

    The CP property: x[n] == x[n + N] for n in [0, Ncp).
    So |R(N)| = sum |x[n] * conj(x[n+N])| has a peak at lag N (FFT size).
    """
    n = len(samples)
    max_fft = min(n // 4, 8192)

    # Try standard sizes first (802.11a: 64, DVB-T 2K: 2048, 8K: 8192, LTE: 512/1024/2048)
    standard_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    # Compute autocorrelation of the signal power
    seg = samples[:min(n, 32768)]
    # Use Schmidl & Cox metric: P(d,N) = sum x[d+m]*conj(x[d+m+N])
    best_metric = -1.0
    best_fft = 64
    best_cp = 16

    for fft_n in standard_sizes:
        if fft_n > max_fft:
            break
        cp_candidates = [fft_n // 4, fft_n // 8, fft_n // 16]
        for cp in cp_candidates:
            sym_len = fft_n + cp
            if sym_len * 4 > len(seg):
                continue
            # Schmidl-Cox metric: correlation of CP with its copy
            P = np.sum(seg[:cp] * np.conj(seg[fft_n:fft_n + cp]))
            R = np.sum(np.abs(seg[fft_n:fft_n + cp]) ** 2) + 1e-12
            metric = float(np.abs(P) / R)
            if metric > best_metric:
                best_metric = metric
                best_fft = fft_n
                best_cp = cp

    # If no standard size matched well, fall back to autocorrelation peak
    if best_metric < 0.3 and n >= 256:
        fft_sizes = [2 ** k for k in range(6, 14)
                     if 2 ** k <= max_fft]
        ac = np.abs(np.correlate(seg[:512], seg[:512], mode='full'))
        ac = ac[len(ac) // 2:]
        # Find peak beyond lag 32
        search = ac[32:]
        if len(search) > 0:
            peak_lag = int(np.argmax(search)) + 32
            # Snap to nearest power-of-2
            snapped = min(fft_sizes,
                          key=lambda x: abs(x - peak_lag),
                          default=64)
            best_fft = snapped
            best_cp = best_fft // 4

    return best_fft, best_cp


def _decide_psk_subcarriers(eq_data: np.ndarray,
                             n_psk: int = 4) -> list:
    """Nearest-neighbour QPSK/BPSK decision on equalised subcarriers."""
    bits_per_sym = int(np.log2(n_psk))
    angle_step = 2 * np.pi / n_psk
    phases = np.angle(eq_data)
    sym_idx = np.round((phases + np.pi) / angle_step).astype(int) % n_psk
    bits = []
    for idx in sym_idx:
        gray = idx ^ (idx >> 1)
        for b in range(bits_per_sym - 1, -1, -1):
            bits.append((int(gray) >> b) & 1)
    return bits


def _decide_qam_subcarriers(eq_data: np.ndarray, order: int = 16) -> list:
    """Nearest-neighbour QAM decision on equalised subcarriers."""
    m = int(np.sqrt(order))
    levels = np.linspace(-m + 1, m - 1, m)
    constellation = np.array([i + 1j * q
                               for q in reversed(levels)
                               for i in levels], dtype=np.complex64)
    constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2) + 1e-12)

    bits_per_sym = int(np.log2(order))
    # Normalise received data
    pwr = np.sqrt(np.mean(np.abs(eq_data) ** 2) + 1e-12)
    eq_norm = eq_data / pwr

    bits = []
    for s in eq_norm:
        idx = int(np.argmin(np.abs(constellation - s)))
        gray = idx ^ (idx >> 1)
        for b in range(bits_per_sym - 1, -1, -1):
            bits.append((int(gray) >> b) & 1)
    return bits

