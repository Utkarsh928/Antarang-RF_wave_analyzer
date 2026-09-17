"""
Symbol Rate Estimator — estimates symbol rate from IQ samples.
Uses power spectral density of the squared signal to find spectral lines
at ±symbol_rate/2, which appear due to AM modulation artifacts.
Also uses autocorrelation-based method as fallback.
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import find_peaks


def estimate_symbol_rate(samples: np.ndarray,
                          sample_rate: float,
                          min_rate: float = None,
                          max_rate: float = None) -> float:
    """
    Estimate symbol rate from complex IQ samples.

    Strategy:
    1. Square the signal → spectral lines at ±Rs appear
    2. Find peak in PSD of squared signal
    3. Fallback: autocorrelation zero-crossing method
    4. Fallback: bandwidth / 1.2 heuristic

    Returns estimated symbol rate in Hz.
    """
    if min_rate is None:
        min_rate = sample_rate / 1000.0
    if max_rate is None:
        max_rate = sample_rate / 2.0

    n = len(samples)
    if n < 64:
        return sample_rate / 8.0

    # First: determine signal type (constant vs varying envelope)
    if np.iscomplexobj(samples):
        env_var = float(np.var(np.abs(samples)))
    else:
        env_var = float(np.var(np.abs(samples.astype(np.float32))))
    is_constant_envelope = env_var < 0.01   # PSK/FSK have constant amplitude

    # --- Method 1: Squared/4th-power spectral lines ---
    rate_sq = None
    try:
        rate_sq = _squared_spectrum_method(samples, sample_rate,
                                            min_rate, max_rate)
    except Exception:
        pass

    # --- Method 2: Autocorrelation (phase-based for PSK/FSK) ---
    rate_ac = None
    try:
        rate_ac = _autocorr_method(samples, sample_rate, min_rate, max_rate)
    except Exception:
        pass

    # --- Method 3: Bandwidth-based estimation ---
    rate_bw = None
    try:
        rate_bw = _bandwidth_method(samples, sample_rate, min_rate, max_rate)
    except Exception:
        pass

    # --- Method 4: Differential autocorrelation (QAM/AM rectangular pulses) ---
    rate_diff = None
    if not is_constant_envelope:
        try:
            rate_diff = _diff_autocorr_method(samples, sample_rate, min_rate, max_rate)
        except Exception:
            pass

    if is_constant_envelope:
        # PSK/FSK: trust autocorr first
        if rate_ac is not None:
            if rate_bw is not None:
                rel_diff = abs(rate_ac - rate_bw) / max(rate_ac, rate_bw)
                if rel_diff < 0.15:
                    return (rate_ac + rate_bw) / 2.0
            return rate_ac
        if rate_bw is not None:
            return rate_bw
    else:
        # QAM/AM: try diff autocorr first (exact for rectangular pulses)
        if rate_diff is not None:
            return rate_diff
        # Fall back to bandwidth method
        if rate_bw is not None:
            if rate_ac is not None:
                rel_diff = abs(rate_ac - rate_bw) / max(rate_ac, rate_bw)
                if rel_diff < 0.12:
                    return (rate_ac + rate_bw) / 2.0
            return rate_bw
        if rate_ac is not None:
            return rate_ac

    # Last resort: squared spectrum
    if rate_sq is not None:
        return rate_sq

    return max(min_rate, sample_rate / 8.0)


def _squared_spectrum_method(samples: np.ndarray,
                               sample_rate: float,
                               min_rate: float,
                               max_rate: float):
    """
    Square the envelope → PSD lines appear at symbol rate.
    Works well for QAM and AM signals.
    For constant-amplitude signals (PSK, FSK), falls back to 4th-power method.
    """
    if np.iscomplexobj(samples):
        env = np.abs(samples) ** 2
    else:
        env = samples.astype(np.float32) ** 2

    # Remove DC
    env -= np.mean(env)

    # Check if signal has amplitude variation (QAM/AM) or is constant-envelope (PSK/FSK)
    env_var = float(np.var(env))
    if env_var < 1e-6:
        # Constant-amplitude signal → try 4th-power method (works for PSK)
        return _fourth_power_method(samples, sample_rate, min_rate, max_rate)

    n_fft = min(len(env), 32768)
    n_fft = int(2 ** np.floor(np.log2(n_fft)))
    n_fft = max(n_fft, 512)

    # Pad if signal is shorter than n_fft
    env_seg = env[:n_fft]
    if len(env_seg) < n_fft:
        env_seg = np.pad(env_seg, (0, n_fft - len(env_seg)))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    psd = np.abs(np.fft.rfft(env_seg * np.hanning(n_fft), n_fft)) ** 2

    # Search band
    mask = (freqs >= min_rate) & (freqs <= max_rate)
    if not mask.any():
        return None

    psd_band = psd.copy()
    psd_band[~mask] = 0

    # Find peaks with minimum prominence
    peaks, props = find_peaks(psd_band,
                               prominence=np.max(psd_band[mask]) * 0.1,
                               distance=max(1, int(min_rate /
                                                    (freqs[1] - freqs[0]))))
    if len(peaks) == 0:
        return None

    # Best peak
    best = peaks[np.argmax(psd_band[peaks])]
    symbol_rate = float(freqs[best])

    if min_rate <= symbol_rate <= max_rate:
        return symbol_rate
    return None


def _diff_autocorr_method(samples: np.ndarray,
                           sample_rate: float,
                           min_rate: float,
                           max_rate: float):
    """
    Differential autocorrelation method for rectangular-pulse signals.

    For signals generated by np.repeat(symbols, sps):
    - Within each symbol, consecutive samples are identical → |diff| ≈ 0
    - At symbol boundaries, samples change → |diff| is large
    - The autocorrelation of |diff| has a strong peak at lag = sps

    This is extremely accurate for non-filtered QAM/AM signals.
    Also works for filtered signals where amplitude transitions are visible.
    """
    if not np.iscomplexobj(samples):
        samples = samples.astype(np.complex64)

    # Magnitude of sample-to-sample differences
    diffs = np.abs(np.diff(samples))
    if len(diffs) < 4:
        return None

    d = diffs - np.mean(diffs)
    n_ac = min(len(d), 4096)
    d = d[:n_ac]

    if np.max(np.abs(d)) < 1e-10:
        return None

    # Normalized autocorrelation
    corr = np.correlate(d, d, mode='full')
    corr = corr[len(corr) // 2:]
    if corr[0] > 0:
        corr = corr / corr[0]

    # Search for the fundamental period
    min_lag = max(2, int(sample_rate / max_rate))
    max_lag = min(len(corr) - 1, int(sample_rate / min_rate))

    if min_lag >= max_lag:
        return None

    # Start one before min_lag to catch edge peaks
    search_start = max(1, min_lag - 1)
    search = corr[search_start:max_lag]

    peaks, _ = find_peaks(search, height=0.05)

    if len(peaks) == 0:
        peaks, _ = find_peaks(search, height=0.01)

    if len(peaks) == 0:
        return None

    # Pick the first significant peak
    peak_vals = search[peaks]
    threshold = np.percentile(peak_vals, 50)
    qualifying = [peaks[i] for i, v in enumerate(peak_vals) if v >= threshold]
    if not qualifying:
        qualifying = [peaks[0]]

    first_lag = qualifying[0] + search_start
    symbol_rate = sample_rate / first_lag

    if min_rate <= symbol_rate <= max_rate:
        return symbol_rate
    return None


def _bandwidth_method(samples: np.ndarray,
                       sample_rate: float,
                       min_rate: float,
                       max_rate: float):
    """
    Estimate symbol rate from signal bandwidth.

    For RRC-filtered signals: bandwidth ≈ symbol_rate × (1 + rolloff)
    Typical rolloff = 0.2–0.5, so symbol_rate ≈ bandwidth / 1.35 (midpoint)

    Also handles the case where the -3dB bandwidth is measured.
    """
    if np.iscomplexobj(samples):
        sig = samples
    else:
        from scipy.signal import hilbert
        sig = hilbert(samples.astype(np.float32)).astype(np.complex64)

    n_fft = len(sig)
    if n_fft > 32768:
        n_fft = 32768
    # Use actual signal length for best frequency resolution on short signals
    # (no power-of-2 rounding for bandwidth measurement)

    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sample_rate))
    sig_seg = sig[:n_fft].astype(np.complex64)
    psd = np.abs(np.fft.fftshift(np.fft.fft(sig_seg *
                                              np.hanning(n_fft), n_fft))) ** 2

    # Find the bandwidth at 10dB below peak
    peak_power = np.max(psd)
    threshold_power = peak_power * 0.1   # 10 dB down

    above = psd > threshold_power
    if not above.any():
        return None

    indices = np.where(above)[0]
    f_low  = float(freqs[indices[0]])
    f_high = float(freqs[indices[-1]])
    bandwidth = abs(f_high - f_low)

    if bandwidth < 10:
        return None

    # Calibrated for rectangular-pulse QAM (np.repeat, no RRC filter):
    # bandwidth at -10dB ≈ 1.086 × symbol_rate
    symbol_rate = bandwidth / 1.086

    if min_rate <= symbol_rate <= max_rate:
        return symbol_rate
    return max(min_rate, min(max_rate, symbol_rate))


def _fourth_power_method(samples: np.ndarray,
                          sample_rate: float,
                          min_rate: float,
                          max_rate: float):
    """
    4th-power spectral method for constant-amplitude signals (PSK, FSK).

    For BPSK: x^2 removes modulation → spectral line at 2*carrier ± Rs
    For QPSK: x^4 removes modulation → spectral line at Rs
    The phase-transition rate equals the symbol rate.

    We use the instantaneous phase derivative (instantaneous frequency)
    which shows spectral energy at the symbol rate due to phase jumps.
    """
    if not np.iscomplexobj(samples):
        from scipy.signal import hilbert
        samples = hilbert(samples.astype(np.float32)).astype(np.complex64)

    n = len(samples)

    # Method A: instantaneous frequency spectrum
    phase = np.unwrap(np.angle(samples))
    inst_freq = np.abs(np.diff(phase)) * sample_rate / (2 * np.pi)
    inst_freq -= np.mean(inst_freq)

    n_fft = min(len(inst_freq), 32768)
    n_fft = int(2 ** np.floor(np.log2(n_fft)))
    n_fft = max(n_fft, 256)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    psd = np.abs(np.fft.rfft(inst_freq[:n_fft] *
                               np.hanning(n_fft), n_fft)) ** 2

    mask = (freqs >= min_rate) & (freqs <= max_rate)
    if not mask.any():
        return None

    psd_band = psd.copy()
    psd_band[~mask] = 0
    max_power = float(np.max(psd_band[mask]))
    if max_power < 1e-30:
        return None

    peaks, _ = find_peaks(psd_band,
                           prominence=max_power * 0.08,
                           distance=max(1, int(min_rate / (freqs[1] - freqs[0] + 1e-9))))
    if len(peaks) > 0:
        best = peaks[np.argmax(psd_band[peaks])]
        rate = float(freqs[best])
        if min_rate <= rate <= max_rate:
            return rate

    # Method B: power of phase-transition signal
    # Look for periodicity in absolute phase differences
    abs_phase_diff = np.abs(np.diff(phase))
    # Clip to [0, pi] — PSK transitions appear as large spikes
    transition_signal = abs_phase_diff - np.mean(abs_phase_diff)

    n_fft2 = min(len(transition_signal), 32768)
    n_fft2 = int(2 ** np.floor(np.log2(n_fft2)))
    n_fft2 = max(n_fft2, 256)

    freqs2 = np.fft.rfftfreq(n_fft2, d=1.0 / sample_rate)
    psd2 = np.abs(np.fft.rfft(transition_signal[:n_fft2] *
                                np.hanning(n_fft2), n_fft2)) ** 2

    mask2 = (freqs2 >= min_rate) & (freqs2 <= max_rate)
    if not mask2.any():
        return None

    psd2_band = psd2.copy()
    psd2_band[~mask2] = 0
    max_p2 = float(np.max(psd2_band[mask2]))
    if max_p2 < 1e-30:
        return None

    peaks2, _ = find_peaks(psd2_band,
                            prominence=max_p2 * 0.08,
                            distance=max(1, int(min_rate / (freqs2[1] - freqs2[0] + 1e-9))))
    if len(peaks2) > 0:
        best2 = peaks2[np.argmax(psd2_band[peaks2])]
        rate2 = float(freqs2[best2])
        if min_rate <= rate2 <= max_rate:
            return rate2

    return None


def _autocorr_method(samples: np.ndarray,
                      sample_rate: float,
                      min_rate: float,
                      max_rate: float):
    """
    Autocorrelation-based symbol period estimation.

    For PSK/FSK: uses the instantaneous phase (not amplitude)
    For QAM/AM: uses the amplitude envelope
    Symbol period ≈ lag of first significant autocorrelation peak.
    """
    if np.iscomplexobj(samples):
        # Try phase-based for constant-envelope signals first
        env = np.abs(samples)
        env_var = float(np.var(env))

        if env_var < 1e-4:
            # Constant envelope → use phase derivative
            phase = np.unwrap(np.angle(samples))
            sig = np.abs(np.diff(phase))   # magnitude of phase changes
        else:
            # QAM/AM: use SQUARED envelope (second-order) for cleaner peaks
            # First-order envelope is distorted by RRC filtering
            sig = np.abs(samples) ** 2
    else:
        sig = np.abs(samples.astype(np.float32)) ** 2

    sig = sig - np.mean(sig)
    n = min(len(sig), 8192)
    sig = sig[:n]

    if np.max(np.abs(sig)) < 1e-10:
        return None

    sig = sig - np.mean(sig)

    # Normalized autocorrelation
    corr = np.correlate(sig, sig, mode='full')
    corr = corr[len(corr) // 2:]
    if corr[0] > 0:
        corr = corr / corr[0]

    # Find first significant peak after lag 0
    min_lag = max(1, int(sample_rate / max_rate))
    max_lag = min(len(corr) - 1, int(sample_rate / min_rate))

    if min_lag >= max_lag:
        return None

    # Start search one before min_lag so the true minimum-period
    # peak (exactly at min_lag) can be detected as a local maximum
    search_start = max(1, min_lag - 1)
    search = corr[search_start:max_lag]

    # Strategy: find ALL peaks, then pick the smallest lag that has
    # a correlation value in the top 50% of all peak values.
    peaks, _ = find_peaks(search, height=0.03)

    if len(peaks) == 0:
        peaks, _ = find_peaks(search, height=0.01)

    if len(peaks) == 0:
        return None

    peak_vals = search[peaks]
    threshold = np.percentile(peak_vals, 50)

    qualifying = [peaks[i] for i, v in enumerate(peak_vals) if v >= threshold]
    if not qualifying:
        qualifying = list(peaks[:3])

    first_peak_lag = qualifying[0] + search_start
    symbol_rate = sample_rate / first_peak_lag

    if min_rate <= symbol_rate <= max_rate:
        return symbol_rate
    return None


def estimate_samples_per_symbol(samples: np.ndarray,
                                  sample_rate: float) -> int:
    """
    Returns integer samples-per-symbol estimate.
    Minimum 2 (Nyquist), maximum sample_rate/100.
    """
    sr = estimate_symbol_rate(samples, sample_rate)
    sps = max(2, int(round(sample_rate / sr)))
    return sps


def estimate_fec_type(bits: np.ndarray) -> str:
    """
    Robust FEC type detection from a decoded bit stream.
    Uses multiple independent evidence signals rather than simple heuristics.

    Strategy:
      1. Block-length checks: RS and LDPC have strict block sizes
      2. Polynomial syndrome check: convolutional codes leave signature
      3. Bit-transition density: Viterbi-decoded streams have ~40-60% transitions
      4. Parity density: LDPC/RS have characteristic parity overhead ratios

    Returns: 'viterbi' | 'reed-solomon' | 'ldpc' | 'concatenated'
    """
    if bits is None or len(bits) < 64:
        return 'viterbi'

    n = len(bits)
    bits_u8 = bits.astype(np.uint8)

    # ── 1. LDPC: check for standard 802.11n/DVB-S2/LTE block sizes ──────
    # IEEE 802.11n: 648, 1296, 1944
    # DVB-S2: 64800, 16200
    # LTE turbo block sizes: 40-6144
    ldpc_sizes = [648, 1296, 1944, 4320, 8640, 16200, 64800]
    for size in ldpc_sizes:
        if n % size == 0:
            # Verify with parity check: for rate-1/2 LDPC ~50% are parity bits
            k = n // 2
            data_half = bits_u8[:k]
            parity_half = bits_u8[k:]
            # Hamming weight of parity vs data should be similar for LDPC
            data_weight = float(np.mean(data_half))
            parity_weight = float(np.mean(parity_half))
            if abs(data_weight - 0.5) < 0.15 and abs(parity_weight - 0.5) < 0.15:
                return 'ldpc'

    # ── 2. Reed-Solomon: check byte-aligned block structure ──────────────
    # Try direct RS(255, 223) syndrome verification
    try:
        import reedsolo
        rsc = reedsolo.RSCodec(32)
        if len(bits_u8) >= 255 * 8:
            byte_arr = np.packbits(bits_u8[:255 * 8])
            try:
                decoded, _, errata = rsc.decode(bytes(byte_arr))
                if len(errata) <= 16:
                    return 'reed-solomon'
            except reedsolo.ReedSolomonError:
                pass
    except Exception:
        pass

    # RS(255,223): codeword = 255 bytes, data = 223 bytes, parity = 32 bytes
    # RS(255,239): parity = 16 bytes
    # Note: RS encoded data has byte-level structure (8-bit symbols)
    rs_params = [(255, 223, 32), (255, 239, 16), (204, 188, 16)]
    
    # First check: is length compatible with RS block size?
    for (n_rs, k_rs, t_rs) in rs_params:
        block_bits = n_rs * 8
        if n >= block_bits and n % block_bits == 0:
            n_blocks = n // block_bits
            
            # RS has byte-level structure - check byte alignment patterns
            # Convert bits to bytes
            if len(bits_u8) % 8 == 0:
                bytes_arr = np.packbits(bits_u8)
                
                # RS characteristic: byte values are distributed
                # (not clustered like Viterbi bit streams)
                byte_hist = np.bincount(bytes_arr, minlength=256).astype(float)
                byte_hist_norm = byte_hist / (len(bytes_arr) + 1e-12)
                
                # Measure uniformity of byte distribution
                # RS should have more uniform byte distribution than Viterbi
                entropy = -np.sum(byte_hist_norm[byte_hist_norm > 0] * 
                                 np.log2(byte_hist_norm[byte_hist_norm > 0] + 1e-12))
                
                # High entropy (>4.5) indicates uniform byte distribution → likely RS
                # Also check that we have many different byte values
                unique_bytes = np.sum(byte_hist > 0)
                
                if entropy > 4.5 and unique_bytes > 80:
                    return 'reed-solomon'
                
                # Additional check: RS has periodic structure at block boundaries
                if n_blocks >= 2:
                    # Check variance between blocks
                    block_weights = []
                    for b in range(min(n_blocks, 4)):
                        block = bits_u8[b * block_bits: (b + 1) * block_bits]
                        block_weights.append(float(np.mean(block)))
                    
                    # RS blocks are independent → higher variance than Viterbi
                    block_variance = float(np.var(block_weights))
                    if block_variance > 0.002 and entropy > 4.5:
                        return 'reed-solomon'

    # ── 3. Convolutional + Viterbi: characteristic transition density ────
    # Hard-decision Viterbi output has ~40-60% bit transitions
    # AND the run-length distribution follows a geometric distribution
    transitions = int(np.sum(np.diff(bits_u8) != 0))
    transition_rate = transitions / max(n - 1, 1)

    # Run-length analysis
    runs = _compute_run_lengths(bits_u8)
    if len(runs) > 0:
        mean_run = float(np.mean(runs))
        # Viterbi output from K=7: run lengths ~2-4 bits (matches 1/2 rate code)
        viterbi_like = (0.35 <= transition_rate <= 0.65) and (1.5 <= mean_run <= 5.0)
    else:
        viterbi_like = 0.35 <= transition_rate <= 0.65

    # ── 4. Concatenated code detection ──────────────────────────────────
    # Concatenated = Viterbi inner + RS outer
    # Look for: transition density consistent with Viterbi AND block-level structure
    if viterbi_like and n >= 255 * 8:
        # Check if there's a periodic block structure at RS block boundaries
        block_bits = 255 * 8
        if n >= 2 * block_bits:
            n_blocks_check = min(n // block_bits, 8)
            block_means = []
            for b in range(n_blocks_check):
                blk = bits_u8[b * block_bits: (b + 1) * block_bits]
                block_means.append(float(np.mean(blk)))
            variance_of_means = float(np.var(block_means))
            # Concatenated codes: each RS block is decoded independently → low variance
            if variance_of_means < 0.005:
                return 'concatenated'

    if viterbi_like:
        return 'viterbi'

    # ── 5. Default: unknown / minimal FEC — use Viterbi as safe default ─
    return 'viterbi'


def _compute_run_lengths(bits: np.ndarray) -> np.ndarray:
    """Return array of run lengths in a binary bit stream."""
    if len(bits) == 0:
        return np.array([], dtype=np.int32)
    runs = []
    current = 1
    for i in range(1, len(bits)):
        if bits[i] == bits[i - 1]:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    return np.array(runs, dtype=np.int32)
