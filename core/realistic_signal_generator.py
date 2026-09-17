"""
Realistic RF Signal Generator with Pulse Shaping

Generates realistic digital communication signals with:
- Root-Raised Cosine (RRC) pulse shaping
- Additive White Gaussian Noise (AWGN)
- Carrier frequency offsets
- Timing jitter
- Multipath fading (optional)

Used for realistic testing of parameter estimation and auto-detection algorithms.
"""
import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, Optional


def generate_realistic_signal(
    modulation: str,
    n_symbols: int,
    sps: int = 8,
    snr_db: float = 20.0,
    carrier_freq_hz: float = 0.0,
    sample_rate_hz: float = 100000.0,
    rrc_rolloff: float = 0.35,
    rrc_span: int = 10,
    timing_jitter_std: float = 0.0,
    freq_offset_hz: float = 0.0,
    add_multipath: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Generate a realistic RF signal with pulse shaping and impairments.
    
    Args:
        modulation: 'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', etc.
        n_symbols: Number of symbols to generate
        sps: Samples per symbol
        snr_db: Signal-to-noise ratio in dB
        carrier_freq_hz: Carrier frequency offset
        sample_rate_hz: Sample rate
        rrc_rolloff: RRC rolloff factor (0.2-0.5 typical)
        rrc_span: RRC filter span in symbols
        timing_jitter_std: Standard deviation of timing jitter (fraction of symbol period)
        freq_offset_hz: Frequency offset to add
        add_multipath: Add multipath channel model
    
    Returns:
        signal: Complex IQ samples
        metadata: Dictionary with ground truth parameters
    """
    # Generate random symbols
    symbols = _generate_symbols(modulation, n_symbols)
    
    # Upsample to sps
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    
    # Apply RRC pulse shaping
    rrc_taps = _rrc_filter(sps, rrc_rolloff, rrc_span)
    shaped = scipy_signal.lfilter(rrc_taps, 1.0, upsampled)
    
    # Add timing jitter if specified
    if timing_jitter_std > 0:
        shaped = _add_timing_jitter(shaped, sps, timing_jitter_std)
    
    # Add carrier frequency offset
    if carrier_freq_hz != 0 or freq_offset_hz != 0:
        total_freq = carrier_freq_hz + freq_offset_hz
        t = np.arange(len(shaped)) / sample_rate_hz
        shaped = shaped * np.exp(2j * np.pi * total_freq * t)
    
    # Add multipath fading if specified
    if add_multipath:
        shaped = _add_multipath_channel(shaped, sps)
    
    # Add AWGN noise
    signal_noisy = _add_awgn(shaped, snr_db)
    
    # Calculate actual symbol rate
    symbol_rate_hz = sample_rate_hz / sps
    
    # Metadata
    metadata = {
        'modulation': modulation,
        'n_symbols': n_symbols,
        'sps': sps,
        'snr_db': snr_db,
        'sample_rate_hz': sample_rate_hz,
        'symbol_rate_hz': symbol_rate_hz,
        'carrier_freq_hz': carrier_freq_hz + freq_offset_hz,
        'rrc_rolloff': rrc_rolloff,
        'has_pulse_shaping': True,
        'has_multipath': add_multipath,
        'timing_jitter_std': timing_jitter_std
    }
    
    return signal_noisy, metadata


def _generate_symbols(modulation: str, n_symbols: int) -> np.ndarray:
    """Generate random modulation symbols."""
    mod = modulation.upper()
    
    if 'BPSK' in mod:
        # BPSK: ±1
        bits = np.random.randint(0, 2, n_symbols)
        symbols = 2 * bits - 1
        return symbols.astype(complex)
    
    elif 'QPSK' in mod or mod == '4PSK':
        # QPSK: exp(j*π/4 * (2k+1)) for k=0,1,2,3
        bits = np.random.randint(0, 4, n_symbols)
        angles = np.pi/4 + bits * np.pi/2
        return np.exp(1j * angles)
    
    elif '8PSK' in mod:
        # 8PSK: exp(j*2π*k/8) for k=0..7
        bits = np.random.randint(0, 8, n_symbols)
        angles = 2 * np.pi * bits / 8
        return np.exp(1j * angles)
    
    elif '16QAM' in mod:
        # 16-QAM: ±1, ±3 on I and Q
        i_vals = np.random.choice([-3, -1, 1, 3], n_symbols)
        q_vals = np.random.choice([-3, -1, 1, 3], n_symbols)
        symbols = (i_vals + 1j * q_vals) / np.sqrt(10)  # Normalize
        return symbols
    
    elif '64QAM' in mod:
        # 64-QAM: ±1, ±3, ±5, ±7 on I and Q
        vals = [-7, -5, -3, -1, 1, 3, 5, 7]
        i_vals = np.random.choice(vals, n_symbols)
        q_vals = np.random.choice(vals, n_symbols)
        symbols = (i_vals + 1j * q_vals) / np.sqrt(42)  # Normalize
        return symbols
    
    else:
        # Default: QPSK
        bits = np.random.randint(0, 4, n_symbols)
        angles = np.pi/4 + bits * np.pi/2
        return np.exp(1j * angles)


def _rrc_filter(sps: int, rolloff: float, span: int) -> np.ndarray:
    """
    Root-Raised Cosine (RRC) filter design.
    
    Args:
        sps: Samples per symbol
        rolloff: Rolloff factor (0 < rolloff < 1), typical 0.35
        span: Filter span in symbols (typical 6-12)
    
    Returns:
        Filter taps (normalized)
    """
    n_taps = span * sps + 1
    t = np.arange(-(n_taps // 2), (n_taps // 2) + 1) / float(sps)
    
    # RRC formula
    h = np.zeros(len(t))
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = (1.0 + rolloff * (4.0 / np.pi - 1.0))
        elif abs(ti) == 1.0 / (4.0 * rolloff):
            h[i] = (rolloff / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * rolloff)) +
                (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * rolloff))
            )
        else:
            numerator = np.sin(np.pi * ti * (1.0 - rolloff)) + \
                       4.0 * rolloff * ti * np.cos(np.pi * ti * (1.0 + rolloff))
            denominator = np.pi * ti * (1.0 - (4.0 * rolloff * ti) ** 2)
            h[i] = numerator / denominator
    
    # Normalize energy
    h = h / np.sqrt(np.sum(h ** 2))
    return h


def _add_timing_jitter(signal: np.ndarray, sps: int, jitter_std: float) -> np.ndarray:
    """
    Add timing jitter by fractional delay interpolation.
    
    Args:
        signal: Input signal
        sps: Samples per symbol
        jitter_std: Standard deviation of jitter (fraction of symbol period)
    
    Returns:
        Signal with timing jitter
    """
    n = len(signal)
    jitter_samples = np.random.normal(0, jitter_std * sps, size=n // sps + 1)
    
    # Interpolate jitter to all samples
    jitter_interp = np.interp(
        np.arange(n),
        np.arange(0, n, sps),
        jitter_samples
    )
    
    # Apply fractional delay using sinc interpolation (simplified)
    jittered = np.zeros_like(signal)
    for i in range(n):
        delay = jitter_interp[i]
        idx = int(np.floor(i - delay))
        if 0 <= idx < n:
            jittered[i] = signal[idx]
        else:
            jittered[i] = signal[i]
    
    return jittered


def _add_multipath_channel(signal: np.ndarray, sps: int) -> np.ndarray:
    """
    Add simple 2-tap multipath channel.
    
    h(t) = δ(t) + α*δ(t - τ)
    
    Args:
        signal: Input signal
        sps: Samples per symbol
    
    Returns:
        Signal with multipath
    """
    # Simple 2-path model
    delay_samples = int(sps * 0.5)  # Half symbol delay
    alpha = 0.3  # Path gain
    
    multipath = signal.copy()
    multipath[delay_samples:] += alpha * signal[:-delay_samples]
    
    return multipath


def _add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Additive White Gaussian Noise.
    
    Args:
        signal: Input signal
        snr_db: Desired SNR in dB
    
    Returns:
        Noisy signal
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    
    return signal + noise


def generate_fec_coded_interleaved_signal(
    fec_type: str,
    interleaving_type: str,
    n_bits: int = 2000,
    interleaving_depth: int = 8
) -> Tuple[np.ndarray, dict]:
    """
    Generate FEC-coded and interleaved bit stream for testing auto-detection.
    
    This creates STRUCTURED data (not random) which is needed for:
    - FEC type auto-detection
    - Interleaving type auto-detection
    
    Args:
        fec_type: 'viterbi', 'reed-solomon', 'ldpc', 'concatenated'
        interleaving_type: 'block', 'convolutional', 'diagonal', 'pseudorandom'
        n_bits: Number of data bits to encode
        interleaving_depth: Interleaver depth parameter
    
    Returns:
        bits: Interleaved coded bit stream
        metadata: Ground truth parameters
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from core.fec.encoder import encode_fec
    from core.interleaving.interleaver import interleave
    
    # Generate structured data (not random - has patterns)
    # Mix of: repeating patterns, pseudo-random, and structured content
    data_bits = np.zeros(n_bits, dtype=np.uint8)
    
    # 30% repeating pattern
    pattern = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    n_pattern = int(n_bits * 0.3)
    for i in range(0, n_pattern, len(pattern)):
        end = min(i + len(pattern), n_pattern)
        data_bits[i:end] = pattern[:end-i]
    
    # 40% pseudo-random
    n_random = int(n_bits * 0.4)
    data_bits[n_pattern:n_pattern+n_random] = np.random.randint(0, 2, n_random)
    
    # 30% structured (alternating with burst)
    n_structured = n_bits - n_pattern - n_random
    data_bits[n_pattern+n_random:] = np.tile([1,1,1,0,0,0], n_structured//6 + 1)[:n_structured]
    
    # Apply FEC encoding
    coded_bits = encode_fec(data_bits, fec_type)
    
    # Apply interleaving
    interleaved_bits = interleave(coded_bits, interleaving_type, rows=interleaving_depth)
    
    metadata = {
        'fec_type': fec_type,
        'interleaving_type': interleaving_type,
        'interleaving_depth': interleaving_depth,
        'n_data_bits': n_bits,
        'n_coded_bits': len(coded_bits),
        'n_interleaved_bits': len(interleaved_bits),
        'coding_rate': n_bits / len(coded_bits),
        'is_structured_data': True
    }
    
    return interleaved_bits, metadata


# Quick test
if __name__ == '__main__':
    print("Generating realistic QPSK signal with RRC pulse shaping...")
    signal, meta = generate_realistic_signal(
        'QPSK',
        n_symbols=1000,
        sps=8,
        snr_db=20.0,
        rrc_rolloff=0.35,
        carrier_freq_hz=0.0,
        sample_rate_hz=100000.0
    )
    
    print(f"Generated {len(signal)} samples")
    print(f"True symbol rate: {meta['symbol_rate_hz']} Hz")
    print(f"True SPS: {meta['sps']}")
    print(f"Has pulse shaping: {meta['has_pulse_shaping']}")
    print(f"Signal power: {np.mean(np.abs(signal)**2):.4f}")
    
    print("\nGenerating FEC-coded interleaved signal...")
    bits, meta2 = generate_fec_coded_interleaved_signal(
        'viterbi',
        'block',
        n_bits=1000,
        interleaving_depth=8
    )
    
    print(f"Data bits: {meta2['n_data_bits']}")
    print(f"Coded bits: {meta2['n_coded_bits']}")
    print(f"Interleaved bits: {meta2['n_interleaved_bits']}")
    print(f"Coding rate: {meta2['coding_rate']:.3f}")
    print(f"Is structured: {meta2['is_structured_data']}")
