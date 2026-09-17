"""
Bit Stream Correlator — finds headers, sync words, and payloads.
Uses sliding window cross-correlation.

When core/fast_correlator.dll is available, find_sync_word() and
auto_correlate() use the C++ implementation (~50x faster on long streams).
Falls back to NumPy silently.
"""
import numpy as np
from typing import List, Tuple, Optional

# Try C++ fast correlator (optional, falls back to NumPy)
try:
    from core.cpp_extensions import (
        find_sync_word as _cpp_find_sync,
        xcorr_bits as _cpp_xcorr,
        cpp_available as _cpp_avail,
    )
    _USE_CPP_CORRELATOR = _cpp_avail().get('fast_correlator', False)
except ImportError:
    _USE_CPP_CORRELATOR = False


# Common sync words / preambles used in real protocols
KNOWN_SYNC_WORDS = {
    'HDLC':         np.array([0,1,1,1,1,1,1,0], dtype=np.uint8),
    'AX.25':        np.array([0,1,1,1,1,1,1,0], dtype=np.uint8),
    'APRS':         np.array([0,1,1,1,1,1,1,0], dtype=np.uint8),
    'CCSDS':        np.array([0,1,0,1,1,0,0,1,1,0,1,1,1,0,0,1,
                               1,1,0,1,1,0,1,1,0,0,1,1,0,1,0,0],
                              dtype=np.uint8),
    '802.11':       np.array([1,0,1,1], dtype=np.uint8),
    'Generic 0x7E': np.array([0,1,1,1,1,1,1,0], dtype=np.uint8),
    'Barker-7':     np.array([1,1,1,0,0,1,0], dtype=np.uint8),
    'Barker-11':    np.array([1,1,1,0,0,0,1,0,0,1,0], dtype=np.uint8),
    'Barker-13':    np.array([1,1,1,1,1,0,0,1,1,0,1,0,1], dtype=np.uint8),
}


def correlate_bits(bits: np.ndarray,
                   sync_word: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, int, int, str]:
    """
    Find sync word / preamble in a bit stream via cross-correlation.
    Uses C++ fast_correlator.dll when available (~50x faster on long streams).

    Returns (correlation_curve, header_offset, payload_offset, sync_name)
    """
    if sync_word is None:
        return auto_correlate(bits)

    if _USE_CPP_CORRELATOR:
        pos, score = _cpp_find_sync(bits, sync_word, allow_complement=True)
        # Build a stub correlation array for compatibility
        corr = np.zeros(max(1, len(bits) - len(sync_word) + 1), dtype=np.float32)
        if pos < len(corr):
            corr[pos] = score
        return corr, int(pos), int(pos) + len(sync_word), 'custom'

    corr = _sliding_correlate(bits, sync_word)
    peak_idx = int(np.argmax(corr))
    return corr, peak_idx, peak_idx + len(sync_word), 'custom'


def auto_correlate(bits: np.ndarray
                   ) -> Tuple[np.ndarray, int, int, str]:
    """
    Try all known sync words and return the best match.
    Uses C++ fast_correlator.dll when available.
    """
    try:
        from core.bitstream.sync_db import load_all
        all_sync_words = load_all()
    except Exception:
        all_sync_words = KNOWN_SYNC_WORDS

    best_score  = -1.0
    best_corr   = np.zeros(len(bits))
    best_offset = 0
    best_payload= 0
    best_name   = 'unknown'

    for name, sw in all_sync_words.items():
        if len(sw) >= len(bits):
            continue

        if _USE_CPP_CORRELATOR:
            pos, score = _cpp_find_sync(bits, sw, allow_complement=True)
            normalised = float(score)
            if normalised > best_score:
                best_score  = normalised
                best_offset = int(pos)
                best_payload= int(pos) + len(sw)
                best_name   = name
                best_corr   = np.zeros(max(1, len(bits)-len(sw)+1), dtype=np.float32)
                if best_offset < len(best_corr):
                    best_corr[best_offset] = score
        else:
            corr = _sliding_correlate(bits, sw)
            peak_val   = float(np.max(corr))
            normalised = peak_val / len(sw)
            if normalised > best_score:
                best_score  = normalised
                best_corr   = corr
                best_offset = int(np.argmax(corr))
                best_payload= best_offset + len(sw)
                best_name   = name

    return best_corr, best_offset, best_payload, best_name


def _sliding_correlate(bits: np.ndarray,
                       pattern: np.ndarray) -> np.ndarray:
    """
    Sliding window cross-correlation between bits and pattern.
    Returns an array of match scores (0 = no match, len(pattern) = perfect).
    """
    n = len(bits)
    m = len(pattern)
    if m > n:
        return np.zeros(n)

    # Convert to bipolar: 0→-1, 1→+1
    b = 2.0 * bits.astype(np.float32) - 1.0
    p = 2.0 * pattern.astype(np.float32) - 1.0

    # Use numpy correlate (fast for short patterns)
    corr = np.correlate(b, p, mode='full')
    # Trim to same length as input
    start = m - 1
    corr = corr[start:start + n]
    # Normalize to [0, len(pattern)]
    corr = (corr + len(pattern)) / 2.0
    return corr.astype(np.float32)


def extract_frames(bits: np.ndarray,
                   sync_word: np.ndarray,
                   frame_length: int = None
                   ) -> List[Tuple[int, np.ndarray]]:
    """
    Find all occurrences of sync_word in bits and extract frames.
    Returns list of (offset, frame_bits) tuples.
    """
    corr = _sliding_correlate(bits, sync_word)
    threshold = 0.8 * len(sync_word)

    # Find peaks above threshold
    above = corr > threshold
    frames = []
    i = 0
    while i < len(above):
        if above[i]:
            start = i + len(sync_word)
            if frame_length is not None:
                end = start + frame_length
                if end <= len(bits):
                    frames.append((i, bits[start:end]))
            else:
                # Find next sync word
                next_sync = i + len(sync_word)
                while next_sync < len(above) and not above[next_sync]:
                    next_sync += 1
                if next_sync < len(bits):
                    frames.append((i, bits[start:next_sync]))
            i = start
        else:
            i += 1

    return frames


def bits_to_hex(bits: np.ndarray) -> str:
    """Convert bit array to hex string."""
    pad = (8 - len(bits) % 8) % 8
    if pad:
        bits = np.pad(bits, (0, pad))
    byte_arr = np.packbits(bits.astype(np.uint8))
    return ' '.join(f'{b:02X}' for b in byte_arr)


def bits_to_ascii(bits: np.ndarray) -> str:
    """Convert bit array to ASCII string, replacing non-printable with '.'"""
    pad = (8 - len(bits) % 8) % 8
    if pad:
        bits = np.pad(bits, (0, pad))
    byte_arr = np.packbits(bits.astype(np.uint8))
    result = ''
    for b in byte_arr:
        if 32 <= b <= 126:
            result += chr(b)
        else:
            result += '.'
    return result
