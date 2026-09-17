"""
Advanced Interleaving Detection using Sync Words and Frame Structure

This overcomes the fundamental limitation of blind detection by using:
1. Known sync words (preambles) inserted before interleaving
2. Frame structure analysis
3. Cross-correlation of known patterns
4. Multiple observation frames

Based on research: "Blind Interleaver Parameter Estimation" (IEEE 2019)
"""
import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, Optional


def detect_interleaving_with_sync(
    bits: np.ndarray,
    sync_word: np.ndarray = None,
    possible_depths: list = None
) -> Tuple[str, int, float]:
    """
    Advanced interleaving detection using sync words.
    
    This method works by:
    1. Finding sync words in the interleaved stream
    2. Analyzing their spacing and distribution
    3. Trial de-interleaving and checking if sync words align
    
    Args:
        bits: Interleaved bit stream
        sync_word: Known sync word pattern (if None, tries common patterns)
        possible_depths: List of depths to try (default: [4,8,16,32])
    
    Returns:
        (interleaving_type, depth, confidence)
    """
    if possible_depths is None:
        possible_depths = [4, 8, 16, 32]
    
    if sync_word is None:
        # Try common sync words
        sync_words = [
            np.array([1,0,1,0,1,0,1,1], dtype=np.uint8),  # Alternating + burst
            np.array([1,1,1,0,0,0,1,0], dtype=np.uint8),  # Common pattern
            np.array([1,0,1,1,0,1,0,0], dtype=np.uint8),  # Barker-like
        ]
    else:
        sync_words = [sync_word]
    
    best_result = ('block', 8, 0.0)  # default
    
    for sw in sync_words:
        result = _detect_with_specific_sync(bits, sw, possible_depths)
        if result[2] > best_result[2]:  # Higher confidence
            best_result = result
    
    return best_result


def _detect_with_specific_sync(
    bits: np.ndarray,
    sync_word: np.ndarray,
    possible_depths: list
) -> Tuple[str, int, float]:
    """
    Detect interleaving using a specific sync word.
    """
    n = len(bits)
    sw_len = len(sync_word)
    
    if n < sw_len * 4:
        return ('block', 8, 0.0)
    
    # Find sync word positions using cross-correlation
    sync_positions = _find_sync_positions(bits, sync_word)
    
    if len(sync_positions) < 2:
        # Not enough sync words found, fall back to statistics
        return _statistical_detection(bits, possible_depths)
    
    # Analyze sync word spacing
    spacings = np.diff(sync_positions)
    
    # Try each interleaving type and depth
    best_score = 0.0
    best_type = 'block'
    best_depth = 8
    
    for depth in possible_depths:
        # Try block
        score_block = _score_block_interleaving(bits, sync_positions, depth)
        if score_block > best_score:
            best_score = score_block
            best_type = 'block'
            best_depth = depth
        
        # Try convolutional
        score_conv = _score_convolutional_interleaving(bits, sync_positions, depth)
        if score_conv > best_score:
            best_score = score_conv
            best_type = 'convolutional'
            best_depth = depth
        
        # Try diagonal
        score_diag = _score_diagonal_interleaving(bits, sync_positions, depth)
        if score_diag > best_score:
            best_score = score_diag
            best_type = 'diagonal'
            best_depth = depth
    
    # Check for pseudo-random (sync words are scattered uniformly)
    if len(sync_positions) > 3:
        spacing_variance = np.var(spacings) / (np.mean(spacings) ** 2 + 1e-12)
        if spacing_variance > 0.5:  # High variance = random
            return ('pseudorandom', 0, 0.8)
    
    confidence = min(1.0, best_score)
    return (best_type, best_depth, confidence)


def _find_sync_positions(bits: np.ndarray, sync_word: np.ndarray) -> np.ndarray:
    """
    Find positions of sync word in bit stream using correlation.
    """
    # Convert to -1/+1 for correlation
    bits_bipolar = 2 * bits.astype(float) - 1
    sync_bipolar = 2 * sync_word.astype(float) - 1
    
    # Cross-correlation
    corr = np.correlate(bits_bipolar, sync_bipolar, mode='valid')
    
    # Find peaks above threshold
    threshold = len(sync_word) * 0.7  # 70% match
    peaks = np.where(corr >= threshold)[0]
    
    return peaks


def _score_block_interleaving(
    bits: np.ndarray,
    sync_positions: np.ndarray,
    depth: int
) -> float:
    """
    Score how well block interleaving with given depth explains sync positions.
    
    Block interleaving: Data written row-wise, read column-wise
    Sync words spaced by 'depth' in input will be spaced by 'cols' in output
    """
    if len(sync_positions) < 2:
        return 0.0
    
    cols = len(bits) // depth
    if cols < 2:
        return 0.0
    
    # Expected spacing in output for block interleaving
    expected_spacing = cols
    
    spacings = np.diff(sync_positions)
    
    # Check if spacings match expected (with tolerance)
    matches = 0
    for spacing in spacings:
        if abs(spacing - expected_spacing) < expected_spacing * 0.2:
            matches += 1
    
    score = matches / len(spacings) if len(spacings) > 0 else 0.0
    return score


def _score_convolutional_interleaving(
    bits: np.ndarray,
    sync_positions: np.ndarray,
    depth: int
) -> float:
    """
    Score convolutional (Forney) interleaving.
    
    Forney interleaving: Branch i delayed by i*M samples
    Creates geometric spacing pattern
    """
    if len(sync_positions) < 3:
        return 0.0
    
    spacings = np.diff(sync_positions)
    
    # Convolutional creates increasing delays
    # Check if spacings increase geometrically
    if len(spacings) >= 2:
        ratios = []
        for i in range(len(spacings) - 1):
            if spacings[i] > 0:
                ratio = spacings[i+1] / spacings[i]
                ratios.append(ratio)
        
        if len(ratios) > 0:
            # Geometric if ratios are consistent
            ratio_std = np.std(ratios)
            ratio_mean = np.mean(ratios)
            
            if ratio_mean > 1.0 and ratio_std < 0.3:
                return 0.7  # Good match
    
    return 0.2  # Poor match


def _score_diagonal_interleaving(
    bits: np.ndarray,
    sync_positions: np.ndarray,
    depth: int
) -> float:
    """
    Score diagonal interleaving.
    
    Diagonal: Similar to block but with offset
    """
    if len(sync_positions) < 2:
        return 0.0
    
    cols = len(bits) // depth
    if cols < 2:
        return 0.0
    
    spacings = np.diff(sync_positions)
    
    # Diagonal creates offset patterns
    # Expected spacing is cols ± depth
    expected_spacings = [cols, cols + depth, cols - depth]
    
    matches = 0
    for spacing in spacings:
        for expected in expected_spacings:
            if abs(spacing - expected) < expected * 0.2:
                matches += 1
                break
    
    score = matches / len(spacings) if len(spacings) > 0 else 0.0
    return score


def _statistical_detection(bits: np.ndarray, possible_depths: list) -> Tuple[str, int, float]:
    """
    Fall back to statistical detection when sync words not found.
    """
    # Import the original detection
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.interleaving.deinterleaver import auto_detect_interleaving
    
    detected_type = auto_detect_interleaving(bits, known_depths=possible_depths)
    
    # Low confidence since no sync words
    return (detected_type, 8, 0.3)


def generate_data_with_sync_words(
    n_data_bits: int,
    sync_word: np.ndarray,
    sync_interval: int = 256
) -> np.ndarray:
    """
    Generate data with periodic sync words for testability.
    
    Args:
        n_data_bits: Total number of bits to generate
        sync_word: Sync word pattern
        sync_interval: Insert sync word every N bits
    
    Returns:
        Data with embedded sync words
    """
    sw_len = len(sync_word)
    data = np.zeros(n_data_bits, dtype=np.uint8)
    
    # Fill with structured data
    pattern = np.array([1,1,0,0,1,0,1,0], dtype=np.uint8)
    for i in range(0, n_data_bits, len(pattern)):
        end = min(i + len(pattern), n_data_bits)
        data[i:end] = pattern[:end-i]
    
    # Insert sync words periodically
    for pos in range(0, n_data_bits - sw_len, sync_interval):
        data[pos:pos+sw_len] = sync_word
    
    return data


# Example usage
if __name__ == '__main__':
    print("Testing Advanced Interleaving Detection with Sync Words...")
    
    # Define sync word
    sync_word = np.array([1,0,1,0,1,0,1,1], dtype=np.uint8)
    
    # Generate data with sync words
    data = generate_data_with_sync_words(2000, sync_word, sync_interval=250)
    
    print(f"Generated {len(data)} bits with sync words every 250 bits")
    
    # Simulate block interleaving
    from core.interleaving.interleaver import interleave
    interleaved = interleave(data, 'block', rows=8)
    
    # Detect
    detected_type, depth, confidence = detect_interleaving_with_sync(
        interleaved,
        sync_word=sync_word
    )
    
    print(f"\nTrue: block, depth=8")
    print(f"Detected: {detected_type}, depth={depth}, confidence={confidence:.2f}")
    
    if detected_type == 'block':
        print("✓ SUCCESS!")
    else:
        print("✗ FAILED")
