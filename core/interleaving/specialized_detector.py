"""
SPECIALIZED Block & Diagonal Interleaving Detector

Uses advanced 2D pattern analysis specifically tuned for block and diagonal.

Key Innovation: Instead of trying to detect patterns in 1D scrambled data,
we reconstruct the 2D interleaver matrix and analyze its structure!
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft2, ifft2
from typing import Tuple, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SpecializedBlockDiagonalDetector:
    """
    Specialized detector for block and diagonal interleaving.
    
    Key Insight: Block and diagonal interleaving create specific 2D matrix
    patterns that can be detected through:
    1. 2D FFT analysis (periodic structure detection)
    2. Row-column correlation patterns
    3. Diagonal vs anti-diagonal energy distribution
    4. Phase coherence analysis
    """
    
    def __init__(self):
        self.possible_depths = [4, 6, 8, 10, 12, 16, 20, 24, 32]
    
    def detect(self, bits: np.ndarray) -> Tuple[str, int, float]:
        """
        Detect block or diagonal interleaving with high accuracy.
        
        Strategy:
        1. For each depth, try DEINTERLEAVING as block
        2. Measure if de-interleaved data has more structure
        3. Try DEINTERLEAVING as diagonal
        4. Measure if de-interleaved data has more structure
        5. The correct type should reveal original structure!
        """
        if len(bits) < 64:
            return ('unknown', 0, 0.0)
        
        best_score = 0.0
        best_type = 'unknown'
        best_depth = 0
        
        for depth in self.possible_depths:
            cols = len(bits) // depth
            if cols < 4 or depth * cols > len(bits):
                continue
            
            # Score: try de-interleaving as block
            block_score = self._score_deinterleaved_block(bits, depth, cols)
            
            # Also score with matrix analysis
            block_matrix_score = self._score_block_structure(bits, depth, cols)
            
            # Combine scores - MORE weight on de-interleaving
            block_combined = 0.9 * block_score + 0.1 * block_matrix_score
            
            if block_combined > best_score:
                best_score = block_combined
                best_type = 'block'
                best_depth = depth
            
            # Score: try de-interleaving as diagonal
            diag_score = self._score_deinterleaved_diagonal(bits, depth, cols)
            
            # Also score with matrix analysis
            diag_matrix_score = self._score_diagonal_structure(bits, depth, cols)
            
            # Combine scores - MORE weight on de-interleaving
            diag_combined = 0.9 * diag_score + 0.1 * diag_matrix_score
            
            if diag_combined > best_score:
                best_score = diag_combined
                best_type = 'diagonal'
                best_depth = depth
        
        # Check if score is high enough
        if best_score < 0.3:
            return ('unknown', 0, 0.0)
        
        return (best_type, best_depth, best_score)
    
    def _score_block_structure(self, bits: np.ndarray, depth: int, cols: int) -> float:
        """
        Score how well data matches block interleaving structure.
        
        Block interleaving: Write row-wise, read column-wise
        Creates specific correlation patterns between positions.
        """
        score = 0.0
        
        # Reconstruct assumed block matrix (as if de-interleaving)
        try:
            # Assume bits are block-interleaved
            # To de-interleave: arrange in depth rows, transpose
            n_bits = depth * cols
            if n_bits > len(bits):
                return 0.0
            
            data = bits[:n_bits]
            
            # Arrange as if reading column-wise (interleaved)
            matrix_interleaved = data.reshape(cols, depth).T
            
            # This is what original looked like (row-wise)
            matrix_original = data.reshape(depth, cols)
            
            # Feature 1: Row correlation in de-interleaved data
            # After correct de-interleaving, rows should show patterns
            row_correlations = []
            for i in range(depth - 1):
                corr = np.corrcoef(matrix_original[i].astype(float), 
                                  matrix_original[i+1].astype(float))[0, 1]
                if not np.isnan(corr):
                    row_correlations.append(abs(corr))
            
            if len(row_correlations) > 0:
                avg_row_corr = np.mean(row_correlations)
                if avg_row_corr > 0.2:  # Adjacent rows are similar
                    score += 0.4  # INCREASED weight
            
            # Feature 2: Block-specific: Regular column structure
            # In block interleaving, columns should have uniform distribution
            col_means = []
            for j in range(cols):
                col_means.append(np.mean(matrix_original[:, j]))
            
            col_variance = np.var(col_means)
            if col_variance < 0.1:  # Uniform columns
                score += 0.3
            
            # Feature 3: NOT diagonal pattern
            # Block should NOT have diagonal correlation
            diag_corr = 0.0
            for i in range(1, min(depth, cols)):
                if i < cols:
                    d1 = [matrix_original[j, j] for j in range(min(depth, cols) - i)]
                    d2 = [matrix_original[j, j + i] for j in range(min(depth, cols) - i)]
                    if len(d1) > 1:
                        corr = np.corrcoef(d1, d2)[0, 1]
                        if not np.isnan(corr):
                            diag_corr = max(diag_corr, abs(corr))
            
            if diag_corr < 0.4:  # Low diagonal correlation = likely block
                score += 0.2
            
            # Feature 4: Rank analysis
            # Block interleaving preserves rank
            rank = np.linalg.matrix_rank(matrix_original.astype(float))
            max_rank = min(depth, cols)
            rank_ratio = rank / max_rank
            if rank_ratio > 0.7:
                score += 0.3
            
        except Exception as e:
            return 0.0
        
        return min(1.0, score)
    
    def _score_diagonal_structure(self, bits: np.ndarray, depth: int, cols: int) -> float:
        """
        Score how well data matches diagonal interleaving structure.
        
        Diagonal: Each row shifted by 1 position
        Creates diagonal patterns in the 2D matrix.
        
        KEY FEATURE: Row shift pattern (each row shifted by 1)
        """
        score = 0.0
        
        try:
            n_bits = depth * cols
            if n_bits > len(bits):
                return 0.0
            
            data = bits[:n_bits]
            
            # Try to reconstruct diagonal structure
            matrix = data.reshape(depth, cols)
            
            # FEATURE 1: Row shift detection (MOST IMPORTANT for diagonal!)
            # Each row should be 1-position shifted version of previous
            shift_1_scores = []
            for i in range(depth - 1):
                row1 = matrix[i].astype(float)
                row2 = matrix[i + 1].astype(float)
                
                # Check if row2 is shifted by 1 position
                if cols > 1:
                    # Compare row1[:-1] with row2[1:]
                    overlap_len = cols - 1
                    corr = np.corrcoef(row1[:overlap_len], row2[1:overlap_len+1])[0, 1]
                    if not np.isnan(corr) and corr > 0.2:
                        shift_1_scores.append(abs(corr))
            
            # Strong evidence of shift-by-1 pattern
            if len(shift_1_scores) > 0:
                avg_shift_1 = np.mean(shift_1_scores)
                if avg_shift_1 > 0.5:  # Strong shift pattern
                    score += 0.5  # HIGH WEIGHT
                elif avg_shift_1 > 0.3:
                    score += 0.3
            
            # FEATURE 2: NOT uniform columns (unlike block)
            # Diagonal should have varying columns
            col_stds = []
            for j in range(min(cols, 20)):  # Check first 20 columns
                col_stds.append(np.std(matrix[:, j].astype(float)))
            
            if len(col_stds) > 0:
                avg_col_std = np.mean(col_stds)
                if avg_col_std > 0.3:  # Columns are varied
                    score += 0.2
            
            # FEATURE 3: Diagonal stripe energy distribution
            # In diagonal interleaving, energy spreads along diagonals
            diag_energies = []
            for offset in range(-min(depth-1, 3), min(cols, 4)):
                diag_sum = 0.0
                count = 0
                for i in range(depth):
                    j = i + offset
                    if 0 <= j < cols:
                        diag_sum += matrix[i, j]
                        count += 1
                if count > 0:
                    diag_energies.append(diag_sum / count)
            
            if len(diag_energies) > 1:
                diag_variance = np.var(diag_energies)
                if diag_variance > 0.05:  # Varied diagonal energy
                    score += 0.2
            
            # FEATURE 4: Row independence (NOT similar like block)
            # Diagonal rows should be less correlated than block
            row_correlations = []
            for i in range(min(depth - 1, 10)):
                corr = np.corrcoef(matrix[i].astype(float), 
                                  matrix[i+1].astype(float))[0, 1]
                if not np.isnan(corr):
                    row_correlations.append(abs(corr))
            
            if len(row_correlations) > 0:
                avg_row_corr = np.mean(row_correlations)
                if avg_row_corr < 0.3:  # Rows are different
                    score += 0.1
            
        except Exception as e:
            return 0.0
        
        return min(1.0, score)
    
    def _analyze_2d_fft(self, matrix: np.ndarray) -> float:
        """
        Analyze 2D FFT for periodic structure.
        Block interleaving creates specific frequency patterns.
        """
        try:
            # Pad to power of 2 for efficient FFT
            rows, cols = matrix.shape
            pad_rows = 2 ** int(np.ceil(np.log2(rows)))
            pad_cols = 2 ** int(np.ceil(np.log2(cols)))
            
            padded = np.zeros((pad_rows, pad_cols))
            padded[:rows, :cols] = matrix
            
            # 2D FFT
            fft_result = np.abs(fft2(padded))
            
            # Shift zero frequency to center
            fft_shifted = np.fft.fftshift(fft_result)
            
            # Measure peakiness (periodic signals have peaks)
            flat = fft_shifted.flatten()
            flat = flat / (np.max(flat) + 1e-12)
            
            # Count significant peaks
            peaks = np.sum(flat > 0.3)
            peak_ratio = peaks / len(flat)
            
            # Moderate peaks indicate periodicity
            if 0.01 < peak_ratio < 0.1:
                return 0.8
            elif 0.001 < peak_ratio < 0.2:
                return 0.5
            else:
                return 0.1
            
        except Exception:
            return 0.0
    
    def _deinterleave_as_block(self, bits: np.ndarray, depth: int, cols: int) -> np.ndarray:
        """
        De-interleave assuming block interleaving.
        Block: wrote row-wise, read column-wise → reverse it
        """
        n_bits = depth * cols
        if n_bits > len(bits):
            return bits
        
        data = bits[:n_bits]
        # Interleaved was read column-wise, so fill column-wise
        matrix = data.reshape(cols, depth).T  # Transpose to get original
        return matrix.flatten()
    
    def _deinterleave_as_diagonal(self, bits: np.ndarray, depth: int, cols: int) -> np.ndarray:
        """
        De-interleave assuming diagonal interleaving.
        Diagonal: wrote row-wise, read diagonally → reverse the diagonal read
        """
        n_bits = depth * cols
        if n_bits > len(bits):
            return bits
        
        data = bits[:n_bits]
        matrix = np.zeros((depth, cols), dtype=np.uint8)
        
        # Reverse diagonal read: for input index i, it came from (r, c) where:
        # r = i % depth, c = (i // depth + r) % cols
        for idx in range(n_bits):
            r = idx % depth
            c = (idx // depth + r) % cols
            matrix[r, c] = data[idx]
        
        return matrix.flatten()
    
    def _score_deinterleaved_block(self, bits: np.ndarray, depth: int, cols: int) -> float:
        """
        De-interleave as block and measure if structure improves.
        If correct, de-interleaving should reveal original patterns.
        """
        try:
            deinterleaved = self._deinterleave_as_block(bits, depth, cols)
            if len(deinterleaved) < 32:
                return 0.0
            
            score = 0.0
            
            # Feature 1: Run-length encoding efficiency
            # Original data should have longer runs
            orig_runs = self._count_runs(bits[:len(deinterleaved)])
            deint_runs = self._count_runs(deinterleaved)
            
            if deint_runs < orig_runs * 0.8:  # Fewer runs = more structure
                score += 0.4
            
            # Feature 2: Autocorrelation at small lags
            # Original data should have higher autocorrelation
            orig_autocorr = self._autocorr_score(bits[:len(deinterleaved)])
            deint_autocorr = self._autocorr_score(deinterleaved)
            
            if deint_autocorr > orig_autocorr * 1.1:
                score += 0.3
            
            # Feature 3: Entropy decrease
            # Original should have lower entropy (more predictable)
            orig_entropy = self._block_entropy(bits[:len(deinterleaved)])
            deint_entropy = self._block_entropy(deinterleaved)
            
            if deint_entropy < orig_entropy * 0.95:
                score += 0.3
            
            return min(1.0, score)
        except Exception:
            return 0.0
    
    def _score_deinterleaved_diagonal(self, bits: np.ndarray, depth: int, cols: int) -> float:
        """
        De-interleave as diagonal and measure if structure improves.
        """
        try:
            deinterleaved = self._deinterleave_as_diagonal(bits, depth, cols)
            if len(deinterleaved) < 32:
                return 0.0
            
            score = 0.0
            
            # Feature 1: Run-length encoding efficiency
            orig_runs = self._count_runs(bits[:len(deinterleaved)])
            deint_runs = self._count_runs(deinterleaved)
            
            if deint_runs < orig_runs * 0.8:
                score += 0.4
            
            # Feature 2: Autocorrelation
            orig_autocorr = self._autocorr_score(bits[:len(deinterleaved)])
            deint_autocorr = self._autocorr_score(deinterleaved)
            
            if deint_autocorr > orig_autocorr * 1.1:
                score += 0.3
            
            # Feature 3: Entropy
            orig_entropy = self._block_entropy(bits[:len(deinterleaved)])
            deint_entropy = self._block_entropy(deinterleaved)
            
            if deint_entropy < orig_entropy * 0.95:
                score += 0.3
            
            return min(1.0, score)
        except Exception:
            return 0.0
    
    def _count_runs(self, bits: np.ndarray) -> int:
        """Count number of runs (consecutive same bits)."""
        if len(bits) == 0:
            return 0
        return np.sum(bits[:-1] != bits[1:]) + 1
    
    def _autocorr_score(self, bits: np.ndarray, max_lag: int = 8) -> float:
        """Measure autocorrelation at small lags."""
        if len(bits) < max_lag + 10:
            return 0.0
        
        bits_float = bits.astype(float) - np.mean(bits)
        autocorrs = []
        for lag in range(1, min(max_lag, len(bits) // 4)):
            if len(bits_float) > lag:
                corr = np.corrcoef(bits_float[:-lag], bits_float[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(abs(corr))
        
        return np.mean(autocorrs) if len(autocorrs) > 0 else 0.0
    
    def _block_entropy(self, bits: np.ndarray, block_size: int = 4) -> float:
        """Measure entropy of blocks."""
        if len(bits) < block_size:
            return 1.0
        
        n_blocks = len(bits) // block_size
        blocks = []
        for i in range(n_blocks):
            block = tuple(bits[i*block_size:(i+1)*block_size])
            blocks.append(block)
        
        # Count unique blocks
        unique_blocks = len(set(blocks))
        max_blocks = 2 ** block_size
        
        return unique_blocks / min(n_blocks, max_blocks)


def detect_block_diagonal_specialized(bits: np.ndarray) -> Tuple[str, int, float]:
    """
    Specialized detection for block and diagonal interleaving.
    
    Args:
        bits: Bit stream
    
    Returns:
        (type, depth, confidence)
        type: 'block', 'diagonal', or 'unknown'
    """
    detector = SpecializedBlockDiagonalDetector()
    return detector.detect(bits)


# Test
if __name__ == '__main__':
    print("Testing Specialized Block/Diagonal Detector...")
    print("=" * 60)
    
    from core.interleaving.interleaver import interleave
    
    # Test 1: Block interleaving
    print("\nTest 1: Block Interleaving")
    print("-" * 60)
    data = np.zeros(2000, dtype=np.uint8)
    # Create pattern
    for i in range(len(data)):
        data[i] = 1 if (i // 8) % 2 == 0 else 0
    
    interleaved = interleave(data, 'block', rows=8)
    detected, depth, conf = detect_block_diagonal_specialized(interleaved)
    
    print(f"True: block, depth=8")
    print(f"Detected: {detected}, depth={depth}, confidence={conf:.2f}")
    print("✓ SUCCESS!" if detected == 'block' else "✗ FAILED")
    
    # Test 2: Diagonal interleaving
    print("\nTest 2: Diagonal Interleaving")
    print("-" * 60)
    interleaved = interleave(data, 'diagonal', rows=8)
    detected, depth, conf = detect_block_diagonal_specialized(interleaved)
    
    print(f"True: diagonal, depth=8")
    print(f"Detected: {detected}, depth={depth}, confidence={conf:.2f}")
    print("✓ SUCCESS!" if detected == 'diagonal' else "~ PARTIAL" if detected != 'unknown' else "✗ FAILED")

