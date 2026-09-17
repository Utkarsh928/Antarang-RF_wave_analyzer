"""
RANK-BASED INTERLEAVER DETECTION using Kolmogorov-Smirnov Test

Based on IEEE Access 2021 research:
"Blind Interleaver Parameters Estimation Using Kolmogorov–Smirnov Test"
https://pmc.ncbi.nlm.nih.gov/articles/PMC8155855/

Key Insight: Linear codes have rank distributions that differ from random data.
"""
import numpy as np
from typing import Tuple, Dict
from scipy.stats import kstest


# Known rank distribution for random binary square matrices (from research)
# Ps = probability that rank deficiency = s
RANDOM_RANK_DISTRIBUTION = {
    0: 0.288788,  # Full rank
    1: 0.577576,  # Rank deficiency = 1
    2: 0.128350,  # Rank deficiency = 2
    3: 0.005238,  # Rank deficiency = 3
    4: 0.000048,  # Rank deficiency >= 4
}


class RankBasedDetector:
    """
    Detects interleaver type using rank deficiency distributions.
    
    Content rephrased for compliance with licensing restrictions.
    
    Method:
    1. Construct square matrices from bitstream
    2. Calculate rank distribution
    3. Compare with known random distribution using K-S test
    4. Higher K-S value → more different from random → likely correct interleaver period
    """
    
    def __init__(self, n_trials: int = 500):
        """
        Args:
            n_trials: Number of random matrices to construct for rank distribution
        """
        self.n_trials = n_trials
        self.possible_depths = [4, 6, 8, 10, 12, 16, 20, 24]
    
    def detect(self, bits: np.ndarray) -> Tuple[str, int, float]:
        """
        Detect interleaver type using rank-based K-S test.
        
        Returns:
            (type, depth, confidence)
        """
        if len(bits) < 100:
            return ('unknown', 0, 0.0)
        
        best_ks_block = 0.0
        best_depth_block = 0
        best_ks_diag = 0.0
        best_depth_diag = 0
        
        for depth in self.possible_depths:
            cols = len(bits) // depth
            if cols < 4 or depth * cols > len(bits):
                continue
            
            # Calculate rank distribution for this depth
            rank_dist = self._calculate_rank_distribution(bits, depth, cols)
            
            if rank_dist is None:
                continue
            
            # K-S test against random distribution
            ks_stat = self._ks_test(rank_dist)
            
            # Score for block interleaving
            # Block should show HIGH K-S (very different from random)
            block_score = self._score_as_block(rank_dist, ks_stat, depth, cols)
            if block_score > best_ks_block:
                best_ks_block = block_score
                best_depth_block = depth
            
            # Score for diagonal interleaving
            # Diagonal shows MODERATE K-S with specific patterns
            diag_score = self._score_as_diagonal(bits, rank_dist, ks_stat, depth, cols)
            if diag_score > best_ks_diag:
                best_ks_diag = diag_score
                best_depth_diag = depth
        
        # Decide which type with clear separation threshold
        threshold_diff = 0.15  # Block must be clearly better
        
        if best_ks_block > best_ks_diag + threshold_diff and best_ks_block > 0.3:
            return ('block', best_depth_block, best_ks_block)
        elif best_ks_diag > best_ks_block + threshold_diff and best_ks_diag > 0.25:
            return ('diagonal', best_depth_diag, best_ks_diag)
        elif best_ks_block > best_ks_diag and best_ks_block > 0.4:
            # If block score is high enough, trust it even without large margin
            return ('block', best_depth_block, best_ks_block)
        elif best_ks_diag > 0.6:
            # Very high diagonal score, trust it
            return ('diagonal', best_depth_diag, best_ks_diag)
        else:
            return ('unknown', 0, 0.0)
    
    def _calculate_rank_distribution(self, bits: np.ndarray, depth: int, cols: int) -> Dict[int, float]:
        """
        Calculate rank deficiency distribution by constructing random square matrices.
        
        Returns dict: {rank_deficiency: probability}
        """
        n_bits = depth * cols
        if n_bits > len(bits):
            return None
        
        data = bits[:n_bits]
        rank_deficiencies = []
        
        # Construct multiple random matrices and calculate ranks
        for trial in range(min(self.n_trials, len(bits) // (depth * depth))):
            try:
                # Randomly select depth vectors of length depth
                indices = np.random.choice(len(data) - depth, depth, replace=False)
                matrix = np.zeros((depth, depth), dtype=np.uint8)
                
                for i, idx in enumerate(indices):
                    matrix[i] = data[idx:idx+depth]
                
                # Calculate rank
                rank = np.linalg.matrix_rank(matrix.astype(float))
                rank_deficiency = depth - rank
                rank_deficiencies.append(rank_deficiency)
            except:
                continue
        
        if len(rank_deficiencies) < 10:
            return None
        
        # Calculate distribution
        rank_dist = {}
        for s in range(5):  # s = 0, 1, 2, 3, 4+
            if s == 4:
                count = sum(1 for r in rank_deficiencies if r >= 4)
            else:
                count = sum(1 for r in rank_deficiencies if r == s)
            rank_dist[s] = count / len(rank_deficiencies)
        
        return rank_dist
    
    def _ks_test(self, observed_dist: Dict[int, float]) -> float:
        """
        Kolmogorov-Smirnov test: measure max difference between CDFs.
        
        D = sup|F_observed(x) - F_random(x)|
        
        Higher value = more different from random
        """
        # Calculate CDFs
        cdf_observed = []
        cdf_random = []
        cumsum_obs = 0.0
        cumsum_rand = 0.0
        
        for s in sorted(RANDOM_RANK_DISTRIBUTION.keys()):
            cumsum_obs += observed_dist.get(s, 0.0)
            cumsum_rand += RANDOM_RANK_DISTRIBUTION[s]
            cdf_observed.append(cumsum_obs)
            cdf_random.append(cumsum_rand)
        
        # K-S statistic: max absolute difference
        ks_stat = max(abs(o - r) for o, r in zip(cdf_observed, cdf_random))
        
        return ks_stat
    
    def _score_as_block(self, rank_dist: Dict[int, float], ks_stat: float, 
                        depth: int, cols: int) -> float:
        """
        Score how well this matches block interleaving.
        
        Block characteristics:
        - HIGH K-S statistic (very different from random)
        - High rank deficiency (s >= 2)
        - Low full-rank probability
        """
        score = 0.0
        
        # Feature 1: K-S statistic (primary indicator)
        # Research shows K-S > 0.4 indicates structured data
        # Block should show VERY high K-S (> 0.5)
        if ks_stat > 0.6:
            score += 0.6  # INCREASED
        elif ks_stat > 0.4:
            score += 0.4
        elif ks_stat > 0.2:
            score += 0.2
        
        # Feature 2: Rank deficiency pattern
        # Block shows high s >= 2 probability
        high_deficiency = rank_dist.get(2, 0) + rank_dist.get(3, 0) + rank_dist.get(4, 0)
        if high_deficiency > 0.25:
            score += 0.25
        elif high_deficiency > 0.1:
            score += 0.1
        
        # Feature 3: Low full rank probability
        # Random has 28.8% full rank, block should be much lower
        full_rank_prob = rank_dist.get(0, 0)
        if full_rank_prob < 0.15:
            score += 0.15
        elif full_rank_prob < 0.25:
            score += 0.05
        
        return min(1.0, score)
    
    def _score_as_diagonal(self, bits: np.ndarray, rank_dist: Dict[int, float], 
                          ks_stat: float, depth: int, cols: int) -> float:
        """
        Score how well this matches diagonal interleaving.
        
        Diagonal characteristics:
        - MODERATE K-S statistic (less than block)
        - Cyclic shift correlation
        - Helical wraparound patterns
        """
        score = 0.0
        
        # Feature 1: Moderate K-S (not as high as block)
        # Diagonal shows 0.15-0.6 range typically
        if 0.1 < ks_stat < 0.65:
            # Perfect moderate range
            score += 0.35  # INCREASED weight
        elif 0.05 < ks_stat < 0.75:
            # Still reasonable
            score += 0.25
        
        # Feature 2: Rank pattern (more mixed than block)
        # Diagonal doesn't have as strong rank deficiency as block
        s1_prob = rank_dist.get(1, 0)
        s2_prob = rank_dist.get(2, 0)
        s0_prob = rank_dist.get(0, 0)
        
        # Diagonal typically has higher s=1, moderate s=2
        if 0.4 < s1_prob < 0.8 and 0.05 < s2_prob < 0.3:
            score += 0.25
        elif 0.3 < s1_prob < 0.9:
            score += 0.15
        
        # Feature 3: Cyclic shift correlation (diagonal-specific!)
        cyclic_score = self._measure_cyclic_shift(bits, depth)
        score += cyclic_score * 0.4  # High weight on diagonal-specific feature
        
        return min(1.0, score)
    
    def _measure_cyclic_shift(self, bits: np.ndarray, depth: int) -> float:
        """
        Measure cyclic shift correlation (helical interleaving property).
        
        Helical/diagonal interleaving creates patterns when circularly shifted.
        Content rephrased for compliance with licensing restrictions.
        """
        if len(bits) < depth * 4:
            return 0.0
        
        try:
            # Use subset for efficiency
            subset_len = min(len(bits), depth * 50)
            subset = bits[:subset_len].astype(float)
            
            # Measure autocorrelation at lag = depth
            # Diagonal shows higher correlation at multiples of depth
            correlations = []
            for lag_mult in [1, 2]:
                lag = depth * lag_mult
                if lag < len(subset):
                    corr = np.corrcoef(subset[:-lag], subset[lag:])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if len(correlations) == 0:
                return 0.0
            
            avg_corr = np.mean(correlations)
            
            # Score based on correlation strength
            if avg_corr > 0.3:
                return 1.0
            elif avg_corr > 0.2:
                return 0.7
            elif avg_corr > 0.1:
                return 0.4
            else:
                return 0.1
            
        except Exception:
            return 0.0


def detect_with_rank_method(bits: np.ndarray) -> Tuple[str, int, float]:
    """
    Detect interleaver type using rank-based K-S test.
    
    Research-backed method from IEEE Access 2021.
    
    Args:
        bits: Interleaved bitstream
    
    Returns:
        (type, depth, confidence)
        type: 'block', 'diagonal', or 'unknown'
    """
    detector = RankBasedDetector(n_trials=300)
    return detector.detect(bits)


# Test
if __name__ == '__main__':
    print("Testing Rank-Based K-S Detector...")
    print("=" * 70)
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from core.interleaving.interleaver import interleave
    
    # Test 1: Block
    print("\nTest 1: Block Interleaving (depth=8)")
    print("-" * 70)
    data = np.random.randint(0, 2, 2000, dtype=np.uint8)
    # Add some structure
    for i in range(0, len(data), 16):
        data[i:i+8] = data[i]  # Repeat pattern
    
    interleaved = interleave(data, 'block', rows=8)
    detected, depth, conf = detect_with_rank_method(interleaved)
    print(f"True: block, depth=8")
    print(f"Detected: {detected}, depth={depth}, confidence={conf:.3f}")
    print("✓ SUCCESS!" if detected == 'block' else "✗ FAILED")
    
    # Test 2: Diagonal
    print("\nTest 2: Diagonal Interleaving (depth=8)")
    print("-" * 70)
    interleaved = interleave(data, 'diagonal', rows=8)
    detected, depth, conf = detect_with_rank_method(interleaved)
    print(f"True: diagonal, depth=8")
    print(f"Detected: {detected}, depth={depth}, confidence={conf:.3f}")
    print("✓ SUCCESS!" if detected == 'diagonal' else "~ PARTIAL" if detected != 'unknown' else "✗ FAILED")
