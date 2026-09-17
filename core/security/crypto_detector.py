"""
Cryptographic Data Detector

Implements research-backed encryption detection methods:
1. Shannon entropy analysis (high entropy = likely encrypted)
2. Chi-square randomness test (NIST methodology)
3. Serial correlation analysis
4. Byte frequency distribution analysis
5. NIST SP 800-22 statistical tests (subset)

References:
- "Detecting Encrypted Network Data Using Entropy Analysis" (PMC 2023)
- "Comparison of Entropy Calculation Methods for Ransomware" (arXiv:2210.13376)
- NIST SP 800-22: Statistical Test Suite for Random Number Generators
- "Crypto-Ransomware Detection Using Chi-Square Test" (Springer 2023)

Content rephrased for licensing compliance.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CryptoAnalysis:
    """Results of cryptographic analysis"""
    is_encrypted: bool          # Primary verdict
    confidence: float           # Confidence score (0-1)
    shannon_entropy: float      # Shannon entropy (0-8 bits)
    chi_square_stat: float      # Chi-square statistic
    chi_square_pvalue: float    # P-value (>0.05 = random)
    serial_correlation: float   # Serial correlation (-1 to 1)
    byte_distribution_score: float  # Uniformity score
    cipher_type_hint: str       # Suspected cipher type
    detailed_results: Dict      # Detailed test results


class CryptoDetector:
    """
    Encryption and randomness detector
    
    Uses multiple statistical tests to determine if data is encrypted.
    Well-encrypted data should be indistinguishable from random data.
    """
    
    # Thresholds derived from research literature
    ENTROPY_THRESHOLD = 7.2          # Shannon entropy for encrypted data (>7.2/8.0)
    CHI_SQUARE_PVALUE_MIN = 0.01     # P-value range for randomness
    CHI_SQUARE_PVALUE_MAX = 0.99
    SERIAL_CORRELATION_MAX = 0.1     # Low correlation = random
    
    def __init__(self):
        """Initialize crypto detector"""
        self.expected_byte_freq = 1.0 / 256  # Uniform distribution
    
    def analyze(self, data: bytes) -> CryptoAnalysis:
        """
        Comprehensive cryptographic analysis
        
        Args:
            data: Binary data to analyze
            
        Returns:
            CryptoAnalysis results
        """
        if len(data) < 16:
            # Too short to analyze reliably
            return CryptoAnalysis(
                is_encrypted=False,
                confidence=0.0,
                shannon_entropy=0.0,
                chi_square_stat=0.0,
                chi_square_pvalue=0.0,
                serial_correlation=0.0,
                byte_distribution_score=0.0,
                cipher_type_hint='INSUFFICIENT_DATA',
                detailed_results={}
            )
        
        # Run all tests
        entropy = self.shannon_entropy(data)
        chi_stat, chi_pval = self.chi_square_test(data)
        serial_corr = self.serial_correlation(data)
        byte_dist = self.byte_distribution_uniformity(data)
        
        # Additional NIST-inspired tests
        runs_test_result = self.runs_test(data)
        longest_run = self.longest_run_of_ones(data)
        
        # Aggregate results
        detailed = {
            'entropy': entropy,
            'chi_square': {'statistic': chi_stat, 'pvalue': chi_pval},
            'serial_correlation': serial_corr,
            'byte_uniformity': byte_dist,
            'runs_test': runs_test_result,
            'longest_run': longest_run
        }
        
        # Decision logic (weighted voting)
        scores = []
        
        # Entropy test (high entropy = encrypted)
        if entropy > self.ENTROPY_THRESHOLD:
            scores.append(1.0)
        else:
            scores.append(entropy / 8.0)  # Normalized
        
        # Chi-square test (p-value in middle range = random)
        if self.CHI_SQUARE_PVALUE_MIN < chi_pval < self.CHI_SQUARE_PVALUE_MAX:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Serial correlation (low = random)
        if abs(serial_corr) < self.SERIAL_CORRELATION_MAX:
            scores.append(1.0)
        else:
            scores.append(max(0.0, 1.0 - abs(serial_corr)))
        
        # Byte distribution uniformity (high = random)
        scores.append(byte_dist)
        
        # Runs test (should be close to expected for random data)
        if 0.4 < runs_test_result < 0.6:  # Expected ~0.5 for random
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Average confidence
        confidence = np.mean(scores)
        is_encrypted = confidence > 0.7  # Threshold for encryption verdict
        
        # Infer cipher type
        cipher_hint = self._infer_cipher_type(entropy, serial_corr, byte_dist, len(data))
        
        return CryptoAnalysis(
            is_encrypted=is_encrypted,
            confidence=confidence,
            shannon_entropy=entropy,
            chi_square_stat=chi_stat,
            chi_square_pvalue=chi_pval,
            serial_correlation=serial_corr,
            byte_distribution_score=byte_dist,
            cipher_type_hint=cipher_hint,
            detailed_results=detailed
        )
    
    def shannon_entropy(self, data: bytes) -> float:
        """
        Calculate Shannon entropy in bits
        
        H = -Σ(p_i * log2(p_i))
        
        Returns:
            Entropy value (0 to 8 bits for bytes)
        """
        if len(data) == 0:
            return 0.0
        
        # Count byte frequencies
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        
        # Calculate probabilities
        probabilities = byte_counts / len(data)
        
        # Remove zeros (log undefined for 0)
        probabilities = probabilities[probabilities > 0]
        
        # Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return float(entropy)
    
    def chi_square_test(self, data: bytes) -> Tuple[float, float]:
        """
        Chi-square goodness-of-fit test for uniformity
        
        Tests null hypothesis: data is uniformly distributed
        High p-value (>0.05) = cannot reject uniform distribution (likely random/encrypted)
        
        Based on NIST methodology for randomness testing.
        
        Returns:
            (chi_square_statistic, p_value)
        """
        if len(data) < 256:
            # Need sufficient data for reliable test
            return (0.0, 0.0)
        
        # Observed frequencies
        observed = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        
        # Expected frequency (uniform distribution)
        expected = np.full(256, len(data) / 256.0)
        
        # Chi-square statistic: Σ((O_i - E_i)² / E_i)
        chi_square = np.sum((observed - expected) ** 2 / expected)
        
        # Degrees of freedom = 256 - 1
        dof = 255
        
        # P-value from chi-square distribution
        p_value = 1.0 - stats.chi2.cdf(chi_square, dof)
        
        return (float(chi_square), float(p_value))
    
    def serial_correlation(self, data: bytes) -> float:
        """
        Calculate serial correlation (lag-1 autocorrelation)
        
        Measures correlation between consecutive bytes.
        Encrypted data should have near-zero correlation.
        
        Returns:
            Correlation coefficient (-1 to 1, 0 = no correlation)
        """
        if len(data) < 2:
            return 0.0
        
        arr = np.frombuffer(data, dtype=np.uint8).astype(float)
        
        if len(arr) < 2:
            return 0.0
        
        # Calculate lag-1 autocorrelation
        mean = np.mean(arr)
        
        numerator = np.sum((arr[:-1] - mean) * (arr[1:] - mean))
        denominator = np.sum((arr - mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        
        return float(correlation)
    
    def byte_distribution_uniformity(self, data: bytes) -> float:
        """
        Measure uniformity of byte distribution
        
        Perfect uniformity (all bytes appear equally) = 1.0
        Non-uniform (some bytes more common) = closer to 0.0
        
        Returns:
            Uniformity score (0-1)
        """
        if len(data) == 0:
            return 0.0
        
        # Count unique bytes
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        
        # Calculate standard deviation of frequencies
        expected_count = len(data) / 256.0
        std_dev = np.std(byte_counts)
        
        # Normalize: low std dev = uniform, high std dev = non-uniform
        # For perfectly uniform: std_dev ≈ sqrt(expected_count)
        # For highly non-uniform: std_dev >> sqrt(expected_count)
        
        expected_std = np.sqrt(expected_count)
        
        if expected_std == 0:
            return 0.0
        
        # Score: 1.0 = perfect uniformity, 0.0 = highly non-uniform
        uniformity = np.exp(-abs(std_dev - expected_std) / expected_std)
        
        return float(uniformity)
    
    def runs_test(self, data: bytes) -> float:
        """
        Runs test (NIST SP 800-22)
        
        A "run" is a sequence of identical bits.
        Random data should have a balanced number of runs.
        
        Returns:
            Normalized runs score (0-1, ~0.5 expected for random)
        """
        # Convert bytes to bits
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        
        if len(bits) < 100:
            return 0.5  # Insufficient data
        
        # Count runs
        runs = 1
        for i in range(1, len(bits)):
            if bits[i] != bits[i-1]:
                runs += 1
        
        # Expected runs for random sequence
        n = len(bits)
        ones = np.sum(bits)
        zeros = n - ones
        
        if ones == 0 or zeros == 0:
            return 0.0
        
        expected_runs = ((2 * ones * zeros) / n) + 1
        
        # Normalize
        if expected_runs == 0:
            return 0.0
        
        normalized = runs / expected_runs
        
        return float(min(normalized, 1.0))
    
    def longest_run_of_ones(self, data: bytes) -> int:
        """
        Find longest run of consecutive 1 bits (NIST SP 800-22)
        
        Random data should not have extremely long runs.
        
        Returns:
            Length of longest run
        """
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        
        max_run = 0
        current_run = 0
        
        for bit in bits:
            if bit == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        return int(max_run)
    
    def _infer_cipher_type(
        self,
        entropy: float,
        correlation: float,
        uniformity: float,
        data_length: int
    ) -> str:
        """
        Attempt to infer cipher type from statistical properties
        
        Note: This is heuristic and not definitive
        """
        if entropy < 6.0:
            return 'PLAINTEXT'
        
        if entropy > 7.9 and uniformity > 0.9 and abs(correlation) < 0.05:
            # Very high quality randomness
            if data_length % 16 == 0:
                return 'AES-128/256 (Block Cipher)'
            else:
                return 'STREAM_CIPHER (ChaCha20/Salsa20)'
        
        if 7.2 < entropy < 7.9:
            # Good but not perfect randomness
            if data_length % 8 == 0:
                return 'DES/3DES or Weak Encryption'
            else:
                return 'COMPRESSED or Weak Cipher'
        
        if entropy > 7.0 and uniformity < 0.7:
            return 'COMPRESSED (not encrypted)'
        
        return 'UNKNOWN'
    
    def compare_samples(self, data1: bytes, data2: bytes) -> Dict[str, float]:
        """
        Compare two data samples for similarity
        
        Useful for detecting if different blocks use same encryption key.
        
        Returns:
            Similarity metrics
        """
        result1 = self.analyze(data1)
        result2 = self.analyze(data2)
        
        # Calculate differences
        entropy_diff = abs(result1.shannon_entropy - result2.shannon_entropy)
        correlation_diff = abs(result1.serial_correlation - result2.serial_correlation)
        
        return {
            'entropy_difference': entropy_diff,
            'correlation_difference': correlation_diff,
            'both_encrypted': result1.is_encrypted and result2.is_encrypted,
            'similarity_score': 1.0 - (entropy_diff / 8.0 + correlation_diff) / 2.0
        }


def demonstrate_crypto_detection():
    """Demonstration of encryption detection"""
    print("=== Crypto Detector Demo ===\n")
    
    detector = CryptoDetector()
    
    # Test 1: Plaintext
    plaintext = b"Hello World! This is plaintext data that should not appear encrypted."
    result = detector.analyze(plaintext)
    
    print("Test 1: Plaintext")
    print(f"  Encrypted: {result.is_encrypted}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Entropy: {result.shannon_entropy:.3f} / 8.0")
    print(f"  Chi-square p-value: {result.chi_square_pvalue:.3f}")
    print(f"  Cipher hint: {result.cipher_type_hint}\n")
    
    # Test 2: Random data (simulates encryption)
    random_data = np.random.bytes(256)
    result = detector.analyze(random_data)
    
    print("Test 2: Random Data (Simulated Encryption)")
    print(f"  Encrypted: {result.is_encrypted}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Entropy: {result.shannon_entropy:.3f} / 8.0")
    print(f"  Chi-square p-value: {result.chi_square_pvalue:.3f}")
    print(f"  Serial correlation: {result.serial_correlation:.3f}")
    print(f"  Cipher hint: {result.cipher_type_hint}\n")
    
    # Test 3: Repeated pattern (not encrypted)
    pattern = b"\x00\xFF" * 128
    result = detector.analyze(pattern)
    
    print("Test 3: Repeated Pattern")
    print(f"  Encrypted: {result.is_encrypted}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Entropy: {result.shannon_entropy:.3f} / 8.0")
    print(f"  Byte uniformity: {result.byte_distribution_score:.3f}")
    print(f"  Cipher hint: {result.cipher_type_hint}\n")


if __name__ == '__main__':
    demonstrate_crypto_detection()
