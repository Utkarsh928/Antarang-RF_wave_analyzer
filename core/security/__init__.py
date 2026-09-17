"""
Category 3: Security & Encryption Analysis

This module provides encryption detection and cryptographic analysis.
Research-backed implementation using:
- Shannon entropy analysis
- Chi-square randomness test (NIST methodology)
- NIST SP 800-22 statistical test suite
- Serial correlation analysis
"""

from .crypto_detector import CryptoDetector
from .traffic_analyzer import TrafficAnalyzer

__all__ = [
    'CryptoDetector',
    'TrafficAnalyzer'
]
