"""
Category 3: Protocol & Application Layer Analysis

This module provides protocol identification, frame extraction, and payload decoding.
Research-backed implementation using:
- Deep learning-based frame synchronization (arXiv:2601.05920)
- Binary protocol reverse engineering (NEMESYS, arXiv:2002.03391)
- Automatic state machine inference (arXiv:2412.02540)
"""

from .frame_extractor import FrameExtractor
from .classifier import ProtocolClassifier
from .decoder import PayloadDecoder
from .validator import CRCValidator

__all__ = [
    'FrameExtractor',
    'ProtocolClassifier', 
    'PayloadDecoder',
    'CRCValidator'
]
