"""
Signal metadata container — holds everything we know about a loaded signal.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SignalInfo:
    """All metadata and results for a loaded signal file."""

    # --- File info ---
    file_path: str = ""
    file_name: str = ""
    file_type: str = ""          # "WAV" or "IQ"
    file_size_bytes: int = 0

    # --- Raw signal ---
    samples: Optional[np.ndarray] = field(default=None, repr=False)
    sample_rate: float = 0.0     # Hz
    num_samples: int = 0
    duration_sec: float = 0.0
    dtype: str = ""              # e.g. "float32", "int16"
    channels: int = 1

    # --- Estimated parameters ---
    center_freq_hz: float = 0.0
    bandwidth_hz: float = 0.0
    snr_db: float = 0.0
    noise_floor_db: float = 0.0
    band: str = ""               # 'HF' | 'VHF' | 'UHF' | 'SHF' | 'EHF'

    # --- Modulation ---
    modulation: str = ""         # e.g. "QPSK"
    mod_confidence: float = 0.0  # 0.0 – 1.0
    mod_top3: list = field(default_factory=list)  # [(mod, conf), ...]

    # --- Demodulation ---
    raw_bits: Optional[np.ndarray] = field(default=None, repr=False)
    bit_rate_bps: float = 0.0
    symbol_rate_hz: float = 0.0

    # --- De-interleaving ---
    interleave_type: str = ""
    deinterleaved_bits: Optional[np.ndarray] = field(default=None, repr=False)

    # --- FEC ---
    fec_type: str = ""
    fec_errors_detected: int = 0
    fec_errors_corrected: int = 0
    fec_auto_confidence: float = 0.0   # confidence of auto-detection (0–1)
    corrected_bits: Optional[np.ndarray] = field(default=None, repr=False)

    # --- Bit correlation ---
    header_bits: Optional[np.ndarray] = field(default=None, repr=False)
    payload_bits: Optional[np.ndarray] = field(default=None, repr=False)
    header_offset: int = 0
    payload_offset: int = 0
    sync_word_name: str = ""

    # --- Protocol analysis ---
    protocol_frames: Optional[list] = field(default=None, repr=False)
    num_frames: int = 0
    protocol_type: str = ""
    protocol_confidence: float = 0.0
    protocol_results: Optional[list] = field(default=None, repr=False)

    # --- Security / crypto analysis ---
    crypto_detected: bool = False
    crypto_confidence: float = 0.0
    crypto_type: str = ""
    crypto_entropy: float = 0.0
    crypto_analysis: Optional[object] = field(default=None, repr=False)

    # --- Traffic analysis ---
    traffic_analysis: Optional[dict] = field(default=None, repr=False)

    def is_loaded(self) -> bool:
        return self.samples is not None and len(self.samples) > 0

    def __iter__(self):
        """Allow tuple unpacking (samples, sample_rate, dtype) for backwards compatibility."""
        yield self.samples
        yield self.sample_rate
        yield self.dtype

    def summary(self) -> dict:
        """Return a clean dict of parameters for the AI overview."""
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "sample_rate_hz": self.sample_rate,
            "duration_sec": round(self.duration_sec, 3),
            "num_samples": self.num_samples,
            "dtype": self.dtype,
            "center_freq_hz": round(self.center_freq_hz, 2),
            "bandwidth_hz": round(self.bandwidth_hz, 2),
            "snr_db": round(self.snr_db, 2),
            "noise_floor_db": round(self.noise_floor_db, 2),
            "band": self.band,
            "modulation": self.modulation,
            "mod_confidence_pct": round(self.mod_confidence * 100, 1),
            "mod_top3": self.mod_top3,
            "bit_rate_bps": round(self.bit_rate_bps, 1),
            "symbol_rate_hz": round(self.symbol_rate_hz, 1),
            "interleave_type": self.interleave_type,
            "fec_type": self.fec_type,
            "fec_errors_detected": self.fec_errors_detected,
            "fec_errors_corrected": self.fec_errors_corrected,
            "header_offset": self.header_offset,
            "payload_offset": self.payload_offset,
            "sync_word_name": self.sync_word_name,
            # protocol
            "num_frames": self.num_frames,
            "protocol_type": self.protocol_type,
            "protocol_confidence_pct": round(self.protocol_confidence * 100, 1),
            # security
            "crypto_detected": self.crypto_detected,
            "crypto_confidence_pct": round(self.crypto_confidence * 100, 1),
            "crypto_type": self.crypto_type,
            "crypto_entropy": round(self.crypto_entropy, 4),
        }
