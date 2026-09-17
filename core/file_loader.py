"""
File Loader — detects WAV vs IQ via magic bytes, loads samples into numpy array.
Supports: WAV (8/16/32-bit, mono/stereo), IQ (float32, int16, int8, complex64)
"""
import os
import struct
import wave
import numpy as np
from scipy.io import wavfile
from core.signal_info import SignalInfo


# WAV magic: bytes 0-3 == b'RIFF', bytes 8-11 == b'WAVE'
_WAV_RIFF = b'RIFF'
_WAV_WAVE = b'WAVE'


def detect_file_type(file_path: str) -> str:
    """
    Returns 'WAV' or 'IQ' by inspecting the first 12 bytes.
    Falls back to extension check if file is too short.
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)
        if len(header) >= 12:
            if header[0:4] == _WAV_RIFF and header[8:12] == _WAV_WAVE:
                return 'WAV'
    except OSError:
        pass

    # Extension fallback
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.wav',):
        return 'WAV'
    return 'IQ'


def load_wav(file_path: str) -> SignalInfo:
    """Load a WAV file — handles mono/stereo, 8/16/32-bit PCM and float."""
    info = SignalInfo()
    info.file_path = file_path
    info.file_name = os.path.basename(file_path)
    info.file_type = 'WAV'
    info.file_size_bytes = os.path.getsize(file_path)

    try:
        sample_rate, data = wavfile.read(file_path)
    except Exception as e:
        raise ValueError(f"Cannot read WAV file: {e}")

    info.sample_rate = float(sample_rate)
    info.dtype = str(data.dtype)

    # Convert to float32 in range [-1, 1]
    if data.dtype == np.int8:
        samples = data.astype(np.float32) / 128.0
    elif data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        samples = data.astype(np.float32) / 2147483648.0
    elif data.dtype in (np.float32, np.float64):
        samples = data.astype(np.float32)
    else:
        samples = data.astype(np.float32)

    # Handle stereo — treat as IQ (left=I, right=Q) or take left channel
    if samples.ndim == 2:
        info.channels = samples.shape[1]
        if info.channels >= 2:
            # Stereo WAV with IQ data: ch0=I, ch1=Q → complex
            i_ch = samples[:, 0]
            q_ch = samples[:, 1]
            info.samples = i_ch + 1j * q_ch
        else:
            info.samples = samples[:, 0]
    else:
        info.channels = 1
        info.samples = samples

    info.num_samples = len(info.samples)
    info.duration_sec = info.num_samples / info.sample_rate
    return info


def load_iq(file_path: str, dtype: str = 'float32',
            sample_rate: float = 0.0) -> SignalInfo:
    """
    Load a raw IQ file.
    dtype options: 'float32' (default), 'int16', 'int8', 'complex64'
    Data is stored as interleaved I,Q pairs except for complex64.
    """
    info = SignalInfo()
    info.file_path = file_path
    info.file_name = os.path.basename(file_path)
    info.file_type = 'IQ'
    info.file_size_bytes = os.path.getsize(file_path)
    info.dtype = dtype
    info.sample_rate = float(sample_rate)
    info.channels = 2

    raw = np.fromfile(file_path, dtype=np.dtype(dtype))

    if dtype == 'complex64':
        info.samples = raw.astype(np.complex64)
    elif dtype == 'float32':
        if len(raw) % 2 != 0:
            raw = raw[:-1]
        info.samples = raw[0::2] + 1j * raw[1::2]
        info.samples = info.samples.astype(np.complex64)
    elif dtype == 'int16':
        if len(raw) % 2 != 0:
            raw = raw[:-1]
        iq = raw.astype(np.float32) / 32768.0
        info.samples = iq[0::2] + 1j * iq[1::2]
        info.samples = info.samples.astype(np.complex64)
    elif dtype == 'int8':
        if len(raw) % 2 != 0:
            raw = raw[:-1]
        iq = raw.astype(np.float32) / 128.0
        info.samples = iq[0::2] + 1j * iq[1::2]
        info.samples = info.samples.astype(np.complex64)
    else:
        raise ValueError(f"Unsupported IQ dtype: {dtype}")

    info.num_samples = len(info.samples)
    if info.sample_rate > 0:
        info.duration_sec = info.num_samples / info.sample_rate
    return info


def load_file(file_path: str, iq_dtype: str = 'float32',
              iq_sample_rate: float = 0.0) -> SignalInfo:
    """
    Top-level loader. Auto-detects WAV vs IQ and loads accordingly.
    Returns a populated SignalInfo object.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_type = detect_file_type(file_path)

    if file_type == 'WAV':
        return load_wav(file_path)
    else:
        return load_iq(file_path, dtype=iq_dtype, sample_rate=iq_sample_rate)


# Alias for backwards compatibility
load_signal = load_file

