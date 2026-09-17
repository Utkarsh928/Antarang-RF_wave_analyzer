"""
Chunked Loader — handles large IQ/WAV files without loading them fully into RAM.
Streams chunks for analysis, merges results.
"""
import os
import numpy as np
from typing import Generator, Tuple, Optional
from core.signal_info import SignalInfo


MAX_RAM_BYTES = 512 * 1024 * 1024  # 512 MB threshold before chunking


def should_chunk(file_path: str) -> bool:
    """Return True if file is large enough to require chunking."""
    try:
        return os.path.getsize(file_path) > MAX_RAM_BYTES
    except OSError:
        return False


def get_file_info(file_path: str, dtype: str = 'float32',
                  sample_rate: float = 0.0) -> dict:
    """
    Return file metadata without loading samples.
    """
    size = os.path.getsize(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.wav':
        import wave
        try:
            with wave.open(file_path, 'r') as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_samples = n_frames
                duration = n_samples / sr
        except Exception:
            return {'size': size, 'estimated': True}
        return {
            'sample_rate': float(sr),
            'n_samples': n_samples,
            'num_samples': n_samples,   # alias for compatibility
            'duration_sec': duration,
            'channels': n_channels,
            'bit_depth': sampwidth * 8,
            'size_bytes': size,
            'file_type': 'WAV',
        }
    else:
        # IQ file
        bytes_per_sample = {'float32': 8, 'int16': 4, 'int8': 2,
                             'complex64': 8}.get(dtype, 8)
        n_samples = size // bytes_per_sample
        duration = n_samples / sample_rate if sample_rate > 0 else 0.0
        return {
            'sample_rate': sample_rate,
            'n_samples': n_samples,
            'num_samples': n_samples,   # alias for compatibility
            'duration_sec': duration,
            'channels': 2,
            'dtype': dtype,
            'size_bytes': size,
            'file_type': 'IQ',
        }


def load_chunk_iq(file_path: str, dtype: str = 'float32',
                   offset_samples: int = 0,
                   n_samples: int = 262144) -> np.ndarray:
    """
    Load a chunk of IQ samples from a raw binary file.
    Returns complex64 array.
    """
    np_dtype = np.dtype(dtype)
    if dtype == 'complex64':
        byte_offset = offset_samples * 8
        raw = np.fromfile(file_path, dtype=np_dtype,
                           count=n_samples, offset=byte_offset)
        return raw.astype(np.complex64)
    else:
        # Interleaved I,Q pairs
        bytes_per_pair = np_dtype.itemsize * 2
        byte_offset = offset_samples * bytes_per_pair
        raw = np.fromfile(file_path, dtype=np_dtype,
                           count=n_samples * 2, offset=byte_offset)
        if len(raw) % 2 != 0:
            raw = raw[:-1]
        if dtype == 'float32':
            iq = raw
        elif dtype == 'int16':
            iq = raw.astype(np.float32) / 32768.0
        elif dtype == 'int8':
            iq = raw.astype(np.float32) / 128.0
        else:
            iq = raw.astype(np.float32)
        return (iq[0::2] + 1j * iq[1::2]).astype(np.complex64)


def iter_chunks(file_path: str, dtype: str = 'float32',
                chunk_size: int = 131072,
                overlap: int = 1024) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Iterate over IQ file in overlapping chunks.
    Yields (chunk_index, samples_complex64).
    """
    file_size = os.path.getsize(file_path)
    bytes_per_pair = np.dtype(dtype).itemsize * 2
    if dtype == 'complex64':
        bytes_per_pair = 8
    total_samples = file_size // bytes_per_pair

    offset = 0
    chunk_idx = 0
    while offset < total_samples:
        n = min(chunk_size, total_samples - offset)
        if n < 64:
            break
        chunk = load_chunk_iq(file_path, dtype, offset, n)
        yield chunk_idx, chunk
        offset += max(1, chunk_size - overlap)
        chunk_idx += 1


def load_representative_chunk(file_path: str,
                               dtype: str = 'float32',
                               target_samples: int = 524288) -> np.ndarray:
    """
    Load a representative chunk from a large file for analysis.
    Takes the middle portion (avoids silence at start/end).
    """
    file_size = os.path.getsize(file_path)
    bytes_per_pair = np.dtype(dtype).itemsize * 2
    if dtype == 'complex64':
        bytes_per_pair = 8

    total_samples = file_size // bytes_per_pair

    if total_samples <= target_samples:
        return load_chunk_iq(file_path, dtype, 0, total_samples)

    # Take from 25% into the file to 25%+target
    start = total_samples // 4
    n = min(target_samples, total_samples - start)
    return load_chunk_iq(file_path, dtype, start, n)


def smart_load(file_path: str, dtype: str = 'float32',
               sample_rate: float = 0.0,
               max_samples: int = 2_097_152) -> SignalInfo:
    """
    Smart loader: for small files loads everything,
    for large files loads a representative chunk.
    Always returns a valid SignalInfo.
    """
    from core.file_loader import load_file, detect_file_type

    file_type = detect_file_type(file_path)
    file_size = os.path.getsize(file_path)

    # Under 200MB: load normally
    if file_size < 200 * 1024 * 1024:
        return load_file(file_path, dtype, sample_rate)

    # Large file: load representative chunk
    info = SignalInfo()
    info.file_path = file_path
    info.file_name = os.path.basename(file_path)
    info.file_type = file_type
    info.file_size_bytes = file_size

    if file_type == 'WAV':
        # For large WAV, load middle section
        from scipy.io import wavfile
        sr, data = wavfile.read(file_path)
        info.sample_rate = float(sr)
        info.dtype = str(data.dtype)

        total = len(data)
        start = total // 4
        end = min(start + max_samples, total)
        chunk = data[start:end]

        if chunk.ndim == 2 and chunk.shape[1] >= 2:
            if data.dtype == np.int16:
                chunk = chunk.astype(np.float32) / 32768.0
            info.samples = chunk[:, 0] + 1j * chunk[:, 1]
            info.samples = info.samples.astype(np.complex64)
            info.channels = 2
        else:
            if data.dtype == np.int16:
                chunk = chunk.astype(np.float32) / 32768.0
            info.samples = chunk.astype(np.float32)
            info.channels = 1
    else:
        # Large IQ: load representative chunk
        chunk = load_representative_chunk(file_path, dtype,
                                           target_samples=max_samples)
        info.samples = chunk
        info.sample_rate = float(sample_rate)
        info.dtype = dtype
        info.channels = 2

    info.num_samples = len(info.samples)
    if info.sample_rate > 0:
        info.duration_sec = info.num_samples / info.sample_rate

    return info
