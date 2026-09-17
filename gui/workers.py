"""
Background workers — QThread-based workers for all heavy DSP tasks.
Keeps the GUI fully responsive at all times.
"""
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class AnalysisWorker(QThread):
    """Runs file loading + parameter estimation + modulation classification."""
    progress = pyqtSignal(int, str)          # percent, message
    finished = pyqtSignal(object)            # SignalInfo
    error = pyqtSignal(str)

    def __init__(self, file_path: str, iq_dtype: str = 'float32',
                 iq_sr: float = 0.0, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.iq_dtype = iq_dtype
        self.iq_sr = iq_sr

    def run(self):
        try:
            from core.chunked_loader import smart_load
            from core.parameter_estimator import (estimate_parameters,
                                                    compute_fft,
                                                    compute_spectrogram)
            from core.modulation.classifier import classify_modulation
            from core.signal_filter import remove_dc, normalize_power

            self.progress.emit(10, "Loading file...")
            info = smart_load(self.file_path, self.iq_dtype, self.iq_sr)

            self.progress.emit(30, "Preprocessing signal...")
            # Remove DC offset and normalize — important for real signals
            info.samples = remove_dc(info.samples)
            info.samples = normalize_power(info.samples)

            self.progress.emit(50, "Estimating parameters...")
            info = estimate_parameters(info)

            self.progress.emit(75, "Classifying modulation...")
            if info.samples is not None and len(info.samples) > 0:
                # Normalize before classification — critical for correct features
                from core.signal_filter import normalize_power, remove_dc
                samples_for_clf = remove_dc(info.samples)
                samples_for_clf = normalize_power(samples_for_clf)
                # Use CNN ensemble: CNN at low SNR, feature-based at high SNR
                try:
                    from core.modulation.cnn_classifier import classify_ensemble
                    results = classify_ensemble(samples_for_clf,
                                                snr_estimate=info.snr_db)
                except Exception:
                    from core.modulation.classifier import classify_modulation
                    results = classify_modulation(samples_for_clf)
                if results:
                    info.modulation = results[0][0]
                    info.mod_confidence = results[0][1]
                    info.mod_top3 = results[:3]

            self.progress.emit(100, "Done")
            self.finished.emit(info)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class DemodulateWorker(QThread):
    """Runs demodulation in background."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)   # updated SignalInfo
    error = pyqtSignal(str)

    def __init__(self, info, modulation: str, parent=None):
        super().__init__(parent)
        self.info = info
        self.modulation = modulation

    def run(self):
        try:
            from core.modulation.demodulator import demodulate
            from core.symbol_rate_estimator import estimate_symbol_rate
            from core.signal_filter import auto_preprocess

            self.progress.emit(10, "Preprocessing signal...")
            # Apply carrier recovery, bandpass filter, DC removal
            preprocessed = auto_preprocess(
                self.info.samples,
                self.info.sample_rate,
                center_freq_hz=self.info.center_freq_hz,
                bandwidth_hz=self.info.bandwidth_hz
            )

            self.progress.emit(30, "Estimating symbol rate...")
            mod = self.modulation
            if mod == 'auto' or not mod:
                mod = self.info.modulation or 'QPSK'

            # Apply Costas loop for PSK/QAM — fixes residual phase rotation
            # on real hardware signals (works silently on all signal types)
            psk_order = {'BPSK': 2, 'QPSK': 4, '8PSK': 8,
                         'QAM16': 2, 'QAM64': 2, 'QAM256': 2}.get(
                mod.upper(), 0)
            if psk_order > 0 and np.iscomplexobj(preprocessed):
                try:
                    from core.signal_filter import costas_loop
                    preprocessed = costas_loop(
                        preprocessed,
                        modulation_order=psk_order,
                        loop_bw=0.005)
                except Exception:
                    pass  # non-fatal — continue without Costas

            # Real symbol rate estimation
            symbol_rate = estimate_symbol_rate(
                preprocessed,
                self.info.sample_rate,
                min_rate=self.info.sample_rate / 512.0,
                max_rate=self.info.sample_rate / 2.0
            )
            self.progress.emit(60, f"Symbol rate: {symbol_rate:.0f} Hz — Demodulating {mod}...")

            bits, bit_rate = demodulate(
                preprocessed, mod,
                self.info.sample_rate,
                symbol_rate=symbol_rate
            )

            self.info.raw_bits = bits
            self.info.bit_rate_bps = bit_rate
            self.info.symbol_rate_hz = symbol_rate

            self.progress.emit(100, "Demodulation complete")
            self.finished.emit(self.info)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class DeinterleaveWorker(QThread):
    """Runs de-interleaving in background."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, info, method: str, rows: int = 8,
                 cols: int = 0, parent=None):
        super().__init__(parent)
        self.info = info
        self.method = method
        self.rows = rows
        self.cols = cols if cols > 0 else None

    def run(self):
        try:
            from core.interleaving.deinterleaver import (deinterleave,
                                                          auto_detect_interleaving)
            bits = self.info.raw_bits
            if bits is None or len(bits) == 0:
                self.error.emit("No bits to de-interleave. Run demodulation first.")
                return

            self.progress.emit(20, "De-interleaving...")

            method = self.method
            if method == 'auto':
                method = auto_detect_interleaving(bits)
                self.info.interleave_type = method
            else:
                self.info.interleave_type = method

            result = deinterleave(bits, method, self.rows, self.cols)
            self.info.deinterleaved_bits = result

            self.progress.emit(100, "De-interleaving complete")
            self.finished.emit(self.info)

        except Exception as e:
            self.error.emit(str(e))


class FECWorker(QThread):
    """Runs FEC decoding in background."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, info, fec_type: str, parent=None):
        super().__init__(parent)
        self.info = info
        self.fec_type = fec_type

    def run(self):
        try:
            from core.fec.decoder import decode_fec
            from core.symbol_rate_estimator import estimate_fec_type

            # Use de-interleaved bits if available, else raw bits
            bits = (self.info.deinterleaved_bits
                    if self.info.deinterleaved_bits is not None
                    else self.info.raw_bits)

            if bits is None or len(bits) == 0:
                self.error.emit("No bits for FEC. Run demodulation first.")
                return

            self.progress.emit(20, "FEC decoding...")

            fec = self.fec_type
            if fec == 'auto':
                # Real heuristic-based FEC detection
                fec = estimate_fec_type(bits)
                self.progress.emit(40, f"Detected FEC: {fec}")

            self.info.fec_type = fec
            corrected, detected, fixed = decode_fec(bits, fec)
            self.info.corrected_bits = corrected
            self.info.fec_errors_detected = detected
            self.info.fec_errors_corrected = fixed

            self.progress.emit(100, "FEC decoding complete")
            self.finished.emit(self.info)

        except Exception as e:
            self.error.emit(str(e))


class CorrelateWorker(QThread):
    """Runs bit stream correlation in background."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, info, sync_name: str, parent=None):
        super().__init__(parent)
        self.info = info
        self.sync_name = sync_name

    def run(self):
        try:
            from core.bitstream.correlator import (correlate_bits,
                                                    KNOWN_SYNC_WORDS,
                                                    auto_correlate)

            bits = (self.info.corrected_bits
                    if self.info.corrected_bits is not None
                    else self.info.deinterleaved_bits
                    if self.info.deinterleaved_bits is not None
                    else self.info.raw_bits)

            if bits is None or len(bits) == 0:
                self.error.emit("No bits to correlate. Run demodulation first.")
                return

            self.progress.emit(20, "Correlating bit stream...")

            if self.sync_name == 'auto':
                corr, h_off, p_off, name = auto_correlate(bits)
            else:
                # Look up in full database (built-in + custom)
                try:
                    from core.bitstream.sync_db import load_all
                    all_words = load_all()
                    if self.sync_name in all_words:
                        sw = all_words[self.sync_name]
                        corr, h_off, p_off, name = correlate_bits(bits, sw)
                        name = self.sync_name
                    else:
                        # Fallback to old KNOWN_SYNC_WORDS
                        if self.sync_name in KNOWN_SYNC_WORDS:
                            sw = KNOWN_SYNC_WORDS[self.sync_name]
                            corr, h_off, p_off, name = correlate_bits(bits, sw)
                            name = self.sync_name
                        else:
                            corr, h_off, p_off, name = auto_correlate(bits)
                except Exception:
                    corr, h_off, p_off, name = auto_correlate(bits)

            self.info.header_offset = h_off
            self.info.payload_offset = p_off
            self.info.sync_word_name = name

            self.progress.emit(100, "Correlation complete")
            self.finished.emit(self.info)

        except Exception as e:
            self.error.emit(str(e))


class PlotDataWorker(QThread):
    """Computes FFT and spectrogram data for plotting."""
    finished = pyqtSignal(object, object, object, object, object)
    # (freqs, power_db, spec_freqs, spec_times, spec_power)
    error = pyqtSignal(str)

    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.info = info

    def run(self):
        try:
            from core.parameter_estimator import compute_fft, compute_spectrogram

            samples = self.info.samples
            sr = self.info.sample_rate

            # FFT
            n_fft = min(len(samples), 8192)
            freqs, power_db = compute_fft(samples, sr, n_fft)

            # Spectrogram
            n_seg = min(256, len(samples) // 8)
            n_seg = max(n_seg, 32)
            spec_freqs, spec_times, spec_power = compute_spectrogram(
                samples, sr, nperseg=n_seg)

            self.finished.emit(freqs, power_db,
                               spec_freqs, spec_times, spec_power)

        except Exception as e:
            self.error.emit(str(e))



class ProtocolAnalysisWorker(QThread):
    """
    Runs protocol analysis and security detection in background.
    Category 3: Protocol/Security Analysis
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)   # updated SignalInfo with protocol results
    error = pyqtSignal(str)

    def __init__(self, info, auto_detect: bool = True, 
                 protocol_hint: str = None, parent=None):
        super().__init__(parent)
        self.info = info
        self.auto_detect = auto_detect
        self.protocol_hint = protocol_hint

    def run(self):
        try:
            from core.protocol.classifier import ProtocolClassifier
            from core.protocol.frame_extractor import extract_frames
            from core.security.crypto_detector import CryptoDetector
            from core.security.traffic_analyzer import analyze_traffic_patterns

            # Get corrected bits (or fallback to deinterleaved or raw)
            bits = (self.info.corrected_bits 
                    if self.info.corrected_bits is not None
                    else self.info.deinterleaved_bits
                    if self.info.deinterleaved_bits is not None
                    else self.info.raw_bits)

            if bits is None or len(bits) == 0:
                raise ValueError("No bits available for protocol analysis")

            # Convert bits to bytes
            self.progress.emit(10, "Converting bits to bytes...")
            # Pad to multiple of 8
            padding = (8 - len(bits) % 8) % 8
            if padding > 0:
                bits = np.concatenate([bits, np.zeros(padding, dtype=np.uint8)])
            
            # Pack bits into bytes
            data_bytes = np.packbits(bits).tobytes()

            # Extract frames
            self.progress.emit(25, "Extracting frames...")
            frames = extract_frames(
                data_bytes, 
                min_frame_size=8,
                max_frame_size=2048
            )
            self.info.protocol_frames = frames
            self.info.num_frames = len(frames)

            # Classify protocol
            self.progress.emit(50, "Classifying protocol...")
            classifier = ProtocolClassifier()
            
            if self.auto_detect:
                # Classify each frame
                frame_results = []
                for frame in frames[:10]:  # Analyze first 10 frames
                    result = classifier.classify(frame)
                    frame_results.append(result)
                
                # Determine most common protocol
                if frame_results:
                    protocols = [r.protocol for r in frame_results]
                    from collections import Counter
                    protocol_counts = Counter(protocols)
                    detected_protocol = protocol_counts.most_common(1)[0][0]
                    confidence = protocol_counts.most_common(1)[0][1] / len(frame_results)
                    
                    self.info.protocol_type = detected_protocol
                    self.info.protocol_confidence = confidence
                    self.info.protocol_results = frame_results
                else:
                    self.info.protocol_type = "Unknown"
                    self.info.protocol_confidence = 0.0
                    self.info.protocol_results = []
            else:
                # Use hint
                self.info.protocol_type = self.protocol_hint or "Unknown"
                self.info.protocol_confidence = 1.0 if self.protocol_hint else 0.0

            # Security analysis
            self.progress.emit(75, "Analyzing security...")
            crypto_detector = CryptoDetector()
            crypto_analysis = crypto_detector.analyze(data_bytes)
            
            self.info.crypto_detected = crypto_analysis.is_encrypted
            self.info.crypto_confidence = crypto_analysis.confidence
            self.info.crypto_type = crypto_analysis.cipher_type_hint
            self.info.crypto_entropy = crypto_analysis.shannon_entropy
            self.info.crypto_analysis = crypto_analysis

            # Traffic analysis (if multiple frames)
            if len(frames) > 1:
                self.progress.emit(90, "Analyzing traffic patterns...")
                traffic_info = analyze_traffic_patterns(
                    frames,
                    timestamps=None  # Use frame indices as proxy
                )
                self.info.traffic_analysis = traffic_info

            self.progress.emit(100, "Protocol analysis complete")
            self.finished.emit(self.info)

        except Exception as e:
            import traceback
            self.error.emit(f"Protocol analysis error: {e}\n{traceback.format_exc()}")
