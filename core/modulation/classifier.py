"""
Modulation Classifier — identifies modulation type from IQ samples.
Primary: ML ensemble (GradientBoosting + RandomForest + MLP) — 88% accuracy.
Fallback: Statistical rule-based classifier — works without any training.

Auto-loads ML model from ml_model.pkl if available (train with train_model.py).
"""
import numpy as np
from typing import List, Tuple, Optional

# Lazy-loaded ML model (loaded once, cached)
_ml_model_data = None
_ml_model_checked = False
_ml_model_mtime  = 0.0   # track file mtime for auto-reload

# Supported modulation types (rule-based fallback list)
MODULATIONS = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64',
               'FSK2', 'FSK4', 'GFSK', 'CPFSK', 'AM-DSB',
               'AM-SSB', 'WBFM', 'APSK16', 'APSK32', 'PAM4', 'OOK', 'OFDM']


def _get_ml_model():
    """Lazy-load ML model on first use. Auto-reloads if pkl file is newer."""
    global _ml_model_data, _ml_model_checked, _ml_model_mtime
    import os
    try:
        from core.modulation.ml_classifier import MODEL_PATH
        cur_mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0.0
    except Exception:
        cur_mtime = 0.0

    # ALWAYS reload if file has changed (check mtime every time)
    if _ml_model_checked and cur_mtime > 0 and cur_mtime <= _ml_model_mtime:
        return _ml_model_data

    # (Re)load
    _ml_model_checked = True
    _ml_model_mtime   = cur_mtime
    try:
        from core.modulation.ml_classifier import load_model
        _ml_model_data = load_model()
    except Exception:
        _ml_model_data = None
    return _ml_model_data


def _extract_symbol_segment(samples: np.ndarray,
                              sample_rate: float = None,
                              n_syms: int = 128,
                              target_sps: int = 8) -> np.ndarray:
    """
    Extract a symbol-aligned segment from the signal.

    RadioML training data has ~8-16 samples/symbol, 128-1024 symbols.
    Our signals have 40 samples/symbol at 48kHz.
    Downsampling to match training scale dramatically improves classification.

    Returns a complex64 array of length n_syms * target_sps.
    """
    n = len(samples)
    target_len = n_syms * target_sps  # = 1024 by default

    if n <= target_len:
        return samples

    # Estimate decimation factor: take a window of ~target_len symbols
    # from the middle, decimated to target_sps
    # If signal has 40 sps and we want 8 sps, decimate by 5
    decimate_factor = max(1, n // (target_len * 4))

    if decimate_factor > 1:
        from scipy.signal import decimate
        if np.iscomplexobj(samples):
            r = decimate(samples.real.astype(np.float64),
                         decimate_factor, ftype='fir', zero_phase=True)
            i = decimate(samples.imag.astype(np.float64),
                         decimate_factor, ftype='fir', zero_phase=True)
            decimated = (r + 1j * i).astype(np.complex64)
        else:
            decimated = decimate(samples.astype(np.float64),
                                  decimate_factor, ftype='fir',
                                  zero_phase=True).astype(np.float32)
    else:
        decimated = samples

    # Take middle target_len samples
    n2 = len(decimated)
    if n2 > target_len:
        start = (n2 - target_len) // 2
        return decimated[start:start + target_len].astype(np.complex64)
    return decimated.astype(np.complex64)


def classify_modulation(samples: np.ndarray,
                         n_samples: int = 8192,
                         sample_rate: float = None) -> List[Tuple[str, float]]:
    """
    Classify modulation type from IQ samples.
    Returns list of (modulation_name, confidence) sorted by confidence descending.

    Uses ML model if trained (run train_model.py first).
    Falls back to rule-based if model not available.
    """
    # Always normalize before classification
    if not np.iscomplexobj(samples):
        samples = samples.astype(np.float32) + 0j
    pwr = np.mean(np.abs(samples) ** 2)
    if pwr > 1e-12:
        samples = samples / np.sqrt(pwr)
    # Remove DC only if significant carrier/LO leakage offset is present
    dc_offset = np.mean(samples)
    if abs(dc_offset) > 0.30:
        samples = samples - dc_offset

    # Use middle segment
    n_samples = int(n_samples) if n_samples is not None else 8192
    if len(samples) > n_samples:
        start = int((len(samples) - n_samples) // 2)
        seg = samples[start:start + n_samples]
    else:
        seg = samples
    seg_full = seg  # use same segment for both ML and rule-based

    # ─── Spectral symmetry pre-check ─────────────────────────────────────
    # USB receiver recordings have asymmetric spectra (ratio >> 1).
    # True IQ signals from SDR are spectrally symmetric.
    # Rule-based classifier is UNRELIABLE on asymmetric (USB audio) captures.
    n_fft = min(len(seg), 512)
    spec_pow = np.abs(np.fft.fft(seg[:n_fft])) ** 2
    pos_pw = float(np.sum(spec_pow[1:n_fft//2]))
    neg_pw = float(np.sum(spec_pow[n_fft//2+1:]))
    total_pw = pos_pw + neg_pw + 1e-12
    spectral_asymmetry = abs(pos_pw - neg_pw) / total_pw
    # Symmetric: asymmetry < 0.20 (true IQ). Asymmetric: > 0.30 (USB audio).
    is_symmetric_spectrum = spectral_asymmetry < 0.20

    # Try ML model first — uses decimated segment at RadioML scale
    model_data = _get_ml_model()
    if model_data is not None:
        try:
            from core.modulation.ml_classifier import predict
            ml_results = predict(seg, model_data=model_data)
            ml_top_mod  = ml_results[0][0]  if ml_results else ''
            ml_top_conf = ml_results[0][1]  if ml_results else 0.0

            # If ML confidence is high (>= 55%), trust ML directly
            if ml_top_conf >= 0.55:
                return ml_results

            # Symmetric spectrum + strong confidence
            if is_symmetric_spectrum and ml_top_conf >= 0.50:
                return ml_results

            # ── Hybrid blend (moderate confidence / consensus needed) ─────
            rb_results  = _rule_based_classify(seg_full)
            rb_top_mod  = rb_results[0][0] if rb_results else ''
            rb_top_conf = rb_results[0][1] if rb_results else 0.0

            # If rule-based top score is strong (> 0.25) AND ML agrees
            if rb_top_conf > 0.25 and rb_top_mod == ml_top_mod:
                return rb_results

            # Fair, uniform blend across all classes: 65% ML + 35% rule-based
            combined = {}
            for mod, conf in ml_results:
                combined[mod] = conf * 0.65
            for mod, conf in rb_results:
                combined[mod] = combined.get(mod, 0.0) + conf * 0.35
            total = sum(combined.values()) + 1e-12
            return sorted([(m, v/total) for m, v in combined.items()],
                          key=lambda x: x[1], reverse=True)
        except Exception:
            pass

    # Fallback: rule-based only
    return _rule_based_classify(seg_full)


def _extract_features(samples: np.ndarray) -> dict:
    """Extract higher-order statistics and spectral features from IQ samples."""
    if not np.iscomplexobj(samples):
        samples = samples + 0j

    # Normalize
    pwr = np.mean(np.abs(samples) ** 2)
    if pwr > 0:
        samples = samples / np.sqrt(pwr)

    n = len(samples)
    amp = np.abs(samples)
    phase = np.unwrap(np.angle(samples))

    # --- Amplitude statistics ---
    amp_mean = float(np.mean(amp))
    amp_var = float(np.var(amp))
    amp_kurt = float(_kurtosis(amp))

    # --- Phase statistics ---
    phase_diff = np.diff(phase)
    phase_var = float(np.var(phase_diff))
    phase_kurt = float(_kurtosis(phase_diff))

    # --- Higher-order cumulants (use abs to keep real-valued) ---
    c20 = complex(np.mean(samples ** 2))
    c21 = float(np.mean(np.abs(samples) ** 2))
    c40 = complex(np.mean(samples ** 4))
    c41 = complex(np.mean(np.abs(samples) ** 2 * samples ** 2))
    c42 = float(np.mean(np.abs(samples) ** 4)) - 2 * c21 ** 2 - abs(c20) ** 2

    # --- Spectral features ---
    n_fft = min(n, 1024)
    spec = np.abs(np.fft.fft(samples[:n_fft])) ** 2
    spec_norm = spec / (np.sum(spec) + 1e-12)
    spec_entropy = float(-np.sum(spec_norm * np.log2(spec_norm + 1e-12)))

    # Spectral flatness
    geom_mean = float(np.exp(np.mean(np.log(spec + 1e-12))))
    arith_mean = float(np.mean(spec) + 1e-12)
    spec_flatness = geom_mean / arith_mean

    return {
        'amp_var': amp_var,
        'amp_kurt': amp_kurt,
        'amp_mean': amp_mean,
        'phase_var': phase_var,
        'phase_kurt': phase_kurt,
        'c20_abs': abs(c20),
        'c40_abs': abs(c40),
        'c42': c42,
        'spec_entropy': spec_entropy,
        'spec_flatness': spec_flatness,
    }


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis."""
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4) - 3.0)


def _count_phase_clusters(samples: np.ndarray, n_bins: int = 36) -> int:
    """Count distinct phase state clusters using histogram peak detection."""
    if not np.iscomplexobj(samples):
        return 1
    amp = np.abs(samples)
    threshold = np.percentile(amp, 30)
    strong = samples[amp > threshold]
    if len(strong) < 20:
        return 1
    phases = np.angle(strong)
    hist, _ = np.histogram(phases, bins=n_bins, range=(-np.pi, np.pi))
    hist_norm = hist.astype(float) / (hist.max() + 1e-12)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(hist_norm, height=0.3, distance=2)
    return max(1, len(peaks))


def _compute_if_bimodality(samples: np.ndarray) -> float:
    """Physical bimodality coefficient of instantaneous frequency."""
    if len(samples) < 10:
        return 0.0
    phase = np.unwrap(np.angle(samples))
    inst_freq = np.diff(phase) / (2 * np.pi)
    from core.modulation.feature_extractor import _bimodality_coefficient
    return _bimodality_coefficient(inst_freq)


def _compute_if_tone_count(samples: np.ndarray) -> int:
    """
    Count discrete tones in the instantaneous frequency spectrum.
    FSK signals have 2 or 4 distinct IF tones; PSK has continuous IF.
    Returns estimated number of IF frequency states.
    """
    if not np.iscomplexobj(samples):
        return 0
    phase = np.unwrap(np.angle(samples))
    inst_freq = np.diff(phase)
    if len(inst_freq) < 32:
        return 0

    # Histogram of instantaneous frequency
    n_bins = 50
    hist, edges = np.histogram(inst_freq, bins=n_bins)
    hist = hist.astype(float)
    hist_norm = hist / (hist.max() + 1e-12)

    from scipy.signal import find_peaks
    # FSK: 2-4 sharp peaks; PSK: spread distribution
    peaks, props = find_peaks(hist_norm, height=0.25, distance=3)
    return len(peaks)


def _compute_phase_transition_rate(samples: np.ndarray,
                                    sps_estimate: int = 8) -> float:
    """
    Estimate phase transition rate (transitions per symbol).
    BPSK: 0.5 transitions/symbol (every other symbol)
    QPSK: ~1 transition/symbol
    8PSK: ~1.5 transitions/symbol
    FSK: frequent small transitions
    """
    if not np.iscomplexobj(samples):
        return 0.0
    phase = np.unwrap(np.angle(samples))
    # Downsample to ~symbol rate: look at every sps_estimate samples
    phase_ds = phase[::sps_estimate]
    transitions = np.abs(np.diff(phase_ds))
    # Count significant phase changes (> pi/4 = 45 degrees)
    significant = np.sum(transitions > np.pi / 4)
    rate = float(significant) / max(len(phase_ds) - 1, 1)
    return rate


def _rule_based_classify(samples: np.ndarray) -> List[Tuple[str, float]]:
    """
    Rule-based classifier using amplitude, phase, IF, and spectral features.

    Calibrated against real-world SDRAngel IQ captures (Sept 2026):
      - radar.sdriq  : AM-SSB  (amp_kurt≈3320, amp_var≈0.96, iq_ratio≈0.65)
      - no84.wav     : GFSK    (amp_var≈0.047, n_if_tones≈11, spec_flat≈0.56)
      - dsd.wav      : 4FSK    (amp_var≈0.002, n_if_tones≈11, spec_flat≈0.44)
    """
    if not np.iscomplexobj(samples):
        samples = samples + 0j

    pwr = np.mean(np.abs(samples) ** 2)
    if pwr > 1e-12:
        samples = samples / np.sqrt(pwr)
    samples = samples - np.mean(samples)

    n = min(len(samples), 8192)
    seg = samples[:n]

    amp        = np.abs(seg)
    phase      = np.unwrap(np.angle(seg))
    inst_freq  = np.diff(phase)

    # ── Core statistics ──────────────────────────────────────────────────
    amp_var   = float(np.var(amp))
    amp_kurt  = float(_kurtosis(amp))
    amp_mean  = float(np.mean(amp))

    c21 = float(np.mean(np.abs(seg)**2))
    c42 = float(np.mean(np.abs(seg)**4) - 2*c21**2 - abs(np.mean(seg**2))**2)
    c20 = float(np.abs(np.mean(seg**2)))

    # ── Spectral ─────────────────────────────────────────────────────────
    n_fft = min(n, 4096)
    spec      = np.abs(np.fft.fft(seg[:n_fft]))**2
    spec_norm = spec / (spec.sum() + 1e-12)
    spec_entropy  = float(-np.sum(spec_norm * np.log2(spec_norm + 1e-12)))
    geom          = float(np.exp(np.mean(np.log(spec + 1e-12))))
    spec_flatness = geom / (float(np.mean(spec)) + 1e-12)

    # ── Instantaneous frequency ──────────────────────────────────────────
    if_std   = float(np.std(inst_freq))
    if_range = float(np.max(inst_freq) - np.min(inst_freq))
    if_var   = float(np.var(inst_freq))

    # ── I/Q asymmetry (high for SSB / PAM4, low for balanced IQ mods) ───
    real_pwr = float(np.mean(seg.real**2))
    imag_pwr = float(np.mean(seg.imag**2))
    iq_ratio = float(abs(real_pwr - imag_pwr) / (real_pwr + imag_pwr + 1e-12))

    phase_var   = float(np.var(inst_freq))
    ph_mean_abs = float(np.mean(np.abs(inst_freq)))

    # ── Higher-level features ────────────────────────────────────────────
    n_if_tones = _compute_if_tone_count(seg)
    if_bmc     = _compute_if_bimodality(seg)

    # Constant-envelope threshold: raised to 0.06 so that borderline GFSK
    # captures (amp_var ≈ 0.047) correctly land in the const-env branch.
    is_const_env = bool(amp_var < 0.06)

    scores = {}

    # ════════════════════════════════════════════════════════════════════
    # CONSTANT-ENVELOPE branch  (PSK, FSK, GFSK, CPFSK, WBFM)
    # ════════════════════════════════════════════════════════════════════
    if is_const_env:

        # ── FM family: discriminate by n_if_tones + spectral shape ──────
        #
        # Real-world observations (SDRAngel captures):
        #   WBFM  broadcast FM : n_if_tones 5-8,  spec_flat 0.01-0.04
        #   GFSK  AX.25/packet : n_if_tones 8-13, spec_flat 0.40-0.65
        #   4FSK  C4FM/YSF     : n_if_tones 8-13, spec_flat 0.35-0.55
        #   2FSK               : n_if_tones 1-3,  spec_flat varies
        #   CPFSK              : n_if_tones 1-3,  c20 high
        #
        # Key insight: real GFSK/4FSK IQ captures have FLAT spectra
        # (spec_flatness 0.3–0.7) because the FM carrier sweeps the full
        # bandwidth evenly.  Synthetic training data has much lower flatness.
        # The old thresholds (0.03–0.10) were calibrated only on synthetic.

        # WBFM: broadcast FM — few IF tones, low spectral flatness
        scores['WBFM'] = _score([
            _in_range(n_if_tones,    5.0,  8.0),
            _in_range(spec_entropy,  5.0,  8.5),
            _in_range(spec_flatness, 0.01, 0.06),  # real WBFM: 0.028
        ])

        # GFSK: Gaussian FSK (AX.25, Bluetooth) — many IF tones, FLAT spectrum
        # Measured on no84.wav: n_if_tones=11, spec_flat=0.556, spec_ent=11.36
        scores['GFSK'] = _score([
            _in_range(n_if_tones,    7.0,  14.0),  # many due to Gaussian smoothing
            _in_range(spec_entropy,  9.0,  13.0),
            _in_range(spec_flatness, 0.30,  0.70),  # real GFSK: 0.35–0.65
        ])

        # 4FSK / C4FM: 4-level FSK (YSF digital voice) — many IF tones, flat
        # Measured on dsd.wav: n_if_tones=11, spec_flat=0.441, spec_ent=10.06
        scores['FSK4'] = _score([
            _in_range(n_if_tones,    7.0,  14.0),
            _in_range(spec_entropy,  8.5,  12.5),
            _in_range(spec_flatness, 0.25,  0.60),  # real 4FSK: 0.35–0.55
        ])


        # 2FSK: binary FSK — requires verified physical 2-tone bimodality
        scores['FSK2'] = _score([
            _in_range(if_bmc, 0.25, 1.0),        # Physically verified 2-tone bimodality
            _in_range(n_if_tones, 1.0, 4.0),     # 1-4 IF tones
            _in_range(spec_entropy, 3.0, 10.0),  # Broader entropy range
            _in_range(if_var, 0.005, 0.5),       # Add IF variance check
            _in_range(c42, -1.20, 0.2),          # Exclude BPSK (c42 < -1.25)
        ])

        # CPFSK: continuous-phase FSK — 2 tones, elevated c20
        scores['CPFSK'] = _score([
            _in_range(n_if_tones, 1.5, 3.5),
            _in_range(c20, 0.3, 0.9),
        ])

        # PSK family: 1-2 IF tones, strictly unimodal IF distribution, characteristic C42
        scores['BPSK'] = _score([
            _in_range(n_if_tones, 0.0, 2.0),
            _in_range(if_bmc, 0.0, 0.20),        # Strictly unimodal IF (rejects FSK2)
            _in_range(c42, -2.5, -1.25),         # Pure BPSK cumulant signature
        ])

        scores['8PSK'] = _score([
            _in_range(n_if_tones, 0.0, 2.0),
            _in_range(if_bmc, 0.0, 0.20),
            _in_range(c42, -0.75, -0.15),
        ])

        scores['QPSK'] = _score([
            _in_range(n_if_tones, 0.0, 2.0),
            _in_range(if_bmc, 0.0, 0.20),
            _in_range(c42, -1.25, -0.65),
        ])

    # ════════════════════════════════════════════════════════════════════
    # VARYING-AMPLITUDE branch  (QAM, AM, OOK, PAM4, APSK, OFDM)
    # ════════════════════════════════════════════════════════════════════
    else:
        # ── AM-SSB ────────────────────────────────────────────────────────
        # Real SSB IQ capture characteristics (radar.sdriq):
        #   amp_kurt ≈ 3320  (extremely impulsive — radar returns / voice peaks)
        #   amp_var  ≈ 0.96  (high amplitude variation)
        #   iq_ratio ≈ 0.65  (strong I/Q asymmetry — single sideband)
        #   n_if_tones = 1   (single audio carrier / narrow IF content)
        #   c42 can be very large positive (impulsive signal)
        #
        # High amp_kurt (>5) + high iq_ratio (>0.3) uniquely identifies SSB.
        scores['AM-SSB'] = _score([
            _in_range(amp_kurt,    5.0, 1e7),   # impulsive / peaky audio
            _in_range(iq_ratio,    0.3, 1.0),   # I/Q power asymmetry
            _in_range(amp_var,     0.1, 1.5),   # varying amplitude
            _in_range(n_if_tones,  0.0, 3.0),   # few IF tones (audio / CW)
        ])

        # ── FSK/GFSK fallback for borderline const-envelope signals ──────
        # Some real GFSK captures (e.g. no84.wav with amp_var=0.047) barely
        # exceed the const-envelope threshold due to capture/processing artifacts.
        # Add GFSK/4FSK/2FSK rules here too, but with lower priority than const-env.
        
        # FSK2 fallback: fewer IF tones + digital characteristics + verified bimodality
        scores['FSK2'] = _score([
            _in_range(if_bmc,        0.25,  1.0),  # Must have physical bimodality
            _in_range(n_if_tones,    1.0,   4.0),  # fewer IF tones than GFSK
            _in_range(spec_entropy,  3.0,  10.0),
            _in_range(iq_ratio,      0.0,   0.15), # balanced I/Q (not SSB)
            _in_range(amp_var,       0.03,  0.15), # borderline const-env
            _in_range(if_var,        0.005, 0.5),  # frequency modulation present
        ]) * 0.7  # Lower priority than const-env branch

        # GFSK fallback: many IF tones + flat spectrum + low iq_ratio
        scores['GFSK'] = _score([
            _in_range(n_if_tones,    7.0,  14.0),  # many IF tones
            _in_range(spec_entropy,  9.0,  13.0),
            _in_range(spec_flatness, 0.30,  0.70),
            _in_range(iq_ratio,      0.0,   0.1),   # balanced I/Q (not SSB)
            _in_range(amp_var,       0.03,  0.1),   # borderline const-env
        ]) * 0.7  # Lower priority than const-env branch

        # 4FSK fallback
        scores['FSK4'] = _score([
            _in_range(n_if_tones,    7.0,  14.0),
            _in_range(spec_entropy,  8.5,  12.5),
            _in_range(spec_flatness, 0.25,  0.60),
            _in_range(iq_ratio,      0.0,   0.1),
            _in_range(amp_var,       0.001, 0.1),
        ]) * 0.7

        # ── AM-DSB ─────────────────────────────────────────────────────────
        # Symmetric double-sideband: lower iq_ratio, narrow spectrum
        scores['AM-DSB'] = _score([
            _in_range(amp_var,       0.05, 0.5),
            _in_range(spec_entropy,  0.5,  5.0),
            _in_range(iq_ratio,      0.0,  0.4),   # more balanced than SSB
            _in_range(amp_kurt,      0.0,  5.0),   # not extremely impulsive
        ])

        # ── OOK ────────────────────────────────────────────────────────────
        scores['OOK'] = _score([
            _in_range(c42,     -0.15, 0.5),
            _in_range(amp_var,  0.2,  0.55),
            _in_range(amp_kurt, 0.0,  5.0),
        ])

        # ── PAM4 ───────────────────────────────────────────────────────────
        scores['PAM4'] = _score([
            _in_range(c42,         -0.55, -0.2),
            _in_range(spec_entropy, 7.0,  11.0),
            _in_range(iq_ratio,     0.5,   1.0),
        ])

        # ── QAM16 ──────────────────────────────────────────────────────────
        scores['QAM16'] = _score([
            _in_range(amp_var,  0.09, 0.15),
            _in_range(c42,      -0.75, -0.30),
            _in_range(amp_kurt, -1.5, -0.1),
        ])

        # ── QAM64 ─────────────────────────────────────────────────────────
        scores['QAM64'] = _score([
            _in_range(amp_var, 0.1, 0.5),
            _in_range(c42,     -0.5, 0.1),
        ])

        # ── APSK16 ──────────────────────────────────────────────────────
        scores['APSK16'] = _score([
            _in_range(amp_var, 0.06, 0.10),
            _in_range(c42,     -0.85, -0.60),
        ])

        # ── APSK32 ──────────────────────────────────────────────────────
        scores['APSK32'] = _score([
            _in_range(amp_var, 0.1, 0.6),
            _in_range(c42,     -0.5, 0.1),
        ])

        # ── OFDM ─── tightened to avoid GFSK false positives ────────────────
        # Problem: OFDM was matching GFSK signals with flat spectra.
        # Solution: add n_if_tones constraint to exclude FSK-family signals.
        # Real OFDM has many subcarriers but NOT the discrete IF tones of FSK.
        scores['OFDM'] = _score([
            _in_range(spec_entropy,  9.5, 14.0),
            _in_range(spec_flatness, 0.3,  0.95),
            _in_range(amp_kurt,      -1.5,  0.5),  # near-Gaussian amplitude
            _in_range(iq_ratio,       0.0,  0.15), # balanced I/Q
            _in_range(n_if_tones,     0.0,  5.0),  # exclude FSK signals
        ])

    # ── Fill zeros for any classes not scored in this branch ─────────────
    for mod in ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'FSK2', 'FSK4',
                'GFSK', 'CPFSK', 'AM-DSB', 'AM-SSB', 'WBFM', 'APSK16',
                'APSK32', 'PAM4', 'OOK', 'OFDM']:
        if mod not in scores:
            scores[mod] = 1e-6

    total = sum(scores.values()) + 1e-12
    probs = {k: v / total for k, v in scores.items()}
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)

def _in_range(val: float, lo: float, hi: float) -> float:
    """Returns 1.0 if val in [lo, hi], else scaled penalty."""
    if lo <= val <= hi:
        return 1.0
    elif val < lo:
        return max(0.0, 1.0 - (lo - val) / (hi - lo + 1e-12))
    else:
        return max(0.0, 1.0 - (val - hi) / (hi - lo + 1e-12))


def _score(criteria: list) -> float:
    """Geometric mean of criteria scores — all must be satisfied."""
    if not criteria:
        return 0.0
    product = 1.0
    for c in criteria:
        product *= max(c, 1e-6)
    return float(product ** (1.0 / len(criteria)))
