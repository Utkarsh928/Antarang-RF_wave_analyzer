"""
SDR Streaming — real-time IQ capture from RTL-SDR, HackRF, Airspy, USRP.

Backends tried in order:
  1. SoapySDR  — universal: HackRF / Airspy / USRP / PlutoSDR / RTL-SDR
  2. pyrtlsdr  — RTL-SDR only (pure pip, no system libs needed on Windows)
  3. Simulated — software-generated signals (always available, full quality)

Install for RTL-SDR (Windows):
    pip install pyrtlsdr
    Then install Zadig driver: https://zadig.akeo.ie/

Install for HackRF / Airspy / USRP:
    conda install -c conda-forge soapysdr-module-rtlsdr   # RTL via SoapySDR
    conda install -c conda-forge soapysdr-module-hackrf   # HackRF

Simulated signal types available without any hardware:
    FM Radio, ADS-B Aircraft, BPSK, QPSK, 8PSK, QAM16, QAM64,
    FSK2 (LoRa-like), GFSK (Bluetooth-like), AM-DSB, NOAA Weather,
    ISM Band, Aircraft VHF, Custom noise
"""

import numpy as np
import threading
import time
from typing import Optional, Callable, List, Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Simulated signal generator — realistic IQ for each signal type
# ─────────────────────────────────────────────────────────────────────────────

class SignalSimulator:
    """
    Generates realistic IQ samples for a given signal type.
    All outputs are complex64 at the requested sample rate.
    Includes proper carrier offset, noise, fading, and channel effects.
    """

    SIGNAL_TYPES = [
        "BPSK",
        "QPSK",
        "8PSK",
        "QAM16",
        "QAM64",
        "FSK2 (LoRa-like)",
        "GFSK (Bluetooth-like)",
        "AM-DSB (Broadcast)",
        "WBFM (FM Radio)",
        "ADS-B Aircraft (1090 MHz)",
        "NOAA Weather Satellite",
        "ACARS Aircraft Data",
        "ISM Band Device",
        "WiFi Preamble (802.11)",
        "Radar Pulse",
        "CW Morse",
        "Noise Only",
        "Multi-Signal (3 carriers)",
    ]

    def __init__(self, signal_type: str = "BPSK",
                 sample_rate: float = 48000.0,
                 center_freq: float = 0.0,
                 snr_db: float = 15.0,
                 seed: int = 42):
        self.signal_type  = signal_type
        self.sample_rate  = float(sample_rate)
        self.center_freq  = float(center_freq)
        self.snr_db       = float(snr_db)
        self._rng         = np.random.RandomState(seed)
        self._phase_acc   = 0.0   # for continuous phase FSK

    def generate(self, n_samples: int,
                 signal_type: str = None) -> np.ndarray:
        """
        Generate n_samples of complex64 IQ at self.sample_rate.

        Parameters
        ----------
        n_samples   : number of samples to generate
        signal_type : override the signal type set at construction time.
                      If None, uses self.signal_type.
        """
        st = (signal_type if signal_type is not None
              else self.signal_type).upper()

        if "BPSK" in st:
            sig = self._gen_psk(n_samples, M=2)
        elif "QPSK" in st:
            sig = self._gen_psk(n_samples, M=4)
        elif "8PSK" in st:
            sig = self._gen_psk(n_samples, M=8)
        elif "QAM16" in st:
            sig = self._gen_qam(n_samples, M=16)
        elif "QAM64" in st:
            sig = self._gen_qam(n_samples, M=64)
        elif "FSK2" in st or "LORA" in st:
            sig = self._gen_fsk(n_samples, deviation=0.25)
        elif "GFSK" in st or "BLUETOOTH" in st:
            sig = self._gen_gfsk(n_samples)
        elif "AM-DSB" in st or "AM DSB" in st:
            sig = self._gen_am_dsb(n_samples)
        elif "WBFM" in st or "FM RADIO" in st:
            sig = self._gen_wbfm(n_samples)
        elif "ADS-B" in st or "AIRCRAFT" in st:
            sig = self._gen_adsb(n_samples)
        elif "NOAA" in st or "WEATHER" in st:
            sig = self._gen_noaa(n_samples)
        elif "ACARS" in st:
            sig = self._gen_acars(n_samples)
        elif "ISM" in st:
            sig = self._gen_ism(n_samples)
        elif "WIFI" in st or "802.11" in st:
            sig = self._gen_wifi(n_samples)
        elif "RADAR" in st:
            sig = self._gen_radar_pulse(n_samples)
        elif "CW" in st or "MORSE" in st:
            sig = self._gen_cw(n_samples)
        elif "NOISE" in st:
            sig = self._gen_noise(n_samples)
        elif "MULTI" in st:
            sig = self._gen_multi_signal(n_samples)
        else:
            sig = self._gen_psk(n_samples, M=4)

        # Apply channel: carrier offset + noise + slight fading
        sig = self._apply_channel(sig, n_samples)
        return sig.astype(np.complex64)

    # ── Signal generators ─────────────────────────────────────────────────

    def _gen_psk(self, n: int, M: int) -> np.ndarray:
        """Phase Shift Keying with matched filter shaping."""
        sps = max(4, int(self.sample_rate / 50000))   # ~50 kbaud
        n_syms = max(1, n // sps)
        bits = self._rng.randint(0, M, n_syms)
        angles = (2 * np.pi * bits / M) + (np.pi / M if M > 2 else 0)
        syms = np.exp(1j * angles)
        # Upsample + raised-cosine filter
        upsampled = np.zeros(n_syms * sps, dtype=np.complex128)
        upsampled[::sps] = syms
        h = self._rrc_filter(sps, alpha=0.35, span=8)
        shaped = np.convolve(upsampled, h, mode='same')
        return shaped[:n]

    def _gen_qam(self, n: int, M: int) -> np.ndarray:
        """QAM with square constellation."""
        sps = max(4, int(self.sample_rate / 50000))
        n_syms = max(1, n // sps)
        sq = int(np.sqrt(M))
        levels = np.linspace(-sq+1, sq-1, sq)
        i_syms = self._rng.choice(levels, n_syms)
        q_syms = self._rng.choice(levels, n_syms)
        syms = (i_syms + 1j * q_syms) / (sq - 1)
        upsampled = np.zeros(n_syms * sps, dtype=np.complex128)
        upsampled[::sps] = syms
        h = self._rrc_filter(sps, alpha=0.25, span=8)
        shaped = np.convolve(upsampled, h, mode='same')
        return shaped[:n]

    def _gen_fsk(self, n: int, deviation: float = 0.25) -> np.ndarray:
        """Continuous-phase FSK (CPFSK) — like LoRa chirp spread."""
        sps = max(8, int(self.sample_rate / 25000))
        n_syms = max(1, n // sps)
        bits = self._rng.randint(0, 2, n_syms)
        # Instantaneous frequency: +/- deviation * symbol_rate
        sym_rate = self.sample_rate / sps
        f_dev = deviation * sym_rate
        freqs = np.where(bits == 0, -f_dev, f_dev)
        # Upsample
        freq_seq = np.repeat(freqs, sps)[:n]
        # Integrate to get phase (continuous phase FSK)
        phase = self._phase_acc + 2 * np.pi * np.cumsum(freq_seq) / self.sample_rate
        self._phase_acc = float(phase[-1]) % (2 * np.pi)
        return np.exp(1j * phase)

    def _gen_gfsk(self, n: int) -> np.ndarray:
        """Gaussian FSK (Bluetooth-like, BT=0.5)."""
        from scipy.signal.windows import gaussian  # SciPy >= 1.1 location
        sps = max(8, int(self.sample_rate / 25000))
        n_syms = max(1, n // sps)
        bits = self._rng.randint(0, 2, n_syms)
        nrz = 2 * bits.astype(np.float64) - 1
        # Gaussian filter BT=0.5
        bt = 0.5
        win_len = 6 * sps + 1
        sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt) * sps
        h_gauss = gaussian(win_len, std=sigma)
        h_gauss /= h_gauss.sum()
        nrz_up = np.repeat(nrz, sps)
        smoothed = np.convolve(nrz_up, h_gauss, mode='same')
        f_dev = 0.5 * self.sample_rate / sps * 0.35
        phase = 2 * np.pi * f_dev * np.cumsum(smoothed[:n]) / self.sample_rate
        return np.exp(1j * phase)

    def _gen_am_dsb(self, n: int) -> np.ndarray:
        """AM Double Sideband — voice-like audio modulating a carrier."""
        t = np.arange(n) / self.sample_rate
        # Simulated voice: mix of audio tones
        audio = (0.5 * np.sin(2*np.pi*300*t) +
                 0.3 * np.sin(2*np.pi*800*t) +
                 0.2 * np.sin(2*np.pi*1500*t))
        audio += 0.1 * self._rng.randn(n)
        carrier_freq = 5000.0
        carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
        return ((1.0 + 0.8 * audio) * carrier).astype(np.complex128)

    def _gen_wbfm(self, n: int) -> np.ndarray:
        """Wideband FM — like an FM radio station."""
        t = np.arange(n) / self.sample_rate
        # Composite audio: pilot + stereo + program
        audio = (np.sin(2*np.pi*1000*t) * 0.4 +
                 np.sin(2*np.pi*2500*t) * 0.3 +
                 0.1 * self._rng.randn(n))
        # FM: integrate audio to get phase
        f_dev = 75000.0  # 75 kHz peak deviation (broadcast standard)
        phase = 2 * np.pi * f_dev * np.cumsum(audio) / self.sample_rate
        return np.exp(1j * phase)

    def _gen_adsb(self, n: int) -> np.ndarray:
        """ADS-B 1090ES Mode S — short squitter burst followed by silence."""
        # ADS-B: PPM (pulse position modulation) at 1 Msps effective
        # Burst: 8 µs preamble + 112 bit message = 120 µs
        burst_samples = int(120e-6 * self.sample_rate)
        silence_samples = int(400e-6 * self.sample_rate)  # inter-message gap
        period = burst_samples + silence_samples

        sig = np.zeros(n, dtype=np.complex128)
        # Preamble: specific pulse pattern
        preamble_pattern = [1,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0]
        chip = max(1, int(0.5e-6 * self.sample_rate))
        for i in range(0, n, period):
            # Preamble
            for j, bit in enumerate(preamble_pattern):
                start = i + j * chip
                if start + chip < n and bit:
                    sig[start:start+chip] = 1.0 + 0j
            # Random payload bits (112 bits)
            payload = self._rng.randint(0, 2, 112)
            for j, bit in enumerate(payload):
                base = i + len(preamble_pattern)*chip + j*2*chip
                if base + 2*chip < n:
                    if bit:
                        sig[base:base+chip] = 1.0 + 0j
                    else:
                        sig[base+chip:base+2*chip] = 1.0 + 0j
        return sig

    def _gen_noaa(self, n: int) -> np.ndarray:
        """NOAA APT weather satellite — 2400 Hz subcarrier, AM modulated."""
        t = np.arange(n) / self.sample_rate
        # APT: 2400 Hz subcarrier AM modulated with image lines
        image = (np.sin(2*np.pi*0.5*t) * 0.5 +   # slow image scan
                 self._rng.randn(n) * 0.1)
        subcarrier = np.sin(2*np.pi*2400*t)
        return ((1 + 0.8 * image) * subcarrier + 1j * 0).astype(np.complex128)

    def _gen_acars(self, n: int) -> np.ndarray:
        """ACARS aircraft data — AM/FSK at 2400 bps."""
        t = np.arange(n) / self.sample_rate
        sps = max(1, int(self.sample_rate / 2400))
        n_bits = max(1, n // sps)
        bits = self._rng.randint(0, 2, n_bits)
        freq_seq = np.repeat(np.where(bits == 0, 1200.0, 2400.0), sps)[:n]
        phase = 2 * np.pi * np.cumsum(freq_seq) / self.sample_rate
        carrier = np.sin(2*np.pi*1000*t)
        return (carrier * np.cos(phase) + 0j).astype(np.complex128)

    def _gen_ism(self, n: int) -> np.ndarray:
        """ISM band IoT device — OOK (on-off keying) bursts."""
        sps = max(4, int(self.sample_rate / 10000))
        n_bits = max(1, n // sps)
        bits = self._rng.randint(0, 2, n_bits)
        # Add sync preamble (10101010...)
        preamble = np.tile([1,0], 16)
        payload = np.concatenate([preamble, bits[:n_bits-32]])
        ook = np.repeat(payload, sps).astype(np.float64)[:n]
        t = np.arange(n) / self.sample_rate
        carrier = np.exp(1j * 2 * np.pi * 10000 * t)
        return (ook * carrier)

    def _gen_wifi(self, n: int) -> np.ndarray:
        """WiFi 802.11 preamble — OFDM short training sequence."""
        # IEEE 802.11a OFDM preamble: 10 short training symbols
        # Each OFDM symbol: 64 subcarriers
        fft_size = 64
        short_ts = np.array([0,0,0,0,0,0,0,0,
                               1+1j,0,0,0,-1-1j,0,0,0,
                               1+1j,0,0,0,-1-1j,0,0,0,
                               -1-1j,0,0,0,1+1j,0,0,0,
                               0,0,0,0,-1-1j,0,0,0,
                               -1-1j,0,0,0,1+1j,0,0,0,
                               1+1j,0,0,0,1+1j,0,0,0,
                               1+1j,0,0,0,0,0,0,0],
                              dtype=np.complex128) / 6.0
        sym = np.fft.ifft(short_ts) * fft_size
        preamble = np.tile(sym, 10)[:n]
        # Pad with QPSK data
        remaining = n - len(preamble)
        if remaining > 0:
            data = self._gen_psk(remaining, M=4)
            result = np.concatenate([preamble, data])
        else:
            result = preamble[:n]
        return result

    def _gen_radar_pulse(self, n: int) -> np.ndarray:
        """Radar chirp pulse — linear frequency modulation."""
        pulse_samples = int(10e-6 * self.sample_rate)   # 10 µs pulse
        pri_samples   = int(1e-3  * self.sample_rate)   # 1 ms PRI
        sig = np.zeros(n, dtype=np.complex128)
        for start in range(0, n, pri_samples):
            end = min(start + pulse_samples, n)
            p_len = end - start
            if p_len < 2:
                continue
            t = np.arange(p_len) / self.sample_rate
            bw = 10e6 / (self.sample_rate / self.sample_rate)
            # LFM chirp
            phase = np.pi * bw * t**2 / (pulse_samples / self.sample_rate)
            sig[start:end] = np.exp(1j * phase)
        return sig

    def _gen_cw(self, n: int) -> np.ndarray:
        """CW Morse code — carrier keyed on/off."""
        # Dit = 100ms, Dah = 300ms  (scaled to sample rate)
        dit  = max(1, int(0.05 * self.sample_rate))
        dah  = 3 * dit
        gap  = dit
        # SOS pattern
        pattern = ([dit,gap]*3 + [gap] + [dah,gap]*3 + [gap] + [dit,gap]*3)
        sig = np.zeros(n, dtype=np.complex128)
        pos = 0
        idx = 0
        while pos < n:
            dur = pattern[idx % len(pattern)]
            on  = (idx % 2 == 0)
            end = min(pos + dur, n)
            if on:
                t = np.arange(end - pos) / self.sample_rate
                sig[pos:end] = np.exp(1j * 2 * np.pi * 800 * t)
            pos = end
            idx += 1
        return sig

    def _gen_noise(self, n: int) -> np.ndarray:
        """Pure Gaussian noise — useful for testing noise floor detection."""
        return (self._rng.randn(n) + 1j * self._rng.randn(n)) / np.sqrt(2)

    def _gen_multi_signal(self, n: int) -> np.ndarray:
        """Three simultaneous signals on different sub-frequencies."""
        t = np.arange(n) / self.sample_rate
        bw = self.sample_rate / 4

        def _make(sig_type, seed, n_out):
            """Generate exactly n_out samples, pad or trim as needed."""
            gen = SignalSimulator(sig_type, self.sample_rate, self.center_freq,
                                  snr_db=18, seed=seed)
            raw = gen.generate(n_out + 512)   # generate extra, then trim
            return raw[:n_out].astype(np.complex128)

        s1 = _make('BPSK',           11, n) * np.exp(-1j * 2 * np.pi * bw/2 * t)
        s2 = _make('WBFM (FM Radio)',99, n) * 0.5
        s3 = _make('FSK2 (LoRa-like)',77, n) * np.exp(1j * 2 * np.pi * bw/2 * t)

        return (s1 + s2 + s3).astype(np.complex128)

    # ── Channel model ──────────────────────────────────────────────────────

    def _apply_channel(self, sig: np.ndarray, n: int) -> np.ndarray:
        """Add carrier frequency offset, AWGN, and mild fading. Always returns exactly n samples."""
        sig = np.asarray(sig, dtype=np.complex128)
        # Guarantee exact length by padding or trimming
        if len(sig) > n:
            sig = sig[:n]
        elif len(sig) < n:
            sig = np.concatenate([sig, np.zeros(n - len(sig), dtype=np.complex128)])

        # Normalise power
        pwr = np.mean(np.abs(sig)**2)
        if pwr > 0:
            sig /= np.sqrt(pwr)

        # Carrier frequency offset (realistic: ±5–20 ppm)
        ppm = self._rng.uniform(-15, 15)
        fo  = self.center_freq * ppm * 1e-6
        t   = np.arange(n) / self.sample_rate
        sig = sig * np.exp(1j * 2 * np.pi * fo * t)

        # AWGN
        snr_lin = 10 ** (self.snr_db / 10)
        noise_std = 1.0 / np.sqrt(2 * snr_lin)
        noise = (self._rng.randn(n) + 1j * self._rng.randn(n)) * noise_std
        sig = sig + noise

        # Mild slow fading (Rayleigh, coherence ~1000 samples)
        if n > 100:
            fade_len = max(1, n // 100)
            fade_env = self._rng.rayleigh(1.0, fade_len)
            fade_interp = np.interp(np.arange(n),
                                     np.linspace(0, n-1, fade_len),
                                     fade_env)
            sig = sig * fade_interp / (fade_interp.mean() + 1e-9)

        return sig.astype(np.complex64)

    # ── Root-raised cosine filter ──────────────────────────────────────────

    @staticmethod
    def _rrc_filter(sps: int, alpha: float = 0.35, span: int = 8) -> np.ndarray:
        """Root-raised cosine FIR filter coefficients."""
        n = span * sps + 1
        t = np.arange(-(span * sps) // 2, (span * sps) // 2 + 1) / sps
        h = np.zeros(n)
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = (1 + alpha * (4/np.pi - 1))
            elif abs(abs(ti) - 1/(4*alpha)) < 1e-6:
                h[i] = (alpha/np.sqrt(2)) * (
                    (1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) +
                    (1 - 2/np.pi) * np.cos(np.pi/(4*alpha)))
            else:
                num = np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))
                den = np.pi*ti*(1 - (4*alpha*ti)**2)
                h[i] = num / (den + 1e-30)
        h /= (np.sqrt(sps) * np.linalg.norm(h) + 1e-30)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_hardware() -> Dict:
    """
    Probe all SDR backends and return a status dict.
    Returns:
        {
          'rtlsdr':  {'available': bool, 'installed': bool, 'device_count': int, 'error': str},
          'soapysdr': {'available': bool, 'installed': bool, 'devices': list, 'error': str},
          'any_real': bool,
        }
    """
    result = {
        'rtlsdr':  {'available': False, 'installed': False, 'device_count': 0, 'error': ''},
        'soapysdr':{'available': False, 'installed': False, 'devices': [], 'error': ''},
        'any_real': False,
    }

    # Check pyrtlsdr
    try:
        import rtlsdr
        result['rtlsdr']['installed'] = True
        count = rtlsdr.RtlSdr.get_device_count()
        result['rtlsdr']['device_count'] = count
        result['rtlsdr']['available'] = count > 0
    except ImportError:
        result['rtlsdr']['error'] = 'pyrtlsdr not installed  →  pip install pyrtlsdr'
    except Exception as e:
        result['rtlsdr']['installed'] = True
        result['rtlsdr']['error'] = str(e)

    # Check SoapySDR
    try:
        import SoapySDR
        result['soapysdr']['installed'] = True
        devs = list(SoapySDR.Device.enumerate())
        result['soapysdr']['devices'] = devs
        result['soapysdr']['available'] = len(devs) > 0
    except ImportError:
        result['soapysdr']['error'] = 'SoapySDR not installed  →  conda install -c conda-forge soapysdr'
    except Exception as e:
        result['soapysdr']['installed'] = True
        result['soapysdr']['error'] = str(e)

    result['any_real'] = (result['rtlsdr']['available'] or
                          result['soapysdr']['available'])
    return result


def list_devices() -> List[Dict]:
    """
    Return all available SDR devices + GNU Radio ZMQ source + Simulated.
    Each dict: {backend, label, driver, serial, args}
    """
    devices = []

    # SoapySDR devices
    try:
        import SoapySDR
        for r in SoapySDR.Device.enumerate():
            label = r.get('label', r.get('driver', 'Unknown SoapySDR device'))
            devices.append({
                'backend': 'SoapySDR',
                'label':   f"[SoapySDR]  {label}",
                'driver':  r.get('driver', ''),
                'serial':  r.get('serial', ''),
                'args':    dict(r),
            })
    except Exception:
        pass

    # pyrtlsdr devices
    try:
        import rtlsdr
        for i in range(rtlsdr.RtlSdr.get_device_count()):
            try:
                name = rtlsdr.RtlSdr.get_device_name(i)
            except Exception:
                name = f"RTL2832U #{i}"
            devices.append({
                'backend': 'pyrtlsdr',
                'label':   f"[RTL-SDR]  {name}  (index {i})",
                'driver':  'rtlsdr',
                'serial':  str(i),
                'args':    {'index': i},
            })
    except Exception:
        pass

    # GNU Radio ZMQ source (if ZMQ available)
    from core.gnu_radio.zmq_source import zmq_available
    if zmq_available():
        devices.append({
            'backend': 'GNU Radio ZMQ',
            'label':   '[GNU Radio ZMQ]  tcp://localhost:5555  (any hardware via GNU Radio)',
            'driver':  'zmq',
            'serial':  '0',
            'args':    {'address': 'tcp://localhost:5555'},
        })

    # Always add simulated
    devices.append({
        'backend': 'Simulated',
        'label':   '[Simulated]  Software Test Signal  (no hardware needed)',
        'driver':  'simulated',
        'serial':  '0',
        'args':    {},
    })

    return devices


# ─────────────────────────────────────────────────────────────────────────────
# SDR Stream Worker
# ─────────────────────────────────────────────────────────────────────────────

class SDRStreamWorker:
    """
    Streams IQ samples from any SDR backend into a callback.
    Thread-safe. Automatically picks backend from device_info.

    Usage:
        worker = SDRStreamWorker(device_info, sample_rate=2.4e6,
                                 center_freq=433.92e6, gain=30,
                                 signal_type='QPSK', snr_db=15)
        worker.start(callback=lambda chunk: ...)
        ...
        worker.stop()
    """

    CHUNK_SIZE = 262144   # ~109 ms at 2.4 MHz

    def __init__(self, device_info: dict,
                 sample_rate: float  = 2_400_000,
                 center_freq: float  = 100_000_000,
                 gain: float         = 30.0,
                 signal_type: str    = 'QPSK',
                 snr_db: float       = 15.0):
        self.device_info  = device_info
        self.sample_rate  = float(sample_rate)
        self.center_freq  = float(center_freq)
        self.gain         = float(gain)
        self.signal_type  = signal_type
        self.snr_db       = float(snr_db)
        self._running     = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable] = None
        self._error: Optional[str] = None

    def start(self, callback: Callable[[np.ndarray], None]):
        if self._running:
            return
        self._callback = callback
        self._running  = True
        self._error    = None

        backend = self.device_info.get('backend', 'Simulated')
        target = {
            'SoapySDR':        self._stream_soapy,
            'pyrtlsdr':        self._stream_rtlsdr,
            'GNU Radio ZMQ':   self._stream_zmq,
            'Simulated':       self._stream_simulated,
        }.get(backend, self._stream_simulated)

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=4.0)
            self._thread = None

    @property
    def error(self) -> Optional[str]:
        return self._error

    # ── SoapySDR backend ───────────────────────────────────────────────────

    def _stream_soapy(self):
        try:
            import SoapySDR
            args  = self.device_info.get('args', {})
            sdr   = SoapySDR.Device(args)
            ch    = 0
            sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, ch, self.sample_rate)
            sdr.setFrequency(SoapySDR.SOAPY_SDR_RX,  ch, self.center_freq)
            sdr.setGain(SoapySDR.SOAPY_SDR_RX,       ch, self.gain)
            sdr.setAntenna(SoapySDR.SOAPY_SDR_RX,    ch, "RX")

            stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX,
                                      SoapySDR.SOAPY_SDR_CF32)
            sdr.activateStream(stream)
            buf = np.zeros(self.CHUNK_SIZE, dtype=np.complex64)

            while self._running:
                sr = sdr.readStream(stream, [buf], len(buf), timeoutUs=1_000_000)
                if sr.ret > 0 and self._callback:
                    self._callback(buf[:sr.ret].copy())

            sdr.deactivateStream(stream)
            sdr.closeStream(stream)
        except Exception as e:
            self._error = f"SoapySDR error: {e}"
            self._running = False

    # ── GNU Radio ZMQ backend ──────────────────────────────────────────────
    def _stream_zmq(self):
        """Receive IQ from any GNU Radio flowgraph via ZMQ PULL socket."""
        from core.gnu_radio.zmq_source import ZMQSource
        address = self.device_info.get('args', {}).get(
            'address', 'tcp://localhost:5555')
        src = ZMQSource(address, self.sample_rate)
        src.start(lambda chunk: self._callback(chunk) if self._callback else None)
        while self._running:
            time.sleep(0.05)
        src.stop()
        if src.error:
            self._error = src.error

    # ── pyrtlsdr backend ───────────────────────────────────────────────────

    def _stream_rtlsdr(self):
        try:
            import rtlsdr
            idx = self.device_info.get('args', {}).get('index', 0)
            sdr = rtlsdr.RtlSdr(idx)

            # Configure device
            sdr.sample_rate  = self.sample_rate
            sdr.center_freq  = self.center_freq
            sdr.freq_correction = 60    # PPM correction (adjust for your dongle)

            # RTL-SDR gain: 'auto' or specific value in tenths of dB
            try:
                sdr.gain = self.gain
            except Exception:
                sdr.gain = 'auto'

            # Streaming
            while self._running:
                raw = sdr.read_samples(self.CHUNK_SIZE)
                if self._callback:
                    self._callback(raw.astype(np.complex64))

            sdr.close()

        except ImportError:
            self._error = ("pyrtlsdr not installed.\n"
                           "Run: pip install pyrtlsdr\n"
                           "Then install Zadig USB driver from https://zadig.akeo.ie/")
            self._running = False
        except Exception as e:
            self._error = (f"RTL-SDR error: {e}\n\n"
                           "Common causes:\n"
                           "• Driver not installed → use Zadig to install WinUSB\n"
                           "• Device in use by another application\n"
                           "• Wrong device index")
            self._running = False

    # ── Simulated backend ──────────────────────────────────────────────────

    def _stream_simulated(self):
        """
        Generates realistic IQ samples for the selected signal type.
        Paced to real-time so the pipeline behaves exactly as with hardware.
        """
        sim = SignalSimulator(
            signal_type  = self.signal_type,
            sample_rate  = self.sample_rate,
            center_freq  = self.center_freq,
            snr_db       = self.snr_db,
            seed         = int(time.time()) % 10000,
        )
        chunk_duration = self.CHUNK_SIZE / self.sample_rate

        while self._running:
            t0  = time.time()
            sig = sim.generate(self.CHUNK_SIZE)
            if self._callback:
                self._callback(sig)
            elapsed = time.time() - t0
            sleep_t = max(0.0, chunk_duration - elapsed)
            time.sleep(sleep_t)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: single-shot capture
# ─────────────────────────────────────────────────────────────────────────────

def capture_once(device_info: dict,
                 sample_rate: float  = 2_400_000,
                 center_freq: float  = 100_000_000,
                 gain: float         = 30.0,
                 duration_sec: float = 1.0,
                 signal_type: str    = 'QPSK',
                 snr_db: float       = 15.0) -> np.ndarray:
    """
    Capture duration_sec seconds of IQ and return as complex64 array.
    Works with real hardware or simulation transparently.
    """
    n_target = int(sample_rate * duration_sec)
    chunks:  List[np.ndarray] = []
    collected = 0
    lock = threading.Lock()

    def collect(chunk: np.ndarray):
        nonlocal collected
        with lock:
            chunks.append(chunk.copy())
            collected += len(chunk)

    worker = SDRStreamWorker(device_info, sample_rate, center_freq, gain,
                              signal_type, snr_db)
    worker.start(collect)

    deadline = time.time() + duration_sec + 3.0
    while collected < n_target and time.time() < deadline:
        time.sleep(0.05)
    worker.stop()

    if not chunks:
        return np.zeros(n_target, dtype=np.complex64)

    result = np.concatenate(chunks)
    return result[:n_target].astype(np.complex64)
