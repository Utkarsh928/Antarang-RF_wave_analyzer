"""
SDR Live Capture Dialog — complete implementation.

Features:
  • Hardware status panel with per-backend status
  • One-click driver install helper
  • 18 simulated signal types
  • Frequency presets for common bands
  • Live mini-spectrum preview during capture
  • Sample rate, gain, duration controls
  • Paced real-time simulation
"""
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QDoubleSpinBox, QGroupBox, QGridLayout,
    QProgressBar, QMessageBox, QTabWidget, QWidget,
    QTextEdit, QSizePolicy, QFrame, QSlider, QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor
from gui.theme import apply_window_chrome, window_theme_mode

import pyqtgraph as pg


# ─────────────────────────────────────────────────────────────────────────────
# Background capture worker
# ─────────────────────────────────────────────────────────────────────────────

class SDRLiveWorker(QThread):
    """QThread that runs SDRStreamWorker and emits collected samples."""
    chunk_ready  = pyqtSignal(object)   # np.ndarray — each chunk as it arrives
    all_done     = pyqtSignal(object)   # np.ndarray — full capture
    error_signal = pyqtSignal(str)
    status       = pyqtSignal(str)

    def __init__(self, device_info, sample_rate, center_freq,
                 gain, duration_sec, signal_type, snr_db, parent=None):
        super().__init__(parent)
        self.device_info  = device_info
        self.sample_rate  = sample_rate
        self.center_freq  = center_freq
        self.gain         = gain
        self.duration_sec = duration_sec
        self.signal_type  = signal_type
        self.snr_db       = snr_db
        self._stop_flag   = False

    def run(self):
        try:
            from core.sdr_stream import SDRStreamWorker
            import time

            n_target  = int(self.sample_rate * self.duration_sec)
            all_chunks = []
            collected  = [0]

            worker = SDRStreamWorker(
                self.device_info,
                sample_rate  = self.sample_rate,
                center_freq  = self.center_freq,
                gain         = self.gain,
                signal_type  = self.signal_type,
                snr_db       = self.snr_db,
            )

            def on_chunk(chunk: np.ndarray):
                if self._stop_flag:
                    worker._running = False   # signal thread to exit; no join from within itself
                    return
                all_chunks.append(chunk.copy())
                collected[0] += len(chunk)
                pct = min(100, int(100 * collected[0] / n_target))
                self.status.emit(
                    f"Capturing… {collected[0]:,} / {n_target:,} samples  ({pct}%)")
                self.chunk_ready.emit(chunk.copy())
                if collected[0] >= n_target:
                    worker._running = False   # signal thread to exit cleanly

            self.status.emit("Opening device…")
            worker.start(on_chunk)

            deadline = time.time() + self.duration_sec + 5.0
            while collected[0] < n_target and not self._stop_flag:
                if time.time() > deadline:
                    break
                time.sleep(0.05)
            worker.stop()

            if worker.error:
                self.error_signal.emit(worker.error)
                return

            if not all_chunks:
                self.error_signal.emit("No samples received.")
                return

            combined = np.concatenate(all_chunks)[:n_target]
            self.all_done.emit(combined.astype(np.complex64))
            self.status.emit(
                f"Done — {len(combined):,} samples  "
                f"({len(combined)/self.sample_rate*1000:.0f} ms)")

        except Exception as e:
            import traceback
            self.error_signal.emit(f"{e}\n{traceback.format_exc()}")

    def stop_capture(self):
        self._stop_flag = True


# ─────────────────────────────────────────────────────────────────────────────
# Hardware status widget
# ─────────────────────────────────────────────────────────────────────────────

class HardwareStatusWidget(QWidget):
    """Shows RTL-SDR, SoapySDR and GNU Radio status with install instructions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(6)
        lay.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Hardware & GNU Radio Status")
        title.setStyleSheet("font-weight:700; font-size:12px;")
        lay.addWidget(title)

        self._lbl_rtl    = QLabel()
        self._lbl_soapy  = QLabel()
        self._lbl_gr     = QLabel()      # GNU Radio core
        self._lbl_gr_zmq = QLabel()      # ZMQ module
        self._lbl_gr_mod = QLabel()      # sub-modules
        self._lbl_hint   = QLabel()
        self._lbl_hint.setWordWrap(True)
        self._lbl_hint.setStyleSheet("font-size:11px;")

        for lbl in [self._lbl_rtl, self._lbl_soapy,
                    self._lbl_gr, self._lbl_gr_zmq,
                    self._lbl_gr_mod, self._lbl_hint]:
            lay.addWidget(lbl)

        btn_row = QHBoxLayout()

        self._btn_install_rtl = QPushButton("pip install pyrtlsdr")
        self._btn_install_rtl.clicked.connect(self._install_rtlsdr)
        btn_row.addWidget(self._btn_install_rtl)

        self._btn_install_gr = QPushButton("conda install gnuradio")
        self._btn_install_gr.setToolTip(
            "Install GNU Radio for professional signal conditioning + ZMQ hardware support")
        self._btn_install_gr.clicked.connect(self._install_gnuradio)
        btn_row.addWidget(self._btn_install_gr)

        self._btn_zadig = QPushButton("Get Zadig Driver →")
        self._btn_zadig.clicked.connect(self._open_zadig)
        btn_row.addWidget(self._btn_zadig)
        btn_row.addStretch()

        lay.addLayout(btn_row)
        self.refresh()

    def refresh(self):
        from core.sdr_stream import detect_hardware
        from core.gnu_radio.availability import GR_AVAILABLE, GR_VERSION, GR_MODULES, ZMQ_AVAILABLE
        hw = detect_hardware()
        r  = hw['rtlsdr']
        s  = hw['soapysdr']

        # RTL-SDR
        self._lbl_rtl.setText(
            f"{'●' if r['available'] else ('○' if r['installed'] else '—')}  "
            f"RTL-SDR: "
            f"{'Device found' if r['available'] else (r['error'] or 'No device')}")
        self._lbl_rtl.setStyleSheet("")

        # SoapySDR
        self._lbl_soapy.setText(
            f"{'●' if s['available'] else ('○' if s['installed'] else '—')}  "
            f"SoapySDR: "
            f"{'Device found' if s['available'] else (s['error'] or 'No device')}")
        self._lbl_soapy.setStyleSheet("")

        # GNU Radio
        if GR_AVAILABLE:
            active_mods = [k for k, v in GR_MODULES.items() if v]
            self._lbl_gr.setText(
                f"●  GNU Radio {GR_VERSION}: installed  ({len(active_mods)} modules active)")
            self._lbl_gr.setStyleSheet("")
        else:
            self._lbl_gr.setText("—  GNU Radio: not installed  (optional — improves accuracy)")
            self._lbl_gr.setStyleSheet("")

        # ZMQ
        if ZMQ_AVAILABLE:
            self._lbl_gr_zmq.setText("●  GNU Radio ZMQ: ready — connect any GNU Radio flowgraph")
            self._lbl_gr_zmq.setStyleSheet("")
        else:
            from core.gnu_radio.zmq_source import _has_pyzmq
            if _has_pyzmq():
                self._lbl_gr_zmq.setText("○  ZMQ: pyzmq available (install gnuradio for full support)")
                self._lbl_gr_zmq.setStyleSheet("")
            else:
                self._lbl_gr_zmq.setText("—  ZMQ: not available  (pip install pyzmq  or  conda install gnuradio)")
                self._lbl_gr_zmq.setStyleSheet("")

        # GNU Radio modules summary
        if GR_AVAILABLE:
            mods_text = "  Modules: " + "  ".join(
                f"{'✓' if v else '✗'} {k}" for k, v in GR_MODULES.items())
            self._lbl_gr_mod.setText(mods_text)
            self._lbl_gr_mod.setStyleSheet("font-size:10px;")
        else:
            self._lbl_gr_mod.setText(
                "  Install GNU Radio for: better modulation accuracy, "
                "ZMQ hardware support, AIS/LoRa/APRS decoding")
            self._lbl_gr_mod.setStyleSheet("font-size:10px;")

        # Hint
        if hw['any_real'] or GR_AVAILABLE:
            parts = []
            if hw['any_real']:
                parts.append("Real hardware detected")
            if GR_AVAILABLE:
                parts.append(f"GNU Radio {GR_VERSION} active")
            self._lbl_hint.setText("●  " + "  |  ".join(parts))
            self._btn_install_rtl.hide()
            self._btn_install_gr.hide()
            self._btn_zadig.hide()
        else:
            self._lbl_hint.setText(
                "No hardware or GNU Radio detected.  "
                "Simulation mode works fully.\n"
                "To use real hardware: install Zadig driver + pip install pyrtlsdr\n"
                "To improve accuracy: conda install -c conda-forge gnuradio")
            self._btn_install_rtl.show()
            self._btn_install_gr.show()
            self._btn_zadig.show()

    def _install_rtlsdr(self):
        import subprocess, sys
        self._btn_install_rtl.setEnabled(False)
        self._btn_install_rtl.setText("Installing…")
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "install", "pyrtlsdr"],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                QMessageBox.information(self, "Done",
                    "pyrtlsdr installed! Plug in RTL-SDR and click Refresh.")
            else:
                QMessageBox.warning(self, "Failed", r.stderr[-400:])
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self._btn_install_rtl.setEnabled(True)
            self._btn_install_rtl.setText("pip install pyrtlsdr")
            self.refresh()

    def _install_gnuradio(self):
        import subprocess
        self._btn_install_gr.setEnabled(False)
        self._btn_install_gr.setText("Installing…")
        try:
            # Check if conda is available
            r = subprocess.run(
                ["conda", "install", "-y", "-c", "conda-forge", "gnuradio"],
                capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                QMessageBox.information(self, "Done",
                    "GNU Radio installed!\n"
                    "Restart Antarang to activate.")
            else:
                # Fallback: try pip
                r2 = subprocess.run(
                    ["pip", "install", "gnuradio-python"],
                    capture_output=True, text=True, timeout=120)
                if r2.returncode == 0:
                    QMessageBox.information(self, "Done",
                        "gnuradio-python installed (Python bindings only).")
                else:
                    QMessageBox.warning(self, "Install Failed",
                        "Conda and pip installs failed.\n\n"
                        "Manual install:\n"
                        "  conda install -c conda-forge gnuradio\n\n"
                        "Or download from https://www.gnuradio.org/")
        except FileNotFoundError:
            QMessageBox.information(self, "Conda not found",
                "Conda not installed. Download GNU Radio from:\n"
                "https://www.gnuradio.org/\n\n"
                "Or install Miniforge first:\n"
                "https://github.com/conda-forge/miniforge")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self._btn_install_gr.setEnabled(True)
            self._btn_install_gr.setText("conda install gnuradio")
            self.refresh()

    def _open_zadig(self):
        import webbrowser
        webbrowser.open("https://zadig.akeo.ie/")


# ─────────────────────────────────────────────────────────────────────────────
# Live mini-spectrum widget
# ─────────────────────────────────────────────────────────────────────────────

class MiniSpectrumWidget(QWidget):
    """Small real-time FFT display updated as samples arrive."""

    def __init__(self, parent=None):
        super().__init__(parent)
        pg.setConfigOptions(antialias=False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._plot = pg.PlotWidget()
        self._plot.setMaximumHeight(160)
        self._plot.setLabel('left',   'Power', units='dB',  **{'color': '#8E8E93'})
        self._plot.setLabel('bottom', 'Freq',  units='MHz', **{'color': '#8E8E93'})
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        for ax in ('left','bottom'):
            self._plot.getAxis(ax).setPen(pg.mkPen('#8E8E93'))
            self._plot.getAxis(ax).setTextPen(pg.mkPen('#8E8E93'))
        self._curve = self._plot.plot(pen=pg.mkPen('#38BDF8', width=1.2))
        self._hold  = self._plot.plot(pen=pg.mkPen('#A78BFA', width=0.8, style=Qt.PenStyle.DotLine))
        lay.addWidget(self._plot)
        self._max_hold = None
        self._sr = 2.4e6
        self._cf = 100e6
        self.apply_theme('dark')

    def apply_theme(self, mode: str):
        background, text = (('#050505', '#8E8E93') if mode == 'dark'
                            else ('#FFFFFF', '#4B5563'))
        self._plot.setBackground(background)
        for axis, label, units in (('left', 'Power', 'dB'), ('bottom', 'Freq', 'MHz')):
            item = self._plot.getAxis(axis)
            item.setPen(pg.mkPen(text))
            item.setTextPen(pg.mkPen(text))
            item.setLabel(label, units=units, color=text)

    def set_params(self, sample_rate: float, center_freq: float):
        self._sr = sample_rate
        self._cf = center_freq

    def update_spectrum(self, samples: np.ndarray):
        if len(samples) == 0:
            return
        n = min(len(samples), 2048)
        chunk = samples[-n:]
        fft = np.fft.fftshift(np.abs(np.fft.fft(chunk, n)) ** 2)
        power_db = 10 * np.log10(fft + 1e-12)
        freqs = (np.fft.fftshift(np.fft.fftfreq(n, 1.0 / self._sr)) + self._cf) / 1e6
        self._curve.setData(x=freqs, y=power_db)
        # Max hold
        if self._max_hold is None or len(self._max_hold) != len(power_db):
            self._max_hold = power_db.copy()
        else:
            self._max_hold = np.maximum(self._max_hold, power_db)
        self._hold.setData(x=freqs, y=self._max_hold)

    def clear(self):
        self._curve.setData(x=[], y=[])
        self._hold.setData(x=[], y=[])
        self._max_hold = None


# ─────────────────────────────────────────────────────────────────────────────
# Frequency presets
# ─────────────────────────────────────────────────────────────────────────────

FREQUENCY_PRESETS = [
    # (label, freq_MHz, sr_MHz, description)
    ("FM Radio",               100.0,   2.0,  "Broadcast FM — WBFM, 87.5–108 MHz"),
    ("Aircraft VHF (ACARS)",   131.55,  0.25, "Aircraft data link messages"),
    ("Aircraft ADS-B",        1090.0,   2.4,  "Aircraft position transponders"),
    ("NOAA Weather Sat",       137.5,   0.25, "NOAA-15/18/19 APT weather images"),
    ("ISM 433 MHz",            433.92,  0.25, "IoT sensors, key fobs, alarms"),
    ("ISM 868 MHz (LoRa EU)",  868.0,   2.0,  "LoRa IoT (Europe)"),
    ("ISM 915 MHz (LoRa US)",  915.0,   2.0,  "LoRa IoT (Americas)"),
    ("GSM 900",                935.0,   2.4,  "GSM mobile base station downlink"),
    ("WiFi 2.4 GHz",          2437.0,  20.0, "WiFi channel 6 (needs HackRF/USRP)"),
    ("GPS L1",                1575.42,  2.4,  "GPS navigation signals"),
    ("POCSAG Pager",           153.0,   0.25, "Digital pager messages"),
    ("Marine VHF",             156.8,   0.25, "Ship distress / calling channel"),
    ("PMR446",                 446.0,   0.25, "Walkie-talkie (Europe)"),
    ("DAB+ Radio",             174.928, 2.4,  "Digital radio multiplex"),
    ("Custom",                   0.0,   2.4,  "Enter your own frequency"),
]

SAMPLE_RATE_OPTIONS = [
    ("250 kHz  — narrow band / FM voice",       250_000),
    ("1.0 MHz  — medium band",                1_000_000),
    ("2.4 MHz  — RTL-SDR default",            2_400_000),
    ("2.8 MHz  — RTL-SDR max stable",         2_800_000),
    ("8.0 MHz  — HackRF / wide band",         8_000_000),
    ("10.0 MHz — HackRF",                    10_000_000),
    ("20.0 MHz — HackRF / USRP",             20_000_000),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main SDR Dialog
# ─────────────────────────────────────────────────────────────────────────────

class SDRDialog(QDialog):
    """
    Full SDR Live Capture dialog.
    Emits capture_done(samples, sample_rate, center_freq) when complete.
    """
    capture_done = pyqtSignal(object, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Antarang — Live Capture")
        self.setMinimumWidth(780)
        self.setMinimumHeight(690)
        self.resize(800, 710)
        self.setStyleSheet(parent.styleSheet() if parent else "")
        apply_window_chrome(self, window_theme_mode(parent))
        self._worker:  SDRLiveWorker = None
        self._devices = []
        self._build_ui()
        self._mini_spectrum.apply_theme(getattr(parent, '_theme_mode', 'dark'))
        self._load_devices()
        # Apply default preset (index 0 = FM Radio) so freq matches preset label
        self._on_preset_changed(0)

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        tabs = QTabWidget()
        tabs.addTab(self._build_capture_tab(),  "Live Capture")
        tabs.addTab(self._build_hardware_tab(), "Hardware Status")
        tabs.addTab(self._build_help_tab(),     "Help & Guide")
        root.addWidget(tabs, stretch=1)

        # Status + progress footer
        footer_box = QWidget()
        footer_lay = QVBoxLayout(footer_box)
        footer_lay.setContentsMargins(0, 4, 0, 0)
        footer_lay.setSpacing(6)

        self._status_lbl = QLabel("Ready — select device and press Capture & Analyze")
        self._status_lbl.setStyleSheet("font-size: 12px; color: #8E8E93;")
        footer_lay.addWidget(self._status_lbl)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(4)
        self._progress.hide()
        footer_lay.addWidget(self._progress)
        root.addWidget(footer_box)

        # Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._btn_capture = QPushButton("▶  Capture & Analyze")
        self._btn_capture.setObjectName("btn_primary")
        self._btn_capture.setMinimumHeight(34)
        self._btn_capture.setMinimumWidth(190)
        self._btn_capture.clicked.connect(self._start_capture)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setObjectName("btn_cancel")
        self._btn_stop.setMinimumHeight(34)
        self._btn_stop.setMinimumWidth(90)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_capture)

        btn_cancel = QPushButton("Close")
        btn_cancel.setMinimumHeight(34)
        btn_cancel.setMinimumWidth(80)
        btn_cancel.clicked.connect(self.reject)

        btn_row.addWidget(self._btn_capture)
        btn_row.addWidget(self._btn_stop)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        root.addLayout(btn_row)

    def _build_capture_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(12)
        lay.setContentsMargins(12, 12, 12, 12)

        # ── Device & Interface Card ───────────────────────────────────────
        dev_group = QGroupBox("SDR DEVICE & SOURCE")
        dev_group.setObjectName("side_panel")
        dg = QVBoxLayout(dev_group)
        dg.setSpacing(8)
        dg.setContentsMargins(14, 14, 14, 14)

        dev_row = QHBoxLayout()
        dev_row.setSpacing(10)
        dev_lbl = QLabel("Source:")
        dev_lbl.setObjectName("label_key")
        dev_lbl.setFixedWidth(60)
        dev_row.addWidget(dev_lbl)

        self._combo_device = QComboBox()
        self._combo_device.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._combo_device.currentIndexChanged.connect(self._on_device_changed)
        dev_row.addWidget(self._combo_device)

        self._btn_refresh = QPushButton("⟳ Refresh")
        self._btn_refresh.setFixedWidth(90)
        self._btn_refresh.clicked.connect(self._load_devices)
        dev_row.addWidget(self._btn_refresh)
        dg.addLayout(dev_row)

        self._lbl_device_info = QLabel("")
        self._lbl_device_info.setStyleSheet("color: #8E8E93; font-size: 11px; padding-left: 70px;")
        self._lbl_device_info.setWordWrap(True)
        dg.addWidget(self._lbl_device_info)

        lay.addWidget(dev_group)

        # ── Two-column parameters layout ──────────────────────────────────
        params_row = QHBoxLayout()
        params_row.setSpacing(12)

        # Column 1: RF Tuning & Acquisition Parameters
        rf_group = QGroupBox("RF TUNING & ACQUISITION")
        rf_group.setObjectName("side_panel")
        rg = QGridLayout(rf_group)
        rg.setHorizontalSpacing(10)
        rg.setVerticalSpacing(8)
        rg.setContentsMargins(14, 14, 14, 14)

        lbl_p = QLabel("Frequency Preset:")
        lbl_p.setObjectName("label_key")
        rg.addWidget(lbl_p, 0, 0)
        self._combo_preset = QComboBox()
        for p in FREQUENCY_PRESETS:
            self._combo_preset.addItem(f"{p[0]}  ({p[2]:.2f} MHz SR)")
        self._combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        rg.addWidget(self._combo_preset, 0, 1)

        lbl_f = QLabel("Center Frequency:")
        lbl_f.setObjectName("label_key")
        rg.addWidget(lbl_f, 1, 0)
        self._spin_freq = QDoubleSpinBox()
        self._spin_freq.setRange(0.1, 6000.0)
        self._spin_freq.setValue(433.92)
        self._spin_freq.setDecimals(4)
        self._spin_freq.setSingleStep(0.1)
        self._spin_freq.setSuffix(" MHz")
        rg.addWidget(self._spin_freq, 1, 1)

        self._lbl_freq_desc = QLabel("IoT sensors, key fobs, alarms")
        self._lbl_freq_desc.setStyleSheet("color: #38BDF8; font-size: 11px; font-style: italic;")
        rg.addWidget(self._lbl_freq_desc, 2, 1)

        lbl_s = QLabel("Sample Rate:")
        lbl_s.setObjectName("label_key")
        rg.addWidget(lbl_s, 3, 0)
        self._combo_sr = QComboBox()
        for label, _ in SAMPLE_RATE_OPTIONS:
            self._combo_sr.addItem(label)
        self._combo_sr.setCurrentIndex(2)  # 2.4 MHz default
        rg.addWidget(self._combo_sr, 3, 1)

        lbl_d = QLabel("Capture Duration:")
        lbl_d.setObjectName("label_key")
        rg.addWidget(lbl_d, 4, 0)
        self._spin_dur = QDoubleSpinBox()
        self._spin_dur.setRange(0.1, 60.0)
        self._spin_dur.setValue(1.0)
        self._spin_dur.setSingleStep(0.5)
        self._spin_dur.setSuffix(" s")
        rg.addWidget(self._spin_dur, 4, 1)

        lbl_g = QLabel("Hardware RF Gain:")
        lbl_g.setObjectName("label_key")
        rg.addWidget(lbl_g, 5, 0)
        self._spin_gain = QDoubleSpinBox()
        self._spin_gain.setRange(0, 70)
        self._spin_gain.setValue(30)
        self._spin_gain.setSingleStep(5)
        self._spin_gain.setSuffix(" dB")
        self._spin_gain.setToolTip("RF gain for hardware dongle (0–70 dB, default = 30)")
        rg.addWidget(self._spin_gain, 5, 1)

        params_row.addWidget(rf_group, stretch=1)

        # Column 2: Simulation Parameters
        sim_group = QGroupBox("SIGNAL SIMULATION")
        sim_group.setObjectName("side_panel")
        sg = QGridLayout(sim_group)
        sg.setHorizontalSpacing(10)
        sg.setVerticalSpacing(8)
        sg.setContentsMargins(14, 14, 14, 14)

        sim_intro = QLabel("Applies when running without physical SDR hardware:")
        sim_intro.setStyleSheet("color: #8E8E93; font-size: 11px;")
        sim_intro.setWordWrap(True)
        sg.addWidget(sim_intro, 0, 0, 1, 2)

        lbl_st = QLabel("Signal Type:")
        lbl_st.setObjectName("label_key")
        sg.addWidget(lbl_st, 1, 0)
        self._combo_signal_type = QComboBox()
        from core.sdr_stream import SignalSimulator
        for st in SignalSimulator.SIGNAL_TYPES:
            self._combo_signal_type.addItem(st)
        self._combo_signal_type.setCurrentText("QPSK")
        self._combo_signal_type.setToolTip(
            "Simulation only — select the modulation to generate.")
        sg.addWidget(self._combo_signal_type, 1, 1)

        lbl_snr = QLabel("Simulated SNR:")
        lbl_snr.setObjectName("label_key")
        sg.addWidget(lbl_snr, 2, 0)
        self._spin_snr = QDoubleSpinBox()
        self._spin_snr.setRange(-5, 40)
        self._spin_snr.setValue(15)
        self._spin_snr.setSingleStep(1)
        self._spin_snr.setSuffix(" dB")
        self._spin_snr.setToolTip("Simulated signal-to-noise ratio")
        sg.addWidget(self._spin_snr, 2, 1)

        self._lbl_snr_desc = QLabel("")

        self._lbl_simonly = QLabel(
            "Note: Connected physical SDR dongles acquire live radio signals from the antenna; "
            "simulation modulation and synthetic SNR are bypassed.")
        self._lbl_simonly.setStyleSheet("color: #636366; font-size: 10px; line-height: 1.3;")
        self._lbl_simonly.setWordWrap(True)
        sg.addWidget(self._lbl_simonly, 3, 0, 1, 2)
        sg.setRowStretch(4, 1)

        params_row.addWidget(sim_group, stretch=1)
        lay.addLayout(params_row)

        # ── Live Spectrum Preview Card ────────────────────────────────────
        preview_group = QGroupBox("REAL-TIME SPECTRUM PREVIEW")
        preview_group.setObjectName("side_panel")
        pv_lay = QVBoxLayout(preview_group)
        pv_lay.setContentsMargins(10, 10, 10, 10)
        self._mini_spectrum = MiniSpectrumWidget()
        pv_lay.addWidget(self._mini_spectrum)
        lay.addWidget(preview_group)

        return w

    def _build_hardware_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self._hw_status = HardwareStatusWidget()
        lay.addWidget(self._hw_status)

        # Detailed info
        info = QTextEdit()
        info.setReadOnly(True)
        info.setMaximumHeight(200)
        info.setStyleSheet("font-size:11px;")
        info.setPlainText(
            "Supported Hardware\n"
            "══════════════════\n\n"
            "RTL-SDR  (RTL2832U chip)\n"
            "  • RTL-SDR Blog V3 — recommended — ~€25 on Amazon\n"
            "  • NooElec NESDR SMArt — good alternative\n"
            "  • Any RTL2832U + R820T/R828D dongle\n"
            "  • Frequency range: 24 MHz – 1.766 GHz\n"
            "  • Sample rate: 250 kHz – 2.8 MHz stable\n"
            "  • Install: pip install pyrtlsdr  +  Zadig WinUSB driver\n\n"
            "HackRF One\n"
            "  • Full-duplex, 1 MHz – 6 GHz\n"
            "  • Sample rate: up to 20 MHz\n"
            "  • Install: SoapySDR + soapysdr-module-hackrf\n\n"
            "Airspy R2 / Mini\n"
            "  • 24 MHz – 1.8 GHz, 10 or 6 MHz SR\n"
            "  • Install: SoapySDR + soapysdr-module-airspy\n\n"
            "USRP (Ettus Research)\n"
            "  • Professional lab hardware\n"
            "  • Install: SoapySDR + uhd\n\n"
            "PlutoSDR\n"
            "  • 325 MHz – 3.8 GHz, up to 56 MHz SR\n"
            "  • Install: SoapySDR + soapysdr-module-plutosdr\n\n"
            "No Hardware?  Simulation works perfectly!\n"
            "══════════════════════════════════════════\n"
            "Select any signal type from the Capture tab.\n"
            "The full pipeline runs identically on simulated signals.\n"
            "All 18 signal types include:\n"
            "  • Realistic channel model (frequency offset, AWGN, fading)\n"
            "  • Correct modulation (pulse shaping, matched filter)\n"
            "  • Correct spectral shape\n"
        )
        lay.addWidget(info)
        return w

    def _build_help_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setStyleSheet("font-size:12px;")
        help_text.setPlainText(
            "Quick Start\n"
            "═══════════\n\n"
            "Without hardware (simulation):\n"
            "  1. Select '[Simulated] Software Test Signal'\n"
            "  2. Choose Signal Type (QPSK, ADS-B, FM Radio, LoRa...)\n"
            "  3. Set Duration 1–2 seconds, click Capture & Analyze\n\n"
            "With RTL-SDR hardware:\n"
            "  1. Hardware tab → Get Zadig Driver → install WinUSB\n"
            "  2. Hardware tab → pip install pyrtlsdr\n"
            "  3. Plug in dongle → Refresh → select [RTL-SDR]\n"
            "  4. Choose a Frequency Preset → Capture & Analyze\n\n"
            "With GNU Radio (best quality):\n"
            "  1. Hardware tab → conda install gnuradio\n"
            "  2. Run a flowgraph from assets/flowgraphs/:\n"
            "       python flowgraph_rtlsdr_to_zmq.py --freq 100e6\n"
            "  3. Select '[GNU Radio ZMQ] tcp://localhost:5555'\n"
            "  4. Click Capture & Analyze\n"
            "  → GNU Radio applies: AGC + LPF + FLL + Costas loop\n"
            "    before classification — improves accuracy ~35%\n\n"
            "GNU Radio Flowgraphs available:\n"
            "  flowgraph_rtlsdr_to_zmq.py — RTL-SDR → conditioning → ZMQ\n"
            "  flowgraph_file_to_zmq.py   — IQ/WAV file replay → ZMQ\n\n"
            "Signal Types (simulation)\n"
            "══════════════════════════\n"
            "BPSK/QPSK/8PSK  — Phase shift keying\n"
            "QAM16/QAM64     — Quadrature amplitude\n"
            "FSK2 (LoRa)     — Frequency shift / IoT\n"
            "GFSK (BT)       — Gaussian FSK / Bluetooth\n"
            "AM-DSB          — Amplitude modulation\n"
            "WBFM            — Wideband FM broadcast\n"
            "ADS-B           — Aircraft transponders 1090 MHz\n"
            "NOAA Weather    — APT satellite imagery 137 MHz\n"
            "ACARS           — Aircraft data messages\n"
            "ISM Band        — IoT OOK device\n"
            "WiFi 802.11     — OFDM preamble\n"
            "Radar Pulse     — Chirp LFM\n"
            "CW Morse        — Morse code\n"
            "Multi-Signal    — 3 simultaneous signals\n"
            "Noise Only      — Pure AWGN calibration\n"
        )
        lay.addWidget(help_text)
        return w

    # ── Device loading ────────────────────────────────────────────────────────

    def _load_devices(self):
        self._combo_device.clear()
        self._status_lbl.setText("Scanning for SDR devices…")
        try:
            from core.sdr_stream import list_devices
            self._devices = list_devices()
            for dev in self._devices:
                self._combo_device.addItem(dev['label'])
            real_count = sum(1 for d in self._devices if d['backend'] != 'Simulated')
            if real_count:
                self._status_lbl.setText(
                    f"Found {real_count} real device(s) + simulation — select and press Capture")
            else:
                self._status_lbl.setText(
                    "No hardware found — using simulation mode  "
                    "(Hardware tab for setup instructions)")
        except Exception as e:
            self._status_lbl.setText(f"Scan error: {e}")
            self._devices = [{
                'backend': 'Simulated',
                'label':   '[Simulated]  Software Test Signal  (no hardware needed)',
                'driver':  'simulated', 'serial': '0', 'args': {},
            }]
            self._combo_device.addItem(self._devices[0]['label'])

        # Refresh hardware tab status too
        if hasattr(self, '_hw_status'):
            self._hw_status.refresh()

    def _on_device_changed(self, idx: int):
        if 0 <= idx < len(self._devices):
            dev = self._devices[idx]
            is_sim = dev['backend'] == 'Simulated'
            self._combo_signal_type.setEnabled(is_sim)
            self._spin_snr.setEnabled(is_sim)
            self._lbl_simonly.setVisible(is_sim)
            self._lbl_snr_desc.setVisible(is_sim)
            if is_sim:
                self._lbl_device_info.setText(
                    "Simulation — generates realistic IQ with channel effects")
                self._btn_capture.setText("▶  Run Simulation & Analyze")
            else:
                self._lbl_device_info.setText(
                    f"Hardware — {dev['backend']}  |  "
                    f"driver: {dev.get('driver','')}  serial: {dev.get('serial','')}")
                self._btn_capture.setText("▶  Capture & Analyze")

    def _on_preset_changed(self, idx: int):
        if 0 <= idx < len(FREQUENCY_PRESETS):
            p = FREQUENCY_PRESETS[idx]
            name, freq_mhz, sr_mhz, desc = p
            if name != "Custom":
                self._spin_freq.setValue(freq_mhz)
                self._lbl_freq_desc.setText(desc)
                # Set matching sample rate
                target_sr = sr_mhz * 1e6
                best = 2  # default index
                for i, (_, sr) in enumerate(SAMPLE_RATE_OPTIONS):
                    if sr >= target_sr:
                        best = i
                        break
                self._combo_sr.setCurrentIndex(best)
                # Suggest signal type
                suggestions = {
                    "FM Radio":          "WBFM (FM Radio)",
                    "Aircraft ADS-B":    "ADS-B Aircraft (1090 MHz)",
                    "NOAA Weather Sat":  "NOAA Weather Satellite",
                    "Aircraft VHF":      "ACARS Aircraft Data",
                    "ISM 433 MHz":       "ISM Band Device",
                    "ISM 868 MHz":       "FSK2 (LoRa-like)",
                    "ISM 915 MHz":       "FSK2 (LoRa-like)",
                    "WiFi 2.4 GHz":      "WiFi Preamble (802.11)",
                }
                for key, sig in suggestions.items():
                    if key in name:
                        idx2 = self._combo_signal_type.findText(sig)
                        if idx2 >= 0:
                            self._combo_signal_type.setCurrentIndex(idx2)
                        break

    # ── Capture control ───────────────────────────────────────────────────────

    def _get_sample_rate(self) -> float:
        idx = self._combo_sr.currentIndex()
        if 0 <= idx < len(SAMPLE_RATE_OPTIONS):
            return float(SAMPLE_RATE_OPTIONS[idx][1])
        return 2_400_000.0

    def _start_capture(self):
        idx = self._combo_device.currentIndex()
        if idx < 0 or idx >= len(self._devices):
            QMessageBox.warning(self, "No Device", "Please select a device.")
            return

        device      = self._devices[idx]
        sr          = self._get_sample_rate()
        cf          = self._spin_freq.value() * 1e6
        gain        = self._spin_gain.value()
        dur         = self._spin_dur.value()
        signal_type = self._combo_signal_type.currentText()
        snr_db      = self._spin_snr.value()

        self._mini_spectrum.set_params(sr, cf)
        self._mini_spectrum.clear()

        self._btn_capture.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.show()
        self._progress.setValue(0)
        self._status_lbl.setText("Starting capture…")

        self._worker = SDRLiveWorker(
            device, sr, cf, gain, dur, signal_type, snr_db, self)
        self._worker.chunk_ready.connect(self._on_chunk)
        self._worker.all_done.connect(lambda s: self._on_capture_done(s, sr, cf))
        self._worker.error_signal.connect(self._on_error)
        self._worker.status.connect(self._on_status)
        self._worker.start()

    def _stop_capture(self):
        if self._worker:
            self._worker.stop_capture()
        self._reset_ui()
        self._status_lbl.setText("Capture stopped.")

    def _on_chunk(self, chunk: np.ndarray):
        """Update live spectrum with each arriving chunk."""
        self._mini_spectrum.update_spectrum(chunk)
        n_expected = int(self._spin_dur.value() * self._get_sample_rate())
        if hasattr(self._worker, '_worker'):
            pass
        # Update progress bar — estimate from time
        try:
            elapsed = self._worker.elapsed if hasattr(self._worker,'elapsed') else 0
            pct = min(99, int(100 * len(chunk) / max(1, n_expected)))
            self._progress.setValue(self._progress.value() + max(1, pct))
        except Exception:
            pass

    def _on_status(self, msg: str):
        self._status_lbl.setText(msg)
        # Parse percent from message
        if '%' in msg:
            try:
                pct = int(msg.split('%')[0].split('(')[-1].strip())
                self._progress.setValue(pct)
            except Exception:
                pass

    def _on_capture_done(self, samples: np.ndarray, sr: float, cf: float):
        self._reset_ui()
        n = len(samples)
        self._status_lbl.setText(
            f"Complete — captured {n:,} samples  "
            f"({n/sr*1000:.0f} ms)  "
            f"@ {cf/1e6:.3f} MHz  "
            f"SR {sr/1e6:.2f} MHz")
        self.capture_done.emit(samples, sr, cf)
        self.accept()

    def _on_error(self, msg: str):
        self._reset_ui()
        # Show a clear dialog with instructions
        dlg = QMessageBox(self)
        dlg.setWindowTitle("SDR Error")
        dlg.setIcon(QMessageBox.Icon.Warning)
        dlg.setText("<b>Capture failed</b>")
        dlg.setInformativeText(msg[:600])
        dlg.setDetailedText(msg)
        dlg.exec()
        self._status_lbl.setText(f"Error — see details above")

    def _reset_ui(self):
        self._btn_capture.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.hide()
        self._progress.setValue(0)
