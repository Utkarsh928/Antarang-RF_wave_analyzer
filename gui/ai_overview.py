"""
AI Overview Panel for Tarang Signal Analyzer.
Clean, consumer-friendly interface for signal analysis and plot interpretation.
All provider selection, fallback routing, and retry logic occur automatically behind the scenes.
"""
import base64
import logging
import threading
from typing import Optional, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLineEdit, QCheckBox, QLabel,
    QFrame, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QTextCursor, QTextCharFormat

from core.ai.config import AIConfig
from core.ai.provider_base import AIResponse
from core.ai.provider_manager import AIProviderManager
from core.ai.context_builder import AIContextBuilder
from core.ai.model_manager import AIModelManager
from gui.offline_ai_dialog import OfflineAIDialog

logger = logging.getLogger("Tarang.AI.Overview")


# ─────────────────────────────────────────────────────────────────────────────
# Background AI Worker
# ─────────────────────────────────────────────────────────────────────────────
class _AIWorker(QObject):
    token_received = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(
        self,
        provider_manager: AIProviderManager,
        prompt: str,
        context: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
    ):
        super().__init__()
        self.pm = provider_manager
        self.prompt = prompt
        self.context = context
        self.image_bytes = image_bytes
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            response = self.pm.generate(
                prompt=self.prompt,
                context=self.context,
                image_bytes=self.image_bytes,
                on_token=self._on_token,
                cancel_check=lambda: self._cancelled,
            )
            self.finished.emit(response)
        except Exception as e:
            logger.exception("AI Worker unexpected error")
            err_resp = AIResponse(
                provider="System",
                model="None",
                success=False,
                error="Online AI is temporarily unavailable.\n\nPlease try again.",
            )
            self.finished.emit(err_resp)

    def _on_token(self, token: str):
        if not self._cancelled:
            self.token_received.emit(token)


# ─────────────────────────────────────────────────────────────────────────────
# Main AI Overview Widget
# ─────────────────────────────────────────────────────────────────────────────
class AIOverviewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._config = AIConfig()
        self._provider_manager = AIProviderManager(self._config)
        self._context_builder = AIContextBuilder()
        self._model_mgr = AIModelManager(self._config)

        self._analysis_data: Optional[Any] = None
        self._current_plot_bytes: Optional[bytes] = None
        self._current_plot_name: str = ""
        self._current_tab: str = ""

        self._worker: Optional[_AIWorker] = None
        self._thread: Optional[threading.Thread] = None
        self._theme_mode: str = "dark"
        self._is_streaming: bool = False

        self._build_ui()
        self.refresh_status()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ── Header Bar ───────────────────────────────────────────────
        header_layout = QHBoxLayout()
        title = QLabel("AI Overview")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Simple mode selector (Automatic / Online / Offline)
        self._combo_mode = QComboBox()
        self._combo_mode.addItems(["Automatic", "Online", "Offline"])
        self._combo_mode.setFont(QFont("Segoe UI", 9))
        self._combo_mode.setToolTip("Select AI execution mode")
        self._combo_mode.currentTextChanged.connect(self._on_mode_changed)
        header_layout.addWidget(self._combo_mode)

        layout.addLayout(header_layout)

        # ── Status & Offline AI Strip ────────────────────────────────
        status_bar = QHBoxLayout()

        self._lbl_status = QLabel("Online AI: Checking...")
        self._lbl_status.setFont(QFont("Segoe UI", 9))
        status_bar.addWidget(self._lbl_status)

        status_bar.addStretch()

        self._btn_offline_ai = QPushButton("Offline AI")
        self._btn_offline_ai.setObjectName("btn_offline")
        self._btn_offline_ai.setFont(QFont("Segoe UI", 8))
        self._btn_offline_ai.clicked.connect(self._open_offline_dialog)
        status_bar.addWidget(self._btn_offline_ai)

        layout.addLayout(status_bar)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("separator")
        layout.addWidget(sep)

        # ── Chat Area ─────────────────────────────────────────────────
        self._chat_area = QTextEdit()
        self._chat_area.setReadOnly(True)
        self._chat_area.setFont(QFont("Segoe UI", 10))
        self._chat_area.setMinimumHeight(280)
        layout.addWidget(self._chat_area, stretch=1)

        # ── Options Row ───────────────────────────────────────────────
        opts_layout = QHBoxLayout()
        self._chk_use_analysis = QCheckBox("Include signal measurements")
        self._chk_use_analysis.setChecked(True)
        self._chk_use_analysis.setToolTip("Pass live RF parameters to AI")
        opts_layout.addWidget(self._chk_use_analysis)

        opts_layout.addStretch()

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_generation)
        opts_layout.addWidget(self._btn_stop)

        self._btn_clear = QPushButton("Clear Chat")
        self._btn_clear.clicked.connect(self._clear_chat)
        opts_layout.addWidget(self._btn_clear)

        layout.addLayout(opts_layout)

        # ── Input Row ─────────────────────────────────────────────────
        input_layout = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask anything about the current signal...")
        self._input.setFont(QFont("Segoe UI", 10))
        self._input.returnPressed.connect(self._send_text)
        input_layout.addWidget(self._input, stretch=1)

        self._btn_send_plot = QPushButton("Analyze Plot")
        self._btn_send_plot.setObjectName("btn_ai")
        self._btn_send_plot.setToolTip("Send current plot view to AI for analysis")
        self._btn_send_plot.clicked.connect(self._send_with_plot)
        self._btn_send_plot.setEnabled(False)
        input_layout.addWidget(self._btn_send_plot)

        self._btn_send = QPushButton("Send")
        self._btn_send.setObjectName("btn_ai")
        self._btn_send.clicked.connect(self._send_text)
        input_layout.addWidget(self._btn_send)

        layout.addLayout(input_layout)

        self.apply_theme("dark")
        self._show_welcome_message()

    def refresh_status(self):
        """Updates clean consumer status indicators."""
        mode = self._config.ai_mode
        self._combo_mode.blockSignals(True)
        if mode == AIConfig.MODE_OFFLINE:
            self._combo_mode.setCurrentText("Offline")
        elif mode == AIConfig.MODE_ONLINE:
            self._combo_mode.setCurrentText("Online")
        else:
            self._combo_mode.setCurrentText("Automatic")
        self._combo_mode.blockSignals(False)

        # Check if online providers or offline model are available
        statuses = self._provider_manager.get_provider_statuses()
        online_available = any(
            statuses.get(k, {}).get("configured", False)
            for k in [AIConfig.PROVIDER_GEMINI, AIConfig.PROVIDER_GROQ, AIConfig.PROVIDER_OPENROUTER, AIConfig.PROVIDER_CEREBRAS]
        )
        offline_installed = self._model_mgr.is_installed()

        if mode == AIConfig.MODE_OFFLINE:
            if offline_installed:
                self._lbl_status.setText("<span style='color:#10B981; font-weight:bold;'>●</span> Offline AI: Ready")
            else:
                self._lbl_status.setText("<span style='color:#EF4444;'>○</span> Offline AI: Not installed")
        else:
            if online_available:
                self._lbl_status.setText("<span style='color:#10B981; font-weight:bold;'>●</span> Online AI: Available")
            elif offline_installed and mode == AIConfig.MODE_AUTO:
                self._lbl_status.setText("<span style='color:#10B981; font-weight:bold;'>●</span> Offline AI: Ready")
            else:
                self._lbl_status.setText("<span style='color:#8E8E93;'>○</span> Online AI: Unavailable")

        # Offline button label
        if offline_installed:
            self._btn_offline_ai.setText("Offline AI: Ready")
        else:
            self._btn_offline_ai.setText("Get Offline AI")

    def _on_mode_changed(self, text: str):
        mode_map = {
            "Automatic": AIConfig.MODE_AUTO,
            "Online": AIConfig.MODE_ONLINE,
            "Offline": AIConfig.MODE_OFFLINE,
        }
        mode = mode_map.get(text, AIConfig.MODE_AUTO)
        self._config.set_qsetting("mode", mode)
        self.refresh_status()

    def _open_offline_dialog(self):
        dlg = OfflineAIDialog(self, self._config)
        dlg.sig_offline_status_changed.connect(lambda _: self.refresh_status())
        dlg.exec()
        self.refresh_status()

    def _show_welcome_message(self):
        self._append_ai_message(
            "Hello! I am your Tarang signal intelligence assistant.\n\n"
            "Ask me anything about your signal, detected modulation, SNR, bandwidth, or protocols.\n"
            "You can also click 'Analyze Plot' while viewing Spectrum, Waterfall, or Constellation to inspect the visual display."
        )

    def set_analysis_data(self, data: Any):
        self._analysis_data = data

    def set_signal_info(self, signal_info: Any):
        self._analysis_data = signal_info

    def set_current_plot_image(self, tab_name: str, image_bytes: Optional[bytes]):
        self._current_tab = tab_name
        self._current_plot_name = tab_name
        self._current_plot_bytes = image_bytes
        self._btn_send_plot.setEnabled(bool(image_bytes))

    def apply_theme(self, mode: str):
        """Preserves conversation text and readability across theme changes."""
        self._theme_mode = mode
        is_dark = (mode == "dark")

        bg = "#050505" if is_dark else "#FFFFFF"
        text = "#F2F2F7" if is_dark else "#111318"
        border = "#2C2C2E" if is_dark else "#D9DDE3"

        self._chat_area.setStyleSheet(
            f"QTextEdit {{ background: {bg}; color: {text}; border: 1px solid {border}; border-radius: 8px; padding: 6px; }}"
        )
        self._btn_offline_ai.setStyleSheet(
            f"QPushButton#btn_offline {{ border: 1px solid {border}; border-radius: 6px; padding: 2px 8px; background: transparent; color: {text}; }}"
        )

        cursor = self._chat_area.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(text))
        cursor.mergeCharFormat(fmt)
        self._chat_area.setTextColor(QColor(text))

        self.refresh_status()

    def _clear_chat(self):
        self._chat_area.clear()
        self._show_welcome_message()

    def _stop_generation(self):
        if self._worker:
            self._worker.cancel()
            self._append_system_note("[Generation cancelled]")
        self._is_streaming = False
        self._btn_stop.setEnabled(False)
        self._btn_send.setEnabled(True)
        if self._current_plot_bytes:
            self._btn_send_plot.setEnabled(True)

    def _send_text(self):
        question = self._input.text().strip()
        if not question:
            return
        self._input.clear()
        self._execute_query(question, image_bytes=None)

    def _send_with_plot(self):
        question = self._input.text().strip()
        if not question:
            tab = self._current_tab or "signal"
            question = f"Analyze the current {tab} plot display and explain what you see."
        self._input.clear()
        self._execute_query(question, image_bytes=self._current_plot_bytes)

    def _execute_query(self, question: str, image_bytes: Optional[bytes]):
        if self._worker:
            self._worker.cancel()

        context_str = None
        if self._chk_use_analysis.isChecked() and self._analysis_data:
            context_str = self._context_builder.build_context(self._analysis_data)

        self._append_user_message(question)
        if image_bytes:
            tab = self._current_tab or "Plot"
            self._append_system_note(f"[{tab} display attached]")

        self._append_ai_prefix()

        self._btn_send.setEnabled(False)
        self._btn_send_plot.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._is_streaming = True

        self._worker = _AIWorker(
            provider_manager=self._provider_manager,
            prompt=question,
            context=context_str,
            image_bytes=image_bytes,
        )
        self._worker.token_received.connect(self._on_token)
        self._worker.finished.connect(self._on_finished)

        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def _on_token(self, token: str):
        is_dark = (self._theme_mode == "dark")
        self._chat_area.setTextColor(QColor("#F2F2F7" if is_dark else "#111318"))
        self._chat_area.moveCursor(QTextCursor.MoveOperation.End)
        self._chat_area.insertPlainText(token)
        self._chat_area.ensureCursorVisible()

    def _on_finished(self, response: AIResponse):
        self._is_streaming = False
        self._btn_stop.setEnabled(False)
        self._btn_send.setEnabled(True)
        if self._current_plot_bytes:
            self._btn_send_plot.setEnabled(True)

        if not response.success:
            self._append_error_message(response.error or "AI service is currently unavailable.")

        self._chat_area.append("")
        self.refresh_status()

    def _append_user_message(self, text: str):
        is_dark = (self._theme_mode == "dark")
        self._chat_area.setTextColor(QColor("#3B82F6" if is_dark else "#2563EB"))
        self._chat_area.setFontWeight(700)
        self._chat_area.append("You: ")
        self._chat_area.setFontWeight(400)
        self._chat_area.setTextColor(QColor("#F2F2F7" if is_dark else "#111318"))
        self._chat_area.insertPlainText(text)
        self._chat_area.append("")

    def _append_ai_prefix(self):
        is_dark = (self._theme_mode == "dark")
        self._chat_area.setTextColor(QColor("#10B981" if is_dark else "#059669"))
        self._chat_area.setFontWeight(700)
        self._chat_area.append("AI: ")
        self._chat_area.setFontWeight(400)
        self._chat_area.setTextColor(QColor("#F2F2F7" if is_dark else "#111318"))

    def _append_ai_message(self, text: str):
        self._append_ai_prefix()
        self._chat_area.insertPlainText(text)
        self._chat_area.append("")

    def _append_system_note(self, text: str):
        is_dark = (self._theme_mode == "dark")
        self._chat_area.setTextColor(QColor("#AEAEB2" if is_dark else "#6B7280"))
        self._chat_area.append(text)
        self._chat_area.setTextColor(QColor("#F2F2F7" if is_dark else "#111318"))

    def _append_error_message(self, error_text: str):
        self._chat_area.setTextColor(QColor("#EF4444"))
        self._chat_area.append(f"\n{error_text}\n")
        is_dark = (self._theme_mode == "dark")
        self._chat_area.setTextColor(QColor("#F2F2F7" if is_dark else "#111318"))


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback responses
# ─────────────────────────────────────────────────────────────────────────────
_RULES = [
    (['waterfall', 'spectrogram'],
     "A waterfall plot shows your signal over time. The horizontal axis is "
     "frequency, the vertical axis is time (newest at top or bottom). "
     "Color indicates signal power: blue = weak, yellow/red = strong. "
     "A bright horizontal band means a signal is present at that frequency. "
     "Vertical lines suggest brief bursts. Drift means the frequency is changing."),

    (['spectrum', 'fft', 'frequency'],
     "The spectrum plot shows signal power vs frequency (FFT). "
     "Each peak is a signal component. The tallest peak is your carrier. "
     "Width of the peak relates to bandwidth. "
     "The flat region at the bottom is your noise floor. "
     "SNR = difference between signal peak and noise floor in dB."),

    (['constellation', 'iq', 'scatter'],
     "The constellation plot shows IQ (In-phase vs Quadrature) symbols. "
     "Each dot is a received symbol. BPSK has 2 clusters, QPSK has 4, "
     "QAM16 has 16, QAM64 has 64. Tight clusters = clean signal. "
     "Spread/blurry clusters = noise or wrong modulation type selected. "
     "Rotation = phase offset (usually harmless)."),

    (['snr', 'signal to noise', 'noise'],
     "SNR (Signal-to-Noise Ratio) measures signal quality in dB. "
     "Above 20 dB = excellent, reliable decoding. "
     "10-20 dB = good, some errors possible. "
     "5-10 dB = marginal, expect FEC errors. "
     "Below 5 dB = poor, decoding may fail."),

    (['bpsk', 'qpsk', '8psk', 'psk', 'phase shift'],
     "PSK (Phase Shift Keying) encodes data in the phase of the carrier. "
     "BPSK uses 2 phases (1 bit/symbol). QPSK uses 4 phases (2 bits/symbol). "
     "8PSK uses 8 phases (3 bits/symbol). More phases = more data rate "
     "but needs better SNR. Look for circular clusters in the constellation."),

    (['qam', 'quadrature amplitude'],
     "QAM (Quadrature Amplitude Modulation) encodes data in both phase AND "
     "amplitude. QAM16 has 16 points (4 bits/symbol), QAM64 has 64 (6 bits/symbol). "
     "In the constellation you see a grid pattern. Requires good SNR "
     "— QAM64 needs ~25+ dB for reliable decoding."),

    (['fsk', 'frequency shift'],
     "FSK (Frequency Shift Keying) encodes bits by shifting the carrier frequency. "
     "2-FSK uses 2 frequencies (1 bit/symbol). 4-FSK uses 4 (2 bits/symbol). "
     "Constant amplitude, so robust to amplitude noise. "
     "In spectrum you'll see two or more peaks separated by the deviation."),

    (['fec', 'error correction', 'viterbi', 'reed solomon', 'ldpc'],
     "FEC (Forward Error Correction) adds redundancy so the receiver can "
     "fix bit errors without retransmission. "
     "Viterbi decodes convolutional codes — good for continuous streams. "
     "Reed-Solomon works on blocks of bytes — great for burst errors. "
     "LDPC (Low Density Parity Check) is modern and very efficient — used in 5G, WiFi. "
     "Concatenated combines two codes for extra protection."),

    (['interleaving', 'deinterleave', 'interleaver'],
     "Interleaving spreads burst errors across multiple codewords so FEC can "
     "correct them. Block interleaving writes by rows, reads by columns. "
     "Convolutional interleaving uses delay lines of increasing length. "
     "Diagonal and pseudo-random are more complex variants. "
     "De-interleaving reverses this process before FEC decoding."),

    (['header', 'payload', 'sync', 'correlation', 'preamble'],
     "The bit stream has a header (control info) and payload (actual data). "
     "A sync word or preamble marks where each frame starts. "
     "Cross-correlation slides the known pattern across the bit stream "
     "and finds where it matches best — that's the header location. "
     "Common sync words: 0x7E (HDLC), Barker codes (radar), CCSDS (space comms)."),

    (['iq', 'in-phase', 'quadrature', 'complex'],
     "IQ signals are complex-valued: I = In-phase (real), Q = Quadrature (imaginary). "
     "Together they can represent any amplitude and phase. "
     "IQ files store raw complex samples as interleaved float32 pairs: I0,Q0,I1,Q1... "
     "WAV files can store IQ as stereo: left channel = I, right channel = Q."),

    (['wav', 'wave', 'audio'],
     "WAV files store sampled audio or IQ data. "
     "They have a RIFF header with sample rate, bit depth, and channel count. "
     "8-bit, 16-bit, or 32-bit samples are common. "
     "For IQ signals, stereo WAV is used: left=I, right=Q. "
     "Sample rate tells you the highest frequency you can represent (Nyquist = SR/2)."),
]


def _rule_based_response(question: str) -> str:
    q = question.lower()
    for keywords, response in _RULES:
        if any(kw in q for kw in keywords):
            return response
    return (
        "That's a great question about signal analysis. "
        "For the most detailed answer, make sure Ollama is running "
        "with llava:7b loaded ('ollama run llava:7b'). "
        "In general: start by checking the spectrum for signal presence, "
        "then the waterfall for timing, then the constellation for modulation type. "
        "Run the pipeline: Demodulate → De-interleave → FEC → Correlate. "
        "If you share more details I can give a more specific answer."
    )
