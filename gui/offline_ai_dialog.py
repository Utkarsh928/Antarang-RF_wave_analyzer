"""
Offline AI Dialog for Tarang Signal Analyzer.
Presents a simple, consumer-friendly interface for optional local AI (Qwen2.5-1.5B).
Hardware auditing, dynamic manifest resolution, and local inference testing
are handled entirely on background worker threads without ever freezing the GUI.
"""
try:
    from core.ai.runtime_manager import safe_backend_init
    safe_backend_init()
except Exception:
    pass

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QFont

from core.ai.config import AIConfig
from core.ai.model_manager import AIModelManager, ModelStatus, ErrorCategory
from core.ai.compatibility import AICompatibilityChecker, CompatibilityResult
from gui.theme import apply_window_chrome, window_theme_mode

logger = logging.getLogger("Tarang.AI.OfflineDialog")


# ─────────────────────────────────────────────────────────────────────────────
# Background Workers (Guarantees zero GUI freezing)
# ─────────────────────────────────────────────────────────────────────────────
class _CompatibilityWorker(QObject):
    finished = pyqtSignal(object)

    def __init__(self, target_dir):
        super().__init__()
        self.target_dir = target_dir

    def run(self):
        res = AICompatibilityChecker.check_compatibility(self.target_dir)
        self.finished.emit(res)


class _DownloadWorker(QObject):
    progress_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, model_manager: AIModelManager):
        super().__init__()
        self.mm = model_manager

    def run(self):
        success, msg = self.mm.download_and_install(
            progress_callback=self._on_progress
        )
        self.finished_signal.emit(success, msg)

    def _on_progress(self, data: dict):
        self.progress_signal.emit(data)


class _SetupWorker(QObject):
    progress_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, model_manager: AIModelManager):
        super().__init__()
        self.mm = model_manager

    def run(self):
        success, msg = self.mm.verify_and_setup(
            progress_callback=self._on_progress
        )
        self.finished_signal.emit(success, msg)

    def _on_progress(self, data: dict):
        self.progress_signal.emit(data)


# ─────────────────────────────────────────────────────────────────────────────
# Main Dialog
# ─────────────────────────────────────────────────────────────────────────────
class OfflineAIDialog(QDialog):
    sig_offline_status_changed = pyqtSignal(bool)

    def __init__(self, parent=None, config: Optional[AIConfig] = None):
        super().__init__(parent)
        self.config = config or AIConfig()
        self.mm = AIModelManager(self.config)

        self._compat_thread: Optional[QThread] = None
        self._compat_worker: Optional[_CompatibilityWorker] = None

        self._download_thread: Optional[QThread] = None
        self._download_worker: Optional[_DownloadWorker] = None

        self._compat_result: Optional[CompatibilityResult] = None
        self._state: str = "IDLE"

        self.setWindowTitle("Offline AI")
        self.setMinimumWidth(460)
        self.setMinimumHeight(330)
        self.setStyleSheet(parent.styleSheet() if parent else "")
        apply_window_chrome(self, window_theme_mode(parent))

        self._build_ui()
        self._refresh_state()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)

        # Title & Subtitle
        title = QLabel("Offline AI")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel("Run signal intelligence locally on this computer with complete privacy.")
        desc.setStyleSheet("color: #8E8E93; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Main Status Card
        self._card = QFrame()
        self._card.setObjectName("offline_status_card")
        self._card.setStyleSheet(
            "#offline_status_card { background: rgba(128, 128, 128, 0.08); border-radius: 10px; padding: 14px; }"
        )
        card_lay = QVBoxLayout(self._card)
        card_lay.setSpacing(8)

        self._lbl_status = QLabel("Checking Offline AI status...")
        self._lbl_status.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        self._lbl_status.setWordWrap(True)
        card_lay.addWidget(self._lbl_status)

        self._lbl_details = QLabel("")
        self._lbl_details.setStyleSheet("color: #8E8E93; font-size: 11px;")
        self._lbl_details.setWordWrap(True)
        card_lay.addWidget(self._lbl_details)

        # Progress elements
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(6)
        card_lay.addWidget(self._progress_bar)

        self._lbl_progress = QLabel("")
        self._lbl_progress.setStyleSheet("color: #8E8E93; font-size: 10px;")
        self._lbl_progress.setVisible(False)
        card_lay.addWidget(self._lbl_progress)

        layout.addWidget(self._card)

        # Action Buttons
        self._btn_row = QHBoxLayout()

        self._btn_primary = QPushButton("Download Offline AI")
        self._btn_primary.setObjectName("btn_primary")
        self._btn_primary.clicked.connect(self._on_primary_clicked)
        self._btn_row.addWidget(self._btn_primary)

        self._btn_secondary = QPushButton("Close")
        self._btn_secondary.clicked.connect(self._on_secondary_clicked)
        self._btn_row.addWidget(self._btn_secondary)

        layout.addStretch()
        layout.addLayout(self._btn_row)

    def _refresh_state(self):
        if self.mm.is_installed():
            self._state = "READY"
            self._lbl_status.setText("Offline AI Ready")
            self._lbl_status.setStyleSheet("color: #10B981; font-size: 12px; font-weight: bold;")
            self._lbl_details.setText("Local model is installed and available for on-device signal analysis.")
            self._btn_primary.setText("Use Offline AI")
            self._btn_primary.setEnabled(True)
            self._btn_secondary.setText("Remove Model")
            self._btn_secondary.setEnabled(True)
            self._progress_bar.setVisible(False)
            self._lbl_progress.setVisible(False)
        else:
            self._state = "NOT_INSTALLED"
            self._lbl_status.setText("Offline AI not installed")
            self._lbl_status.setStyleSheet("color: #F2F2F7; font-size: 12px; font-weight: bold;")
            self._lbl_details.setText("A lightweight ~1.04 GB download is required for local offline analysis.")
            self._btn_primary.setText("Download Offline AI")
            self._btn_primary.setEnabled(True)
            self._btn_secondary.setText("Close")
            self._btn_secondary.setEnabled(True)
            self._progress_bar.setVisible(False)
            self._lbl_progress.setVisible(False)

    def _on_primary_clicked(self):
        if self.mm.is_installed():
            # Set mode to OFFLINE and close
            self.config.set_qsetting("mode", AIConfig.MODE_OFFLINE)
            self.sig_offline_status_changed.emit(True)
            self.accept()
            return

        if self._state in ("DOWNLOADING", "SETTING_UP"):
            return

        if self._state == "RUNTIME_UNAVAILABLE" or self._btn_primary.text() == "Retry":
            # Re-run compatibility / runtime check
            self._run_compat_check()
            return

        if self._state == "SETUP_FAILED" or self._btn_primary.text() in ("Retry Setup", "Set Up Offline AI"):
            self._start_setup_only()
            return

        if self._state in ("CONFIRM_DOWNLOAD", "CANCELLED", "FAILED") or self._btn_primary.text() in (
            "Download", "Resume Download", "Retry Download", "Download & Install"
        ):
            self._start_download()
            return

        # Start Async Compatibility Check (Never blocks GUI)
        self._run_compat_check()

    def _run_compat_check(self):
        self._state = "CHECKING_COMPATIBILITY"
        self._lbl_status.setText("Checking compatibility...")
        self._lbl_status.setStyleSheet("color: #3B82F6; font-size: 12px; font-weight: bold;")
        self._lbl_details.setText("Checking hardware and runtime compatibility...")
        self._btn_primary.setEnabled(False)

        self._compat_thread = QThread(self)
        self._compat_worker = _CompatibilityWorker(self.mm.model_dir)
        self._compat_worker.moveToThread(self._compat_thread)
        self._compat_thread.started.connect(self._compat_worker.run)
        self._compat_worker.finished.connect(self._on_compat_finished)
        self._compat_thread.start()

    def _on_compat_finished(self, res: CompatibilityResult):
        self._compat_result = res
        if self._compat_thread:
            self._compat_thread.quit()
            self._compat_thread.wait(500)
            self._compat_thread = None

        self._btn_primary.setEnabled(True)

        if res.status in ("RUNTIME_UNAVAILABLE", "RUNTIME_BROKEN"):
            self._state = "RUNTIME_UNAVAILABLE"
            self._lbl_status.setText("Offline AI setup couldn't be completed.")
            self._lbl_status.setStyleSheet("color: #EF4444; font-weight: bold;")
            self._lbl_details.setText("Tarang couldn't start its local AI engine.\n\nOnline AI remains available.")
            self._btn_primary.setText("Retry")
            self._btn_secondary.setText("Use Online AI")
            logger.error(f"Runtime pre-flight failed: {res.details.get('runtime_message') or res.details.get('runtime_error')}")
            return

        if res.status == "HARDWARE_UNSUPPORTED":
            self._state = "HARDWARE_UNSUPPORTED"
            self._lbl_status.setText("Offline AI cannot be installed on this computer.")
            self._lbl_status.setStyleSheet("color: #EF4444; font-weight: bold;")
            self._lbl_details.setText(res.summary)
            self._btn_primary.setText("Use Online AI")
            self._btn_secondary.setText("Close")
            return

        if res.status == "INSUFFICIENT DISK SPACE":
            self._state = "INSUFFICIENT_SPACE"
            self._lbl_status.setText("Insufficient disk space for Offline AI.")
            self._lbl_status.setStyleSheet("color: #EF4444; font-weight: bold;")
            self._lbl_details.setText(f"{res.summary}\n\nPlease free up disk space or use Online AI.")
            self._btn_primary.setText("Use Online AI")
            self._btn_secondary.setText("Close")
            return

        if res.status == "LIMITED":
            self._state = "CONFIRM_DOWNLOAD"
            self._lbl_status.setText("Compatible")
            self._lbl_status.setStyleSheet("color: #F59E0B; font-weight: bold;")
            self._lbl_details.setText("Download size: approximately 1.04 GB\n\nSystem memory is constrained.")
            self._btn_primary.setText("Download")
            self._btn_secondary.setText("Use Online AI")
            return

        # READY or READY — CPU MODE
        self._state = "CONFIRM_DOWNLOAD"
        self._lbl_status.setText("Compatible")
        self._lbl_status.setStyleSheet("color: #10B981; font-weight: bold;")
        self._lbl_details.setText("Download size: approximately 1.04 GB")
        self._btn_primary.setText("Download")
        self._btn_secondary.setText("Cancel")

    def _start_download(self):
        self._state = "DOWNLOADING"
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._lbl_progress.setVisible(True)
        self._lbl_status.setText("Downloading Offline AI...")
        self._lbl_status.setStyleSheet("color: #3B82F6; font-weight: bold;")
        self._lbl_details.setText("Downloading Qwen2.5-1.5B model files...")

        self._btn_primary.setEnabled(False)
        self._btn_secondary.setText("Cancel")

        self._download_thread = QThread(self)
        self._download_worker = _DownloadWorker(self.mm)
        self._download_worker.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.progress_signal.connect(self._on_download_progress)
        self._download_worker.finished_signal.connect(self._on_download_finished)
        self._download_thread.start()

    def _start_setup_only(self):
        self._state = "SETTING_UP"
        self._progress_bar.setValue(99)
        self._progress_bar.setVisible(True)
        self._lbl_progress.setVisible(True)
        self._lbl_status.setText("Initializing Offline AI...")
        self._lbl_status.setStyleSheet("color: #3B82F6; font-weight: bold;")
        self._lbl_details.setText("Setting up local inference engine...")

        self._btn_primary.setEnabled(False)
        self._btn_secondary.setText("Close")
        self._btn_secondary.setEnabled(True)

        self._download_thread = QThread(self)
        self._download_worker = _SetupWorker(self.mm)
        self._download_worker.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.progress_signal.connect(self._on_download_progress)
        self._download_worker.finished_signal.connect(self._on_download_finished)
        self._download_thread.start()

    def _on_download_progress(self, data: dict):
        status = data.get("status")
        pct = data.get("percentage", 0.0)
        self._progress_bar.setValue(int(pct))

        if status == ModelStatus.VERIFYING:
            self._lbl_status.setText("Verifying...")
            self._lbl_details.setText("Checking file integrity...")
            self._lbl_progress.setText("99%")
        elif status == ModelStatus.PREPARING:
            self._lbl_status.setText("Preparing Offline AI...")
            self._lbl_details.setText("Preparing local inference engine...")
            self._lbl_progress.setText("Finalizing...")
        elif status == ModelStatus.LOADING:
            self._lbl_status.setText("Loading Offline AI...")
            self._lbl_details.setText("Loading model into memory...")
            self._lbl_progress.setText("Finalizing...")
        elif status == ModelStatus.TESTING:
            self._lbl_status.setText("Testing Offline AI...")
            self._lbl_details.setText("Performing initial local runtime test...")
            self._lbl_progress.setText("Finalizing...")
        elif status == ModelStatus.DOWNLOADING:
            speed = data.get("speed_mb_s", 0.0)
            eta = data.get("eta_seconds", 0)
            dl_mb = round(data.get("downloaded_bytes", 0) / (1024 * 1024), 1)
            tot_mb = round(data.get("total_bytes", 0) / (1024 * 1024), 1)
            eta_str = f"~{eta // 60}m {eta % 60}s" if eta >= 60 else f"{eta}s"
            self._lbl_status.setText("Downloading Offline AI...")
            self._lbl_details.setText(f"{speed} MB/s — ETA: {eta_str}")
            self._lbl_progress.setText(
                f"{pct:.0f}%\n{dl_mb} MB / {tot_mb} MB"
            )

    def _on_download_finished(self, success: bool, msg: str):
        if self._download_thread:
            self._download_thread.quit()
            self._download_thread.wait(500)
            self._download_thread = None

        self._btn_primary.setEnabled(True)
        self._btn_secondary.setEnabled(True)

        # Always hide progress bar and label upon completion to prevent lingering text
        self._progress_bar.setVisible(False)
        self._lbl_progress.setVisible(False)

        # Readiness verification: if not reported success, check if model is actually installed and usable
        if not success and self.mm.is_installed():
            try:
                llm, _ = self.mm.load_model()
                if llm is not None:
                    test_res = llm("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", max_tokens=1)
                    if test_res and "choices" in test_res:
                        success = True
                        msg = "Offline AI Ready"
                        self.mm._status = ModelStatus.READY
                        self.mm._status_message = "Offline AI Ready"
            except Exception as e:
                logger.debug(f"Readiness verification check: {e}")

        if success or (self.mm.is_installed() and self.mm.status == ModelStatus.READY):
            self._state = "READY"
            self._lbl_status.setText("Offline AI Ready")
            self._lbl_status.setStyleSheet("color: #10B981; font-size: 12px; font-weight: bold;")
            self._lbl_details.setText("Installation complete. Local AI is ready for on-device analysis.")
            self.config.set_qsetting("mode", AIConfig.MODE_OFFLINE)
            self.sig_offline_status_changed.emit(True)
            self._btn_primary.setText("Use Offline AI")
            self._btn_secondary.setText("Remove Model")
        else:
            if "cancel" in msg.lower():
                self._state = "CANCELLED"
                self._lbl_status.setText("Download cancelled")
                self._lbl_status.setStyleSheet("color: #F59E0B; font-weight: bold;")
                self._lbl_details.setText("You can resume the download at any time.")
                self._btn_primary.setText("Resume Download")
                self._btn_secondary.setText("Close")
            elif self.mm.status == ModelStatus.SETUP_FAILED or "setup failed" in msg.lower():
                self._state = "SETUP_FAILED"
                self._lbl_status.setText("Offline AI setup couldn't be completed.")
                self._lbl_status.setStyleSheet("color: #EF4444; font-weight: bold;")
                self._lbl_details.setText("Tarang couldn't start its local AI engine.\n\nOnline AI remains available.")
                self._btn_primary.setText("Retry")
                self._btn_secondary.setText("Use Online AI")
                logger.error(f"Offline AI setup failure: {msg}")
            else:
                self._state = "FAILED"
                self._lbl_status.setText("Download failed")
                self._lbl_status.setStyleSheet("color: #EF4444; font-weight: bold;")
                clean_reason = msg
                if "404" in clean_reason:
                    clean_reason = "Model weights could not be reached on repository server."
                elif "space" in clean_reason.lower():
                    clean_reason = "Insufficient disk space to complete download."
                elif any(err in clean_reason.lower() for err in ["timeout", "connection", "network"]):
                    clean_reason = "Network connection was interrupted. Click Retry Download to resume."
                elif "llama" in clean_reason.lower() or "runtime" in clean_reason.lower() or "exception" in clean_reason.lower():
                    clean_reason = "Local runtime initialization failed."
                self._lbl_details.setText(f"Reason: {clean_reason}")
                self._btn_primary.setText("Retry Download")
                self._btn_secondary.setText("Close")

    def _on_secondary_clicked(self):
        if self._btn_secondary.text() == "Cancel":
            if self._state == "DOWNLOADING":
                self.mm.cancel_download()
                self._lbl_details.setText("Cancelling download...")
                self._btn_secondary.setEnabled(False)
                return
            self.reject()
            return

        if self._btn_secondary.text() == "Use Online AI":
            self.config.set_qsetting("mode", AIConfig.MODE_AUTO)
            self.sig_offline_status_changed.emit(False)
            self.accept()
            return

        if self._btn_secondary.text() == "Remove Model":
            ret = QMessageBox.question(
                self,
                "Remove Model",
                "Remove the offline AI model to free disk space?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if ret == QMessageBox.StandardButton.Yes:
                self.mm.remove_model()
                self.config.set_qsetting("mode", AIConfig.MODE_AUTO)
                self.sig_offline_status_changed.emit(False)
                self._refresh_state()
            return

        self.reject()

    def closeEvent(self, event):
        """Ensure threads and downloads are cleanly shut down on close."""
        if self._state == "DOWNLOADING":
            self.mm.cancel_download()
        if self._compat_thread and self._compat_thread.isRunning():
            self._compat_thread.quit()
            self._compat_thread.wait(300)
        if self._download_thread and self._download_thread.isRunning():
            self._download_thread.quit()
            self._download_thread.wait(300)
        super().closeEvent(event)
