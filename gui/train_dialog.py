"""
Training Dialog — lets user train or retrain the ML model from inside the app.
Shows real-time progress, accuracy result, and allows cancellation.
"""
import os
import sys
import threading
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QProgressBar, QTextEdit,
                               QSpinBox, QFormLayout, QGroupBox,
                               QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QFont, QColor
from gui.theme import apply_window_chrome, window_theme_mode


class _TrainWorkerSignals(QObject):
    log_line  = pyqtSignal(str)
    progress  = pyqtSignal(int)
    finished  = pyqtSignal(bool, str)   # success, message


class TrainingDialog(QDialog):
    model_trained = pyqtSignal()  # emitted when training completes successfully

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Antarang — Train Modulation Model")
        self.setMinimumSize(520, 400)
        self.setModal(True)
        if parent:
            self.setStyleSheet(parent.styleSheet())
        apply_window_chrome(self, window_theme_mode(parent))
        self._thread = None
        self._cancelled = False
        self._build_ui()
        self._check_existing_model()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Info
        info = QLabel(
            "Train the ML modulation classifier on synthetic IQ data.\n"
            "More samples = higher accuracy but longer training time.\n"
            "After training, the app uses ML instead of rule-based classification."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 12px;")
        layout.addWidget(info)

        # Settings
        settings_group = QGroupBox("TRAINING CONFIGURATION")
        form = QFormLayout(settings_group)

        self._spin_samples = QSpinBox()
        self._spin_samples.setRange(50, 1000)
        self._spin_samples.setValue(300)
        self._spin_samples.setSingleStep(50)
        self._spin_samples.setToolTip(
            "Samples per modulation class.\n"
            "300 → ~88% accuracy, ~3 min\n"
            "500 → ~90% accuracy, ~6 min\n"
            "100 → ~75% accuracy, ~1 min (fast test)")
        form.addRow("Samples per class:", self._spin_samples)

        self._lbl_estimate = QLabel("Est. time: ~3 min  |  ~3,300 total samples")
        self._lbl_estimate.setStyleSheet("font-size: 11px;")
        form.addRow("", self._lbl_estimate)
        self._spin_samples.valueChanged.connect(self._update_estimate)

        layout.addWidget(settings_group)

        # Model status
        self._status_lbl = QLabel("Model: checking...")
        self._status_lbl.setStyleSheet("")
        layout.addWidget(self._status_lbl)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        layout.addWidget(self._progress)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 10))
        self._log.setMaximumHeight(160)
        layout.addWidget(self._log)

        # Buttons
        btn_layout = QHBoxLayout()
        self._btn_train = QPushButton("Start Training")
        self._btn_train.setObjectName("btn_primary")
        self._btn_train.clicked.connect(self._start_training)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self._cancel)
        self._btn_cancel.setEnabled(False)

        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.reject)

        btn_layout.addWidget(self._btn_train)
        btn_layout.addWidget(self._btn_cancel)
        btn_layout.addStretch()
        btn_layout.addWidget(self._btn_close)
        layout.addLayout(btn_layout)

    def _update_estimate(self, n: int):
        total = n * 11
        mins = max(1, int(n * 0.015))
        self._lbl_estimate.setText(
            f"Est. time: ~{mins} min  |  ~{total:,} total samples")

    def _check_existing_model(self):
        """Check if model exists and show status."""
        try:
            from core.modulation.ml_classifier import get_model_info
            info = get_model_info()
            if info.get('available'):
                n_cls = info.get('n_classes', 0)
                n_samp = info.get('n_train_samples', 0)
                mb = info.get('size_mb', 0)
                self._status_lbl.setText(
                    f"Model: ✓ Available  |  {n_cls} classes  |  "
                    f"{n_samp} samples  |  {mb:.1f} MB")
                self._status_lbl.setStyleSheet("font-weight: 600;")
            else:
                self._status_lbl.setText("Model: ✗ Not trained yet")
                self._status_lbl.setStyleSheet("")
        except Exception:
            self._status_lbl.setText("Model: status unknown")
            self._status_lbl.setStyleSheet("")

    def _log_line(self, text: str):
        self._log.append(text)
        # Keep scrolled to bottom
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _start_training(self):
        n = self._spin_samples.value()
        self._btn_train.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._btn_close.setEnabled(False)
        self._progress.setValue(0)
        self._log.clear()
        self._cancelled = False
        self._log_line(f"Starting training: {n} samples/class, 11 classes...")

        # The trainer has no per-epoch callback, so show Qt's honest indeterminate state.
        self._progress.setRange(0, 0)

        signals = _TrainWorkerSignals()
        signals.log_line.connect(self._log_line)
        signals.finished.connect(self._on_finished)

        self._thread = threading.Thread(
            target=self._run_training,
            args=(n, signals),
            daemon=True
        )
        self._thread.start()

    def _run_training(self, n: int, signals: _TrainWorkerSignals):
        try:
            import time
            from core.modulation.ml_classifier import train_model, MODEL_PATH

            signals.log_line.emit("  Generating synthetic dataset...")
            signals.log_line.emit(f"  {n * 11} total samples across 11 SNR levels...")

            # Redirect stdout capture via simple wrapper
            pipeline = train_model(n_per_class=n, verbose=False)

            signals.finished.emit(True,
                                   f"Training complete! "
                                   f"Model saved to ml_model.pkl")

        except Exception as e:
            import traceback
            signals.finished.emit(False, f"Error: {e}\n{traceback.format_exc()}")

    def _on_finished(self, success: bool, message: str):
        self._progress.setRange(0, 100)
        self._progress.setValue(100 if success else 0)
        self._btn_train.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._btn_close.setEnabled(True)

        if success:
            self._log_line(f"✓ {message}")
            self._log_line("")
            self._log_line("Running accuracy evaluation...")
            self._run_evaluation()
            self._check_existing_model()
            self.model_trained.emit()
        else:
            self._log_line(f"✗ {message}")
            self._status_lbl.setText("Training failed")
            self._status_lbl.setStyleSheet("")

    def _run_evaluation(self):
        """Run a quick accuracy check after training."""
        try:
            from core.modulation.training_data import (
                MODULATION_LABELS, generate_dataset)
            from core.modulation.feature_extractor import extract_batch
            from core.modulation.ml_classifier import load_model, MODEL_PATH
            import numpy as np

            model_data = load_model(MODEL_PATH)
            if model_data is None:
                return

            pipeline = model_data['pipeline']
            # Quick eval — 30 samples/class, seed=999 (different from training)
            X_raw, y_true = generate_dataset(n_per_class=30, seed=999)
            X = extract_batch([X_raw[i] for i in range(len(X_raw))])
            y_pred = pipeline.predict(X)
            overall = float(np.mean(y_pred == y_true)) * 100

            self._log_line(f"  Overall accuracy: {overall:.1f}%")

            # Per-class
            for i, label in enumerate(MODULATION_LABELS):
                mask = y_true == i
                if mask.sum() > 0:
                    acc = float(np.mean(y_pred[mask] == i)) * 100
                    bar = "█" * int(acc / 10)
                    self._log_line(f"  {label:<10} {bar:<10} {acc:.0f}%")

        except Exception as e:
            self._log_line(f"  Evaluation error: {e}")

    def _cancel(self):
        self._cancelled = True
        self._log_line("Cancellation requested; the current training run will finish before closing.")
        self._btn_cancel.setEnabled(False)
