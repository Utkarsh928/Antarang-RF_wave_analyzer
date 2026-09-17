"""
Exporter — exports results to PDF, PNG, TXT, JSON.
"""
import os
import json
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QCheckBox,
                               QPushButton, QLabel, QFileDialog, QProgressBar,
                               QGroupBox, QGridLayout, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from core.signal_info import SignalInfo
from core.bitstream.correlator import bits_to_hex, bits_to_ascii
from gui.theme import apply_window_chrome, theme_stylesheet, window_theme_mode


class ExportDialog(QDialog):
    def __init__(self, info: SignalInfo,
                 spectrum_widget=None,
                 waterfall_widget=None,
                 constellation_widget=None,
                 protocol_widget=None,   # Fix #9: accept protocol widget
                 parent=None):
        super().__init__(parent)
        self._info = info
        self._spectrum_w = spectrum_widget
        self._waterfall_w = waterfall_widget
        self._constellation_w = constellation_widget
        self._protocol_w = protocol_widget    # Fix #9
        self.setWindowTitle("Antarang — Export Results")
        self.setMinimumWidth(420)
        self.setObjectName("export_dialog")
        self.apply_theme(window_theme_mode(parent))
        self._build_ui()

    def apply_theme(self, mode: str):
        """Use the current centralized theme instead of a copied parent stylesheet."""
        self.setStyleSheet(theme_stylesheet(mode))
        colors = {
            "window": "#F5F6F8",
            "surface": "#FFFFFF",
            "text": "#111318",
        } if mode == "light" else {
            "window": "#080809",
            "surface": "#101011",
            "text": "#F2F2F7",
        }
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(colors["window"]))
        palette.setColor(QPalette.ColorRole.Base, QColor(colors["surface"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Text, QColor(colors["text"]))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        apply_window_chrome(self, mode)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Select what to export:")
        title.setObjectName("label_section")
        layout.addWidget(title)

        group = QGroupBox("Export Options")
        grid = QGridLayout(group)

        self._checks = {}
        options = [
            ('pdf_report',    'Full PDF Report'),
            ('spectrum_png',  'Spectrum Plot (PNG)'),
            ('waterfall_png', 'Waterfall Plot (PNG)'),
            ('const_png',     'Constellation Plot (PNG)'),
            ('raw_bits',      'Raw Bit Stream (TXT)'),
            ('corrected_bits','Corrected Bit Stream (TXT)'),
            ('payload_txt',   'Decoded Payload (TXT + HEX)'),
            ('params_json',   'Signal Parameters (JSON)'),
            ('protocol_txt',  'Protocol Analysis (TXT)'),   # Fix #9
            ('protocol_png',  'Protocol View (PNG)'),       # Fix #9
        ]
        for i, (key, label) in enumerate(options):
            chk = QCheckBox(label)
            chk.setChecked(True)
            grid.addWidget(chk, i // 2, i % 2)
            self._checks[key] = chk

        layout.addWidget(group)

        # Progress
        self._progress = QProgressBar()
        self._progress.hide()
        layout.addWidget(self._progress)

        # Buttons
        btn_layout = QHBoxLayout()
        self._btn_select_all = QPushButton("Select All")
        self._btn_select_all.clicked.connect(
            lambda: [c.setChecked(True) for c in self._checks.values()])
        self._btn_none = QPushButton("Select None")
        self._btn_none.clicked.connect(
            lambda: [c.setChecked(False) for c in self._checks.values()])
        self._btn_export = QPushButton("Export Selected")
        self._btn_export.setObjectName("btn_primary")
        self._btn_export.clicked.connect(self._do_export)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self._btn_select_all)
        btn_layout.addWidget(self._btn_none)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(self._btn_export)
        layout.addLayout(btn_layout)

    def _do_export(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Export Folder", os.path.expanduser("~"))
        if not folder:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = (os.path.splitext(self._info.file_name)[0]
                     or "signal") + f"_{timestamp}"

        self._progress.show()
        self._progress.setValue(0)
        exported = []
        total = sum(1 for c in self._checks.values() if c.isChecked())
        done = 0

        try:
            if self._checks['params_json'].isChecked():
                path = os.path.join(folder, f"{base_name}_params.json")
                self._export_json(path)
                exported.append(path)
                done += 1
                self._progress.setValue(int(done / total * 100))

            if self._checks['raw_bits'].isChecked():
                if self._info.raw_bits is not None:
                    path = os.path.join(folder, f"{base_name}_raw_bits.txt")
                    self._export_bits_txt(path, self._info.raw_bits, "Raw Bits")
                    exported.append(path)
                done += 1
                self._progress.setValue(int(done / total * 100))

            if self._checks['corrected_bits'].isChecked():
                if self._info.corrected_bits is not None:
                    path = os.path.join(folder,
                                        f"{base_name}_corrected_bits.txt")
                    self._export_bits_txt(path, self._info.corrected_bits,
                                          "Corrected Bits")
                    exported.append(path)
                done += 1
                self._progress.setValue(int(done / total * 100))

            if self._checks['payload_txt'].isChecked():
                bits = self._info.corrected_bits or self._info.raw_bits
                if bits is not None and self._info.payload_offset > 0:
                    payload = bits[self._info.payload_offset:]
                    path = os.path.join(folder, f"{base_name}_payload.txt")
                    self._export_payload(path, payload)
                    exported.append(path)
                done += 1
                self._progress.setValue(int(done / total * 100))

            if self._checks['spectrum_png'].isChecked():
                path = os.path.join(folder, f"{base_name}_spectrum.png")
                self._export_plot_png(path, self._spectrum_w, 'spectrum')
                exported.append(path)
                done += 1
                self._progress.setValue(int(done / total * 100))

            if self._checks['waterfall_png'].isChecked():
                path = os.path.join(folder, f"{base_name}_waterfall.png")
                self._export_plot_png(path, self._waterfall_w, 'waterfall')
                exported.append(path)
                done += 1
                self._progress.setValue(int(done / total * 100))

            if self._checks['const_png'].isChecked():
                path = os.path.join(folder, f"{base_name}_constellation.png")
                self._export_plot_png(path, self._constellation_w,
                                      'constellation')
                exported.append(path)
                done += 1
                self._progress.setValue(int(done / total * 100))

            if self._checks['pdf_report'].isChecked():
                path = os.path.join(folder, f"{base_name}_report.pdf")
                self._export_pdf(path, folder, base_name)
                exported.append(path)
                done += 1
                self._progress.setValue(100)

            # Fix #9: Protocol analysis text export
            if self._checks['protocol_txt'].isChecked():
                if hasattr(self._info, 'protocol_type') and self._info.protocol_type:
                    path = os.path.join(folder, f"{base_name}_protocol.txt")
                    self._export_protocol_txt(path)
                    exported.append(path)
                done += 1
                self._progress.setValue(int(done / total * 100))

            # Fix #9: Protocol widget screenshot
            if self._checks['protocol_png'].isChecked():
                if self._protocol_w is not None:
                    path = os.path.join(folder, f"{base_name}_protocol.png")
                    self._export_plot_png(path, self._protocol_w, 'protocol')
                    exported.append(path)
                done += 1
                self._progress.setValue(int(done / total * 100))

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            return

        self._progress.hide()
        QMessageBox.information(
            self, "Export Complete",
            f"Exported {len(exported)} file(s) to:\n{folder}"
        )
        self.accept()

    def _export_json(self, path: str):
        data = self._info.summary()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _export_bits_txt(self, path: str, bits: np.ndarray, title: str):
        MAX_BITS = 10_000
        truncated = len(bits) > MAX_BITS
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"Signal Analyzer Pro — {title}\n")
            f.write(f"File: {self._info.file_name}\n")
            f.write(f"Total bits: {len(bits):,}\n")
            # Fix #8: explicit truncation warning
            if truncated:
                f.write(f"NOTE: Output truncated to {MAX_BITS:,} bits "
                        f"(full capture: {len(bits):,} bits)\n")
            f.write("\nBinary:\n")
            for i in range(0, min(len(bits), MAX_BITS), 64):
                chunk = bits[i:i+64]
                f.write(' '.join(
                    ''.join(str(b) for b in chunk[j:j+8])
                    for j in range(0, len(chunk), 8)
                ) + '\n')
            f.write("\nHex:\n")
            f.write(bits_to_hex(bits[:MAX_BITS]) + '\n')
        # Fix #8: show warning in UI if truncated
        if truncated:
            QMessageBox.information(
                self,
                "Bits Export — Truncated",
                f"The bit stream was truncated to {MAX_BITS:,} bits.\n\n"
                f"Full capture has {len(bits):,} bits.\n\n"
                "This limit exists to keep export files manageable. "
                "The full bit data is still available in the application."
            )

    def _export_payload(self, path: str, bits: np.ndarray):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"Signal Analyzer Pro — Payload\n")
            f.write(f"File: {self._info.file_name}\n")
            f.write(f"Payload offset: bit {self._info.payload_offset}\n")
            f.write(f"Payload length: {len(bits)} bits\n\n")
            f.write("Hex:\n")
            f.write(bits_to_hex(bits) + '\n\n')
            f.write("ASCII:\n")
            f.write(bits_to_ascii(bits) + '\n')

    def _export_protocol_txt(self, path: str):
        """Fix #9: Export protocol and security analysis results as text."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("Signal Analyzer Pro — Protocol Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"File: {self._info.file_name}\n")
            f.write(f"Protocol: {getattr(self._info, 'protocol_type', '—')}\n")
            conf = getattr(self._info, 'protocol_confidence', 0.0)
            f.write(f"Confidence: {conf*100:.1f}%\n")
            f.write(f"Frames: {getattr(self._info, 'num_frames', 0)}\n\n")
            f.write("Security Analysis:\n")
            f.write("-" * 30 + "\n")
            encrypted = getattr(self._info, 'crypto_detected', False)
            f.write(f"Encryption: {'DETECTED' if encrypted else 'Not Detected'}\n")
            f.write(f"Type: {getattr(self._info, 'crypto_type', '—')}\n")
            entropy = getattr(self._info, 'crypto_entropy', 0.0)
            f.write(f"Shannon Entropy: {entropy:.4f} bits/byte\n")
            crypto_conf = getattr(self._info, 'crypto_confidence', 0.0)
            f.write(f"Detection Confidence: {crypto_conf*100:.1f}%\n\n")
            # Detailed stats if available
            ca = getattr(self._info, 'crypto_analysis', None)
            if ca:
                f.write("Statistical Tests:\n")
                if hasattr(ca, 'chi_square_stat'):
                    f.write(f"  Chi-Square Stat: {ca.chi_square_stat:.4f}\n")
                if hasattr(ca, 'chi_square_pvalue'):
                    f.write(f"  Chi-Square P-Value: {ca.chi_square_pvalue:.4f}\n")
                if hasattr(ca, 'serial_correlation'):
                    f.write(f"  Serial Correlation: {ca.serial_correlation:.4f}\n")
                if hasattr(ca, 'byte_distribution_score'):
                    f.write(f"  Byte Distribution Score: {ca.byte_distribution_score:.4f}\n")
            f.write("\nFrame List:\n")
            f.write("-" * 30 + "\n")
            frames = getattr(self._info, 'protocol_frames', [])
            results = getattr(self._info, 'protocol_results', [])
            for i, frame in enumerate(frames):
                proto = results[i].protocol if i < len(results) else '—'
                c = results[i].confidence if i < len(results) else 0.0
                preview = frame[:8].hex() if frame else ''
                f.write(f"  Frame {i+1:3d}: {len(frame):4d} bytes  "
                        f"{proto} ({c*100:.0f}%)  {preview}...\n")

    def _export_plot_png(self, path: str, widget, name: str):
        if widget is None:
            return
        try:
            import pyqtgraph as pg
            import pyqtgraph.exporters
            if hasattr(widget, '_plot_widget'):
                exporter = pg.exporters.ImageExporter(
                    widget._plot_widget.plotItem)
                exporter.parameters()['width'] = 1400
                exporter.export(path)   # pass file path — not BytesIO
                return
        except Exception:
            pass
        # Qt screenshot fallback — always works
        try:
            pixmap = widget.grab()
            pixmap.save(path, 'PNG')
        except Exception:
            pass

    def _export_pdf(self, path: str, folder: str, base_name: str):
        try:
            from reportlab.lib.pagesizes import A4
        except ImportError:
            # Fix #3: Show actionable warning instead of silently skipping
            QMessageBox.warning(
                self,
                "PDF Export — Missing Library",
                "PDF export requires the 'reportlab' package.\n\n"
                "Install it with:\n"
                "    pip install reportlab\n\n"
                "All other selected exports have been saved."
            )
            return
        try:
            from reportlab.lib import colors
            from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                             Spacer, Table, TableStyle,
                                             Image as RLImage)
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm

            doc = SimpleDocTemplate(path, pagesize=A4,
                                    rightMargin=2*cm, leftMargin=2*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()
            story = []

            title_style = ParagraphStyle(
                'Title', parent=styles['Title'],
                fontSize=18, textColor=colors.HexColor('#242426'))
            story.append(Paragraph("Antarang — Analysis Report",
                                   title_style))
            story.append(Spacer(1, 0.5*cm))
            story.append(Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles['Normal']))
            story.append(Spacer(1, 0.5*cm))

            params = self._info.summary()
            table_data = [['Parameter', 'Value']]
            for k, v in params.items():
                if not isinstance(v, (list, dict)):
                    table_data.append([k.replace('_', ' ').title(), str(v)])

            t = Table(table_data, colWidths=[8*cm, 8*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#242426')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.HexColor('#F2F2F7'), colors.white]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.5*cm))

            for plot_name in ['spectrum', 'waterfall', 'constellation']:
                img_path = os.path.join(folder, f"{base_name}_{plot_name}.png")
                if os.path.exists(img_path):
                    story.append(Paragraph(plot_name.title() + " Plot",
                                           styles['Heading2']))
                    story.append(RLImage(img_path, width=16*cm, height=8*cm))
                    story.append(Spacer(1, 0.3*cm))

            # Fix #9 (protocol section in PDF) — added below
            if hasattr(self._info, 'protocol_type') and self._info.protocol_type:
                story.append(Paragraph("Protocol Analysis", styles['Heading2']))
                proto_data = [['Field', 'Value']]
                proto_data.append(['Protocol', getattr(self._info, 'protocol_type', '—')])
                proto_data.append(['Confidence',
                    f"{getattr(self._info, 'protocol_confidence', 0)*100:.1f}%"])
                proto_data.append(['Frames', str(getattr(self._info, 'num_frames', 0))])
                if hasattr(self._info, 'crypto_detected'):
                    proto_data.append(['Encryption',
                        'Detected' if self._info.crypto_detected else 'Not Detected'])
                if hasattr(self._info, 'crypto_entropy'):
                    proto_data.append(['Shannon Entropy',
                        f"{self._info.crypto_entropy:.4f} bits/byte"])
                pt = Table(proto_data, colWidths=[8*cm, 8*cm])
                pt.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#242426')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                     [colors.HexColor('#F2F2F7'), colors.white]),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('PADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(pt)

            doc.build(story)
        except Exception as e:
            raise RuntimeError(f"PDF generation failed: {e}") from e
