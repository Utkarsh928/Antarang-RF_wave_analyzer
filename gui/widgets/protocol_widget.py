"""
Protocol Analysis Widget - Displays protocol classification and security analysis results.
Uses Antarang's monochrome dark theme.
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QTextEdit, QTableWidget, QTableWidgetItem,
                              QGroupBox, QPushButton, QTabWidget, QSplitter,
                              QHeaderView, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QFontDatabase, QPalette

# Semantic data colours remain readable on both application themes.
_COLOR_DETECTED  = "#C63D3D"
_COLOR_CLEAR     = "#2E9D62"
_COLOR_WARNING   = "#B7791F"


class ProtocolWidget(QWidget):
    """
    Widget for displaying protocol analysis results.
    Shows protocol type, frames, security analysis, and traffic patterns.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.apply_theme('dark')
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Main splitter: top (summary) / bottom (details)
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # ═══════════════════════════════════════════════════════════
        # TOP: Protocol Summary
        # ═══════════════════════════════════════════════════════════
        summary_group = QGroupBox("Protocol Summary")
        summary_layout = QVBoxLayout()
        
        # Protocol type and confidence
        protocol_layout = QHBoxLayout()
        protocol_layout.addWidget(QLabel("<b>Detected Protocol:</b>"))
        self._protocol_label = QLabel("—")
        self._protocol_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        protocol_layout.addWidget(self._protocol_label)
        protocol_layout.addStretch()
        
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("<b>Confidence:</b>"))
        self._confidence_label = QLabel("—")
        confidence_layout.addWidget(self._confidence_label)
        confidence_layout.addStretch()
        
        summary_layout.addLayout(protocol_layout)
        summary_layout.addLayout(confidence_layout)
        
        # Frame statistics
        stats_layout = QHBoxLayout()
        self._frames_label = QLabel("Frames: —")
        self._avg_size_label = QLabel("Avg Size: —")
        stats_layout.addWidget(self._frames_label)
        stats_layout.addWidget(self._avg_size_label)
        stats_layout.addStretch()
        summary_layout.addLayout(stats_layout)
        
        summary_group.setLayout(summary_layout)
        splitter.addWidget(summary_group)
        
        # ═══════════════════════════════════════════════════════════
        # BOTTOM: Tabbed Details
        # ═══════════════════════════════════════════════════════════
        self._tabs = QTabWidget()
        
        # Tab 1: Frames
        self._frames_table = QTableWidget()
        self._frames_table.setColumnCount(5)
        self._frames_table.setHorizontalHeaderLabels([
            "Frame #", "Size (bytes)", "Protocol", "Confidence", "Preview"
        ])
        # Fixed narrow columns, Preview stretches
        hdr = self._frames_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._frames_table.setColumnWidth(0, 60)
        self._frames_table.setColumnWidth(1, 90)
        self._frames_table.setColumnWidth(3, 90)
        self._frames_table.itemSelectionChanged.connect(self._on_frame_selected)
        # Empty-state hint
        self._frames_table.setRowCount(1)
        placeholder = QTableWidgetItem("Run Protocol Analysis to populate this view")
        placeholder.setForeground(QColor("#636366"))
        placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
        self._frames_table.setItem(0, 0, placeholder)
        self._frames_table.setSpan(0, 0, 1, 5)
        self._tabs.addTab(self._frames_table, "Frames")
        
        # Tab 2: Frame Details
        frame_detail_widget = QWidget()
        frame_detail_layout = QVBoxLayout(frame_detail_widget)
        
        self._frame_detail_label = QLabel("<i>Select a frame to view details</i>")
        frame_detail_layout.addWidget(self._frame_detail_label)
        
        self._frame_hex_view = QTextEdit()
        self._frame_hex_view.setReadOnly(True)
        _fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        _fixed.setPointSize(9)
        self._frame_hex_view.setFont(_fixed)
        frame_detail_layout.addWidget(self._frame_hex_view)
        
        self._tabs.addTab(frame_detail_widget, "Frame Detail")
        
        # Tab 3: Security Analysis
        security_widget = QWidget()
        security_layout = QVBoxLayout(security_widget)
        
        # Crypto detection
        crypto_group = QGroupBox("Cryptography Detection")
        crypto_layout = QVBoxLayout()
        
        self._crypto_detected_label = QLabel("Encryption: —")
        self._crypto_type_label = QLabel("Type: —")
        self._entropy_label = QLabel("Entropy: —")
        self._crypto_confidence_label = QLabel("Confidence: —")
        
        crypto_layout.addWidget(self._crypto_detected_label)
        crypto_layout.addWidget(self._crypto_type_label)
        crypto_layout.addWidget(self._entropy_label)
        crypto_layout.addWidget(self._crypto_confidence_label)
        crypto_group.setLayout(crypto_layout)
        security_layout.addWidget(crypto_group)
        
        # Security metrics
        metrics_group = QGroupBox("Statistical Tests")
        metrics_layout = QVBoxLayout()
        self._security_metrics_text = QTextEdit()
        self._security_metrics_text.setReadOnly(True)
        # No hardcoded max-height — let the splitter handle sizing
        metrics_layout.addWidget(self._security_metrics_text)
        metrics_group.setLayout(metrics_layout)
        security_layout.addWidget(metrics_group)
        
        security_layout.addStretch()
        self._tabs.addTab(security_widget, "Security")
        
        # Tab 4: Traffic Analysis
        traffic_widget = QWidget()
        traffic_layout = QVBoxLayout(traffic_widget)
        
        self._traffic_text = QTextEdit()
        self._traffic_text.setReadOnly(True)
        traffic_layout.addWidget(self._traffic_text)
        
        self._tabs.addTab(traffic_widget, "Traffic Analysis")
        
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        layout.addWidget(splitter)
        
        # Store current frames for detail view
        self._current_frames = []

    def apply_theme(self, mode: str):
        """Keep the protocol pane, splitters, and tab pages aligned with the theme."""
        if mode == "light":
            window, surface, raised = "#F5F6F8", "#FFFFFF", "#F8F9FB"
            text, muted, border, active = "#111318", "#4B5563", "#D9DDE3", "#F1F3F6"
        else:
            window, surface, raised = "#080809", "#101011", "#18181A"
            text, muted, border, active = "#F2F2F7", "#8E8E93", "#2C2C2E", "#242426"

        self.setObjectName("protocol_widget")
        self.setStyleSheet(f"""
            QWidget#protocol_widget, QWidget#protocol_widget QWidget {{
                background: {window};
                color: {text};
            }}
            QGroupBox {{
                background: transparent;
                color: {muted};
                border-color: {border};
            }}
            QTabWidget::pane {{
                background: {surface};
                border: 1px solid {border};
            }}
            QTabBar::tab {{
                color: {muted};
                background: transparent;
            }}
            QTabBar::tab:selected {{
                color: {text};
                background: {active};
            }}
            QTableWidget, QTextEdit {{
                background: {surface};
                color: {text};
                border: 1px solid {border};
                selection-background-color: {active};
                selection-color: {text};
            }}
            QHeaderView::section {{
                background: {raised};
                color: {muted};
                border-color: {border};
            }}
            QSplitter::handle {{
                background: {border};
            }}
            QScrollBar {{
                background: {raised};
            }}
        """)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(window))
        palette.setColor(QPalette.ColorRole.Base, QColor(surface))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(raised))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(text))
        palette.setColor(QPalette.ColorRole.Text, QColor(text))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        
    def set_protocol_results(self, signal_info):
        """Update display with protocol analysis results."""
        # Update summary
        protocol = getattr(signal_info, 'protocol_type', 'Unknown')
        confidence = getattr(signal_info, 'protocol_confidence', 0.0)
        num_frames = getattr(signal_info, 'num_frames', 0)
        
        self._protocol_label.setText(protocol)
        self._confidence_label.setText(f"{confidence*100:.1f}%")
        
        # Color-code confidence — use theme palette
        if confidence > 0.8:
            self._confidence_label.setStyleSheet(f"color: {_COLOR_CLEAR}; font-weight: bold;")
        elif confidence > 0.5:
            self._confidence_label.setStyleSheet(f"color: {_COLOR_WARNING}; font-weight: bold;")
        else:
            self._confidence_label.setStyleSheet(f"color: {_COLOR_DETECTED}; font-weight: bold;")
        
        # Frame statistics
        frames = getattr(signal_info, 'protocol_frames', [])
        self._current_frames = frames
        self._frames_label.setText(f"Frames: {len(frames)}")
        
        if frames:
            avg_size = sum(len(f) for f in frames) / len(frames)
            self._avg_size_label.setText(f"Avg Size: {avg_size:.1f} bytes")
        else:
            self._avg_size_label.setText("Avg Size: —")
        
        # Update frames table
        self._update_frames_table(signal_info)
        
        # Update security tab
        self._update_security_tab(signal_info)
        
        # Update traffic tab
        self._update_traffic_tab(signal_info)
        
    def _update_frames_table(self, signal_info):
        """Populate frames table."""
        frames = getattr(signal_info, 'protocol_frames', [])
        results = getattr(signal_info, 'protocol_results', [])

        # Clear any span from the placeholder row first
        self._frames_table.clearSpans()
        self._frames_table.setRowCount(len(frames))
        
        for i, frame in enumerate(frames):
            # Frame number
            self._frames_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            
            # Size
            self._frames_table.setItem(i, 1, QTableWidgetItem(str(len(frame))))
            
            # Protocol (if available)
            if i < len(results):
                protocol = results[i].protocol
                conf = results[i].confidence
            else:
                protocol = "—"
                conf = 0.0
            
            self._frames_table.setItem(i, 2, QTableWidgetItem(protocol))
            self._frames_table.setItem(i, 3, QTableWidgetItem(f"{conf*100:.1f}%"))
            
            # Preview (first 16 bytes as hex)
            preview = frame[:16].hex() if len(frame) >= 16 else frame.hex()
            if len(frame) > 16:
                preview += "..."
            self._frames_table.setItem(i, 4, QTableWidgetItem(preview))
        
        self._frames_table.resizeColumnsToContents()
        
    def _update_security_tab(self, signal_info):
        """Update security analysis display."""
        crypto_detected = getattr(signal_info, 'crypto_detected', False)
        crypto_type = getattr(signal_info, 'crypto_type', 'Unknown')
        crypto_conf = getattr(signal_info, 'crypto_confidence', 0.0)
        entropy = getattr(signal_info, 'crypto_entropy', 0.0)
        
        # Update labels
        if crypto_detected:
            self._crypto_detected_label.setText(
                f"Encryption: <b style='color:{_COLOR_DETECTED};'>DETECTED</b>")
        else:
            self._crypto_detected_label.setText(
                f"Encryption: <b style='color:{_COLOR_CLEAR};'>Not Detected</b>")
        
        self._crypto_type_label.setText(f"Type: {crypto_type}")
        self._entropy_label.setText(f"Entropy: {entropy:.4f} bits/byte (max: 8.0)")
        self._crypto_confidence_label.setText(f"Confidence: {crypto_conf*100:.1f}%")
        
        # Detailed metrics
        crypto_analysis = getattr(signal_info, 'crypto_analysis', None)
        if crypto_analysis:
            metrics_text = []
            metrics_text.append(f"Shannon Entropy: {crypto_analysis.shannon_entropy:.4f}")
            
            if hasattr(crypto_analysis, 'chi_square_stat'):
                metrics_text.append(f"Chi-Square: {crypto_analysis.chi_square_stat:.2f}")
            if hasattr(crypto_analysis, 'serial_correlation'):
                metrics_text.append(f"Serial Correlation: {crypto_analysis.serial_correlation:.4f}")
            if hasattr(crypto_analysis, 'byte_distribution_score'):
                metrics_text.append(f"Byte Uniformity: {crypto_analysis.byte_distribution_score:.4f}")
            if hasattr(crypto_analysis, 'runs_score'):
                metrics_text.append(f"Runs Test: {crypto_analysis.runs_score:.4f}")
            
            metrics_text.append("\n<b>Interpretation:</b>")
            if entropy > 7.5:
                metrics_text.append("• Very high entropy suggests strong encryption or compression")
            elif entropy > 7.0:
                metrics_text.append("• High entropy suggests possible encryption")
            elif entropy > 6.0:
                metrics_text.append("• Moderate entropy - possibly weak encryption or structured data")
            else:
                metrics_text.append("• Low entropy - likely unencrypted data")
            
            self._security_metrics_text.setHtml("<br>".join(metrics_text))
        else:
            self._security_metrics_text.setText("No detailed analysis available")
    
    def _update_traffic_tab(self, signal_info):
        """Update traffic analysis display."""
        traffic = getattr(signal_info, 'traffic_analysis', None)
        
        if traffic:
            lines = []
            lines.append("<h3>Traffic Pattern Analysis</h3>")
            lines.append(f"<b>Total Frames:</b> {traffic.get('total_frames', 0)}")
            lines.append(f"<b>Unique Frame Sizes:</b> {traffic.get('unique_sizes', 0)}")
            lines.append(f"<b>Min/Max Size:</b> {traffic.get('min_size', 0)} / {traffic.get('max_size', 0)} bytes")
            lines.append(f"<b>Average Size:</b> {traffic.get('avg_size', 0):.1f} bytes")
            
            if 'periodicity' in traffic:
                lines.append(f"<b>Periodicity:</b> {traffic['periodicity']}")
            
            if 'burst_detected' in traffic:
                if traffic['burst_detected']:
                    lines.append(f"<b style='color:{_COLOR_WARNING};'>Burst Traffic Detected</b>")
                else:
                    lines.append(f"<b style='color:{_COLOR_CLEAR};'>Steady Traffic Pattern</b>")
            
            self._traffic_text.setHtml("<br>".join(lines))
        else:
            self._traffic_text.setText("No traffic analysis available (single frame or insufficient data)")
    
    def _on_frame_selected(self):
        """Handle frame selection in table."""
        selected_rows = self._frames_table.selectedItems()
        if not selected_rows:
            return
        
        row = self._frames_table.currentRow()
        if row < 0 or row >= len(self._current_frames):
            return
        
        frame = self._current_frames[row]
        
        # Update detail label
        self._frame_detail_label.setText(f"<b>Frame {row+1}</b> ({len(frame)} bytes)")
        
        # Generate hex dump
        hex_dump = self._format_hex_dump(frame)
        self._frame_hex_view.setText(hex_dump)
    
    def _format_hex_dump(self, data: bytes) -> str:
        """Format bytes as hex dump with ASCII."""
        lines = []
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            
            # Offset
            offset = f"{i:08x}  "
            
            # Hex bytes
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            hex_part = hex_part.ljust(48)  # 16 bytes * 3 chars
            
            # ASCII
            ascii_part = "".join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
            
            lines.append(f"{offset}{hex_part}  {ascii_part}")
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear all displayed data."""
        self._protocol_label.setText("—")
        self._confidence_label.setText("—")
        self._frames_label.setText("Frames: —")
        self._avg_size_label.setText("Avg Size: —")
        self._frames_table.setRowCount(0)
        # Restore empty-state placeholder
        self._frames_table.setRowCount(1)
        placeholder = QTableWidgetItem("Run Protocol Analysis to populate this view")
        placeholder.setForeground(QColor("#636366"))
        placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
        self._frames_table.setItem(0, 0, placeholder)
        self._frames_table.setSpan(0, 0, 1, 5)
        self._frame_hex_view.clear()
        self._crypto_detected_label.setText("Encryption: —")
        self._crypto_type_label.setText("Type: —")
        self._entropy_label.setText("Entropy: —")
        self._crypto_confidence_label.setText("Confidence: —")
        self._security_metrics_text.clear()
        self._traffic_text.clear()
        self._current_frames = []
