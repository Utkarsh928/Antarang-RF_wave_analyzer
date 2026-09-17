# SignalAnalyzerPro - User Manual

**Version**: 1.0  
**Date**: 2026-09-08  
**RF Signal Analysis Application**

---

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [User Interface](#user-interface)
5. [Loading Signals](#loading-signals)
6. [Analysis Workflow](#analysis-workflow)
7. [Export Results](#export-results)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## Introduction

SignalAnalyzerPro is a professional RF signal analysis application that provides:
- **Signal Loading**: WAV and IQ file formats
- **Spectral Analysis**: FFT, Waterfall, Constellation
- **Modulation Classification**: 17 modulation types (97% accuracy)
- **Demodulation**: PSK, QAM, FSK, OFDM, AM, FM
- **De-interleaving**: Block, Convolutional, Diagonal, Pseudo-random
- **FEC Decoding**: Viterbi, Reed-Solomon, LDPC, Concatenated
- **Bit Correlation**: Header/Payload identification
- **Protocol Analysis**: Frame extraction, Protocol classification
- **Security Analysis**: Encryption detection, Traffic analysis

### System Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 or higher
- **RAM**: 4 GB minimum (8 GB recommended)
- **Display**: 1920×1080 or higher

---

## Installation

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Required packages**:
- numpy, scipy
- PyQt6
- tensorflow, scikit-learn (for ML classification)
- reedsolo (for Reed-Solomon FEC)

### Step 2: Verify Installation
```bash
python gui/main.py
```

The application window should open.

### Optional: Install SDR Support
For live SDR capture (RTL-SDR, HackRF, USRP):
```bash
pip install pyrtlsdr hackrf-soapy SoapySDR
```

---

## Quick Start

### 5-Minute Tutorial

1. **Launch Application**
   ```bash
   python gui/main.py
   ```

2. **Load Test Signal**
   - Click **📂 Open File**
   - Navigate to `test_signals/`
   - Select `test_bpsk.wav`
   - Click Open

3. **Analyze Signal**
   - Click **▶ Analyze** button
   - Wait ~2 seconds
   - View spectrum, waterfall, constellation

4. **Demodulate**
   - Modulation is auto-detected (BPSK)
   - Click **▶ Demodulate** in Control Panel
   - View bits in "Bits" tab

5. **Export Results**
   - Click **💾 Export** button
   - Select export format
   - Save to file

**Congratulations!** You've completed a basic signal analysis.

---

## User Interface

### Main Window Layout

```
┌────────────────────────────────────────────────────────────┐
│  📂 Open   📡 Live   ▶ Analyze   💾 Export   🤖 AI   ℹ About│
├─────────┬──────────────────────────────────────────────────┤
│ File    │ [📊 Spectrum] [🌊 Waterfall] [⭕ Constellation]  │
│ Info    │                                                  │
│         │              PLOT AREA                           │
│ Status  │                                                  │
│ ▸ Loaded│                                                  │
│ ▸ Demod │                                                  │
│ ▸ FEC   │                                                  │
│ ▸ Corr  │                                                  │
│         ├──────────────────────────────────────────────────┤
│         │ [1.Demod] [2.Deinter] [3.FEC] [4.Corr] [5.Proto]│
│         │   CONTROL PANEL - Pipeline Controls              │
└─────────┴──────────────────────────────────────────────────┘
│ Status: Ready                                              │
└────────────────────────────────────────────────────────────┘
```

### Keyboard Shortcuts
- **Ctrl+O**: Open file
- **Ctrl+R**: Run analysis
- **Ctrl+E**: Export results
- **Ctrl+I**: Toggle AI panel
- **1-5**: Switch tabs (Spectrum, Waterfall, Constellation, Bits, Protocol)
- **F11**: Fullscreen

---

## Loading Signals

### WAV Files
**Format**: Standard audio WAV files  
**Channels**: Mono (real-valued signal)  
**Sample Rate**: Any (extracted from header)

**Example**:
```
File: signal.wav
Format: 16-bit PCM
Sample Rate: 48 kHz
Duration: 10 seconds
```

**How to Load**:
1. Click **📂 Open File**
2. Select `.wav` file
3. Click Open
4. Signal loads automatically

### IQ Files
**Format**: Binary complex samples  
**Data Type**: float32 (complex64) or int16  
**Interleaving**: I,Q,I,Q,I,Q...

**How to Load**:
1. Click **📂 Open File**
2. Select `.iq` file
3. Dialog appears: "IQ Parameters"
4. Enter:
   - **Data Type**: float32, int16, or complex64
   - **Sample Rate**: e.g., 48000 Hz
5. Click OK

**Supported IQ Formats**:
- GNU Radio: complex64 (float32 I/Q)
- SDR#: int16 interleaved
- GQRX: complex64

---

## Analysis Workflow

### Complete Pipeline

```
1. LOAD FILE (WAV or IQ)
   ↓
2. ANALYZE (FFT, Spectrogram, Parameters)
   ↓
3. CLASSIFY MODULATION (Auto-detect)
   ↓
4. DEMODULATE (Bits from samples)
   ↓
5. DE-INTERLEAVE (Optional - remove interleaving)
   ↓
6. FEC DECODE (Optional - error correction)
   ↓
7. CORRELATE (Find header/payload)
   ↓
8. PROTOCOL ANALYSIS (Frame extraction, crypto detection)
   ↓
9. EXPORT RESULTS
```

---

### Step-by-Step Guide

#### 1. Load & Analyze
**Action**: Load signal file  
**Button**: 📂 Open File  
**Result**: Signal loaded into memory

**Action**: Run analysis  
**Button**: ▶ Analyze  
**Result**:
- FFT spectrum computed
- Waterfall spectrogram generated
- Constellation plot (if complex)
- Modulation auto-detected
- Parameters estimated

**What You See**:
- Spectrum tab: FFT power spectrum
- Waterfall tab: Time-frequency visualization
- Constellation tab: I/Q scatter plot
- File panel: Parameters (fs, modulation, SNR)

---

#### 2. Demodulation

**Control Panel Section**: 1. Demodulate

**Options**:
- **Modulation**: Auto-detect (or manual selection)
  - PSK: BPSK, QPSK, 8PSK, 16PSK
  - QAM: QAM16, QAM64, QAM256
  - FSK: FSK2, FSK4, GFSK
  - Other: OFDM, AM, FM, OOK, PAM4

**Action**: Click **▶ Demodulate**

**Result**:
- Bits extracted from signal
- Bit rate calculated
- Bits displayed in "Bits" tab

**Interpretation**:
- Green highlighted: Raw bits
- Bit rate shown in bps
- Use hex/binary view to inspect

---

#### 3. De-interleaving (Optional)

**Control Panel Section**: 2. De-interleave

**When to Use**:
- Signal uses interleaving (common in aerospace, satellite)
- Error bursts need to be spread out

**Options**:
- **Method**: Auto-detect, Block, Convolutional, Diagonal, Pseudo-random
- **Rows/Depth**: Matrix dimension (e.g., 8 for 8×N block)

**Action**: Click **▶ De-interleave**

**Result**:
- Bits re-ordered
- Original bit sequence restored

**Recommendation**:
- Use **Auto-detect** first
- If accuracy is low (<50%), try manual selection
- Common: Block 8×64 or Convolutional depth 4

---

#### 4. FEC Decode (Optional)

**Control Panel Section**: 3. FEC Decode

**When to Use**:
- Signal includes error correction coding
- Want to correct bit errors from noise

**Options**:
- **FEC Type**: Auto-detect, Viterbi, Reed-Solomon, LDPC, Concatenated

**Action**: Click **▶ FEC Decode**

**Result**:
- Errors detected and corrected
- Corrected bits shown
- Error statistics displayed

**Interpretation**:
- **Errors Detected**: Total bit errors found
- **Errors Corrected**: Successfully fixed errors
- **BER**: Bit Error Rate (lower is better)

**Recommendation**:
- Try **Auto-detect** first
- If detection fails (50% accuracy), try:
  - **Viterbi**: Aerospace, satellite (most common)
  - **Reed-Solomon**: Digital TV (DVB), storage
  - **LDPC**: Modern standards (WiFi 802.11n, 5G)
  - **Concatenated**: Deep space, high-reliability

---

#### 5. Bit Correlation

**Control Panel Section**: 4. Bit Correlation

**Purpose**: Find frame boundaries (header/payload)

**Options**:
- **Sync Word**: Auto-detect, HDLC, CCSDS, Barker codes, 802.11, Custom

**Action**: Click **▶ Correlate**

**Result**:
- Header position identified
- Payload position identified
- Sync word name displayed

**Interpretation**:
- **Header @ bit 128**: Sync word found at bit position 128
- **Payload @ bit 256**: Data starts at bit 256

**Custom Sync Words**:
- Click **Manage Sync DB**
- Add your own sync patterns
- Format: hex string (e.g., "7E FF 00")

---

#### 6. Protocol Analysis (NEW!)

**Control Panel Section**: 5. Protocol Analysis

**Purpose**: Identify protocol type and detect encryption

**Options**:
- **Protocol**: Auto-detect, AX.25, CCSDS, Custom
- **Run Security Analysis**: Checkbox (default: ON)

**Action**: Click **▶ Analyze Protocol**

**Result**:
- Protocol tab opens
- Protocol type identified
- Frames extracted
- Encryption detection performed
- Traffic patterns analyzed

**What You See**:

**Protocol Summary**:
```
Detected Protocol: AX.25
Confidence: 87.5%
Frames: 24
Avg Size: 128.3 bytes
```

**Frames Tab**:
- Table of all extracted frames
- Frame number, size, protocol, preview
- Click frame to view hex dump

**Frame Detail Tab**:
- Hex dump with ASCII
- Byte-by-byte view
- Copy hex to clipboard

**Security Tab**:
```
Encryption: DETECTED
Type: Likely AES (block cipher)
Entropy: 7.932 bits/byte
Confidence: 94.2%

Statistical Tests:
- Shannon Entropy: 7.932
- Chi-Square: 245.12
- Serial Correlation: 0.0023
- Byte Uniformity: 0.996
```

**Interpretation**:
- **Entropy > 7.5**: Likely encrypted
- **Entropy 6-7**: Possible encryption or compression
- **Entropy < 6**: Likely unencrypted

**Traffic Analysis Tab**:
```
Total Frames: 24
Unique Sizes: 3
Min/Max: 64 / 256 bytes
Periodicity: High (likely periodic transmission)
Burst Traffic: Not Detected
```

---

## Export Results

### What Can Be Exported
- Spectrum plot (PNG, PDF)
- Waterfall plot (PNG, PDF)
- Constellation plot (PNG, PDF)
- Bit data (binary, hex, text)
- Protocol frames (hex, JSON)
- Analysis report (PDF, Markdown)

### How to Export

1. Click **💾 Export** button
2. Export dialog opens
3. Select items to export:
   - ☑ Spectrum plot
   - ☑ Waterfall plot
   - ☑ Constellation plot
   - ☑ Bit data
   - ☑ Protocol frames
   - ☑ Analysis report
4. Choose export format
5. Click **Export**
6. Select save location
7. Click **Save**

### Export Formats

**Images**: PNG (recommended), PDF, SVG  
**Data**: Binary (.bin), Hex (.txt), JSON (.json)  
**Reports**: PDF (formatted), Markdown (.md)

---

## Troubleshooting

### Problem: File won't load
**Symptoms**: Error message "Cannot load file"

**Solutions**:
1. Check file format (WAV or IQ)
2. For IQ files, verify data type and sample rate
3. Check file size (<2GB recommended)
4. Ensure file is not corrupted

---

### Problem: Classification is wrong
**Symptoms**: Detected modulation doesn't match expectation

**Solutions**:
1. Check SNR (signal-to-noise ratio)
   - SNR < 0 dB: Classification unreliable
   - SNR > 10 dB: Good classification
2. Use manual modulation selection
3. Check if signal is complex (I/Q) or real-valued

---

### Problem: Demodulation produces garbage bits
**Symptoms**: All 0s, all 1s, or random-looking bits

**Solutions**:
1. Verify modulation type is correct
2. Check sample rate (fs must be correct)
3. Check if signal needs carrier recovery
4. Try different modulation types manually

---

### Problem: FEC auto-detection fails
**Symptoms**: "Auto-detect" results in wrong FEC type

**Known Limitation**: FEC auto-detection is 50% accurate

**Solutions**:
1. Use manual FEC type selection
2. Common FEC types:
   - **Viterbi**: Most aerospace, satellite
   - **Reed-Solomon**: DVB-T, DVB-S, CDs
   - **LDPC**: Modern (WiFi, 5G, DVB-S2)
   - **Concatenated**: Deep space, Voyager

---

### Problem: Protocol analysis shows "Unknown"
**Symptoms**: Protocol confidence is 0% or very low

**Reasons**:
1. Signal may not use a known protocol
2. Bits may be incorrect (demodulation issue)
3. Encryption may obscure protocol structure

**Solutions**:
1. Verify demodulation worked correctly
2. Check if signal is encrypted
3. Add custom protocol to database
4. Use manual protocol hint

---

### Problem: Application is slow
**Symptoms**: GUI freezes or is unresponsive

**Causes**:
- Large signal files (>100 MB)
- Complex processing (LDPC decoding)

**Solutions**:
1. Close other applications
2. Use shorter signal clips
3. Disable AI panel (Ctrl+I)
4. Reduce waterfall resolution

---

## FAQ

**Q: What file formats are supported?**  
A: WAV (audio) and IQ (binary complex samples).

**Q: Can I analyze live SDR signals?**  
A: Yes! Click **📡 Live SDR** and select your device.

**Q: How accurate is modulation classification?**  
A: 97.2% at 30dB SNR, 54% at low SNR (<0dB).

**Q: What is FEC auto-detection accuracy?**  
A: 50% (Viterbi and LDPC work well, RS and Concatenated may need manual selection).

**Q: What is interleaving auto-detection accuracy?**  
A: 50% (Convolutional and Pseudo-random work well, Block and Diagonal may need manual).

**Q: Can I add custom protocols?**  
A: Yes! Edit `core/protocol/protocol_database.json`.

**Q: Can I add custom sync words?**  
A: Yes! Click **Manage Sync DB** in Control Panel.

**Q: How do I report bugs?**  
A: Create an issue on GitHub with:
- Signal file (if possible)
- Steps to reproduce
- Error message/screenshot

**Q: Is this software free?**  
A: Yes, open-source under MIT license.

**Q: Can I use this commercially?**  
A: Yes, MIT license allows commercial use.

---

## Keyboard Reference Card

| Key | Action |
|-----|--------|
| **Ctrl+O** | Open file |
| **Ctrl+R** | Run analysis |
| **Ctrl+E** | Export results |
| **Ctrl+I** | Toggle AI panel |
| **1** | Spectrum tab |
| **2** | Waterfall tab |
| **3** | Constellation tab |
| **4** | Bits tab |
| **5** | Protocol tab |
| **F11** | Fullscreen mode |

---

## Tips & Best Practices

### ✓ DO:
- Start with **Auto-detect** for all parameters
- Check signal quality (SNR) before demodulation
- Use manual selection if auto-detect fails
- Export results before closing application
- Save settings between sessions

### ✗ DON'T:
- Don't skip analysis step (always click **▶ Analyze** first)
- Don't expect 100% accuracy on noisy signals
- Don't use FEC decoding without first having bits
- Don't ignore error messages
- Don't forget to export your work

---

## Getting Help

**Documentation**: See `docs/` folder  
**Examples**: See `test_signals/` folder  
**Support**: Open GitHub issue  
**Email**: support@signalanalyzerpro.example  

---

**END OF USER MANUAL**

Version 1.0 | © 2026 SignalAnalyzerPro | MIT License
