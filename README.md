# Antarang (Signal Analyzer Pro) — RF Signal Analysis Platform

[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)]()
[![Tests](https://img.shields.io/badge/Tests-71%2F71%20PASS%20(100%25)-success)]()
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)]()
[![GUI](https://img.shields.io/badge/GUI-PyQt6%20Cupertino%20Monochrome-black)]()

Antarang is an automated, high-precision radio frequency (RF) signal intelligence and analysis workstation engineered for `.wav` and `.iq` captures. It provides automated parameter estimation, 17-class machine-learning modulation recognition, de-interleaving, forward error correction (FEC), bitstream inspection, protocol correlation, and cryptographic analysis.

---

## Architecture Overview: Clean Separation

The codebase follows a strict separation between **Backend** DSP logic and **Frontend** user interface:

```
D:\Tarang\
│
├── core\                         # [BACKEND] Signal Processing & ML Pipeline
│   ├── bitstream\                # Bit extraction, framing, correlation
│   ├── cpp_src\                  # C++ DSP performance acceleration sources
│   ├── fec\                      # Viterbi, Reed-Solomon, LDPC, Concatenated decoders
│   ├── gnu_radio\                # GNU Radio ZMQ stream integration
│   ├── interleaving\             # Block, convolutional, diagonal, pseudo-random de-interleavers
│   ├── modulation\               # ML Ensemble (ml_model.pkl - 1.85 GB), CNN, 17 demodulators
│   ├── protocol\                 # Protocol decoders (AX.25, DVB-S2, HDLC, LTE, etc.)
│   ├── security\                 # Cryptographic cipher detection & traffic entropy
│   ├── fast_correlator.dll       # Compiled native acceleration DLLs
│   ├── signal_features.dll
│   ├── viterbi.dll
│   ├── chunked_loader.py         # Memory-efficient I/Q reader (>1 GB support)
│   ├── parameter_estimator.py    # Sample rate, center frequency, SNR, bandwidth
│   ├── signal_filter.py          # DC offset removal, power normalization
│   └── ...
│
├── gui\                          # [FRONTEND] PyQt6 Modern Interface
│   ├── widgets\                  # Spectrum, waterfall, constellation, bits, protocol widgets
│   ├── translations\             # 13-language localization dictionaries
│   ├── main_window.py            # Primary application shell & tab manager
│   ├── theme.py                  # Cupertino modern monochrome aesthetic
│   ├── branding.py               # Branding, assets, logos
│   ├── workers.py                # Asynchronous background analysis & plot workers
│   └── ...
│
├── assets\                       # [ASSETS] Application icons & branding media
│   └── branding\
│
├── tests\                        # [TEST SUITE] Automated regression & unit test suite
│   ├── test_core.py
│   ├── test_end_to_end_pipeline.py
│   ├── test_ml.py
│   └── ...
│
├── demo_signals\                 # Presentation & demonstration signals
├── test_signals\                 # Synthetic verification signals
├── navtex\                       # Real NAVTEX captures (2-FSK)
├── rtty\                         # Real RTTY captures (2-FSK / USB audio)
├── dsd.2021-12-02T16_44_54_046.wav # Real C4FM/YSF recording (4-FSK, 75 kHz)
│
├── main.py                       # Main application entry point
├── startup.py                    # Pre-flight environment check
├── run.bat                       # One-click Windows launcher
├── functional_test.py            # Official test runner (71/71 tests passing)
├── train_model.py                # ML Model training utility
├── custom_sync_words.json        # Protocol sync definition database
├── requirements.txt              # Production Python dependencies
├── antarang.spec                 # PyInstaller production build specification
├── BUILD_EXE.bat                 # Automated executable packager
└── USER_MANUAL.md                # End-user operator guide
```

---

## Key Capabilities

### 1. Automated Signal Parameter Estimation
* **Sampling Rate ($F_s$) & Bandwidth ($BW$)**: Automatic occupied bandwidth (99% power) and spectral center detection.
* **Carrier Frequency ($f_c$)**: Baseband offset identification.
* **Signal-to-Noise Ratio (SNR)**: In-band vs noise floor power estimation ($dB$).
* **Symbol Rate ($R_s$) & Bit Rate ($R_b$)**: Precise baud rate extraction and bit rate calculation.

### 2. Machine Learning Modulation Classification (17 Classes)
* **High-Accuracy 4-Voter Ensemble**: GradientBoosting + RandomForest + ExtraTrees + Deep 5-Layer MLP (`ml_model.pkl`).
* **Low-SNR ResNet 1D CNN**: Deep neural network for signals below $2\text{ dB}$ SNR.
* **Supported Modulations**:
  * **PSK**: BPSK, QPSK, 8PSK
  * **FSK**: FSK2, FSK4, GFSK, CPFSK
  * **QAM**: QAM16, QAM64
  * **APSK**: APSK16, APSK32
  * **Analog**: AM-DSB, AM-SSB, WBFM
  * **Other**: OOK, PAM4, OFDM

### 3. De-Interleaving & Error Correction
* **4 Interleaver Types**: Block, Convolutional, Diagonal, Pseudo-random.
* **FEC Decoders**: Viterbi (Convolutional), Reed-Solomon (255, 223/239), LDPC (DVB-S2 / 802.11n), Concatenated (RS + Viterbi).

### 4. Protocol & Cryptographic Analysis
* **Frame Boundary Detection**: Preambles, sync words, and header/payload extraction.
* **Built-in Protocols**: CCSDS, AX.25, HDLC, DVB-S2, DMR, P25, GSM, Bluetooth, AIS, ADS-B, etc.
* **Security Inspection**: Shannon entropy, byte distribution chi-square, cipher detection.

---

## Quick Start

### 1. Launch GUI Application
```bat
run.bat
```
or via Python:
```bash
py -3.11 main.py
```

### 2. Run Comprehensive Functional Tests (71 Tests)
```bash
py -3.11 functional_test.py
```
**Result**: `71 passed, 0 failed out of 71 (100% Pass Rate)`.

### 3. Build Windows Executable (.exe)
```bat
BUILD_EXE.bat
```
Output executable is packaged in `dist\Antarang\Antarang.exe`.

---

## Documentation
* [USER_MANUAL.md](file:///D:/Tarang/USER_MANUAL.md): Complete operational instructions.
* [HOW_TO_TRAIN_ML_MODEL.md](file:///D:/Tarang/HOW_TO_TRAIN_ML_MODEL.md): Instructions for retraining the ML model.
