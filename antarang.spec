# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Antarang
# All output goes to D:\AntarangBuild\ — keeps C: drive free

from PyInstaller.utils.hooks import collect_all, collect_data_files
import os

block_cipher = None

SRC = os.path.dirname(os.path.abspath(SPEC))

# ── Collect data files from heavy packages ───────────────────────────────────
scipy_datas,   scipy_bins,   scipy_hidden   = collect_all('scipy')
sklearn_datas, sklearn_bins, sklearn_hidden = collect_all('sklearn')
pyqt6_datas,   pyqt6_bins,   pyqt6_hidden   = collect_all('PyQt6')
llama_datas,   llama_bins,   llama_hidden   = collect_all('llama_cpp')
hf_datas,      hf_bins,      hf_hidden      = collect_all('huggingface_hub')

# ── Files and folders to bundle ───────────────────────────────────────────────
datas = [
    # Assets — logo, icons, splash video
    (os.path.join(SRC, 'assets'),                          'assets'),
    # GUI package
    (os.path.join(SRC, 'gui'),                             'gui'),
    # Core package
    (os.path.join(SRC, 'core'),                            'core'),
    # Bundled local AI inference engine
    (os.path.join(SRC, 'core', 'ai', 'runtime'),           'core/ai/runtime'),

    # Protocol DB
    (os.path.join(SRC, 'core', 'protocol', 'protocol_database.json'),
                                                           'core/protocol'),
    # Pre-trained ML models
    (os.path.join(SRC, 'core', 'modulation', 'cnn_model.pkl'),
                                                           'core/modulation'),
    (os.path.join(SRC, 'core', 'modulation', 'ml_model.pkl'),
                                                           'core/modulation'),
    # Pre-compiled C++ DLLs
    (os.path.join(SRC, 'core', 'fast_correlator.dll'),     '.'),
    (os.path.join(SRC, 'core', 'signal_features.dll'),     '.'),
    (os.path.join(SRC, 'core', 'viterbi.dll'),             '.'),
    (os.path.join(SRC, 'core', 'fast_correlator.dll'),     'core'),
    (os.path.join(SRC, 'core', 'signal_features.dll'),     'core'),
    (os.path.join(SRC, 'core', 'viterbi.dll'),             'core'),
    # Package data files
    *scipy_datas,
    *sklearn_datas,
    *pyqt6_datas,
    *llama_datas,
    *hf_datas,
]

# ── Hidden imports PyInstaller misses ─────────────────────────────────────────
hiddenimports = [
    # ── Core ──
    'core', 'core.file_loader', 'core.signal_info',
    'core.signal_filter', 'core.parameter_estimator',
    'core.chunked_loader', 'core.symbol_rate_estimator',
    'core.cpp_extensions', 'core.sdr_stream',
    'core.realistic_signal_generator',
    # modulation
    'core.modulation', 'core.modulation.classifier',
    'core.modulation.cnn_classifier', 'core.modulation.demodulator',
    'core.modulation.feature_extractor', 'core.modulation.ml_classifier',
    'core.modulation.dataset_loader', 'core.modulation.training_data',
    # interleaving
    'core.interleaving', 'core.interleaving.deinterleaver',
    'core.interleaving.interleaver', 'core.interleaving.advanced_detector',
    'core.interleaving.combined_detector', 'core.interleaving.fec_assisted_detector',
    'core.interleaving.ml_enhanced_detector', 'core.interleaving.rank_detector',
    'core.interleaving.specialized_detector', 'core.interleaving.ultimate_detector',
    # fec
    'core.fec', 'core.fec.decoder', 'core.fec.encoder',
    'core.fec.fec_identifier',
    # bitstream
    'core.bitstream', 'core.bitstream.correlator', 'core.bitstream.sync_db',
    # protocol
    'core.protocol', 'core.protocol.classifier', 'core.protocol.decoder',
    'core.protocol.frame_extractor', 'core.protocol.parsers',
    'core.protocol.validator',
    # security
    'core.security', 'core.security.crypto_detector',
    'core.security.traffic_analyzer',
    # gnu radio
    'core.gnu_radio', 'core.gnu_radio.availability',
    'core.gnu_radio.conditioner', 'core.gnu_radio.zmq_source',
    # ── GUI ──
    'gui', 'gui.main_window', 'gui.branding', 'gui.workers',
    'gui.ai_overview', 'gui.exporter', 'gui.compare_dialog',
    'gui.drop_overlay', 'gui.icons', 'gui.theme',
    'gui.language_manager', 'gui.settings_dialog', 'gui.sdr_dialog',
    'gui.train_dialog', 'gui.sync_editor',
    'gui.widgets', 'gui.widgets.file_panel', 'gui.widgets.control_panel',
    'gui.widgets.spectrum_widget', 'gui.widgets.waterfall_widget',
    'gui.widgets.constellation_widget', 'gui.widgets.bits_widget',
    'gui.widgets.protocol_widget',
    'gui.history_panel', 'gui.language_selector',
    # assets — flowgraphs are plain .py files bundled via datas, not a package
    # ── Scientific ──
    'scipy', 'scipy.signal', 'scipy.fft', 'scipy.stats',
    'scipy.linalg', 'scipy.interpolate', 'scipy.optimize',
    'scipy.signal.windows', 'scipy.special',
    'numpy', 'numpy.core', 'numpy.fft',
    'sklearn', 'sklearn.ensemble', 'sklearn.svm',
    'sklearn.preprocessing', 'sklearn.pipeline',
    'sklearn.neural_network', 'sklearn.neighbors',
    # ── PyQt6 ──
    'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
    'PyQt6.QtMultimedia', 'PyQt6.QtMultimediaWidgets',
    'PyQt6.QtOpenGL', 'PyQt6.QtOpenGLWidgets',
    # ── Other ──
    'PIL', 'PIL.Image', 'PIL.ImageDraw',
    'pickle', 'json', 'struct', 'wave', 'zipfile',
    'urllib', 'urllib.request', 'http', 'http.client',
    # ── AI Overview & Local Offline AI ──
    'core.ai', 'core.ai.runtime_manager', 'core.ai.model_manager',
    'core.ai.local_llava_provider', 'core.ai.offline_qwen_provider',
    'core.ai.compatibility', 'core.ai.config', 'core.ai.context_builder',
    'llama_cpp', 'huggingface_hub',
    *scipy_hidden,
    *sklearn_hidden,
    *pyqt6_hidden,
    *llama_hidden,
    *hf_hidden,
]

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    [os.path.join(SRC, 'main.py')],
    pathex=[SRC],
    binaries=scipy_bins + sklearn_bins + pyqt6_bins + llama_bins + hf_bins,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'IPython', 'jupyter',
        'notebook', 'pandas', 'sympy', 'wx', 'gtk',
        'PyQt5', 'PySide2', 'PySide6',
        # Exclude heavy ML frameworks we don't use directly
        # (they get pulled in as optional deps — we don't need them bundled)
        'torch', 'torchvision', 'torchaudio',
        'onnx', 'onnxruntime',
        'tensorflow', 'keras',
        'jax', 'flax',
        'networkx',
        'pyarrow',
        'dask',
        'numba',
        'cupy',
        'cv2', 'opencv',
        'test', 'tests', 'testing',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Antarang',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # No black terminal window — GUI only
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(SRC, 'assets', 'branding', 'tarang.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=['vcruntime140.dll', 'python3*.dll'],
    name='Antarang',
)
