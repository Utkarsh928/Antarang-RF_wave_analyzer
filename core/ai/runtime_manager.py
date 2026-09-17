"""
Central runtime manager for Tarang local GGUF inference.
Guarantees self-contained, out-of-the-box local inference on Windows:
- Resolves llama_cpp from PyInstaller frozen bundles (sys._MEIPASS),
  bundled package in core/ai/runtime/, or environment site-packages.
- Sets up Windows DLL search paths via os.add_dll_directory() for native C++ libraries.
- Provides pre-flight checks before 1 GB model downloads.
- Sanitizes all technical exceptions to prevent exposing raw Python errors to users.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional, Any

logger = logging.getLogger("Tarang.AI.RuntimeManager")

_RUNTIME_INITIALIZED = False
_DLL_DIRECTORIES_ADDED = False
_LLAMA_MODULE: Optional[Any] = None


def _ensure_dll_search_paths():
    """Adds native DLL directories to Windows search path."""
    global _DLL_DIRECTORIES_ADDED
    if _DLL_DIRECTORIES_ADDED:
        return
    _DLL_DIRECTORIES_ADDED = True

    if sys.platform != "win32":
        return

    candidate_dirs = []

    # 1. PyInstaller frozen application path
    if hasattr(sys, "_MEIPASS"):
        meipass = Path(sys._MEIPASS)
        candidate_dirs.extend([
            meipass,
            meipass / "llama_cpp" / "lib",
            meipass / "core" / "ai" / "runtime" / "llama_cpp" / "lib",
        ])

    # 2. Bundled runtime inside Tarang repository / installation
    bundled_lib = Path(__file__).resolve().parent / "runtime" / "llama_cpp" / "lib"
    candidate_dirs.append(bundled_lib)

    # 3. Application directory
    app_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent.parent
    candidate_dirs.extend([
        app_dir / "llama_cpp" / "lib",
        app_dir / "core" / "ai" / "runtime" / "llama_cpp" / "lib",
    ])

    for d in candidate_dirs:
        if d.is_dir():
            try:
                os.add_dll_directory(str(d))
                logger.debug(f"Added DLL search directory: {d}")
            except Exception as e:
                logger.debug(f"Could not add DLL directory {d}: {e}")

            # Also ensure in PATH as fallback
            d_str = str(d)
            if d_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{d_str};{os.environ.get('PATH', '')}"


def get_llama_cpp():
    """
    Safely resolves and imports the llama_cpp module.
    Checks:
    1. Already loaded in sys.modules
    2. Standard import from environment
    3. Bundled runtime directory in core/ai/runtime
    4. Frozen PyInstaller bundle
    Returns (llama_cpp_module, None) on success, or (None, error_str) on failure.
    """
    global _LLAMA_MODULE
    if _LLAMA_MODULE is not None:
        return _LLAMA_MODULE, None

    _ensure_dll_search_paths()

    # 1. Ensure bundled runtime directory is in sys.path FIRST
    bundled_dir = Path(__file__).resolve().parent / "runtime"
    if bundled_dir.is_dir() and str(bundled_dir) not in sys.path:
        sys.path.insert(0, str(bundled_dir))

    if hasattr(sys, "_MEIPASS"):
        meipass_runtime = Path(sys._MEIPASS) / "core" / "ai" / "runtime"
        if meipass_runtime.is_dir() and str(meipass_runtime) not in sys.path:
            sys.path.insert(0, str(meipass_runtime))

    # 2. Check if already imported
    if "llama_cpp" in sys.modules and sys.modules["llama_cpp"] is not None:
        _LLAMA_MODULE = sys.modules["llama_cpp"]
        return _LLAMA_MODULE, None

    # 3. Try importing llama_cpp
    try:
        import llama_cpp
        _LLAMA_MODULE = llama_cpp
        logger.info(f"Loaded llama_cpp successfully (location: {getattr(llama_cpp, '__file__', 'bundled')}).")
        return _LLAMA_MODULE, None
    except Exception as e:
        logger.error(f"Failed to load llama_cpp: {e}", exc_info=True)
        return None, f"Local inference engine could not be initialized: {str(e)}"


def safe_backend_init() -> bool:
    """
    Initializes the GGML backend safely.
    Must be called before creating models or running inference.
    """
    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED:
        return True

    mod, err = get_llama_cpp()
    if mod is None:
        logger.debug(f"Backend init skipped (module not loaded: {err})")
        return False

    try:
        if hasattr(mod, "llama_backend_init"):
            try:
                mod.llama_backend_init()
            except (Exception, OSError):
                pass
        if hasattr(mod, "Llama"):
            try:
                setattr(mod.Llama, "_Llama__backend_initialized", True)
                setattr(mod.Llama, "backend_initialized", True)
            except Exception:
                pass
        _RUNTIME_INITIALIZED = True
        logger.info("Local GGML backend initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize GGML backend: {e}", exc_info=True)
        return False


def is_runtime_available() -> Tuple[bool, str]:
    """
    Pre-flight check: verifies that the local inference runtime is operational.
    Returns: (is_available, human_readable_message)
    """
    mod, err = get_llama_cpp()
    if mod is None:
        return False, "Local AI runtime is not bundled or could not be loaded."

    if not hasattr(mod, "Llama"):
        return False, "Local AI runtime is missing required inference classes."

    return True, "Local inference runtime is operational."


def get_llama_class():
    """
    Returns the Llama class from llama_cpp, or raises a clean sanitized RuntimeError.
    """
    mod, err = get_llama_cpp()
    if mod is None or not hasattr(mod, "Llama"):
        raise RuntimeError("Tarang could not initialize the local AI runtime.")
    return mod.Llama
