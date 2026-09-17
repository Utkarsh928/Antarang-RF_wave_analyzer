"""Startup checks and dependency recovery for the desktop application."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


_REQUIRED_MODULES = ("PyQt6", "numpy", "pyqtgraph")


def ensure_runtime_dependencies() -> None:
    """Install declared runtime dependencies for the interpreter running us.

    The launcher may be started with ``python``, ``py`` or a virtual
    environment. Installing through ``sys.executable`` keeps the packages
    aligned with that exact interpreter instead of a different Python on PATH.
    """
    missing = _missing_modules()
    if not missing:
        return

    requirements = Path(__file__).with_name("requirements.txt")
    if not requirements.is_file():
        raise RuntimeError(
            f"Missing dependencies ({', '.join(missing)}) and requirements.txt "
            f"was not found at {requirements}."
        )

    print(
        f"Installing missing dependencies ({', '.join(missing)}) for "
        f"{sys.executable}...",
        file=sys.stderr,
    )
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Dependency installation failed. Run this command and launch again:\n"
            f'  "{sys.executable}" -m pip install -r "{requirements}"'
        )

    still_missing = _missing_modules()
    if still_missing:
        raise RuntimeError(
            "Dependency installation completed, but these modules are still "
            f"unavailable: {', '.join(still_missing)}."
        )


def _missing_modules() -> list[str]:
    return [
        module
        for module in _REQUIRED_MODULES
        if importlib.util.find_spec(module) is None
    ]
