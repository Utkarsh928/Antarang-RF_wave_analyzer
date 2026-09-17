"""
GNU Radio integration package for Signal Analyzer Pro.

All imports are optional — when GNU Radio is not installed the package
exposes only the availability flags and pure-Python fallbacks.

Usage anywhere in the project:
    from core.gnu_radio import GR_AVAILABLE, gr_info
    if GR_AVAILABLE:
        from core.gnu_radio.conditioner import GnuRadioConditioner
"""

from core.gnu_radio.availability import (
    GR_AVAILABLE,
    GR_VERSION,
    GR_MODULES,
    ZMQ_AVAILABLE,
    gr_info,
    check_module,
)

__all__ = [
    "GR_AVAILABLE",
    "GR_VERSION",
    "GR_MODULES",
    "ZMQ_AVAILABLE",
    "gr_info",
    "check_module",
]
