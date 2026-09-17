"""
Base class for all Signal Analyzer Pro GNU Radio flowgraphs.

Each flowgraph:
  - Is a self-contained Python class (no .grc file needed)
  - Can be run as a script: python flowgraph_rtlsdr_to_zmq.py
  - Sends IQ to ZMQ Push socket that Signal Analyzer Pro reads
  - Has graceful fallback if GNU Radio is not installed

Usage:
    from assets.flowgraphs.flowgraph_rtlsdr_to_zmq import RTLSDRtoZMQ
    fg = RTLSDRtoZMQ(center_freq=100e6, sample_rate=2.4e6, gain=30)
    fg.start()
    ...
    fg.stop()
"""
from __future__ import annotations
from typing import Optional


class FlowgraphBase:
    """
    Base for all flowgraphs. Manages start/stop and status reporting.
    Subclasses override _build() and _teardown().
    """

    def __init__(self,
                 center_freq:  float = 100e6,
                 sample_rate:  float = 2.4e6,
                 gain:         float = 30.0,
                 zmq_address:  str   = "tcp://127.0.0.1:5555"):
        self.center_freq  = float(center_freq)
        self.sample_rate  = float(sample_rate)
        self.gain         = float(gain)
        self.zmq_address  = zmq_address
        self._tb          = None
        self._running     = False
        self._error: Optional[str] = None

    def start(self):
        if self._running:
            return
        from core.gnu_radio.availability import GR_AVAILABLE
        if not GR_AVAILABLE:
            self._error = (
                "GNU Radio not installed. "
                "Install with: conda install -c conda-forge gnuradio")
            return
        try:
            self._tb = self._build()
            if self._tb:
                self._tb.start()
                self._running = True
        except Exception as e:
            self._error = str(e)

    def stop(self):
        if self._tb and self._running:
            try:
                self._tb.stop()
                self._tb.wait()
            except Exception:
                pass
        self._running = False
        self._tb = None

    def is_running(self) -> bool:
        return self._running

    @property
    def error(self) -> Optional[str]:
        return self._error

    def status(self) -> dict:
        return {
            "running":      self._running,
            "center_freq":  self.center_freq,
            "sample_rate":  self.sample_rate,
            "gain":         self.gain,
            "zmq_address":  self.zmq_address,
            "error":        self._error,
        }

    def _build(self):
        """Override in subclass to build and return a gr.top_block."""
        raise NotImplementedError
