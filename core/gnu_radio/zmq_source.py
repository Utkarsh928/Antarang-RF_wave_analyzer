"""
GNU Radio ZMQ IQ Source
========================
Receives complex64 IQ samples from any GNU Radio flowgraph that has a
ZMQ Push Sink block configured.

This makes Signal Analyzer Pro hardware-agnostic — any device that
GNU Radio supports (RTL-SDR, HackRF, USRP, Airspy, PlutoSDR, LimeSDR,
audio cards via gr-audio, file replay, etc.) can feed samples into the
application over a local ZMQ socket.

Two backends tried in order:
  1. gnuradio.zeromq — native GNU Radio ZMQ binding (best, requires GR)
  2. pyzmq           — raw ZMQ socket reading GR-format frames (requires zmq)
  3. Simulated       — fallback, always works

Usage:
    from core.gnu_radio.zmq_source import ZMQSource, zmq_available

    if zmq_available():
        src = ZMQSource("tcp://localhost:5555", sample_rate=2.4e6)
        src.start(callback=lambda chunk: ...)
        ...
        src.stop()
"""
from __future__ import annotations
import threading
import time
import numpy as np
from typing import Callable, Optional, List

# ── Default connection parameters ─────────────────────────────────────────────
DEFAULT_ADDRESS    = "tcp://localhost:5555"
DEFAULT_SAMPLE_RATE = 2_400_000.0
CHUNK_SIZE          = 262144         # ~109 ms at 2.4 MHz


def zmq_available() -> bool:
    """Return True if at least one ZMQ backend is available."""
    from core.gnu_radio.availability import ZMQ_AVAILABLE
    if ZMQ_AVAILABLE:
        return True
    try:
        import zmq  # pyzmq
        return True
    except ImportError:
        return False


def zmq_info() -> dict:
    """Return dict describing ZMQ availability and connection instructions."""
    from core.gnu_radio.availability import ZMQ_AVAILABLE, GR_AVAILABLE
    has_zmq = zmq_available()
    return {
        "available":       has_zmq,
        "gr_zmq":          ZMQ_AVAILABLE,
        "pyzmq":           _has_pyzmq(),
        "default_address": DEFAULT_ADDRESS,
        "instructions":    _connection_instructions(has_zmq, GR_AVAILABLE),
    }


def _has_pyzmq() -> bool:
    try:
        import zmq
        return True
    except ImportError:
        return False


def _connection_instructions(has_zmq: bool, has_gr: bool) -> str:
    if not has_zmq:
        return (
            "ZMQ not available.\n\n"
            "Install with either:\n"
            "  conda install -c conda-forge gnuradio    (recommended)\n"
            "  pip install pyzmq                        (lightweight)\n\n"
            "Then create a GNU Radio flowgraph with:\n"
            "  [Any SDR source] → [Signal Processing] → [ZMQ Push Sink]\n"
            "  Address: tcp://127.0.0.1:5555\n"
            "  Item size: gr_complex (complex float32)"
        )
    if not has_gr:
        return (
            "pyzmq is available but GNU Radio is not installed.\n"
            "ZMQ reception works but without GNU Radio signal processing.\n\n"
            "To use: create any application that sends complex64 IQ\n"
            "samples over a ZMQ PUSH socket at tcp://localhost:5555"
        )
    return (
        "GNU Radio ZMQ ready.\n\n"
        "Quick start:\n"
        "  1. Open GNU Radio Companion (gnuradio-companion)\n"
        "  2. Add your SDR source block (RTL-SDR, HackRF, etc.)\n"
        "  3. Add a ZMQ Push Sink block\n"
        "  4. Set Address: tcp://127.0.0.1:5555\n"
        "  5. Set Item size: gr_complex\n"
        "  6. Run the flowgraph\n"
        "  7. In Signal Analyzer Pro → Live SDR → GNU Radio ZMQ\n\n"
        "Or use one of the pre-built flowgraphs in assets/flowgraphs/"
    )


# ── ZMQ Source class ──────────────────────────────────────────────────────────

class ZMQSource:
    """
    Receives complex64 IQ from a ZMQ PULL socket.
    Compatible with GNU Radio ZMQ Push Sink (gr_complex = complex float32).

    Automatically uses gnuradio.zeromq if available, else raw pyzmq.
    Falls back to simulated signal if neither is available.
    """

    def __init__(self,
                 address:     str   = DEFAULT_ADDRESS,
                 sample_rate: float = DEFAULT_SAMPLE_RATE,
                 timeout_ms:  int   = 1000):
        self.address     = address
        self.sample_rate = float(sample_rate)
        self.timeout_ms  = timeout_ms
        self._running    = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable] = None
        self._error: Optional[str] = None
        self._backend: str = "none"

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self, callback: Callable[[np.ndarray], None]):
        """Start receiving. callback receives np.complex64 chunks."""
        if self._running:
            return
        self._callback = callback
        self._running  = True
        self._error    = None

        # Pick best available backend
        from core.gnu_radio.availability import ZMQ_AVAILABLE
        if ZMQ_AVAILABLE:
            self._backend = "gnuradio.zeromq"
            target = self._recv_gr_zmq
        elif _has_pyzmq():
            self._backend = "pyzmq"
            target = self._recv_pyzmq
        else:
            self._backend = "simulated"
            target = self._recv_simulated

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop receiving gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def backend(self) -> str:
        return self._backend

    # ── GNU Radio ZMQ backend ──────────────────────────────────────────────

    def _recv_gr_zmq(self):
        """Use gnuradio.zeromq.pull_source in a mini flowgraph."""
        try:
            from gnuradio import gr, blocks
            from gnuradio import zeromq

            tb   = gr.top_block()
            src  = zeromq.pull_source(
                gr.sizeof_gr_complex,
                CHUNK_SIZE,
                self.address,
                self.timeout_ms,
                False,   # pass_tags
                -1,      # high_water_mark
                "",
            )
            snk  = blocks.vector_sink_c(CHUNK_SIZE)
            tb.connect(src, snk)
            tb.start()

            accumulated = []
            acc_len = 0

            while self._running:
                time.sleep(0.05)
                data = snk.data()
                if data:
                    chunk = np.array(data, dtype=np.complex64)
                    snk.reset()
                    if self._callback:
                        self._callback(chunk)

            tb.stop()
            tb.wait()

        except Exception as e:
            self._error = f"GNU Radio ZMQ error: {e}"
            self._running = False

    # ── pyzmq backend ──────────────────────────────────────────────────────

    def _recv_pyzmq(self):
        """
        Raw pyzmq PULL socket reader.
        Expects frames of interleaved float32 [I0, Q0, I1, Q1, ...],
        which is exactly the format GNU Radio ZMQ Push Sink writes.
        """
        try:
            import zmq
            ctx    = zmq.Context()
            socket = ctx.socket(zmq.PULL)
            socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            socket.setsockopt(zmq.LINGER,   0)
            socket.connect(self.address)

            while self._running:
                try:
                    raw = socket.recv()
                    # GNU Radio sends raw bytes of float32 interleaved I/Q
                    floats = np.frombuffer(raw, dtype=np.float32)
                    if len(floats) >= 2:
                        chunk = floats[0::2] + 1j * floats[1::2]
                        chunk = chunk.astype(np.complex64)
                        if self._callback:
                            self._callback(chunk)
                except zmq.Again:
                    continue   # timeout — keep polling
                except Exception as e:
                    self._error = f"ZMQ recv error: {e}"
                    break

            socket.close()
            ctx.term()

        except Exception as e:
            self._error = f"pyzmq error: {e}"
            self._running = False

    # ── Simulated fallback ─────────────────────────────────────────────────

    def _recv_simulated(self):
        """
        When no ZMQ backend is available, simulate a QPSK signal.
        Identical to core/sdr_stream.py simulation but labelled as ZMQ.
        """
        from core.sdr_stream import SignalSimulator
        sim = SignalSimulator("QPSK", self.sample_rate, 100e6, snr_db=15, seed=42)
        chunk_duration = CHUNK_SIZE / self.sample_rate

        while self._running:
            t0  = time.time()
            sig = sim.generate(CHUNK_SIZE)
            if self._callback:
                self._callback(sig)
            elapsed = time.time() - t0
            time.sleep(max(0.0, chunk_duration - elapsed))


# ── Capture-once convenience ───────────────────────────────────────────────────

def capture_from_zmq(
    address:      str   = DEFAULT_ADDRESS,
    sample_rate:  float = DEFAULT_SAMPLE_RATE,
    duration_sec: float = 1.0,
) -> np.ndarray:
    """
    Capture duration_sec seconds of IQ from a ZMQ socket.
    Returns complex64 array. Works with GNU Radio ZMQ Push Sink or pyzmq.
    """
    n_target  = int(sample_rate * duration_sec)
    chunks: List[np.ndarray] = []
    collected = [0]
    lock      = threading.Lock()

    def collect(chunk):
        with lock:
            chunks.append(chunk.copy())
            collected[0] += len(chunk)

    src = ZMQSource(address, sample_rate)
    src.start(collect)

    deadline = time.time() + duration_sec + 3.0
    while collected[0] < n_target and time.time() < deadline:
        time.sleep(0.05)
    src.stop()

    if not chunks:
        return np.zeros(n_target, dtype=np.complex64)

    result = np.concatenate(chunks)[:n_target]
    return result.astype(np.complex64)
