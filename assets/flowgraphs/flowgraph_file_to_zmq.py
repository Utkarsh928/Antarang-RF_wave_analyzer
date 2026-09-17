"""
IQ File → Signal Conditioning → ZMQ Push Sink
==============================================
Reads a .iq/.wav/.bin file and streams conditioned IQ to ZMQ.
Useful for replaying captured signals at any speed.

Run:
    python flowgraph_file_to_zmq.py --file capture.iq --sr 2.4e6
"""
from __future__ import annotations
import argparse
import os
from assets.flowgraphs.flowgraph_base import FlowgraphBase


class FileToZMQ(FlowgraphBase):
    """Replay an IQ file through conditioning to ZMQ."""

    def __init__(self, file_path: str = "",
                 sample_rate: float = 2.4e6,
                 dtype: str = "float32",
                 repeat: bool = True,
                 zmq_address: str = "tcp://127.0.0.1:5555"):
        super().__init__(sample_rate=sample_rate, zmq_address=zmq_address)
        self.file_path = file_path
        self.dtype     = dtype
        self.repeat    = repeat

    def _build(self):
        from gnuradio import gr, blocks, analog
        from gnuradio import filter as gr_filter
        from gnuradio.filter import firdes
        try:
            from gnuradio import zeromq
        except ImportError as e:
            self._error = str(e); return None

        if not os.path.exists(self.file_path):
            self._error = f"File not found: {self.file_path}"; return None

        tb = gr.top_block("File to ZMQ")

        # File source
        src = blocks.file_source(
            gr.sizeof_gr_complex, self.file_path, self.repeat)

        # Throttle to real-time
        throttle = blocks.throttle(gr.sizeof_gr_complex, self.sample_rate)

        # AGC
        agc = analog.agc2_cc(1e-3, 1e-4, 1.0, 1.0)

        # LPF
        lp_taps = firdes.low_pass(
            1.0, self.sample_rate, self.sample_rate * 0.4,
            self.sample_rate * 0.08, firdes.window.WIN_HAMMING)
        lpf = gr_filter.fir_filter_ccc(1, lp_taps)

        # ZMQ sink
        zmq_sink = zeromq.push_sink(
            gr.sizeof_gr_complex, 1,
            self.zmq_address, 100, False, -1, "")

        tb.connect(src, throttle, agc, lpf, zmq_sink)
        return tb


class WavFileToZMQ(FlowgraphBase):
    """Replay a .wav file (stereo = I/Q channels) to ZMQ."""

    def __init__(self, file_path: str = "",
                 sample_rate: float = 48000,
                 zmq_address: str = "tcp://127.0.0.1:5555"):
        super().__init__(sample_rate=sample_rate, zmq_address=zmq_address)
        self.file_path = file_path

    def _build(self):
        from gnuradio import gr, blocks, analog
        try:
            from gnuradio import zeromq
        except ImportError as e:
            self._error = str(e); return None

        tb = gr.top_block("WAV to ZMQ")
        src       = blocks.wavfile_source(self.file_path, False)
        throttle  = blocks.throttle(gr.sizeof_float, self.sample_rate)
        f2c       = blocks.float_to_complex()  # L=I, R=Q
        agc       = analog.agc2_cc(1e-3, 1e-4, 1.0, 1.0)
        zmq_sink  = zeromq.push_sink(
            gr.sizeof_gr_complex, 1,
            self.zmq_address, 100, False, -1, "")

        tb.connect((src, 0), throttle)
        tb.connect(throttle, (f2c, 0))   # I channel
        try:
            tb.connect((src, 1), (f2c, 1))   # Q channel
        except Exception:
            pass
        tb.connect(f2c, agc, zmq_sink)
        return tb


def main():
    p = argparse.ArgumentParser(description="IQ/WAV file → ZMQ for Signal Analyzer Pro")
    p.add_argument("--file",   required=True,            help="Input .iq or .wav file")
    p.add_argument("--sr",     type=float, default=2.4e6)
    p.add_argument("--repeat", action="store_true",      help="Loop file")
    p.add_argument("--addr",   default="tcp://127.0.0.1:5555")
    args = p.parse_args()

    if args.file.endswith(".wav"):
        fg = WavFileToZMQ(args.file, args.sr, args.addr)
    else:
        fg = FileToZMQ(args.file, args.sr, zmq_address=args.addr, repeat=args.repeat)

    fg.start()
    if fg.error:
        print(f"Error: {fg.error}")
        return
    print(f"Streaming {args.file} to {args.addr}  SR={args.sr/1e6:.2f} MHz")
    print("Press Ctrl+C to stop")
    try:
        import time
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    fg.stop()


if __name__ == "__main__":
    main()
