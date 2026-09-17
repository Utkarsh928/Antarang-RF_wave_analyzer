"""
RTL-SDR → Signal Conditioning → ZMQ Push Sink
==============================================
Captures from RTL-SDR dongles, applies professional conditioning,
sends IQ to tcp://127.0.0.1:5555 for Signal Analyzer Pro.

Requires: gnuradio + gr-osmosdr  (conda install -c conda-forge gnuradio gr-osmosdr)
          pyrtlsdr  (pip install pyrtlsdr) + Zadig WinUSB driver

Run standalone:
    python flowgraph_rtlsdr_to_zmq.py --freq 100e6 --sr 2.4e6 --gain 30

Or use from Signal Analyzer Pro automatically when GNU Radio is installed.
"""
from __future__ import annotations
import argparse
from assets.flowgraphs.flowgraph_base import FlowgraphBase


class RTLSDRtoZMQ(FlowgraphBase):
    """
    Full-quality RTL-SDR flowgraph:
      rtlsdr_source → AGC → LPF → FLL → Costas → ZMQ Push Sink
    """

    def _build(self):
        from gnuradio import gr, analog, blocks
        from gnuradio import filter as gr_filter
        from gnuradio.filter import firdes
        try:
            import osmosdr
        except ImportError:
            self._error = (
                "gr-osmosdr not installed. "
                "Run: conda install -c conda-forge gr-osmosdr")
            return None
        try:
            from gnuradio import zeromq
        except ImportError:
            self._error = (
                "gnuradio.zeromq not installed. "
                "Run: conda install -c conda-forge gnuradio")
            return None

        tb = gr.top_block("RTL-SDR to ZMQ")

        # ── RTL-SDR source ────────────────────────────────────────────────
        src = osmosdr.source(args="numchan=1 rtl=0")
        src.set_sample_rate(self.sample_rate)
        src.set_center_freq(self.center_freq)
        src.set_freq_corr(0, 0)
        src.set_gain(self.gain, 0)
        src.set_if_gain(20, 0)
        src.set_bb_gain(20, 0)
        src.set_bandwidth(0, 0)

        # ── AGC ───────────────────────────────────────────────────────────
        agc = analog.agc2_cc(
            attack_rate=1e-3, decay_rate=1e-4,
            reference=1.0, gain=1.0)

        # ── Low-pass filter ───────────────────────────────────────────────
        cutoff = min(self.sample_rate * 0.4, self.sample_rate / 2 * 0.9)
        lp_taps = firdes.low_pass(
            1.0, self.sample_rate, cutoff,
            cutoff * 0.2, firdes.window.WIN_HAMMING)
        lpf = gr_filter.fir_filter_ccc(1, lp_taps)

        # ── FLL (frequency locked loop) ───────────────────────────────────
        sps = max(2, int(self.sample_rate / 50_000))
        fll = analog.fll_band_edge_cc(sps, 0.35, 45, 0.005)

        # ── Costas loop (phase correction) ───────────────────────────────
        costas = analog.costas_loop_cc(0.005, 2, False)

        # ── ZMQ Push Sink ─────────────────────────────────────────────────
        zmq_sink = zeromq.push_sink(
            gr.sizeof_gr_complex, 1,
            self.zmq_address, 100, False, -1, "")

        # ── Connect ───────────────────────────────────────────────────────
        tb.connect(src, agc, lpf, fll, costas, zmq_sink)
        return tb


class RTLSDRSimpletoZMQ(FlowgraphBase):
    """
    Simpler RTL-SDR flowgraph without FLL/Costas — for basic monitoring.
      rtlsdr_source → AGC → ZMQ Push Sink
    """

    def _build(self):
        from gnuradio import gr, analog
        try:
            import osmosdr
            from gnuradio import zeromq
        except ImportError as e:
            self._error = str(e)
            return None

        tb  = gr.top_block("RTL-SDR Simple to ZMQ")
        src = osmosdr.source(args="numchan=1 rtl=0")
        src.set_sample_rate(self.sample_rate)
        src.set_center_freq(self.center_freq)
        src.set_gain(self.gain, 0)
        agc = analog.agc2_cc(1e-3, 1e-4, 1.0, 1.0)
        zmq = zeromq.push_sink(
            gr.sizeof_gr_complex, 1,
            self.zmq_address, 100, False, -1, "")
        tb.connect(src, agc, zmq)
        return tb


# ── Standalone entry point ────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="RTL-SDR → ZMQ flowgraph for Signal Analyzer Pro")
    p.add_argument("--freq",   type=float, default=100e6,  help="Center freq Hz")
    p.add_argument("--sr",     type=float, default=2.4e6,  help="Sample rate Hz")
    p.add_argument("--gain",   type=float, default=30.0,   help="RF gain dB")
    p.add_argument("--addr",   type=str,   default="tcp://127.0.0.1:5555",
                   help="ZMQ push address")
    p.add_argument("--simple", action="store_true",
                   help="Use simple flowgraph (no FLL/Costas)")
    args = p.parse_args()

    cls = RTLSDRSimpletoZMQ if args.simple else RTLSDRtoZMQ
    fg  = cls(center_freq=args.freq, sample_rate=args.sr,
               gain=args.gain, zmq_address=args.addr)
    fg.start()
    if fg.error:
        print(f"Error: {fg.error}")
        return
    print(f"Flowgraph running — sending IQ to {args.addr}")
    print(f"Center: {args.freq/1e6:.3f} MHz  SR: {args.sr/1e6:.2f} MHz  Gain: {args.gain} dB")
    print("Press Ctrl+C to stop")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        fg.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
