"""
Structured AI Context Builder for Tarang.
Converts real SignalInfo data and current UI tab state into structured context.
Never guesses or invents missing values.
"""
from typing import Optional, Dict, Any, Union


class AIContextBuilder:
    """Constructs structured RF telemetry for LLM consumption."""

    @staticmethod
    def build_context(
        signal_data: Optional[Union[Dict[str, Any], Any]] = None,
        current_tab: Optional[str] = None,
        comparison_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build clean markdown telemetry context for the prompt.
        Accepts either a SignalInfo instance or a dictionary from SignalInfo.summary().
        """
        if not signal_data:
            tab_str = f"\nCurrent Active View: {current_tab} Tab\n" if current_tab else ""
            return f"No signal is currently loaded in the Tarang workspace.{tab_str}"

        # If it's a SignalInfo object, call summary()
        if hasattr(signal_data, "summary") and callable(signal_data.summary):
            data = signal_data.summary()
        elif isinstance(signal_data, dict):
            data = signal_data
        else:
            data = {}

        def _fmt(val, suffix="", default="Not available"):
            if val is None or val == "" or val == 0.0 or val == 0:
                # Distinguish explicit 0 from missing where appropriate
                if isinstance(val, (int, float)) and not isinstance(val, bool) and val == 0:
                    return f"0{suffix}"
                return default
            return f"{val}{suffix}"

        # File Telemetry
        fn = _fmt(data.get("file_name"))
        ft = _fmt(data.get("file_type"))
        sr = f"{data['sample_rate_hz'] / 1e3:.1f} kHz" if data.get("sample_rate_hz") else "Not available"
        dur = f"{data['duration_sec']:.3f} s" if data.get("duration_sec") else "Not available"
        ns = f"{data['num_samples']:,}" if data.get("num_samples") else "Not available"

        # RF Parameters
        cf_val = data.get("center_freq_hz") or data.get("center_freq")
        cf = f"{cf_val:.1f} Hz" if cf_val else "Not available"

        bw_val = data.get("bandwidth_hz") or data.get("bandwidth")
        bw = f"{bw_val / 1e3:.2f} kHz" if bw_val else "Not available"

        snr = f"{data['snr_db']:.1f} dB" if data.get("snr_db") is not None else "Not available"
        nf = f"{data['noise_floor_db']:.1f} dB" if data.get("noise_floor_db") is not None else "Not available"
        band = _fmt(data.get("band"))

        # Modulation
        mod = _fmt(data.get("modulation"))
        conf_val = data.get("mod_confidence_pct")
        if conf_val is None:
            raw_c = data.get("mod_confidence")
            if raw_c is not None:
                conf_val = raw_c * 100.0 if raw_c <= 1.0 else raw_c
        elif conf_val > 100.0:
            conf_val = conf_val / 100.0
        conf = f"{conf_val:.1f}%" if conf_val is not None else "Not available"

        top3 = data.get("mod_top3")
        if top3 and isinstance(top3, list) and len(top3) > 0:
            top3_str = ", ".join([f"{m} ({c * 100:.1f}%)" if c <= 1.0 else f"{m} ({c:.1f}%)" for m, c in top3])
        else:
            top3_str = "Not available"

        # Demodulation & Rates
        sym_r = f"{data['symbol_rate_hz']:.1f} baud" if data.get("symbol_rate_hz") else "Not available"
        bit_r = f"{data['bit_rate_bps']:.1f} bps" if data.get("bit_rate_bps") else "Not available"

        # FEC & Interleaving
        interleave = _fmt(data.get("interleave_type"), default="None / Not analyzed")
        fec = _fmt(data.get("fec_type"), default="None / Not analyzed")
        err_det = _fmt(data.get("fec_errors_detected"), default="0")
        err_cor = _fmt(data.get("fec_errors_corrected"), default="0")

        # Correlation & Framing
        sync_w = _fmt(data.get("sync_word_name"), default="None detected")
        h_off = _fmt(data.get("header_offset"), default="Not found")
        p_off = _fmt(data.get("payload_offset"), default="Not found")

        # Protocol
        proto = _fmt(data.get("protocol_type"), default="Unknown / Not identified")
        proto_conf = f"{data['protocol_confidence_pct']:.1f}%" if data.get("protocol_confidence_pct") is not None else "Not available"
        num_frames = _fmt(data.get("num_frames"), default="0")

        # Security
        crypto_det = "YES" if data.get("crypto_detected") else "No"
        crypto_type = _fmt(data.get("crypto_type"), default="Plaintext / Unencrypted")
        crypto_ent = f"{data['crypto_entropy']:.4f}" if data.get("crypto_entropy") is not None else "Not available"

        lines = [
            "### Current Signal Telemetry (Measured by Tarang RF Engine):",
            "```json",
            "{",
            f'  "file": {{ "name": "{fn}", "type": "{ft}", "sample_rate": "{sr}", "duration": "{dur}", "samples": "{ns}" }},',
            f'  "rf_measurements": {{ "center_freq": "{cf}", "bandwidth": "{bw}", "snr": "{snr}", "noise_floor": "{nf}", "band": "{band}" }},',
            f'  "modulation": {{ "detected": "{mod}", "confidence": "{conf}", "top_candidates": "{top3_str}" }},',
            f'  "rates": {{ "symbol_rate": "{sym_r}", "bit_rate": "{bit_r}" }},',
            f'  "coding": {{ "interleaving": "{interleave}", "fec": "{fec}", "errors_detected": {err_det}, "errors_corrected": {err_cor} }},',
            f'  "framing": {{ "sync_word": "{sync_w}", "header_offset": "{h_off}", "payload_offset": "{p_off}" }},',
            f'  "protocol": {{ "type": "{proto}", "confidence": "{proto_conf}", "frames_decoded": {num_frames} }},',
            f'  "security": {{ "encrypted": "{crypto_det}", "cipher_type": "{crypto_type}", "shannon_entropy": "{crypto_ent}" }}',
            "}",
            "```",
        ]

        if current_tab:
            lines.append(f"\n**Active Workspace Visual Tab:** {current_tab} View")

        if comparison_data:
            lines.append("\n**Signal Comparison Data:** Available")

        return "\n".join(lines)
