"""
Sync Word Database — extensible registry of known frame headers.

Operators can add custom protocols via the GUI or directly editing
custom_sync_words.json in the project directory.

Built-in protocols cover:
  - Aerospace/defense: CCSDS, AX.25, HDLC
  - Wireless: 802.11, Bluetooth, Zigbee, LoRa
  - Barker codes (radar/comms)
  - DVB-S2 super-frame sync
  - Common military patterns

Custom protocols: stored in custom_sync_words.json, loaded at runtime.
"""
import os
import json
import numpy as np
from typing import Dict, Optional

# ── Built-in protocol database ────────────────────────────────────────────────

BUILTIN_SYNC_WORDS: Dict[str, dict] = {
    # --- Standard digital comms ---
    'HDLC/AX.25':   {
        'bits': [0,1,1,1,1,1,1,0],
        'description': 'HDLC flag / AX.25 packet radio',
        'band': 'VHF/UHF', 'protocol': 'AX.25'
    },
    'CCSDS-32':     {
        'bits': [0,1,0,1,1,0,0,1,1,0,1,1,1,0,0,1,
                 1,1,0,1,1,0,1,1,0,0,1,1,0,1,0,0],
        'description': 'CCSDS Telemetry Transfer Frame sync marker',
        'band': 'UHF/SHF', 'protocol': 'CCSDS'
    },
    '802.11-SFD':   {
        'bits': [1,0,1,1],
        'description': 'IEEE 802.11 PLCP start-of-frame delimiter',
        'band': 'UHF', 'protocol': '802.11'
    },
    'Barker-7':     {
        'bits': [1,1,1,0,0,1,0],
        'description': 'Barker-7 spreading code (802.11b, radar)',
        'band': 'any', 'protocol': 'Barker'
    },
    'Barker-11':    {
        'bits': [1,1,1,0,0,0,1,0,0,1,0],
        'description': 'Barker-11 (802.11b 11 Mbps)',
        'band': 'UHF', 'protocol': 'Barker'
    },
    'Barker-13':    {
        'bits': [1,1,1,1,1,0,0,1,1,0,1,0,1],
        'description': 'Barker-13 (radar, WLAN)',
        'band': 'any', 'protocol': 'Barker'
    },
    # --- Satellite ---
    'DVB-S2-SOF':   {
        'bits': [0,1,1,1,0,1,0,0,1,0,0,0,1,0,1,1,
                 0,1,0,0,0,1,1,1,0,0,1,0,0,0,0,0],
        'description': 'DVB-S2 Physical Layer Frame Start of Frame',
        'band': 'SHF', 'protocol': 'DVB-S2'
    },
    # --- IoT / sub-GHz ---
    'LoRa-Preamble':{
        'bits': [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
        'description': 'LoRa-style alternating preamble pattern',
        'band': 'UHF', 'protocol': 'LoRa'
    },
    'Zigbee-SHR':   {
        'bits': [0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,1],
        'description': 'IEEE 802.15.4 Zigbee SHR start delimiter',
        'band': 'UHF', 'protocol': 'Zigbee'
    },
    # --- HF ---
    'STANAG-4285':  {
        'bits': [1,0,1,1,0,1,1,0,1,0,0,1,0,1,1,0,
                 0,1,0,0,0,1,1,1,0,1,0,1,1,0,0,1],
        'description': 'STANAG 4285 HF modem synchronisation sequence',
        'band': 'HF', 'protocol': 'STANAG-4285'
    },
    'ALE-400':      {
        'bits': [1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,0],
        'description': 'MIL-STD-188-141 ALE preamble pattern',
        'band': 'HF', 'protocol': 'ALE'
    },
}

_CUSTOM_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))),
    'custom_sync_words.json'
)


def load_all() -> Dict[str, np.ndarray]:
    """
    Load all sync words (built-in + custom).
    Returns dict: name -> np.ndarray uint8.
    """
    result = {}

    # Built-in
    for name, entry in BUILTIN_SYNC_WORDS.items():
        result[name] = np.array(entry['bits'], dtype=np.uint8)

    # Custom (from JSON file)
    custom = load_custom()
    for name, entry in custom.items():
        result[name] = np.array(entry['bits'], dtype=np.uint8)

    return result


def load_custom() -> Dict[str, dict]:
    """Load user-defined sync words from JSON file."""
    if not os.path.exists(_CUSTOM_DB_PATH):
        return {}
    try:
        with open(_CUSTOM_DB_PATH, 'r') as f:
            data = json.load(f)
        # Validate entries
        valid = {}
        for name, entry in data.items():
            if isinstance(entry.get('bits'), list) and len(entry['bits']) >= 4:
                valid[name] = entry
        return valid
    except Exception:
        return {}


def save_custom(custom_db: Dict[str, dict]):
    """Save custom sync word database to JSON."""
    with open(_CUSTOM_DB_PATH, 'w') as f:
        json.dump(custom_db, f, indent=2)


def add_custom(name: str, bits: list,
                description: str = '',
                band: str = 'any',
                protocol: str = 'Custom'):
    """Add a new custom sync word entry."""
    db = load_custom()
    db[name] = {
        'bits':        [int(b) for b in bits],
        'description': description,
        'band':        band,
        'protocol':    protocol,
    }
    save_custom(db)


def remove_custom(name: str):
    """Remove a custom sync word by name."""
    db = load_custom()
    db.pop(name, None)
    save_custom(db)


def bits_from_hex(hex_str: str) -> list:
    """
    Convert hex string to bit list.
    Example: '7E' -> [0,1,1,1,1,1,1,0]
    """
    hex_str = hex_str.replace(' ', '').replace('0x', '')
    bits = []
    for ch in hex_str:
        val = int(ch, 16)
        for i in range(3, -1, -1):
            bits.append((val >> i) & 1)
    return bits


def bits_from_binary_str(bin_str: str) -> list:
    """
    Convert binary string to bit list.
    Example: '01111110' -> [0,1,1,1,1,1,1,0]
    Accepts spaces: '0111 1110' also works.
    """
    return [int(b) for b in bin_str.replace(' ', '') if b in '01']


def get_builtin_names() -> list:
    return list(BUILTIN_SYNC_WORDS.keys())


def get_builtin_info(name: str) -> Optional[dict]:
    return BUILTIN_SYNC_WORDS.get(name)


def get_sync_word(name: str) -> Optional[np.ndarray]:
    """
    Get a sync word by name (built-in or custom).
    Returns np.ndarray uint8 or None if not found.
    """
    all_words = load_all()
    return all_words.get(name, None)
