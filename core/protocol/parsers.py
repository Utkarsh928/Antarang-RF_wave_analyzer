"""
Protocol-Specific Parsers

Implements detailed parsers for common protocols:
- HDLC (High-Level Data Link Control) - ISO 13239
- CCSDS (Space Data Systems) - CCSDS 732.0-B-3
- AX.25 (Amateur Packet Radio) - AX.25 v2.2
- X.25 (Packet Switched Networks)

References:
- ISO/IEC 13239:2002 (HDLC)
- CCSDS 732.0-B-3 (TM Synchronization and Channel Coding)
- AX.25 Link Access Protocol for Amateur Packet Radio v2.2
- ITU-T X.25 Recommendation
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class FrameType(Enum):
    """HDLC frame types"""
    INFORMATION = "I"      # Information frame
    SUPERVISORY = "S"      # Supervisory frame
    UNNUMBERED = "U"       # Unnumbered frame


@dataclass
class HDLCFrame:
    """Parsed HDLC frame"""
    flag_start: int        # Opening flag (0x7E)
    address: int           # Address field
    control: int           # Control field
    frame_type: FrameType  # Frame type
    payload: bytes         # Information field
    fcs: int              # Frame Check Sequence
    flag_end: int         # Closing flag
    valid: bool           # FCS validation result
    
    # Decoded control fields
    ns: Optional[int] = None      # Send sequence number (I-frames)
    nr: Optional[int] = None      # Receive sequence number
    p_f: Optional[bool] = None    # Poll/Final bit
    command: Optional[str] = None  # Command name (S/U frames)


@dataclass
class CCSDSPacket:
    """Parsed CCSDS Telemetry packet"""
    # Primary Header
    version: int                # Version (should be 000)
    packet_type: int           # 0=TM, 1=TC
    secondary_header_flag: int  # Secondary header present
    apid: int                  # Application Process ID
    sequence_flags: int        # Sequence flags
    sequence_count: int        # Packet sequence count
    data_length: int           # Data field length - 1
    
    # Secondary Header (if present)
    secondary_header: Optional[bytes] = None
    
    # Data Field
    user_data: bytes = bytes()
    
    # Validation
    valid: bool = True


@dataclass
class AX25Frame:
    """Parsed AX.25 frame"""
    destination: str       # Destination callsign
    source: str           # Source callsign
    repeaters: List[str]  # Digipeater path
    control: int          # Control field
    frame_type: FrameType  # Frame type
    pid: Optional[int]    # Protocol ID (if I/UI frame)
    payload: bytes        # Information field
    fcs: int             # Frame Check Sequence
    valid: bool          # FCS validation


class HDLCParser:
    """
    HDLC Frame Parser
    
    Supports:
    - Information frames (I-frames)
    - Supervisory frames (RR, RNR, REJ, SREJ)
    - Unnumbered frames (SABM, DISC, UA, DM, FRMR, UI)
    """
    
    FLAG = 0x7E
    
    # Control field patterns
    I_FRAME_MASK = 0x01  # xxxxxxx0
    S_FRAME_MASK = 0x03  # xxxxxx01
    U_FRAME_MASK = 0x03  # xxxxxx11
    
    # Supervisory commands
    S_COMMANDS = {
        0x00: "RR",   # Receive Ready
        0x04: "RNR",  # Receive Not Ready
        0x08: "REJ",  # Reject
        0x0C: "SREJ"  # Selective Reject
    }
    
    # Unnumbered commands
    U_COMMANDS = {
        0x2F: "SABM",  # Set Asynchronous Balanced Mode
        0x43: "DISC",  # Disconnect
        0x63: "UA",    # Unnumbered Acknowledgment
        0x0F: "DM",    # Disconnected Mode
        0x87: "FRMR",  # Frame Reject
        0x03: "UI"     # Unnumbered Information
    }
    
    def parse(self, data: bytes) -> Optional[HDLCFrame]:
        """
        Parse HDLC frame
        
        Frame format:
        FLAG | ADDRESS | CONTROL | INFORMATION | FCS | FLAG
        """
        if len(data) < 6:  # Minimum: FLAG + ADDR + CTRL + FCS + FLAG
            return None
        
        # Check flags
        if data[0] != self.FLAG or data[-1] != self.FLAG:
            return None
        
        # Extract fields
        address = data[1]
        control = data[2]
        
        # Determine frame type and extract info
        frame_type, ns, nr, p_f, command = self._parse_control(control)
        
        # Extract payload and FCS
        if len(data) >= 6:
            payload = data[3:-3]
            fcs = (data[-3] << 8) | data[-2]
        else:
            payload = bytes()
            fcs = 0
        
        # TODO: Validate FCS
        valid = True  # Placeholder
        
        return HDLCFrame(
            flag_start=data[0],
            address=address,
            control=control,
            frame_type=frame_type,
            payload=payload,
            fcs=fcs,
            flag_end=data[-1],
            valid=valid,
            ns=ns,
            nr=nr,
            p_f=p_f,
            command=command
        )
    
    def _parse_control(self, control: int) -> Tuple[FrameType, Optional[int], Optional[int], bool, Optional[str]]:
        """Parse control field"""
        
        # Check frame type
        if (control & self.I_FRAME_MASK) == 0:
            # Information frame: N(S), N(R), P
            # Format: N(R) P N(S) 0
            ns = (control >> 1) & 0x07  # Bits 1-3
            nr = (control >> 5) & 0x07  # Bits 5-7
            p_f = bool(control & 0x10)   # Bit 4
            return (FrameType.INFORMATION, ns, nr, p_f, None)
        
        elif (control & self.S_FRAME_MASK) == 0x01:
            # Supervisory frame: S, N(R), P/F
            # Format: N(R) P/F S S 0 1
            nr = (control >> 5) & 0x07
            p_f = bool(control & 0x10)
            s_type = control & 0x0C
            command = self.S_COMMANDS.get(s_type, "UNKNOWN")
            return (FrameType.SUPERVISORY, None, nr, p_f, command)
        
        else:
            # Unnumbered frame: M, P/F
            # Format: M M P/F M M 1 1
            p_f = bool(control & 0x10)
            command = self.U_COMMANDS.get(control & 0xEF, "UNKNOWN")
            return (FrameType.UNNUMBERED, None, None, p_f, command)


class CCSDSParser:
    """
    CCSDS Telemetry Packet Parser
    
    Supports CCSDS Space Packet Protocol (CCSDS 133.0-B-2)
    """
    
    def parse(self, data: bytes) -> Optional[CCSDSPacket]:
        """
        Parse CCSDS packet
        
        Primary Header (6 bytes):
        - Packet Identification (2 bytes)
        - Packet Sequence Control (2 bytes)  
        - Packet Data Length (2 bytes)
        """
        if len(data) < 6:
            return None
        
        # Packet Identification (bits 0-15)
        packet_id = (data[0] << 8) | data[1]
        version = (packet_id >> 13) & 0x07           # Bits 0-2
        packet_type = (packet_id >> 12) & 0x01       # Bit 3
        secondary_hdr = (packet_id >> 11) & 0x01     # Bit 4
        apid = packet_id & 0x7FF                     # Bits 5-15
        
        # Packet Sequence Control (bits 16-31)
        seq_control = (data[2] << 8) | data[3]
        sequence_flags = (seq_control >> 14) & 0x03  # Bits 0-1
        sequence_count = seq_control & 0x3FFF        # Bits 2-15
        
        # Packet Data Length (bits 32-47)
        data_length = (data[4] << 8) | data[5]
        
        # Extract secondary header if present
        secondary_header = None
        user_data_start = 6
        
        if secondary_hdr:
            # Secondary header format varies by APID
            # Typically 6-10 bytes, contains timestamp
            # For simplicity, we'll extract a fixed length
            if len(data) > 16:
                secondary_header = data[6:16]
                user_data_start = 16
        
        # Extract user data
        if len(data) > user_data_start:
            user_data = data[user_data_start:]
        else:
            user_data = bytes()
        
        # Validation
        valid = version == 0  # Should always be 0
        
        return CCSDSPacket(
            version=version,
            packet_type=packet_type,
            secondary_header_flag=secondary_hdr,
            apid=apid,
            sequence_flags=sequence_flags,
            sequence_count=sequence_count,
            data_length=data_length,
            secondary_header=secondary_header,
            user_data=user_data,
            valid=valid
        )


class AX25Parser:
    """
    AX.25 Packet Radio Frame Parser
    
    Supports AX.25 v2.2 specification
    """
    
    FLAG = 0x7E
    
    def parse(self, data: bytes) -> Optional[AX25Frame]:
        """
        Parse AX.25 frame
        
        Frame format:
        FLAG | ADDRESS | ADDRESS | ... | CONTROL | PID | INFO | FCS | FLAG
        
        Address field: 7 bytes per callsign (6 + SSID)
        """
        if len(data) < 17:  # Minimum: FLAG + 2 addresses + CTRL + FCS + FLAG
            return None
        
        # Check flags
        if data[0] != self.FLAG or data[-1] != self.FLAG:
            return None
        
        pos = 1
        
        # Parse destination address (7 bytes)
        dest = self._decode_address(data[pos:pos+7])
        pos += 7
        
        # Parse source address (7 bytes)
        source = self._decode_address(data[pos:pos+7])
        pos += 7
        
        # Parse repeater addresses (optional, up to 8)
        repeaters = []
        while pos < len(data) - 3:  # Leave room for CTRL, FCS, FLAG
            # Check if this is the last address (SSID byte bit 0 = 1)
            if data[pos + 6] & 0x01:
                break
            
            repeater = self._decode_address(data[pos:pos+7])
            repeaters.append(repeater)
            pos += 7
            
            if len(repeaters) >= 8:  # Max 8 repeaters
                break
        
        # Parse control field
        if pos >= len(data) - 3:
            return None
        
        control = data[pos]
        pos += 1
        
        # Determine frame type
        frame_type = self._get_frame_type(control)
        
        # Parse PID (Protocol ID) if I or UI frame
        pid = None
        if frame_type == FrameType.INFORMATION or (control & 0xEF) == 0x03:  # UI
            if pos < len(data) - 3:
                pid = data[pos]
                pos += 1
        
        # Extract payload
        if pos < len(data) - 3:
            payload = data[pos:-3]
        else:
            payload = bytes()
        
        # Extract FCS (last 2 bytes before flag)
        fcs = (data[-3] << 8) | data[-2]
        
        # TODO: Validate FCS
        valid = True
        
        return AX25Frame(
            destination=dest,
            source=source,
            repeaters=repeaters,
            control=control,
            frame_type=frame_type,
            pid=pid,
            payload=payload,
            fcs=fcs,
            valid=valid
        )
    
    def _decode_address(self, addr_bytes: bytes) -> str:
        """Decode AX.25 address field (7 bytes)"""
        if len(addr_bytes) != 7:
            return "INVALID"
        
        # Extract callsign (6 bytes, each shifted left by 1)
        callsign = ''
        for i in range(6):
            char = (addr_bytes[i] >> 1) & 0x7F
            if char != 0x20:  # Skip space padding
                callsign += chr(char)
        
        # Extract SSID (last byte, bits 1-4)
        ssid = (addr_bytes[6] >> 1) & 0x0F
        
        # Format callsign-SSID
        if ssid > 0:
            return f"{callsign.strip()}-{ssid}"
        return callsign.strip()
    
    def _get_frame_type(self, control: int) -> FrameType:
        """Determine frame type from control byte"""
        if (control & 0x01) == 0:
            return FrameType.INFORMATION
        elif (control & 0x03) == 0x01:
            return FrameType.SUPERVISORY
        else:
            return FrameType.UNNUMBERED


def demonstrate_parsers():
    """Demonstration of protocol parsers"""
    print("=== Protocol Parsers Demo ===\n")
    
    # Test HDLC Parser
    print("1. HDLC Parser")
    hdlc_data = bytes([
        0x7E,  # Flag
        0xFF,  # Address (all stations)
        0x03,  # Control (UI - Unnumbered Information)
        0x48, 0x65, 0x6C, 0x6C, 0x6F,  # "Hello"
        0x00, 0x00,  # FCS (dummy)
        0x7E   # Flag
    ])
    
    hdlc_parser = HDLCParser()
    hdlc_frame = hdlc_parser.parse(hdlc_data)
    
    if hdlc_frame:
        print(f"  Frame Type: {hdlc_frame.frame_type.value}")
        print(f"  Address: 0x{hdlc_frame.address:02X}")
        print(f"  Command: {hdlc_frame.command}")
        print(f"  Payload: {hdlc_frame.payload.decode('ascii', errors='ignore')}")
        print(f"  Valid: {hdlc_frame.valid}\n")
    
    # Test CCSDS Parser
    print("2. CCSDS Parser")
    ccsds_data = bytes([
        0x08, 0x01,  # Packet ID (Version=0, Type=0, SecHdr=0, APID=1)
        0xC0, 0x00,  # Sequence Control (Flags=11, Count=0)
        0x00, 0x0F,  # Data Length (15 bytes)
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,  # User data
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E
    ])
    
    ccsds_parser = CCSDSParser()
    ccsds_packet = ccsds_parser.parse(ccsds_data)
    
    if ccsds_packet:
        print(f"  Version: {ccsds_packet.version}")
        print(f"  APID: {ccsds_packet.apid}")
        print(f"  Sequence Count: {ccsds_packet.sequence_count}")
        print(f"  Data Length: {ccsds_packet.data_length}")
        print(f"  Valid: {ccsds_packet.valid}\n")
    
    print("3. AX.25 Parser")
    # Creating a realistic AX.25 frame is complex, showing structure
    print("  (AX.25 frame structure demonstrated in code)")
    print("  Format: FLAG | DEST | SOURCE | REPEATERS | CTRL | PID | INFO | FCS | FLAG")
    print("  Callsigns encoded as 7 bytes each (6 chars + SSID)")


if __name__ == '__main__':
    demonstrate_parsers()


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level convenience functions (thin wrappers around the parser classes)
# ─────────────────────────────────────────────────────────────────────────────

_hdlc_parser   = HDLCParser()
_ccsds_parser  = CCSDSParser()
_ax25_parser   = AX25Parser()


def parse_hdlc(data: bytes) -> Optional[HDLCFrame]:
    """Parse an HDLC frame from raw bytes. Returns HDLCFrame or None."""
    return _hdlc_parser.parse(data)


def parse_ccsds(data: bytes) -> Optional[CCSDSPacket]:
    """Parse a CCSDS space packet from raw bytes. Returns CCSDSPacket or None."""
    return _ccsds_parser.parse(data)


def parse_ax25(data: bytes) -> Optional[AX25Frame]:
    """Parse an AX.25 frame from raw bytes. Returns AX25Frame or None."""
    return _ax25_parser.parse(data)


def parse_auto(data: bytes):
    """
    Auto-detect and parse the protocol from raw bytes.
    Tries AX.25, HDLC, CCSDS in order. Returns the first successful parse
    or None if no parser matched.
    """
    if len(data) < 4:
        return None

    # AX.25: starts and ends with 0x7E, minimum 17 bytes
    if data[0] == 0x7E and len(data) >= 17:
        result = _ax25_parser.parse(data)
        if result is not None:
            return result

    # HDLC: starts and ends with 0x7E, minimum 6 bytes
    if data[0] == 0x7E and len(data) >= 6:
        result = _hdlc_parser.parse(data)
        if result is not None:
            return result

    # CCSDS: minimum 6 bytes
    if len(data) >= 6:
        result = _ccsds_parser.parse(data)
        if result is not None:
            return result

    return None
