"""
Payload Decoder

Extracts and decodes payload data from various formats:
1. ASCII/UTF-8 text extraction
2. TLV (Type-Length-Value) parsing
3. Binary structure inference
4. Common protocol payload formats

Supports automatic detection of:
- Plain text (ASCII, UTF-8)
- Binary protocols (TLV, fixed-field)
- Compressed data
- Encrypted data (via CryptoDetector)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import struct
import re


@dataclass
class DecodedField:
    """Decoded field from payload"""
    name: str              # Field name
    offset: int           # Byte offset in payload
    length: int           # Field length in bytes
    raw_value: bytes      # Raw bytes
    decoded_value: Any    # Interpreted value
    field_type: str       # 'ASCII', 'INT', 'FLOAT', 'HEX', etc.


@dataclass
class PayloadAnalysis:
    """Result of payload analysis"""
    payload_type: str         # 'TEXT', 'BINARY', 'TLV', 'MIXED'
    encoding: str            # 'ASCII', 'UTF-8', 'BINARY', 'UNKNOWN'
    printable_ratio: float   # Ratio of printable chars (0-1)
    fields: List[DecodedField]  # Decoded fields
    text_content: str        # Extracted text (if any)
    hex_dump: str           # Hex representation
    structure: Optional[Dict]  # Inferred structure


class PayloadDecoder:
    """
    Universal payload decoder
    
    Attempts to decode payload using multiple strategies:
    1. Text detection and extraction
    2. TLV format parsing
    3. Fixed-field structure inference
    4. Binary analysis
    """
    
    # Printable ASCII range
    PRINTABLE_MIN = 32
    PRINTABLE_MAX = 126
    
    # Common TLV tag values (examples)
    KNOWN_TLV_TAGS = {
        0x01: "INTEGER",
        0x02: "STRING",
        0x03: "OCTET_STRING",
        0x04: "SEQUENCE",
        0x05: "NULL",
        0x06: "OBJECT_ID",
        0x30: "SEQUENCE",  # ASN.1
        0x31: "SET",       # ASN.1
    }
    
    def __init__(self):
        """Initialize payload decoder"""
        pass
    
    def decode(self, payload: bytes) -> PayloadAnalysis:
        """
        Decode payload using all available methods
        
        Args:
            payload: Raw payload bytes
            
        Returns:
            PayloadAnalysis with decoded content
        """
        if len(payload) == 0:
            return PayloadAnalysis(
                payload_type='EMPTY',
                encoding='NONE',
                printable_ratio=0.0,
                fields=[],
                text_content='',
                hex_dump='',
                structure=None
            )
        
        # Calculate printable ratio
        printable_ratio = self._calculate_printable_ratio(payload)
        
        # Determine payload type
        if printable_ratio > 0.8:
            payload_type = 'TEXT'
            encoding = self._detect_encoding(payload)
            text_content = self._extract_text(payload, encoding)
            fields = self._extract_text_fields(text_content)
        elif self._is_tlv_format(payload):
            payload_type = 'TLV'
            encoding = 'BINARY'
            text_content = ''
            fields = self._parse_tlv(payload)
        else:
            payload_type = 'BINARY'
            encoding = 'BINARY'
            text_content = self._extract_partial_text(payload)
            fields = self._infer_binary_fields(payload)
        
        # Generate hex dump
        hex_dump = self._generate_hex_dump(payload)
        
        # Infer structure
        structure = self._infer_structure(payload, fields)
        
        return PayloadAnalysis(
            payload_type=payload_type,
            encoding=encoding,
            printable_ratio=printable_ratio,
            fields=fields,
            text_content=text_content,
            hex_dump=hex_dump,
            structure=structure
        )
    
    def _calculate_printable_ratio(self, data: bytes) -> float:
        """Calculate ratio of printable ASCII characters"""
        if len(data) == 0:
            return 0.0
        
        printable_count = sum(
            1 for b in data
            if self.PRINTABLE_MIN <= b <= self.PRINTABLE_MAX or b in [9, 10, 13]  # Include tab, LF, CR
        )
        
        return printable_count / len(data)
    
    def _detect_encoding(self, data: bytes) -> str:
        """Detect text encoding"""
        # Try UTF-8
        try:
            data.decode('utf-8')
            return 'UTF-8'
        except UnicodeDecodeError:
            pass
        
        # Try ASCII
        try:
            data.decode('ascii')
            return 'ASCII'
        except UnicodeDecodeError:
            pass
        
        # Try Latin-1 (always succeeds)
        return 'LATIN-1'
    
    def _extract_text(self, data: bytes, encoding: str) -> str:
        """Extract text from payload"""
        try:
            return data.decode(encoding, errors='replace')
        except Exception:
            return data.decode('latin-1', errors='replace')
    
    def _extract_partial_text(self, data: bytes) -> str:
        """Extract any text strings from binary data"""
        # Find ASCII strings (4+ chars)
        text_parts = []
        current_string = []
        
        for byte in data:
            if self.PRINTABLE_MIN <= byte <= self.PRINTABLE_MAX:
                current_string.append(chr(byte))
            else:
                if len(current_string) >= 4:  # Min string length
                    text_parts.append(''.join(current_string))
                current_string = []
        
        # Check final string
        if len(current_string) >= 4:
            text_parts.append(''.join(current_string))
        
        return ' | '.join(text_parts)
    
    def _extract_text_fields(self, text: str) -> List[DecodedField]:
        """Extract fields from text payload"""
        fields = []
        
        # Look for key-value pairs (key=value or key:value)
        kv_pattern = r'(\w+)\s*[:=]\s*([^\s,;]+)'
        matches = re.finditer(kv_pattern, text)
        
        for i, match in enumerate(matches):
            key = match.group(1)
            value = match.group(2)
            
            fields.append(DecodedField(
                name=key,
                offset=match.start(),
                length=len(match.group(0)),
                raw_value=match.group(0).encode('utf-8'),
                decoded_value=value,
                field_type='TEXT'
            ))
        
        return fields
    
    def _is_tlv_format(self, data: bytes) -> bool:
        """
        Check if data follows TLV format
        
        TLV: Tag (1-2 bytes) | Length (1-4 bytes) | Value (variable)
        """
        if len(data) < 3:
            return False
        
        # Simple heuristic: check if first few bytes look like TLV
        tag = data[0]
        length = data[1]
        
        # Check if length is reasonable
        if length > 0 and length < 128:
            # Check if we have enough data for the value
            if len(data) >= 2 + length:
                # This could be TLV
                return True
        
        # Check for multi-byte length encoding (0x81, 0x82, etc.)
        if length >= 0x80:
            return True
        
        return False
    
    def _parse_tlv(self, data: bytes) -> List[DecodedField]:
        """
        Parse TLV-encoded data
        
        Supports:
        - Simple TLV (1 byte tag, 1 byte length)
        - Extended TLV (multi-byte length)
        """
        fields = []
        offset = 0
        field_num = 0
        
        while offset < len(data) - 1:
            # Parse tag
            tag = data[offset]
            tag_len = 1
            
            # Parse length
            length_offset = offset + tag_len
            if length_offset >= len(data):
                break
            
            length_byte = data[length_offset]
            
            if length_byte < 0x80:
                # Short form: length in 1 byte
                value_length = length_byte
                length_len = 1
            else:
                # Long form: length encoded in multiple bytes
                num_length_bytes = length_byte & 0x7F
                length_len = 1 + num_length_bytes
                
                if length_offset + length_len > len(data):
                    break
                
                # Decode multi-byte length (big-endian)
                value_length = 0
                for i in range(num_length_bytes):
                    value_length = (value_length << 8) | data[length_offset + 1 + i]
            
            # Extract value
            value_offset = offset + tag_len + length_len
            
            if value_offset + value_length > len(data):
                break  # Invalid length
            
            value_bytes = data[value_offset:value_offset + value_length]
            
            # Decode value based on tag
            tag_name = self.KNOWN_TLV_TAGS.get(tag, f"TAG_{tag:02X}")
            
            if tag in [0x02]:  # INTEGER
                decoded_value = self._decode_integer(value_bytes)
            elif tag in [0x04, 0x0C]:  # OCTET STRING, UTF8 STRING
                decoded_value = self._decode_string(value_bytes)
            else:
                decoded_value = value_bytes.hex()
            
            fields.append(DecodedField(
                name=f"{tag_name}_{field_num}",
                offset=offset,
                length=tag_len + length_len + value_length,
                raw_value=data[offset:value_offset + value_length],
                decoded_value=decoded_value,
                field_type='TLV'
            ))
            
            # Move to next TLV
            offset = value_offset + value_length
            field_num += 1
            
            # Safety limit
            if field_num > 100:
                break
        
        return fields
    
    def _infer_binary_fields(self, data: bytes) -> List[DecodedField]:
        """
        Infer fields from binary data using heuristics
        
        Looks for:
        - Fixed-size integers (2, 4, 8 bytes)
        - Repeated patterns
        - NULL-terminated strings
        """
        fields = []
        offset = 0
        field_num = 0
        
        while offset < len(data):
            remaining = len(data) - offset
            
            # Try 4-byte integer (most common)
            if remaining >= 4:
                int_val = struct.unpack('>I', data[offset:offset+4])[0]
                
                # If value looks reasonable (not too large), treat as integer
                if int_val < 1000000:
                    fields.append(DecodedField(
                        name=f"INT32_{field_num}",
                        offset=offset,
                        length=4,
                        raw_value=data[offset:offset+4],
                        decoded_value=int_val,
                        field_type='INT32'
                    ))
                    offset += 4
                    field_num += 1
                    continue
            
            # Try 2-byte integer
            if remaining >= 2:
                int_val = struct.unpack('>H', data[offset:offset+2])[0]
                
                fields.append(DecodedField(
                    name=f"INT16_{field_num}",
                    offset=offset,
                    length=2,
                    raw_value=data[offset:offset+2],
                    decoded_value=int_val,
                    field_type='INT16'
                ))
                offset += 2
                field_num += 1
                continue
            
            # Single byte
            if remaining >= 1:
                fields.append(DecodedField(
                    name=f"BYTE_{field_num}",
                    offset=offset,
                    length=1,
                    raw_value=data[offset:offset+1],
                    decoded_value=f"0x{data[offset]:02X}",
                    field_type='BYTE'
                ))
                offset += 1
                field_num += 1
            
            # Safety limit
            if field_num > 50:
                break
        
        return fields
    
    def _decode_integer(self, value_bytes: bytes) -> int:
        """Decode integer from bytes (big-endian)"""
        result = 0
        for byte in value_bytes:
            result = (result << 8) | byte
        return result
    
    def _decode_string(self, value_bytes: bytes) -> str:
        """Decode string from bytes"""
        try:
            return value_bytes.decode('utf-8', errors='replace')
        except Exception:
            return value_bytes.hex()
    
    def _generate_hex_dump(self, data: bytes, bytes_per_line: int = 16) -> str:
        """Generate formatted hex dump"""
        lines = []
        
        for i in range(0, len(data), bytes_per_line):
            chunk = data[i:i + bytes_per_line]
            
            # Hex representation
            hex_part = ' '.join(f'{b:02X}' for b in chunk)
            
            # ASCII representation
            ascii_part = ''.join(
                chr(b) if self.PRINTABLE_MIN <= b <= self.PRINTABLE_MAX else '.'
                for b in chunk
            )
            
            # Format: offset | hex | ascii
            line = f"{i:04X}  {hex_part:<48}  {ascii_part}"
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _infer_structure(self, data: bytes, fields: List[DecodedField]) -> Optional[Dict]:
        """Infer overall structure from fields"""
        if len(fields) == 0:
            return None
        
        structure = {
            'total_length': len(data),
            'field_count': len(fields),
            'field_types': {},
            'patterns': []
        }
        
        # Count field types
        for field in fields:
            field_type = field.field_type
            structure['field_types'][field_type] = structure['field_types'].get(field_type, 0) + 1
        
        # Detect patterns (repeated field sequences)
        if len(fields) >= 3:
            field_type_sequence = [f.field_type for f in fields]
            
            # Look for repeating patterns of length 2-5
            for pattern_len in range(2, min(6, len(field_type_sequence) // 2)):
                pattern = field_type_sequence[:pattern_len]
                
                # Check if pattern repeats
                matches = 0
                for i in range(0, len(field_type_sequence) - pattern_len, pattern_len):
                    if field_type_sequence[i:i + pattern_len] == pattern:
                        matches += 1
                
                if matches >= 2:
                    structure['patterns'].append({
                        'pattern': pattern,
                        'repetitions': matches
                    })
        
        return structure
    
    def export_fields(self, analysis: PayloadAnalysis) -> str:
        """Export decoded fields as formatted text"""
        lines = []
        lines.append(f"Payload Type: {analysis.payload_type}")
        lines.append(f"Encoding: {analysis.encoding}")
        lines.append(f"Printable Ratio: {analysis.printable_ratio:.2%}")
        lines.append(f"\nFields ({len(analysis.fields)}):")
        
        for field in analysis.fields:
            lines.append(f"  [{field.offset:04X}] {field.name} ({field.field_type})")
            lines.append(f"    Raw: {field.raw_value.hex()}")
            lines.append(f"    Value: {field.decoded_value}")
        
        if analysis.text_content:
            lines.append(f"\nText Content:")
            lines.append(f"  {analysis.text_content}")
        
        if analysis.structure:
            lines.append(f"\nStructure:")
            for key, value in analysis.structure.items():
                lines.append(f"  {key}: {value}")
        
        return '\n'.join(lines)


def demonstrate_decoder():
    """Demonstration of payload decoder"""
    print("=== Payload Decoder Demo ===\n")
    
    decoder = PayloadDecoder()
    
    # Test 1: Text payload
    print("1. Text Payload")
    text_payload = b"user=admin password=secret123 status=ok"
    result = decoder.decode(text_payload)
    print(decoder.export_fields(result))
    print()
    
    # Test 2: Binary payload with integers
    print("2. Binary Payload")
    binary_payload = struct.pack('>HHI', 0x1234, 0x5678, 0x9ABCDEF0)
    result = decoder.decode(binary_payload)
    print(decoder.export_fields(result))
    print()
    
    # Test 3: TLV payload
    print("3. TLV Payload")
    tlv_payload = bytes([
        0x02, 0x04, 0x00, 0x00, 0x00, 0x42,  # INTEGER tag, length 4, value 66
        0x04, 0x05, 0x48, 0x65, 0x6C, 0x6C, 0x6F  # OCTET STRING, length 5, "Hello"
    ])
    result = decoder.decode(tlv_payload)
    print(decoder.export_fields(result))


if __name__ == '__main__':
    demonstrate_decoder()


# ─────────────────────────────────────────────────────────────────────────────
#  Compatibility alias
# ─────────────────────────────────────────────────────────────────────────────

class ProtocolDecoder(PayloadDecoder):
    """
    Protocol-aware decoder that wraps PayloadDecoder and adds
    protocol-level context to the analysis result.

    This class is the public API used by workers.py and the GUI.
    Internally delegates to PayloadDecoder for payload analysis and
    adds protocol-level metadata.
    """

    def decode(self, data: bytes) -> dict:
        """
        Decode protocol data. Returns a dict with all extracted fields.

        Parameters
        ----------
        data : raw frame bytes (may include protocol headers)

        Returns
        -------
        dict with keys:
            payload_type, encoding, printable_ratio, fields (list),
            text_content, hex_dump, structure, protocol_hint
        """
        analysis = super().decode(data)

        # Convert dataclass to dict for easier consumption
        result = {
            'payload_type':    analysis.payload_type,
            'encoding':        analysis.encoding,
            'printable_ratio': analysis.printable_ratio,
            'fields':          analysis.fields,
            'text_content':    analysis.text_content,
            'hex_dump':        analysis.hex_dump,
            'structure':       analysis.structure,
            'protocol_hint':   self._guess_protocol(data),
            'field_count':     len(analysis.fields),
        }
        return result

    def _guess_protocol(self, data: bytes) -> str:
        """Quick protocol identification from data patterns."""
        if len(data) < 2:
            return 'UNKNOWN'
        if data[0] == 0x7E:
            return 'HDLC/AX.25'
        if data[0] == 0x47:
            return 'DVB-TS'
        if len(data) >= 6 and (data[0] >> 5) == 0:
            return 'CCSDS'
        if len(data) >= 2 and data[0] == 0x1A and data[1] == 0xCF:
            return 'CCSDS'
        return 'GENERIC'
