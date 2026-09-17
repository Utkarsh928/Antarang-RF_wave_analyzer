"""
CRC Validator for Frame Integrity Verification

Implements CRC-16, CRC-32, and CRC-CCITT algorithms used in:
- HDLC frames (CRC-CCITT/CRC-16)
- AX.25 frames (CRC-CCITT)
- CCSDS packets (CRC-CCITT)
- Ethernet frames (CRC-32)

Reference: ITU-T Recommendation V.41 (CRC-CCITT)
"""

import numpy as np
from typing import Tuple, Dict


class CRCValidator:
    """Frame integrity validator using multiple CRC algorithms"""
    
    # Polynomial definitions (standard implementations)
    CRC16_POLYNOMIAL = 0x8005      # CRC-16-IBM/ANSI
    CRC32_POLYNOMIAL = 0x04C11DB7  # CRC-32 (Ethernet)
    CRC_CCITT_POLYNOMIAL = 0x1021  # CRC-CCITT (X.25, HDLC, AX.25)
    
    def __init__(self):
        """Initialize CRC validator with lookup tables for fast computation"""
        self.crc16_table = self._build_crc16_table()
        self.crc32_table = self._build_crc32_table()
        self.crc_ccitt_table = self._build_crc_ccitt_table()
    
    def _build_crc16_table(self) -> np.ndarray:
        """Build CRC-16 lookup table"""
        table = np.zeros(256, dtype=np.uint16)
        for i in range(256):
            crc = i << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ self.CRC16_POLYNOMIAL
                else:
                    crc = crc << 1
                crc &= 0xFFFF
            table[i] = crc
        return table
    
    def _build_crc32_table(self) -> np.ndarray:
        """Build CRC-32 lookup table"""
        table = np.zeros(256, dtype=np.uint32)
        for i in range(256):
            crc = i << 24
            for _ in range(8):
                if crc & 0x80000000:
                    crc = (crc << 1) ^ self.CRC32_POLYNOMIAL
                else:
                    crc = crc << 1
                crc &= 0xFFFFFFFF
            table[i] = crc
        return table
    
    def _build_crc_ccitt_table(self) -> np.ndarray:
        """Build CRC-CCITT lookup table"""
        table = np.zeros(256, dtype=np.uint16)
        for i in range(256):
            crc = i << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ self.CRC_CCITT_POLYNOMIAL
                else:
                    crc = crc << 1
                crc &= 0xFFFF
            table[i] = crc
        return table
    
    def calculate_crc16(self, data: bytes, init: int = 0xFFFF) -> int:
        """
        Calculate CRC-16 (IBM/ANSI polynomial)
        
        Args:
            data: Input bytes
            init: Initial CRC value (default 0xFFFF)
            
        Returns:
            16-bit CRC value
        """
        crc = init
        for byte in data:
            index = ((crc >> 8) ^ byte) & 0xFF
            crc = ((crc << 8) ^ self.crc16_table[index]) & 0xFFFF
        return crc
    
    def calculate_crc32(self, data: bytes, init: int = 0xFFFFFFFF) -> int:
        """
        Calculate CRC-32 (Ethernet polynomial)
        
        Args:
            data: Input bytes
            init: Initial CRC value (default 0xFFFFFFFF)
            
        Returns:
            32-bit CRC value
        """
        crc = init
        for byte in data:
            index = ((crc >> 24) ^ byte) & 0xFF
            crc = ((crc << 8) ^ self.crc32_table[index]) & 0xFFFFFFFF
        return crc ^ 0xFFFFFFFF
    
    def calculate_crc_ccitt(self, data: bytes, init: int = 0xFFFF) -> int:
        """
        Calculate CRC-CCITT (X.25, HDLC, AX.25 standard)
        
        Args:
            data: Input bytes
            init: Initial CRC value (default 0xFFFF for X.25)
            
        Returns:
            16-bit CRC-CCITT value
        """
        crc = init
        for byte in data:
            index = ((crc >> 8) ^ byte) & 0xFF
            crc = ((crc << 8) ^ self.crc_ccitt_table[index]) & 0xFFFF
        return crc ^ 0xFFFF  # Final XOR for X.25
    
    def verify_frame(self, data: bytes) -> Dict[str, any]:
        """
        Verify frame integrity using all CRC methods
        
        Tries CRC-16, CRC-32, and CRC-CCITT to determine which (if any) is valid.
        
        Args:
            data: Complete frame including CRC bytes
            
        Returns:
            Dictionary with verification results:
            {
                'valid': bool,
                'crc_type': str ('CRC-16', 'CRC-32', 'CRC-CCITT', or 'NONE'),
                'crc_received': int,
                'crc_calculated': int,
                'payload': bytes (data without CRC)
            }
        """
        if len(data) < 3:
            return {
                'valid': False,
                'crc_type': 'NONE',
                'error': 'Data too short for CRC'
            }
        
        results = []
        
        # Try CRC-16 (last 2 bytes)
        if len(data) >= 2:
            payload = data[:-2]
            crc_received = (data[-2] << 8) | data[-1]
            crc_calc = self.calculate_crc16(payload)
            
            results.append({
                'valid': crc_received == crc_calc,
                'crc_type': 'CRC-16',
                'crc_received': crc_received,
                'crc_calculated': crc_calc,
                'payload': payload,
                'match_quality': 1.0 if crc_received == crc_calc else 0.0
            })
        
        # Try CRC-CCITT (last 2 bytes, different algorithm)
        if len(data) >= 2:
            payload = data[:-2]
            crc_received = (data[-2] << 8) | data[-1]
            crc_calc = self.calculate_crc_ccitt(payload)
            
            results.append({
                'valid': crc_received == crc_calc,
                'crc_type': 'CRC-CCITT',
                'crc_received': crc_received,
                'crc_calculated': crc_calc,
                'payload': payload,
                'match_quality': 1.0 if crc_received == crc_calc else 0.0
            })
        
        # Try CRC-32 (last 4 bytes)
        if len(data) >= 4:
            payload = data[:-4]
            crc_received = (data[-4] << 24) | (data[-3] << 16) | (data[-2] << 8) | data[-1]
            crc_calc = self.calculate_crc32(payload)
            
            results.append({
                'valid': crc_received == crc_calc,
                'crc_type': 'CRC-32',
                'crc_received': crc_received,
                'crc_calculated': crc_calc,
                'payload': payload,
                'match_quality': 1.0 if crc_received == crc_calc else 0.0
            })
        
        # Return first valid match, or best guess
        valid_results = [r for r in results if r['valid']]
        
        if valid_results:
            return valid_results[0]  # Return first valid CRC
        else:
            # Return most likely (though invalid)
            return sorted(results, key=lambda x: x['match_quality'], reverse=True)[0]
    
    def validate_batch(self, frames: list) -> list:
        """
        Validate multiple frames
        
        Args:
            frames: List of byte arrays
            
        Returns:
            List of validation results
        """
        return [self.verify_frame(frame) for frame in frames]
    
    def get_crc_statistics(self, frames: list) -> Dict[str, any]:
        """
        Analyze CRC usage across multiple frames
        
        Args:
            frames: List of byte arrays
            
        Returns:
            Statistics dictionary with CRC type distribution
        """
        results = self.validate_batch(frames)
        
        crc_counts = {'CRC-16': 0, 'CRC-32': 0, 'CRC-CCITT': 0, 'NONE': 0}
        valid_count = 0
        
        for result in results:
            if result.get('valid', False):
                crc_counts[result['crc_type']] += 1
                valid_count += 1
            else:
                crc_counts['NONE'] += 1
        
        total = len(frames)
        
        return {
            'total_frames': total,
            'valid_frames': valid_count,
            'invalid_frames': total - valid_count,
            'validity_rate': valid_count / total if total > 0 else 0.0,
            'crc_type_distribution': crc_counts,
            'most_common_crc': max(crc_counts.items(), key=lambda x: x[1])[0]
        }


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Convert bit array to bytes for CRC calculation
    
    Args:
        bits: Binary array (0s and 1s)
        
    Returns:
        Packed bytes
    """
    if len(bits) % 8 != 0:
        # Pad to byte boundary
        padding = 8 - (len(bits) % 8)
        bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
    
    # Pack bits into bytes
    bytes_array = np.packbits(bits).tobytes()
    return bytes_array


def validate_frame_from_bits(bits: np.ndarray) -> Dict[str, any]:
    """
    Convenience function to validate frame from bit array
    
    Args:
        bits: Binary array
        
    Returns:
        Validation result dictionary
    """
    data = bits_to_bytes(bits)
    validator = CRCValidator()
    return validator.verify_frame(data)
