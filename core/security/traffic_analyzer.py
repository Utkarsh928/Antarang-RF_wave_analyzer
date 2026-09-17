"""
Traffic Pattern Analyzer

Analyzes temporal patterns in signal transmissions:
1. Burst detection and characterization
2. Inter-frame gap analysis  
3. Transmission timing patterns
4. Channel utilization statistics
5. Multi-signal correlation

Useful for:
- Protocol behavior analysis
- Anomaly detection
- Communication pattern fingerprinting
- Network reconnaissance
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import signal, stats
from collections import Counter


@dataclass
class Burst:
    """Detected transmission burst"""
    start_time: float      # Start time (seconds)
    end_time: float        # End time (seconds)
    duration: float        # Duration (seconds)
    frame_count: int       # Number of frames in burst
    data_rate: float       # Average data rate (bits/sec)
    inter_frame_gaps: List[float]  # Gaps between frames (seconds)


@dataclass
class TimingPattern:
    """Detected timing pattern"""
    pattern_type: str      # 'PERIODIC', 'RANDOM', 'BURSTY'
    period: Optional[float]  # Period if periodic (seconds)
    jitter: float          # Timing variation (seconds)
    confidence: float      # Detection confidence (0-1)
    characteristics: Dict   # Additional pattern features


@dataclass
class TrafficStatistics:
    """Overall traffic statistics"""
    total_frames: int
    total_duration: float      # seconds
    average_frame_rate: float  # frames/sec
    total_data_bits: int
    average_data_rate: float   # bits/sec
    channel_utilization: float  # 0-1
    bursts: List[Burst]
    timing_pattern: TimingPattern
    inter_frame_gaps: np.ndarray
    gap_statistics: Dict


class TrafficAnalyzer:
    """
    Analyze temporal patterns in signal transmissions
    
    Processes frame timing information to detect:
    - Burst patterns
    - Periodic transmissions
    - Random vs. scheduled traffic
    - Communication patterns
    """
    
    def __init__(
        self,
        burst_threshold: float = 0.1,  # Max gap in burst (seconds)
        min_burst_frames: int = 3       # Min frames to count as burst
    ):
        """
        Initialize traffic analyzer
        
        Args:
            burst_threshold: Maximum inter-frame gap to consider part of same burst
            min_burst_frames: Minimum frames to count as a burst
        """
        self.burst_threshold = burst_threshold
        self.min_burst_frames = min_burst_frames
    
    def analyze(
        self,
        frame_times: List[float],
        frame_lengths: List[int],
        sample_rate: Optional[float] = None
    ) -> TrafficStatistics:
        """
        Analyze traffic patterns
        
        Args:
            frame_times: List of frame start times (seconds)
            frame_lengths: List of frame lengths (bits)
            sample_rate: Optional sample rate for utilization calculation
            
        Returns:
            TrafficStatistics
        """
        if len(frame_times) == 0:
            return self._empty_statistics()
        
        # Sort by time
        sorted_indices = np.argsort(frame_times)
        frame_times = np.array(frame_times)[sorted_indices]
        frame_lengths = np.array(frame_lengths)[sorted_indices]
        
        # Calculate inter-frame gaps
        if len(frame_times) > 1:
            gaps = np.diff(frame_times)
        else:
            gaps = np.array([])
        
        # Detect bursts
        bursts = self._detect_bursts(frame_times, frame_lengths, gaps)
        
        # Detect timing patterns
        timing_pattern = self._detect_timing_pattern(frame_times, gaps)
        
        # Calculate statistics
        total_duration = frame_times[-1] - frame_times[0] if len(frame_times) > 1 else 0.0
        total_frames = len(frame_times)
        average_frame_rate = total_frames / total_duration if total_duration > 0 else 0.0
        
        total_data_bits = np.sum(frame_lengths)
        average_data_rate = total_data_bits / total_duration if total_duration > 0 else 0.0
        
        # Channel utilization (if sample rate known)
        if sample_rate and total_duration > 0:
            channel_capacity = sample_rate * total_duration  # Total possible bits
            channel_utilization = total_data_bits / channel_capacity
        else:
            channel_utilization = 0.0
        
        # Gap statistics
        gap_stats = self._calculate_gap_statistics(gaps)
        
        return TrafficStatistics(
            total_frames=total_frames,
            total_duration=total_duration,
            average_frame_rate=average_frame_rate,
            total_data_bits=int(total_data_bits),
            average_data_rate=average_data_rate,
            channel_utilization=channel_utilization,
            bursts=bursts,
            timing_pattern=timing_pattern,
            inter_frame_gaps=gaps,
            gap_statistics=gap_stats
        )
    
    def _detect_bursts(
        self,
        frame_times: np.ndarray,
        frame_lengths: np.ndarray,
        gaps: np.ndarray
    ) -> List[Burst]:
        """
        Detect transmission bursts
        
        A burst is a sequence of frames with small inter-frame gaps.
        """
        if len(gaps) == 0:
            return []
        
        bursts = []
        burst_start = 0
        burst_frames = [0]  # Include first frame
        
        for i, gap in enumerate(gaps):
            if gap <= self.burst_threshold:
                # Continue burst
                burst_frames.append(i + 1)
            else:
                # End of burst
                if len(burst_frames) >= self.min_burst_frames:
                    # Record burst
                    burst = self._create_burst(
                        frame_times,
                        frame_lengths,
                        burst_frames
                    )
                    bursts.append(burst)
                
                # Start new burst
                burst_start = i + 1
                burst_frames = [i + 1]
        
        # Handle final burst
        if len(burst_frames) >= self.min_burst_frames:
            burst = self._create_burst(
                frame_times,
                frame_lengths,
                burst_frames
            )
            bursts.append(burst)
        
        return bursts
    
    def _create_burst(
        self,
        frame_times: np.ndarray,
        frame_lengths: np.ndarray,
        burst_frame_indices: List[int]
    ) -> Burst:
        """Create burst object from frame indices"""
        start_time = frame_times[burst_frame_indices[0]]
        end_time = frame_times[burst_frame_indices[-1]]
        duration = end_time - start_time
        
        frame_count = len(burst_frame_indices)
        
        # Calculate inter-frame gaps within burst
        if len(burst_frame_indices) > 1:
            burst_times = frame_times[burst_frame_indices]
            inter_frame_gaps = list(np.diff(burst_times))
        else:
            inter_frame_gaps = []
        
        # Calculate data rate
        total_bits = np.sum(frame_lengths[burst_frame_indices])
        data_rate = total_bits / duration if duration > 0 else 0.0
        
        return Burst(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            frame_count=frame_count,
            data_rate=data_rate,
            inter_frame_gaps=inter_frame_gaps
        )
    
    def _detect_timing_pattern(
        self,
        frame_times: np.ndarray,
        gaps: np.ndarray
    ) -> TimingPattern:
        """
        Detect timing pattern in frame transmissions
        
        Classifies as:
        - PERIODIC: Regular intervals (e.g., beacon frames)
        - BURSTY: Clustered transmissions
        - RANDOM: No clear pattern
        """
        if len(gaps) < 3:
            return TimingPattern(
                pattern_type='INSUFFICIENT_DATA',
                period=None,
                jitter=0.0,
                confidence=0.0,
                characteristics={}
            )
        
        # Test for periodicity using autocorrelation
        period, periodic_confidence = self._test_periodicity(gaps)
        
        # Calculate jitter (standard deviation of gaps)
        jitter = float(np.std(gaps))
        
        # Test for burstiness (coefficient of variation)
        mean_gap = np.mean(gaps)
        cv = jitter / mean_gap if mean_gap > 0 else 0.0
        
        # Classification
        if periodic_confidence > 0.7:
            pattern_type = 'PERIODIC'
            confidence = periodic_confidence
        elif cv > 1.5:  # High coefficient of variation
            pattern_type = 'BURSTY'
            confidence = 0.8
        else:
            pattern_type = 'RANDOM'
            confidence = 0.6
        
        characteristics = {
            'mean_gap': float(mean_gap),
            'coefficient_of_variation': float(cv),
            'gap_entropy': self._calculate_gap_entropy(gaps)
        }
        
        return TimingPattern(
            pattern_type=pattern_type,
            period=period,
            jitter=jitter,
            confidence=confidence,
            characteristics=characteristics
        )
    
    def _test_periodicity(self, gaps: np.ndarray) -> Tuple[Optional[float], float]:
        """
        Test if gaps show periodic pattern
        
        Uses autocorrelation to detect periodicity.
        
        Returns:
            (period, confidence)
        """
        if len(gaps) < 10:
            return (None, 0.0)
        
        # Check if gaps are roughly uniform (periodic transmission)
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        
        # Coefficient of variation (low = periodic)
        cv = std_gap / mean_gap if mean_gap > 0 else float('inf')
        
        if cv < 0.2:  # Very regular
            # Periodic with high confidence
            return (mean_gap, 0.9)
        elif cv < 0.5:
            # Somewhat periodic
            return (mean_gap, 0.7)
        else:
            # Not periodic
            return (None, 0.0)
    
    def _calculate_gap_entropy(self, gaps: np.ndarray) -> float:
        """
        Calculate entropy of gap distribution
        
        High entropy = random/unpredictable
        Low entropy = predictable pattern
        """
        if len(gaps) < 2:
            return 0.0
        
        # Bin gaps into categories
        hist, _ = np.histogram(gaps, bins=20)
        
        # Calculate probabilities
        probabilities = hist / np.sum(hist)
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        
        # Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Normalize to 0-1
        max_entropy = np.log2(len(probabilities)) if len(probabilities) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(normalized_entropy)
    
    def _calculate_gap_statistics(self, gaps: np.ndarray) -> Dict:
        """Calculate detailed gap statistics"""
        if len(gaps) == 0:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'q25': 0.0,
                'q75': 0.0
            }
        
        return {
            'min': float(np.min(gaps)),
            'max': float(np.max(gaps)),
            'mean': float(np.mean(gaps)),
            'median': float(np.median(gaps)),
            'std': float(np.std(gaps)),
            'q25': float(np.percentile(gaps, 25)),
            'q75': float(np.percentile(gaps, 75))
        }
    
    def _empty_statistics(self) -> TrafficStatistics:
        """Return empty statistics object"""
        return TrafficStatistics(
            total_frames=0,
            total_duration=0.0,
            average_frame_rate=0.0,
            total_data_bits=0,
            average_data_rate=0.0,
            channel_utilization=0.0,
            bursts=[],
            timing_pattern=TimingPattern(
                pattern_type='NO_DATA',
                period=None,
                jitter=0.0,
                confidence=0.0,
                characteristics={}
            ),
            inter_frame_gaps=np.array([]),
            gap_statistics={}
        )
    
    def compare_traffic_patterns(
        self,
        stats1: TrafficStatistics,
        stats2: TrafficStatistics
    ) -> Dict[str, float]:
        """
        Compare two traffic patterns for similarity
        
        Useful for protocol fingerprinting or anomaly detection.
        
        Returns:
            Similarity metrics (0-1, higher = more similar)
        """
        similarities = {}
        
        # Compare frame rates
        if stats1.average_frame_rate > 0 and stats2.average_frame_rate > 0:
            rate_ratio = min(stats1.average_frame_rate, stats2.average_frame_rate) / \
                        max(stats1.average_frame_rate, stats2.average_frame_rate)
            similarities['frame_rate'] = rate_ratio
        
        # Compare timing patterns
        if stats1.timing_pattern.pattern_type == stats2.timing_pattern.pattern_type:
            similarities['timing_pattern'] = 1.0
        else:
            similarities['timing_pattern'] = 0.0
        
        # Compare burst characteristics
        burst_similarity = self._compare_burst_patterns(stats1.bursts, stats2.bursts)
        similarities['burst_pattern'] = burst_similarity
        
        # Overall similarity
        similarities['overall'] = np.mean(list(similarities.values()))
        
        return similarities
    
    def _compare_burst_patterns(self, bursts1: List[Burst], bursts2: List[Burst]) -> float:
        """Compare burst patterns between two traffic samples"""
        if len(bursts1) == 0 or len(bursts2) == 0:
            return 0.0 if len(bursts1) != len(bursts2) else 1.0
        
        # Compare average burst duration
        avg_dur1 = np.mean([b.duration for b in bursts1])
        avg_dur2 = np.mean([b.duration for b in bursts2])
        
        duration_similarity = min(avg_dur1, avg_dur2) / max(avg_dur1, avg_dur2)
        
        # Compare average burst size (frame count)
        avg_size1 = np.mean([b.frame_count for b in bursts1])
        avg_size2 = np.mean([b.frame_count for b in bursts2])
        
        size_similarity = min(avg_size1, avg_size2) / max(avg_size1, avg_size2)
        
        return (duration_similarity + size_similarity) / 2.0
    
    def export_report(self, stats: TrafficStatistics) -> str:
        """Generate human-readable traffic analysis report"""
        lines = []
        
        lines.append("=== Traffic Analysis Report ===\n")
        
        # Overall statistics
        lines.append("Overall Statistics:")
        lines.append(f"  Total Frames: {stats.total_frames}")
        lines.append(f"  Duration: {stats.total_duration:.3f} seconds")
        lines.append(f"  Frame Rate: {stats.average_frame_rate:.2f} frames/sec")
        lines.append(f"  Data Rate: {stats.average_data_rate:.2f} bits/sec")
        lines.append(f"  Channel Utilization: {stats.channel_utilization:.2%}\n")
        
        # Timing pattern
        lines.append("Timing Pattern:")
        lines.append(f"  Type: {stats.timing_pattern.pattern_type}")
        lines.append(f"  Confidence: {stats.timing_pattern.confidence:.2%}")
        if stats.timing_pattern.period:
            lines.append(f"  Period: {stats.timing_pattern.period:.6f} seconds")
        lines.append(f"  Jitter: {stats.timing_pattern.jitter:.6f} seconds\n")
        
        # Burst analysis
        lines.append(f"Bursts Detected: {len(stats.bursts)}")
        if len(stats.bursts) > 0:
            avg_burst_duration = np.mean([b.duration for b in stats.bursts])
            avg_burst_frames = np.mean([b.frame_count for b in stats.bursts])
            lines.append(f"  Average Duration: {avg_burst_duration:.3f} seconds")
            lines.append(f"  Average Frames per Burst: {avg_burst_frames:.1f}")
        lines.append("")
        
        # Gap statistics
        if len(stats.inter_frame_gaps) > 0:
            lines.append("Inter-Frame Gap Statistics:")
            for key, value in stats.gap_statistics.items():
                lines.append(f"  {key}: {value:.6f} seconds")
        
        return '\n'.join(lines)


def demonstrate_traffic_analysis():
    """Demonstration of traffic analysis"""
    print("=== Traffic Analyzer Demo ===\n")
    
    analyzer = TrafficAnalyzer(burst_threshold=0.1, min_burst_frames=3)
    
    # Test 1: Periodic transmission
    print("1. Periodic Transmission Pattern")
    frame_times = [i * 0.1 for i in range(20)]  # Every 100ms
    frame_lengths = [256] * 20
    
    stats = analyzer.analyze(frame_times, frame_lengths, sample_rate=10000)
    print(analyzer.export_report(stats))
    
    # Test 2: Bursty transmission
    print("\n2. Bursty Transmission Pattern")
    frame_times = [0.0, 0.01, 0.02, 0.03,  # Burst 1
                   1.0, 1.01, 1.02, 1.03,  # Burst 2
                   2.5, 2.51, 2.52, 2.53]  # Burst 3
    frame_lengths = [128] * 12
    
    stats = analyzer.analyze(frame_times, frame_lengths, sample_rate=10000)
    print(analyzer.export_report(stats))


if __name__ == '__main__':
    demonstrate_traffic_analysis()



def analyze_traffic_patterns(frames: list, timestamps=None) -> dict:
    """
    Simple traffic analysis wrapper for GUI integration.
    
    Args:
        frames: List of frame byte sequences
        timestamps: Optional timestamps for each frame
    
    Returns:
        Dictionary with traffic statistics
    """
    if not frames:
        return {}
    
    # Basic statistics
    frame_sizes = [len(f) for f in frames]
    
    result = {
        'total_frames': len(frames),
        'unique_sizes': len(set(frame_sizes)),
        'min_size': min(frame_sizes),
        'max_size': max(frame_sizes),
        'avg_size': sum(frame_sizes) / len(frame_sizes),
        'total_bytes': sum(frame_sizes)
    }
    
    # Check for burst patterns (simple heuristic)
    if len(frames) > 10:
        size_variance = sum((s - result['avg_size'])**2 for s in frame_sizes) / len(frame_sizes)
        result['burst_detected'] = size_variance > (result['avg_size'] * 0.5)
    else:
        result['burst_detected'] = False
    
    # Check for periodicity (if many frames with same size)
    size_counts = {}
    for s in frame_sizes:
        size_counts[s] = size_counts.get(s, 0) + 1
    
    max_count = max(size_counts.values())
    if max_count > len(frames) * 0.7:  # 70% same size
        result['periodicity'] = 'High (likely periodic transmission)'
    elif max_count > len(frames) * 0.4:
        result['periodicity'] = 'Medium (some periodic patterns)'
    else:
        result['periodicity'] = 'Low (varied frame sizes)'
    
    return result
