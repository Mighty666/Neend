from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SeverityLevel(str, Enum):
    NORMAL = "Normal"
    MILD = "Mild OSA"
    MODERATE = "Moderate OSA"
    SEVERE = "Severe OSA"


@dataclass
class AHIResult:
    ahi: float
    severity: SeverityLevel
    apnea_events: int
    hypopnea_events: int
    snoring_events: int
    total_events: int
    recording_hours: float


class AHICalculator:
    def __init__(self, segment_duration: float = 30.0):
        self.segment_duration = segment_duration

    def calculate_ahi(
        self,
        events_timeline: List[int],
        recording_hours: float = None
    ) -> AHIResult:
        if recording_hours is None:
            total_seconds = len(events_timeline) * self.segment_duration
            recording_hours = total_seconds / 3600
        if recording_hours <= 0:
            recording_hours = 1
        apnea_count = events_timeline.count(3)
        hypopnea_count = events_timeline.count(2)
        snoring_count = events_timeline.count(1)
        total_events = apnea_count + hypopnea_count
        ahi_score = total_events / recording_hours
        severity = self._classify_severity(ahi_score)
        return AHIResult(
            ahi=round(ahi_score, 2),
            severity=severity,
            apnea_events=apnea_count,
            hypopnea_events=hypopnea_count,
            snoring_events=snoring_count,
            total_events=total_events,
            recording_hours=round(recording_hours, 2)
        )

    def _classify_severity(self, ahi: float) -> SeverityLevel:
        if ahi < 5:
            return SeverityLevel.NORMAL
        elif ahi < 15:
            return SeverityLevel.MILD
        elif ahi < 30:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.SEVERE

    def calculate_running_ahi(
        self,
        events_timeline: List[int]
    ) -> Dict:
        result = self.calculate_ahi(events_timeline)
        total_segments = len(events_timeline)
        if total_segments > 0:
            apnea_rate = result.apnea_events / total_segments
            hypopnea_rate = result.hypopnea_events / total_segments
        else:
            apnea_rate = 0
            hypopnea_rate = 0
        return {
            'ahi': result.ahi,
            'severity': result.severity.value,
            'apnea_events': result.apnea_events,
            'hypopnea_events': result.hypopnea_events,
            'snoring_events': result.snoring_events,
            'total_events': result.total_events,
            'recording_hours': result.recording_hours,
            'apnea_rate': round(apnea_rate, 3),
            'hypopnea_rate': round(hypopnea_rate, 3),
            'segments_processed': total_segments
        }

    def get_hourly_breakdown(
        self,
        events_timeline: List[int]
    ) -> List[Dict]:
        segments_per_hour = int(3600 / self.segment_duration)
        hourly_results = []
        for i in range(0, len(events_timeline), segments_per_hour):
            hour_events = events_timeline[i:i + segments_per_hour]
            if hour_events:
                result = self.calculate_ahi(hour_events, recording_hours=1.0)
                hourly_results.append({
                    'hour': i // segments_per_hour + 1,
                    'ahi': result.ahi,
                    'severity': result.severity.value,
                    'apnea_events': result.apnea_events,
                    'hypopnea_events': result.hypopnea_events
                })
        return hourly_results

    def generate_report(
        self,
        events_timeline: List[int],
        user_name: str = "Patient"
    ) -> str:
        result = self.calculate_ahi(events_timeline)
        hourly = self.get_hourly_breakdown(events_timeline)
        report = []
        report.append("=" * 50)
        report.append("SLEEP APNEA ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"\nPatient: {user_name}")
        report.append(f"Recording Duration: {result.recording_hours} hours")
        report.append("")
        report.append("OVERALL RESULTS")
        report.append("-" * 30)
        report.append(f"AHI Score: {result.ahi}")
        report.append(f"Severity: {result.severity.value}")
        report.append("")
        report.append("EVENT SUMMARY")
        report.append("-" * 30)
        report.append(f"Apnea Events: {result.apnea_events}")
        report.append(f"Hypopnea Events: {result.hypopnea_events}")
        report.append(f"Snoring Events: {result.snoring_events}")
        report.append(f"Total Apnea+Hypopnea: {result.total_events}")
        report.append("")
        report.append("HOURLY BREAKDOWN")
        report.append("-" * 30)
        for hour_data in hourly:
            report.append(
                f"Hour {hour_data['hour']}: AHI={hour_data['ahi']:.1f} "
                f"({hour_data['severity']})"
            )
        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        if result.severity == SeverityLevel.SEVERE:
            report.append("URGENT: Severe OSA detected.")
            report.append("   Immediate consultation with sleep specialist recommended.")
            report.append("   Consider CPAP therapy evaluation.")
        elif result.severity == SeverityLevel.MODERATE:
            report.append("Moderate OSA detected.")
            report.append("   Schedule appointment with sleep specialist.")
            report.append("   Lifestyle modifications may help.")
        elif result.severity == SeverityLevel.MILD:
            report.append("Mild OSA detected.")
            report.append("   Consider sleep position changes.")
            report.append("   Weight management if applicable.")
        else:
            report.append("No significant sleep apnea detected.")
            report.append("   Continue monitoring if symptoms persist.")
        report.append("")
        report.append("=" * 50)
        report.append("Note: This is a screening tool. Please consult")
        report.append("a healthcare provider for official diagnosis.")
        report.append("=" * 50)
        return "\n".join(report)
