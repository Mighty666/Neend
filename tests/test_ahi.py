"""
Tests for AHI calculator
"""

import pytest
from src.ml import AHICalculator
from src.ml.ahi_calculator import SeverityLevel


class TestAHICalculator:
    """Tests for AHICalculator class"""

    def setup_method(self):
        self.calculator = AHICalculator(segment_duration=30.0)

    def test_calculate_ahi_normal(self):
        """Test AHI calculation for normal breathing"""
        # 8 hours of normal breathing (no events)
        events = [0] * 960  # 960 segments of 30s = 8 hours
        result = self.calculator.calculate_ahi(events)

        assert result.ahi == 0
        assert result.severity == SeverityLevel.NORMAL
        assert result.apnea_events == 0
        assert result.hypopnea_events == 0

    def test_calculate_ahi_mild(self):
        """Test AHI calculation for mild OSA"""
        # 8 hours with 40-80 events (AHI 5-10)
        events = [0] * 900 + [3] * 40 + [2] * 20  # 60 events in 8 hours = AHI 7.5
        result = self.calculator.calculate_ahi(events)

        assert 5 <= result.ahi < 15
        assert result.severity == SeverityLevel.MILD
        assert result.apnea_events == 40
        assert result.hypopnea_events == 20

    def test_calculate_ahi_severe(self):
        """Test AHI calculation for severe OSA"""
        # 8 hours with 240+ events (AHI >= 30)
        events = [0] * 700 + [3] * 150 + [2] * 110  # 260 events in 8 hours = AHI 32.5
        result = self.calculator.calculate_ahi(events)

        assert result.ahi >= 30
        assert result.severity == SeverityLevel.SEVERE

    def test_calculate_running_ahi(self):
        """Test running AHI calculation"""
        events = [0, 0, 1, 3, 0, 2, 0, 0, 3, 1]
        result = self.calculator.calculate_running_ahi(events)

        assert 'ahi' in result
        assert 'severity' in result
        assert 'segments_processed' in result
        assert result['segments_processed'] == 10

    def test_hourly_breakdown(self):
        """Test hourly AHI breakdown"""
        # Create events for 3 hours
        events = [0] * 120 + [3] * 30 + [0] * 90 + [2] * 30 + [0] * 90
        hourly = self.calculator.get_hourly_breakdown(events)

        assert len(hourly) >= 3
        for hour_data in hourly:
            assert 'hour' in hour_data
            assert 'ahi' in hour_data
            assert 'severity' in hour_data

    def test_generate_report(self):
        """Test report generation"""
        events = [0] * 100 + [3] * 10 + [2] * 10
        report = self.calculator.generate_report(events, "Test Patient")

        assert "Test Patient" in report
        assert "AHI Score" in report
        assert "Severity" in report
        assert "RECOMMENDATIONS" in report

    def test_snoring_not_counted_in_ahi(self):
        """Test that snoring events are not counted in AHI"""
        # Only snoring events
        events = [1] * 100
        result = self.calculator.calculate_ahi(events)

        assert result.ahi == 0  # Snoring doesn't count
        assert result.snoring_events == 100
        assert result.total_events == 0  # Only apnea + hypopnea


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
