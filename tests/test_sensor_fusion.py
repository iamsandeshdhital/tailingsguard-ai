"""
Tests for Sensor Fusion Module
"""

import unittest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sensor_fusion import SensorFusion, SensorReading, DamHealthIndex


class TestSensorFusion(unittest.TestCase):
    """Test cases for SensorFusion."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "dam": {"height_m": 45, "type": "upstream"},
            "safety": {
                "fs_minimum": 1.5,
                "fs_warning": 1.3,
                "fs_critical": 1.1,
                "phreatic_max_ratio": 0.7,
                "min_beach_width_m": 50
            }
        }
        self.fusion = SensorFusion(self.config)

    def test_initialization(self):
        """Test that SensorFusion initializes correctly."""
        self.assertIsNotNone(self.fusion)
        self.assertEqual(self.fusion.fs_minimum, 1.5)
        self.assertEqual(self.fusion.fs_warning, 1.3)
        self.assertEqual(self.fusion.fs_critical, 1.1)

    def test_add_reading(self):
        """Test adding sensor readings."""
        reading = SensorReading(
            timestamp=time.time(),
            sensor_id="piezo_0",
            sensor_type="piezometer",
            value=100.0,
            unit="kPa",
            location="crest"
        )
        self.fusion.add_reading(reading)
        self.assertIn("piezometer_piezo_0", self.fusion.sensor_data)

    def test_health_index_calculation(self):
        """Test health index calculation."""
        # Add some normal readings
        for i in range(5):
            reading = SensorReading(
                timestamp=time.time(),
                sensor_id=f"piezo_{i}",
                sensor_type="piezometer",
                value=80.0,
                unit="kPa",
                location="crest"
            )
            self.fusion.add_reading(reading)

        health = self.fusion.calculate_health_index()
        self.assertIsInstance(health, DamHealthIndex)
        self.assertGreaterEqual(health.overall_score, 0)
        self.assertLessEqual(health.overall_score, 100)
        self.assertIn(health.risk_level, ["low", "medium", "high", "critical"])

    def test_critical_risk_detection(self):
        """Test that critical conditions are detected."""
        # Add dangerous readings
        for i in range(5):
            reading = SensorReading(
                timestamp=time.time(),
                sensor_id=f"piezo_{i}",
                sensor_type="piezometer",
                value=250.0,  # Very high pressure
                unit="kPa",
                location="crest"
            )
            self.fusion.add_reading(reading)

        health = self.fusion.calculate_health_index()
        self.assertIn(health.risk_level, ["high", "critical"])

    def test_factor_of_safety_calculation(self):
        """Test Factor of Safety calculation."""
        # Normal pressure
        reading = SensorReading(
            timestamp=time.time(),
            sensor_id="piezo_0",
            sensor_type="piezometer",
            value=50.0,
            unit="kPa",
            location="crest"
        )
        self.fusion.add_reading(reading)
        fs = self.fusion._calculate_factor_of_safety()
        self.assertGreater(fs, 1.5)

        # High pressure
        self.fusion.sensor_data.clear()
        reading.value = 250.0
        self.fusion.add_reading(reading)
        fs = self.fusion._calculate_factor_of_safety()
        self.assertLess(fs, 1.5)

    def test_phreatic_ratio_calculation(self):
        """Test phreatic surface ratio calculation."""
        reading = SensorReading(
            timestamp=time.time(),
            sensor_id="piezo_0",
            sensor_type="piezometer",
            value=100.0,
            unit="kPa",
            location="crest"
        )
        self.fusion.add_reading(reading)
        ratio = self.fusion._calculate_phreatic_ratio()
        self.assertGreaterEqual(ratio, 0)
        self.assertLessEqual(ratio, 1.0)

    def test_trend_analysis(self):
        """Test trend analysis."""
        # Add some health history
        for i in range(10):
            health = DamHealthIndex(
                timestamp=time.time() - (10 - i) * 3600,
                overall_score=80 - i * 5,
                factor_of_safety=2.0 - i * 0.1,
                phreatic_ratio=0.3 + i * 0.02,
                movement_rate=1.0 + i * 0.5,
                pond_distance_m=100 - i * 5,
                risk_level="low",
                contributing_factors=[]
            )
            self.fusion.health_history.append(health)

        trend = self.fusion.get_trend(hours=24)
        self.assertIn("trend", trend)
        self.assertIn("change", trend)


if __name__ == "__main__":
    unittest.main()
