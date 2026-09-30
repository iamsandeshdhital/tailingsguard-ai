"""
Tests for Sensor Health Monitor Module
"""

import unittest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sensor_health import SensorHealthMonitor, SensorHealth, DataQualityReport


class TestSensorHealthMonitor(unittest.TestCase):
    """Test cases for SensorHealthMonitor."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "dam": {"height_m": 45},
            "sensors": {
                "piezometers": {"count": 12},
                "inclinometers": {"count": 6},
                "max_silence_seconds": 300,
                "drift_threshold": 0.3,
                "min_readings_per_hour": 10
            }
        }
        self.monitor = SensorHealthMonitor(self.config)

    def test_initialization(self):
        """Test that SensorHealthMonitor initializes correctly."""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(self.monitor.max_silence_seconds, 300)

    def test_register_sensor(self):
        """Test sensor registration."""
        self.monitor.register_sensor("piezo_0", "piezometer")
        self.assertIn("piezo_0", self.monitor.sensor_health)
        self.assertEqual(self.monitor.sensor_health["piezo_0"].status, "unknown")

    def test_update_reading(self):
        """Test updating sensor readings."""
        self.monitor.update_reading("piezo_0", 100.0)
        self.assertIn("piezo_0", self.monitor.sensor_health)
        self.assertEqual(self.monitor.sensor_health["piezo_0"].last_reading, 100.0)
        self.assertEqual(self.monitor.sensor_health["piezo_0"].reading_count, 1)

    def test_sensor_status_healthy(self):
        """Test healthy sensor status."""
        # Add multiple readings
        for i in range(20):
            self.monitor.update_reading("piezo_0", 100.0 + i * 0.1)

        status = self.monitor.sensor_health["piezo_0"].status
        self.assertIn(status, ["healthy", "degraded"])

    def test_sensor_status_failed(self):
        """Test failed sensor status (no data)."""
        self.monitor.register_sensor("piezo_0", "piezometer")
        # Don't add any readings - sensor should be failed
        status = self.monitor._determine_status("piezo_0")
        self.assertEqual(status, "failed")

    def test_health_report(self):
        """Test health report generation."""
        # Add some sensors
        for i in range(5):
            self.monitor.update_reading(f"piezo_{i}", 100.0)

        report = self.monitor.get_health_report()
        self.assertIsInstance(report, DataQualityReport)
        self.assertGreater(report.total_sensors, 0)
        self.assertGreaterEqual(report.overall_quality, 0)
        self.assertLessEqual(report.overall_quality, 1)

    def test_drift_detection(self):
        """Test drift detection."""
        # Add baseline readings (minimum 100 for baseline calculation)
        for i in range(100):
            self.monitor.update_reading("piezo_0", 100.0)

        # Add drifted readings
        for i in range(20):
            self.monitor.update_reading("piezo_0", 200.0)

        # Force status update
        status = self.monitor._determine_status("piezo_0")
        self.monitor.sensor_health["piezo_0"].status = status

        drift = self.monitor.sensor_health["piezo_0"].drift_score
        self.assertGreater(drift, 0)


if __name__ == "__main__":
    unittest.main()
