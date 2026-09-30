"""Tests for Sensor Health Monitor Module"""
import unittest
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.sensor_health import SensorHealthMonitor, DataQualityReport

class TestSensorHealthMonitor(unittest.TestCase):
    def setUp(self):
        self.config = {"sensors": {"piezometers": {"count": 12}, "inclinometers": {"count": 6}}}
        self.monitor = SensorHealthMonitor(self.config)

    def test_initialization(self):
        self.assertIsNotNone(self.monitor)

    def test_register_sensor(self):
        self.monitor.register_sensor("piezo_0", "piezometer")
        self.assertIn("piezo_0", self.monitor.sensor_health)

    def test_update_reading(self):
        self.monitor.update_reading("piezo_0", 100.0)
        self.assertEqual(self.monitor.sensor_health["piezo_0"].last_reading, 100.0)

    def test_health_report(self):
        for i in range(5):
            self.monitor.update_reading(f"piezo_{i}", 100.0)
        report = self.monitor.get_health_report()
        self.assertIsInstance(report, DataQualityReport)
        self.assertGreater(report.total_sensors, 0)

if __name__ == "__main__":
    unittest.main()
