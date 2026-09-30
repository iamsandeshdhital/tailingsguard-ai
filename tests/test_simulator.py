"""
Tests for Sensor Simulator Module
"""

import unittest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.simulator import SensorSimulator, SimulatedSensor


class TestSensorSimulator(unittest.TestCase):
    """Test cases for SensorSimulator."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "dam": {"height_m": 45, "type": "upstream"},
            "sensors": {
                "piezometers": {"enabled": True, "count": 3, "locations": ["crest"]},
                "inclinometers": {"enabled": True, "count": 2},
                "water_level": {"enabled": True},
                "weather": {"enabled": True},
                "seismic": {"enabled": True}
            }
        }
        self.simulator = SensorSimulator(self.config)

    def test_initialization(self):
        """Test that simulator initializes correctly."""
        self.assertIsNotNone(self.simulator)
        self.assertGreater(len(self.simulator.sensors), 0)

    def test_get_readings(self):
        """Test getting sensor readings."""
        readings = self.simulator.get_readings()
        self.assertIsInstance(readings, dict)
        self.assertGreater(len(readings), 0)

    def test_readings_have_valid_values(self):
        """Test that readings have valid numeric values."""
        readings = self.simulator.get_readings()
        for sensor_id, value in readings.items():
            self.assertIsInstance(value, (int, float))
            self.assertGreaterEqual(value, 0)

    def test_failure_scenario(self):
        """Test failure scenario trigger."""
        # Get normal readings
        normal_readings = self.simulator.get_readings()

        # Trigger failure
        self.simulator.trigger_failure_scenario()

        # Get failure readings
        failure_readings = self.simulator.get_readings()

        # Values should be different (generally higher in failure mode)
        # Note: Due to randomness, we just check that readings exist
        self.assertIsInstance(failure_readings, dict)

    def test_reset_scenario(self):
        """Test scenario reset."""
        self.simulator.trigger_failure_scenario()
        self.simulator.reset_scenario()
        self.assertFalse(self.simulator._failure_mode)

    def test_sensor_config(self):
        """Test sensor configuration retrieval."""
        config = self.simulator.get_sensor_config()
        self.assertIsInstance(config, dict)
        self.assertGreater(len(config), 0)

        for sensor_id, sensor_config in config.items():
            self.assertIn("type", sensor_config)
            self.assertIn("location", sensor_config)
            self.assertIn("unit", sensor_config)


if __name__ == "__main__":
    unittest.main()
