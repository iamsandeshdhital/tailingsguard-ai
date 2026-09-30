"""
Tests for Data Validation Module
"""

import unittest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_validator import DataValidator, ValidationResult, ValidationReport


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "sensors": {
                "piezometers": {"count": 12},
                "inclinometers": {"count": 6}
            }
        }
        self.validator = DataValidator(self.config)

    def test_initialization(self):
        """Test that DataValidator initializes correctly."""
        self.assertIsNotNone(self.validator)
        self.assertIn("piezometer", self.validator.rules)
        self.assertIn("inclinometer", self.validator.rules)

    def test_valid_reading(self):
        """Test validation of a valid reading."""
        result = self.validator.validate_reading("piezo_0", "piezometer", 100.0)
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.issues), 0)

    def test_out_of_range_low(self):
        """Test validation of below-range reading."""
        result = self.validator.validate_reading("piezo_0", "piezometer", -10.0)
        self.assertFalse(result.is_valid)
        self.assertIsNotNone(result.corrected_value)
        self.assertIn("below minimum", result.issues[0])

    def test_out_of_range_high(self):
        """Test validation of above-range reading."""
        result = self.validator.validate_reading("piezo_0", "piezometer", 600.0)
        self.assertFalse(result.is_valid)
        self.assertIsNotNone(result.corrected_value)
        self.assertIn("above maximum", result.issues[0])

    def test_rate_of_change_validation(self):
        """Test rate of change validation."""
        # First reading
        self.validator.validate_reading("piezo_0", "piezometer", 100.0, time.time())

        # Rapid change
        result = self.validator.validate_reading("piezo_0", "piezometer", 200.0, time.time() + 1)
        # Should have rate of change issue
        self.assertTrue(any("Rate of change" in issue for issue in result.issues))

    def test_outlier_detection(self):
        """Test statistical outlier detection."""
        # Add baseline readings
        for i in range(20):
            self.validator.validate_reading("piezo_0", "piezometer", 100.0 + i * 0.1)

        # Add outlier
        result = self.validator.validate_reading("piezo_0", "piezometer", 500.0)
        self.assertTrue(any("outlier" in issue.lower() for issue in result.issues))

    def test_stuck_sensor_detection(self):
        """Test stuck sensor detection."""
        # Add same value multiple times
        for i in range(10):
            self.validator.validate_reading("piezo_0", "piezometer", 100.0)

        result = self.validator.validate_reading("piezo_0", "piezometer", 100.0)
        self.assertTrue(any("stuck" in issue.lower() for issue in result.issues))

    def test_batch_validation(self):
        """Test batch validation."""
        readings = [
            ("piezo_0", "piezometer", 100.0, time.time()),
            ("piezo_1", "piezometer", 150.0, time.time()),
            ("inclin_0", "inclinometer", 5.0, time.time()),
        ]

        report = self.validator.validate_batch(readings)
        self.assertIsInstance(report, ValidationReport)
        self.assertEqual(report.total_readings, 3)
        self.assertGreaterEqual(report.overall_quality, 0)
        self.assertLessEqual(report.overall_quality, 1)

    def test_validation_stats(self):
        """Test validation statistics."""
        self.validator.validate_reading("piezo_0", "piezometer", 100.0)
        self.validator.validate_reading("piezo_1", "piezometer", -10.0)

        stats = self.validator.get_validation_stats()
        self.assertEqual(stats["total_validations"], 2)
        self.assertEqual(stats["valid"], 1)
        self.assertEqual(stats["corrected"], 1)


if __name__ == "__main__":
    unittest.main()
