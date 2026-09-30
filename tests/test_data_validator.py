"""Tests for Data Validation Module"""
import unittest
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.data_validator import DataValidator, ValidationResult

class TestDataValidator(unittest.TestCase):
    def setUp(self):
        self.config = {"sensors": {"piezometers": {"count": 12}}}
        self.validator = DataValidator(self.config)

    def test_initialization(self):
        self.assertIsNotNone(self.validator)

    def test_valid_reading(self):
        result = self.validator.validate_reading("piezo_0", "piezometer", 100.0)
        self.assertTrue(result.is_valid)

    def test_out_of_range(self):
        result = self.validator.validate_reading("piezo_0", "piezometer", 600.0)
        self.assertFalse(result.is_valid)

    def test_validation_stats(self):
        self.validator.validate_reading("piezo_0", "piezometer", 100.0)
        stats = self.validator.get_validation_stats()
        self.assertEqual(stats["total_validations"], 1)

if __name__ == "__main__":
    unittest.main()
