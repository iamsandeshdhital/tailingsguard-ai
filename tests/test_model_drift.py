"""Tests for Model Drift Detection Module"""
import unittest
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model_drift import ModelDriftDetector, DriftReport

class TestModelDriftDetector(unittest.TestCase):
    def setUp(self):
        self.config = {"models": {"drift_threshold": 0.3, "min_samples_for_drift": 50}}
        self.detector = ModelDriftDetector(self.config)

    def test_initialization(self):
        self.assertIsNotNone(self.detector)

    def test_record_prediction(self):
        self.detector.record_prediction(0.5, {"piezometer": 100.0})
        self.assertEqual(len(self.detector.predictions), 1)

    def test_check_drift_insufficient_data(self):
        for i in range(10):
            self.detector.record_prediction(0.5, {"piezometer": 100.0})
        report = self.detector.check_drift()
        self.assertIsInstance(report, DriftReport)
        self.assertFalse(report.drift_detected)

if __name__ == "__main__":
    unittest.main()
