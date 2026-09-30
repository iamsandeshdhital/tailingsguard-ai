"""
Tests for Model Drift Detection Module
"""

import unittest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_drift import ModelDriftDetector, DriftReport, PredictionRecord


class TestModelDriftDetector(unittest.TestCase):
    """Test cases for ModelDriftDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "models": {
                "drift_threshold": 0.3,
                "min_samples_for_drift": 50,
                "prediction_window": 1000
            }
        }
        self.detector = ModelDriftDetector(self.config)

    def test_initialization(self):
        """Test that ModelDriftDetector initializes correctly."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.drift_threshold, 0.3)

    def test_record_prediction(self):
        """Test recording predictions."""
        self.detector.record_prediction(0.5, {"piezometer": 100.0})
        self.assertEqual(len(self.detector.predictions), 1)

    def test_check_drift_insufficient_data(self):
        """Test drift check with insufficient data."""
        # Add a few predictions
        for i in range(10):
            self.detector.record_prediction(0.5, {"piezometer": 100.0})

        report = self.detector.check_drift()
        self.assertIsInstance(report, DriftReport)
        self.assertFalse(report.drift_detected)

    def test_check_drift_with_data(self):
        """Test drift check with sufficient data."""
        # Add baseline predictions
        for i in range(100):
            self.detector.record_prediction(0.3, {"piezometer": 100.0})

        # Add drifted predictions
        for i in range(100):
            self.detector.record_prediction(0.8, {"piezometer": 200.0})

        # Force drift check
        self.detector.last_drift_check = 0
        report = self.detector.check_drift()
        self.assertIsInstance(report, DriftReport)

    def test_model_performance(self):
        """Test model performance metrics."""
        # No outcomes yet
        perf = self.detector.get_model_performance()
        self.assertEqual(perf["status"], "insufficient_data")

        # Add some predictions and outcomes
        for i in range(10):
            self.detector.record_prediction(0.7 if i < 5 else 0.3, {"piezometer": 100.0})

        # Record outcomes
        for i in range(5):
            self.detector.record_outcome(i, True)  # True positives
        for i in range(5, 10):
            self.detector.record_outcome(i, False)  # True negatives

        perf = self.detector.get_model_performance()
        self.assertIn("precision", perf)
        self.assertIn("recall", perf)
        self.assertIn("f1_score", perf)


if __name__ == "__main__":
    unittest.main()
