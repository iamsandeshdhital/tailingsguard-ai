"""
Tests for Failure Prediction Module
"""

import unittest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.failure_predictor import FailurePredictor, FailurePrediction


class TestFailurePredictor(unittest.TestCase):
    """Test cases for FailurePredictor."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "dam": {"height_m": 45, "type": "upstream"},
            "models": {
                "failure_predictor": {
                    "prediction_horizon_hours": 72
                }
            }
        }
        self.predictor = FailurePredictor(self.config)

    def test_initialization(self):
        """Test that FailurePredictor initializes correctly."""
        self.assertIsNotNone(self.predictor)
        self.assertEqual(self.predictor.prediction_horizon, 72)

    def test_normal_conditions(self):
        """Test prediction under normal conditions."""
        sensor_data = {
            "piezometer": [80.0, 82.0, 79.0, 81.0, 80.0],
            "inclinometer": [0.1, 0.2, 0.1, 0.2, 0.1],
            "water_level": [20.0, 20.1, 20.0, 20.1, 20.0],
            "weather": [5.0, 3.0, 4.0, 5.0, 6.0],
            "seismic": [0.0, 0.0, 0.0, 0.0, 0.0]
        }

        prediction = self.predictor.predict(sensor_data)
        self.assertIsInstance(prediction, FailurePrediction)
        self.assertLess(prediction.probability, 0.5)
        self.assertIsNone(prediction.time_to_failure_hours)

    def test_critical_conditions(self):
        """Test prediction under critical conditions."""
        sensor_data = {
            "piezometer": [200.0, 220.0, 240.0, 260.0, 280.0],  # Rapidly increasing
            "inclinometer": [5.0, 10.0, 15.0, 20.0, 25.0],  # Accelerating
            "water_level": [40.0, 41.0, 42.0, 43.0, 44.0],  # Near crest
            "weather": [120.0, 130.0, 140.0, 150.0, 160.0],  # Heavy rain
            "seismic": [0.0, 0.0, 0.0, 0.0, 0.0]
        }

        prediction = self.predictor.predict(sensor_data)
        self.assertGreater(prediction.probability, 0.5)
        self.assertIsNotNone(prediction.time_to_failure_hours)

    def test_pore_pressure_spike_detection(self):
        """Test detection of pore pressure spikes."""
        sensor_data = {
            "piezometer": [50.0, 60.0, 80.0, 120.0, 180.0],  # 3.6x increase
        }

        score = self.predictor._check_pore_pressure(sensor_data, 1.5)
        self.assertGreater(score, 0)

    def test_acceleration_detection(self):
        """Test detection of accelerating deformation."""
        sensor_data = {
            "inclinometer": [0.1, 0.2, 0.4, 0.8, 1.6],  # Doubling each time
        }

        score = self.predictor._check_acceleration(sensor_data, 2.0)
        self.assertGreater(score, 0)

    def test_pond_encroachment_detection(self):
        """Test detection of pond encroachment."""
        sensor_data = {
            "water_level": [38.0],  # 38/45 = 84% of dam height
        }

        score = self.predictor._check_pond_encroachment(sensor_data, 0.8)
        self.assertGreater(score, 0)

    def test_recommendations_generation(self):
        """Test that recommendations are generated."""
        sensor_data = {
            "piezometer": [250.0, 260.0, 270.0, 280.0, 290.0],
            "inclinometer": [10.0, 15.0, 20.0, 25.0, 30.0],
            "water_level": [42.0, 43.0, 44.0, 44.5, 45.0],
            "weather": [150.0, 160.0, 170.0, 180.0, 190.0],
        }

        prediction = self.predictor.predict(sensor_data)
        self.assertGreater(len(prediction.recommended_actions), 0)

    def test_confidence_calculation(self):
        """Test confidence calculation."""
        # Full sensor coverage
        sensor_data = {
            "piezometer": [80.0] * 10,
            "inclinometer": [0.1] * 10,
            "water_level": [20.0] * 10,
        }
        confidence = self.predictor._calculate_confidence(sensor_data)
        self.assertGreater(confidence, 0.5)

        # No sensor data
        confidence = self.predictor._calculate_confidence({})
        self.assertEqual(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
