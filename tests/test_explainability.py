"""Tests for Explainability Module"""
import unittest
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.explainability import ExplainabilityEngine, Explanation
from core.sensor_fusion import DamHealthIndex
from core.failure_predictor import FailurePrediction

class TestExplainabilityEngine(unittest.TestCase):
    def setUp(self):
        self.config = {"dam": {"height_m": 45}, "safety": {"fs_minimum": 1.5}}
        self.engine = ExplainabilityEngine(self.config)

    def test_initialization(self):
        self.assertIsNotNone(self.engine)

    def test_explain_prediction(self):
        prediction = FailurePrediction(timestamp=time.time(), probability=0.8, time_to_failure_hours=24, confidence=0.9, risk_factors=["Test"], recommended_actions=["Test"])
        health = DamHealthIndex(timestamp=time.time(), overall_score=30.0, factor_of_safety=1.2, phreatic_ratio=0.8, movement_rate=15.0, pond_distance_m=20.0, risk_level="critical", contributing_factors=["Test"])
        sensor_data = {"piezometer": [200.0], "inclinometer": [5.0], "water_level": [40.0], "weather": [100.0], "seismic": [0.0]}
        explanation = self.engine.explain_prediction(prediction, health, sensor_data)
        self.assertIsInstance(explanation, Explanation)
        self.assertIn("CRITICAL", explanation.narrative_summary)

if __name__ == "__main__":
    unittest.main()
