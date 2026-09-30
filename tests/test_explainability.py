"""
Tests for Explainability Module
"""

import unittest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.explainability import ExplainabilityEngine, Explanation
from core.sensor_fusion import DamHealthIndex
from core.failure_predictor import FailurePrediction


class TestExplainabilityEngine(unittest.TestCase):
    """Test cases for ExplainabilityEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "dam": {"height_m": 45, "type": "upstream"},
            "safety": {"fs_minimum": 1.5, "fs_warning": 1.3, "fs_critical": 1.1}
        }
        self.engine = ExplainabilityEngine(self.config)

    def test_initialization(self):
        """Test that ExplainabilityEngine initializes correctly."""
        self.assertIsNotNone(self.engine)
        self.assertIn("piezometer", self.engine.feature_descriptions)
        self.assertIn("low", self.engine.risk_descriptions)

    def test_explain_prediction(self):
        """Test prediction explanation generation."""
        prediction = FailurePrediction(
            timestamp=time.time(),
            probability=0.8,
            time_to_failure_hours=24,
            confidence=0.9,
            risk_factors=["Pore pressure spike"],
            recommended_actions=["Evacuate"]
        )

        health = DamHealthIndex(
            timestamp=time.time(),
            overall_score=30.0,
            factor_of_safety=1.2,
            phreatic_ratio=0.8,
            movement_rate=15.0,
            pond_distance_m=20.0,
            risk_level="critical",
            contributing_factors=["High pore pressure"]
        )

        sensor_data = {
            "piezometer": [200.0, 220.0, 240.0],
            "inclinometer": [5.0, 10.0, 15.0],
            "water_level": [40.0, 41.0, 42.0],
            "weather": [100.0, 120.0, 140.0],
            "seismic": [0.0, 0.0, 0.0]
        }

        explanation = self.engine.explain_prediction(prediction, health, sensor_data)
        self.assertIsInstance(explanation, Explanation)
        self.assertGreater(len(explanation.top_factors), 0)
        self.assertIn("CRITICAL", explanation.narrative_summary)

    def test_feature_importance_calculation(self):
        """Test feature importance calculation."""
        health = DamHealthIndex(
            timestamp=time.time(),
            overall_score=50.0,
            factor_of_safety=1.4,
            phreatic_ratio=0.6,
            movement_rate=5.0,
            pond_distance_m=60.0,
            risk_level="medium",
            contributing_factors=[]
        )

        sensor_data = {
            "piezometer": [150.0],
            "inclinometer": [5.0],
            "water_level": [30.0],
            "weather": [50.0],
            "seismic": [0.0]
        }

        importance = self.engine._calculate_feature_importance(sensor_data, health)
        self.assertIn("piezometer", importance)
        self.assertIn("factor_of_safety", importance)
        self.assertGreaterEqual(importance["piezometer"], 0)
        self.assertLessEqual(importance["piezometer"], 1)

    def test_narrative_generation(self):
        """Test narrative summary generation."""
        prediction = FailurePrediction(
            timestamp=time.time(),
            probability=0.85,
            time_to_failure_hours=12,
            confidence=0.9,
            risk_factors=["Test"],
            recommended_actions=["Test action"]
        )

        health = DamHealthIndex(
            timestamp=time.time(),
            overall_score=25.0,
            factor_of_safety=1.1,
            phreatic_ratio=0.85,
            movement_rate=25.0,
            pond_distance_m=10.0,
            risk_level="critical",
            contributing_factors=["Test factor"]
        )

        top_factors = [
            {"description": "Pore water pressure", "status": "critical"},
            {"description": "Slope movement", "status": "warning"}
        ]

        narrative = self.engine._generate_narrative(prediction, health, top_factors)
        self.assertIn("CRITICAL", narrative)
        self.assertIn("85%", narrative)
        self.assertIn("Pore water pressure", narrative)


if __name__ == "__main__":
    unittest.main()
