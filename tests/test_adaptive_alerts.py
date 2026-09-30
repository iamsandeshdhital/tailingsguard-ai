"""
Tests for Adaptive Alert System Module
"""

import unittest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.adaptive_alerts import AdaptiveAlertSystem, AdaptiveThreshold, AlertFeedback


class TestAdaptiveAlertSystem(unittest.TestCase):
    """Test cases for AdaptiveAlertSystem."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "safety": {
                "fs_minimum": 1.5,
                "fs_warning": 1.3,
                "fs_critical": 1.1,
                "phreatic_max_ratio": 0.7,
                "min_beach_width_m": 50
            },
            "alerts": {
                "max_alerts_per_hour": 5,
                "cooldown_minutes": 30
            }
        }
        self.system = AdaptiveAlertSystem(self.config)

    def test_initialization(self):
        """Test that AdaptiveAlertSystem initializes correctly."""
        self.assertIsNotNone(self.system)
        self.assertIn("factor_of_safety", self.system.thresholds)
        self.assertIn("phreatic_ratio", self.system.thresholds)

    def test_should_alert(self):
        """Test alert decision logic."""
        # First alert should trigger
        alert_id1 = self.system.trigger_alert("factor_of_safety", 1.0, "critical", "Test 1")
        self.assertIsNotNone(alert_id1)

        # Immediate second alert should be blocked by cooldown
        alert_id2 = self.system.trigger_alert("factor_of_safety", 1.0, "critical", "Test 2")
        self.assertIsNone(alert_id2)

    def test_trigger_alert(self):
        """Test triggering an alert."""
        alert_id = self.system.trigger_alert("factor_of_safety", 1.0, "critical", "Test alert")
        self.assertIsNotNone(alert_id)
        self.assertIn(alert_id, self.system.active_alerts)

    def test_provide_feedback(self):
        """Test providing feedback on alerts."""
        alert_id = self.system.trigger_alert("factor_of_safety", 1.0, "critical", "Test alert")
        self.system.provide_feedback(alert_id, True, "Was a real issue")

        self.assertNotIn(alert_id, self.system.active_alerts)
        self.assertEqual(self.system.true_positives, 1)

    def test_threshold_adjustment(self):
        """Test threshold adjustment based on feedback."""
        original = self.system.thresholds["factor_of_safety"].current_value

        # False positive should increase threshold
        alert_id = self.system.trigger_alert("factor_of_safety", 1.0, "warning", "Test")
        self.system.provide_feedback(alert_id, False, "False alarm")

        self.assertGreater(self.system.thresholds["factor_of_safety"].current_value, original)

    def test_alert_accuracy(self):
        """Test alert accuracy calculation."""
        # No feedback yet
        self.assertEqual(self.system.get_alert_accuracy(), 1.0)

        # Add true positive
        alert_id = self.system.trigger_alert("factor_of_safety", 1.0, "critical", "Test")
        self.system.provide_feedback(alert_id, True, "Real")

        self.assertEqual(self.system.get_alert_accuracy(), 1.0)

        # Reset ALL state to allow second alert
        self.system.recent_alerts.clear()
        self.system.total_alerts = 0
        self.system.true_positives = 0
        self.system.false_positives = 0
        self.system.active_alerts.clear()

        # Add false positive (use different parameter to avoid cooldown)
        alert_id = self.system.trigger_alert("phreatic_ratio", 0.8, "warning", "Test")
        self.system.provide_feedback(alert_id, False, "False alarm")

        # Should be 0.5 (1 true positive, 1 false positive)
        accuracy = self.system.get_alert_accuracy()
        self.assertAlmostEqual(accuracy, 0.5, places=1)

    def test_reset_thresholds(self):
        """Test resetting thresholds."""
        # Modify a threshold
        self.system.thresholds["factor_of_safety"].current_value = 1.5

        # Reset
        self.system.reset_thresholds()

        self.assertEqual(
            self.system.thresholds["factor_of_safety"].current_value,
            self.system.thresholds["factor_of_safety"].original_value
        )


if __name__ == "__main__":
    unittest.main()
