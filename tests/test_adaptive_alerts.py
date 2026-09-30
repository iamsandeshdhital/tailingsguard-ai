"""Tests for Adaptive Alert System Module"""
import unittest
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.adaptive_alerts import AdaptiveAlertSystem

class TestAdaptiveAlertSystem(unittest.TestCase):
    def setUp(self):
        self.config = {"safety": {"fs_warning": 1.3, "phreatic_max_ratio": 0.7}, "alerts": {"max_alerts_per_hour": 5, "cooldown_minutes": 30}}
        self.system = AdaptiveAlertSystem(self.config)

    def test_initialization(self):
        self.assertIsNotNone(self.system)

    def test_trigger_alert(self):
        alert_id = self.system.trigger_alert("factor_of_safety", 1.0, "critical", "Test")
        self.assertIsNotNone(alert_id)

    def test_provide_feedback(self):
        alert_id = self.system.trigger_alert("factor_of_safety", 1.0, "critical", "Test")
        self.system.provide_feedback(alert_id, True, "Real")
        self.assertEqual(self.system.true_positives, 1)

    def test_alert_accuracy(self):
        self.assertEqual(self.system.get_alert_accuracy(), 1.0)
        alert_id = self.system.trigger_alert("factor_of_safety", 1.0, "critical", "Test")
        self.system.provide_feedback(alert_id, True, "Real")
        self.assertEqual(self.system.get_alert_accuracy(), 1.0)
        alert_id = self.system.trigger_alert("phreatic_ratio", 0.6, "warning", "Test")
        self.system.provide_feedback(alert_id, False, "False alarm")
        self.assertAlmostEqual(self.system.get_alert_accuracy(), 0.5, places=1)

if __name__ == "__main__":
    unittest.main()
