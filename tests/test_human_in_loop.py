"""Tests for Human-in-the-Loop Module"""
import unittest
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.human_in_loop import HumanInLoop, DecisionStatus

class TestHumanInLoop(unittest.TestCase):
    def setUp(self):
        self.config = {"human_in_loop": {"require_approval_for": ["evacuate"]}, "alerts": {"recipients": []}}
        self.system = HumanInLoop(self.config)

    def test_initialization(self):
        self.assertIsNotNone(self.system)

    def test_request_decision(self):
        decision_id = self.system.request_decision("evacuate", "Test", "critical")
        self.assertIsNotNone(decision_id)

    def test_approve_decision(self):
        decision_id = self.system.request_decision("evacuate", "Test", "critical")
        result = self.system.approve_decision(decision_id, "Manager")
        self.assertTrue(result)

    def test_reject_decision(self):
        decision_id = self.system.request_decision("evacuate", "Test", "critical")
        result = self.system.reject_decision(decision_id, "Manager")
        self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()
