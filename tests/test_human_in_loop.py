"""
Tests for Human-in-the-Loop Module
"""

import unittest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.human_in_loop import HumanInLoop, HumanDecision, DecisionStatus


class TestHumanInLoop(unittest.TestCase):
    """Test cases for HumanInLoop."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "human_in_loop": {
                "require_approval_for": ["evacuate", "lower_pond"],
                "default_timeout_seconds": 300
            },
            "alerts": {
                "recipients": [
                    {"name": "Manager", "email": "manager@mine.com"}
                ]
            }
        }
        self.system = HumanInLoop(self.config)

    def test_initialization(self):
        """Test that HumanInLoop initializes correctly."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.default_timeout, 300)

    def test_request_decision(self):
        """Test requesting a decision."""
        decision_id = self.system.request_decision(
            "evacuate",
            "Failure probability 85%",
            "critical"
        )
        self.assertIsNotNone(decision_id)
        self.assertIn(decision_id, self.system.pending_decisions)
        self.assertEqual(self.system.pending_decisions[decision_id].status, DecisionStatus.PENDING)

    def test_approve_decision(self):
        """Test approving a decision."""
        decision_id = self.system.request_decision("evacuate", "Test reason", "critical")
        result = self.system.approve_decision(decision_id, "Manager", "Approved")
        self.assertTrue(result)
        self.assertNotIn(decision_id, self.system.pending_decisions)
        self.assertEqual(self.system.approved_decisions, 1)

    def test_reject_decision(self):
        """Test rejecting a decision."""
        decision_id = self.system.request_decision("evacuate", "Test reason", "critical")
        result = self.system.reject_decision(decision_id, "Manager", "Rejected")
        self.assertTrue(result)
        self.assertNotIn(decision_id, self.system.pending_decisions)
        self.assertEqual(self.system.rejected_decisions, 1)

    def test_check_timeouts(self):
        """Test timeout checking."""
        # Add a decision with very short timeout
        decision_id = self.system.request_decision("evacuate", "Test", "critical")
        self.system.pending_decisions[decision_id].timeout_seconds = 0.1

        # Wait for timeout
        time.sleep(0.2)
        self.system.check_timeouts()

        self.assertNotIn(decision_id, self.system.pending_decisions)
        self.assertEqual(self.system.timed_out_decisions, 1)

    def test_get_pending_decisions(self):
        """Test getting pending decisions."""
        self.system.request_decision("evacuate", "Test 1", "critical")
        self.system.request_decision("lower_pond", "Test 2", "high")

        pending = self.system.get_pending_decisions()
        self.assertEqual(len(pending), 2)

    def test_get_statistics(self):
        """Test decision statistics."""
        self.system.request_decision("evacuate", "Test", "critical")
        self.system.request_decision("lower_pond", "Test", "high")

        stats = self.system.get_statistics()
        self.assertEqual(stats["total_decisions"], 2)
        self.assertEqual(stats["pending"], 2)


if __name__ == "__main__":
    unittest.main()
