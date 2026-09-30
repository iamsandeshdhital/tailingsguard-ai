"""
Tests for Alert System Module
"""

import unittest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.alert_system import AlertSystem, Alert, AlertLevel


class TestAlertSystem(unittest.TestCase):
    """Test cases for AlertSystem."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "alerts": {
                "enabled": True,
                "escalation_minutes": 15,
                "recipients": [
                    {"name": "Test", "email": "test@test.com", "phone": "1234567890"}
                ]
            }
        }
        self.alert_system = AlertSystem(self.config)

    def test_initialization(self):
        """Test that AlertSystem initializes correctly."""
        self.assertIsNotNone(self.alert_system)
        self.assertEqual(self.alert_system.escalation_minutes, 15)

    def test_create_alert(self):
        """Test creating an alert."""
        alert = self.alert_system.create_alert(
            AlertLevel.HIGH,
            "Test Alert",
            "This is a test alert",
            "TestModule"
        )
        self.assertIsInstance(alert, Alert)
        self.assertEqual(alert.level, AlertLevel.HIGH)
        self.assertEqual(alert.title, "Test Alert")
        self.assertFalse(alert.acknowledged)

    def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        alert = self.alert_system.create_alert(
            AlertLevel.WARNING,
            "Test Alert",
            "Test message",
            "TestModule"
        )
        self.alert_system.acknowledge_alert(alert, "TestUser")
        self.assertTrue(alert.acknowledged)
        self.assertEqual(alert.acknowledged_by, "TestUser")

    def test_get_active_alerts(self):
        """Test getting active (unacknowledged) alerts."""
        alert1 = self.alert_system.create_alert(
            AlertLevel.HIGH, "Alert 1", "Message 1", "Test"
        )
        alert2 = self.alert_system.create_alert(
            AlertLevel.WARNING, "Alert 2", "Message 2", "Test"
        )
        self.alert_system.acknowledge_alert(alert1, "User")

        active = self.alert_system.get_active_alerts()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].title, "Alert 2")

    def test_get_critical_alerts(self):
        """Test getting critical alerts."""
        self.alert_system.create_alert(
            AlertLevel.HIGH, "High Alert", "Message", "Test"
        )
        self.alert_system.create_alert(
            AlertLevel.CRITICAL, "Critical Alert", "Message", "Test"
        )
        self.alert_system.create_alert(
            AlertLevel.EMERGENCY, "Emergency Alert", "Message", "Test"
        )

        critical = self.alert_system.get_critical_alerts()
        self.assertEqual(len(critical), 2)

    def test_alert_summary(self):
        """Test alert summary statistics."""
        self.alert_system.create_alert(AlertLevel.HIGH, "Alert 1", "Msg", "Test")
        self.alert_system.create_alert(AlertLevel.WARNING, "Alert 2", "Msg", "Test")
        self.alert_system.create_alert(AlertLevel.CRITICAL, "Alert 3", "Msg", "Test")

        summary = self.alert_system.get_alert_summary()
        self.assertEqual(summary["total_alerts"], 3)
        self.assertEqual(summary["active_alerts"], 3)
        self.assertEqual(summary["critical_alerts"], 1)


if __name__ == "__main__":
    unittest.main()
