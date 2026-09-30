"""
Tests for Data Logger Module
"""

import unittest
import time
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_logger import DataLogger
from core.sensor_fusion import DamHealthIndex
from core.failure_predictor import FailurePrediction
from core.alert_system import Alert, AlertLevel


class TestDataLogger(unittest.TestCase):
    """Test cases for DataLogger."""

    def setUp(self):
        """Set up test fixtures with temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.config = {
            "database": {
                "path": self.db_path,
                "retention_days": 30
            }
        }
        self.logger = DataLogger(self.config)

    def tearDown(self):
        """Clean up temporary database."""
        try:
            os.remove(self.db_path)
            os.rmdir(self.temp_dir)
        except OSError:
            pass

    def test_database_initialization(self):
        """Test that database is created with correct tables."""
        self.assertTrue(os.path.exists(self.db_path))

        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            self.assertIn("sensor_readings", tables)
            self.assertIn("health_index", tables)
            self.assertIn("predictions", tables)
            self.assertIn("alerts", tables)

    def test_log_sensor_reading(self):
        """Test logging sensor readings."""
        self.logger.log_sensor_reading("piezo_0", "piezometer", 100.0, "kPa", "crest")

        readings = self.logger.get_recent_readings("piezometer", hours=1)
        self.assertEqual(len(readings), 1)
        self.assertEqual(readings[0]["sensor_id"], "piezo_0")
        self.assertEqual(readings[0]["value"], 100.0)

    def test_log_health_index(self):
        """Test logging health index."""
        health = DamHealthIndex(
            timestamp=time.time(),
            overall_score=75.0,
            factor_of_safety=1.8,
            phreatic_ratio=0.4,
            movement_rate=2.0,
            pond_distance_m=80.0,
            risk_level="medium",
            contributing_factors=["Test factor"]
        )
        self.logger.log_health_index(health)

        history = self.logger.get_health_history(hours=1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["overall_score"], 75.0)

    def test_log_prediction(self):
        """Test logging predictions."""
        prediction = FailurePrediction(
            timestamp=time.time(),
            probability=0.3,
            time_to_failure_hours=None,
            confidence=0.8,
            risk_factors=["Test"],
            recommended_actions=["Action 1"]
        )
        self.logger.log_prediction(prediction)

        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions")
            rows = cursor.fetchall()
            self.assertEqual(len(rows), 1)

    def test_log_alert(self):
        """Test logging alerts."""
        alert = Alert(
            timestamp=time.time(),
            level=AlertLevel.HIGH,
            title="Test Alert",
            message="Test message",
            source="Test"
        )
        self.logger.log_alert(alert)

        alerts = self.logger.get_alerts(hours=1)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["title"], "Test Alert")

    def test_cleanup_old_data(self):
        """Test cleanup of old data."""
        # This test just verifies the method runs without error
        self.logger.cleanup_old_data()

    def test_export_compliance_report(self):
        """Test compliance report export."""
        # Add some data
        self.logger.log_sensor_reading("piezo_0", "piezometer", 100.0)
        self.logger.log_health_index(DamHealthIndex(
            timestamp=time.time(),
            overall_score=80.0,
            factor_of_safety=2.0,
            phreatic_ratio=0.3,
            movement_rate=1.0,
            pond_distance_m=100.0,
            risk_level="low",
            contributing_factors=[]
        ))

        report_path = self.logger.export_compliance_report(days=1)
        self.assertTrue(os.path.exists(report_path))

        import json
        with open(report_path, 'r') as f:
            report = json.load(f)
        self.assertIn("total_health_checks", report)
        self.assertIn("health_summary", report)

        # Clean up
        os.remove(report_path)


if __name__ == "__main__":
    unittest.main()
