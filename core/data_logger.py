"""
Data Logger Module
Stores all sensor data, predictions, and alerts for compliance and analysis.
"""

import os
import json
import time
import sqlite3
import threading
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import asdict


class DataLogger:
    """Logs all dam safety data for regulatory compliance."""

    def __init__(self, config: dict):
        self.config = config
        self.db_path = config.get("database", {}).get("path", "data/tailingsguard.db")
        self.retention_days = config.get("database", {}).get("retention_days", 365)

        # Ensure data directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._running = False
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Sensor readings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    sensor_id TEXT NOT NULL,
                    sensor_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    location TEXT,
                    quality REAL DEFAULT 1.0
                )
            ''')

            # Health index table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    overall_score REAL NOT NULL,
                    factor_of_safety REAL,
                    phreatic_ratio REAL,
                    movement_rate REAL,
                    pond_distance_m REAL,
                    risk_level TEXT,
                    contributing_factors TEXT
                )
            ''')

            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    probability REAL NOT NULL,
                    time_to_failure_hours REAL,
                    confidence REAL,
                    risk_factors TEXT,
                    recommended_actions TEXT
                )
            ''')

            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    level TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT,
                    source TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    acknowledged_by TEXT,
                    acknowledged_at REAL
                )
            ''')

            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensor_time ON sensor_readings(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_health_time ON health_index(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(timestamp)')

            conn.commit()

    def start(self):
        """Start data logger."""
        self._running = True
        print(f"[DataLogger] Started. Database: {self.db_path}")

    def stop(self):
        """Stop data logger."""
        self._running = False
        print("[DataLogger] Stopped")

    def log_sensor_reading(self, sensor_id: str, sensor_type: str, value: float,
                           unit: str = "", location: str = "", quality: float = 1.0):
        """Log a sensor reading."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sensor_readings (timestamp, sensor_id, sensor_type, value, unit, location, quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (time.time(), sensor_id, sensor_type, value, unit, location, quality))
                conn.commit()

    def log_health_index(self, health):
        """Log a health index calculation."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO health_index (timestamp, overall_score, factor_of_safety,
                        phreatic_ratio, movement_rate, pond_distance_m, risk_level, contributing_factors)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    health.timestamp,
                    health.overall_score,
                    health.factor_of_safety,
                    health.phreatic_ratio,
                    health.movement_rate,
                    health.pond_distance_m,
                    health.risk_level,
                    json.dumps(health.contributing_factors)
                ))
                conn.commit()

    def log_prediction(self, prediction):
        """Log a failure prediction."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO predictions (timestamp, probability, time_to_failure_hours,
                        confidence, risk_factors, recommended_actions)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    prediction.timestamp,
                    prediction.probability,
                    prediction.time_to_failure_hours,
                    prediction.confidence,
                    json.dumps(prediction.risk_factors),
                    json.dumps(prediction.recommended_actions)
                ))
                conn.commit()

    def log_alert(self, alert):
        """Log an alert."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alerts (timestamp, level, title, message, source, acknowledged)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    alert.timestamp,
                    alert.level.value,
                    alert.title,
                    alert.message,
                    alert.source,
                    1 if alert.acknowledged else 0
                ))
                conn.commit()

    def get_recent_readings(self, sensor_type: str, hours: int = 24) -> List[dict]:
        """Get recent sensor readings."""
        cutoff = time.time() - (hours * 3600)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM sensor_readings
                WHERE sensor_type = ? AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (sensor_type, cutoff))
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_health_history(self, hours: int = 24) -> List[dict]:
        """Get health index history."""
        cutoff = time.time() - (hours * 3600)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM health_index
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff,))
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_alerts(self, acknowledged: Optional[bool] = None, hours: int = 24) -> List[dict]:
        """Get alerts with optional filtering."""
        cutoff = time.time() - (hours * 3600)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if acknowledged is None:
                cursor.execute('''
                    SELECT * FROM alerts WHERE timestamp > ? ORDER BY timestamp DESC
                ''', (cutoff,))
            else:
                cursor.execute('''
                    SELECT * FROM alerts WHERE timestamp > ? AND acknowledged = ? ORDER BY timestamp DESC
                ''', (cutoff, 1 if acknowledged else 0))
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def cleanup_old_data(self):
        """Remove data older than retention period."""
        cutoff = time.time() - (self.retention_days * 86400)
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM sensor_readings WHERE timestamp < ?', (cutoff,))
                cursor.execute('DELETE FROM health_index WHERE timestamp < ?', (cutoff,))
                cursor.execute('DELETE FROM predictions WHERE timestamp < ?', (cutoff,))
                cursor.execute('DELETE FROM alerts WHERE timestamp < ?', (cutoff,))
                conn.commit()
                print(f"[DataLogger] Cleaned up data older than {self.retention_days} days")

    def export_compliance_report(self, days: int = 30) -> str:
        """Export data for regulatory compliance report."""
        health_data = self.get_health_history(days * 24)
        alerts = self.get_alerts(hours=days * 24)

        report = {
            "report_period_days": days,
            "generated_at": time.time(),
            "total_health_checks": len(health_data),
            "total_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a["level"] == "critical"]),
            "health_summary": {
                "average_score": sum(h["overall_score"] for h in health_data) / max(len(health_data), 1),
                "min_score": min((h["overall_score"] for h in health_data), default=0),
                "max_score": max((h["overall_score"] for h in health_data), default=0),
            },
            "alerts": alerts
        }

        report_path = f"compliance_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report_path
