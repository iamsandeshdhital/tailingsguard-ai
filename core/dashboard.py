"""
Dashboard Module
Real-time monitoring dashboard for tailings dam safety.
"""

import time
import threading
from typing import Dict, List, Optional
from collections import deque

import numpy as np


class Dashboard:
    """Real-time dam safety dashboard."""

    def __init__(self, config: dict):
        self.config = config
        self.refresh_rate = config.get("dashboard", {}).get("refresh_rate_seconds", 10)

        # Data storage
        self.health_history: deque = deque(maxlen=1000)
        self.sensor_data: Dict[str, deque] = {}
        self.alerts: List[dict] = []
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        """Start dashboard."""
        self._running = True
        print("[Dashboard] Started real-time monitoring dashboard")

    def stop(self):
        """Stop dashboard."""
        self._running = False
        print("[Dashboard] Stopped")

    def update_health(self, health_index):
        """Update with new health index."""
        with self._lock:
            self.health_history.append(health_index)

    def update_sensor(self, sensor_id: str, value: float):
        """Update sensor data."""
        with self._lock:
            if sensor_id not in self.sensor_data:
                self.sensor_data[sensor_id] = deque(maxlen=100)
            self.sensor_data[sensor_id].append({
                "timestamp": time.time(),
                "value": value
            })

    def add_alert(self, alert: dict):
        """Add alert to dashboard."""
        with self._lock:
            self.alerts.append(alert)

    def get_current_status(self) -> dict:
        """Get current dam status for display."""
        with self._lock:
            if not self.health_history:
                return {"status": "initializing"}

            latest = self.health_history[-1]

            return {
                "health_score": latest.overall_score,
                "risk_level": latest.risk_level,
                "factor_of_safety": latest.factor_of_safety,
                "phreatic_ratio": latest.phreatic_ratio,
                "movement_rate": latest.movement_rate,
                "pond_distance": latest.pond_distance_m,
                "factors": latest.contributing_factors,
                "timestamp": latest.timestamp
            }

    def get_sensor_summary(self) -> dict:
        """Get summary of all sensor readings."""
        with self._lock:
            summary = {}
            for sensor_id, readings in self.sensor_data.items():
                if readings:
                    values = [r["value"] for r in readings]
                    summary[sensor_id] = {
                        "current": values[-1],
                        "min": min(values),
                        "max": max(values),
                        "avg": np.mean(values),
                        "count": len(values)
                    }
            return summary

    def get_trend_data(self, hours: int = 24) -> dict:
        """Get trend data for charts."""
        with self._lock:
            cutoff = time.time() - (hours * 3600)
            recent = [h for h in self.health_history if h.timestamp > cutoff]

            if not recent:
                return {"timestamps": [], "scores": []}

            return {
                "timestamps": [h.timestamp for h in recent],
                "scores": [h.overall_score for h in recent],
                "fs": [h.factor_of_safety for h in recent],
                "phreatic": [h.phreatic_ratio for h in recent],
                "movement": [h.movement_rate for h in recent]
            }

    def display_console(self):
        """Display dashboard in console."""
        status = self.get_current_status()
        sensors = self.get_sensor_summary()

        print("\n" + "=" * 60)
        print("  TailingsGuard AI - Dam Safety Dashboard")
        print("=" * 60)

        if status.get("status") == "initializing":
            print("  Status: Initializing...")
            return

        # Health score with color indicator
        score = status["health_score"]
        risk = status["risk_level"].upper()

        if score >= 80:
            indicator = "[SAFE]"
        elif score >= 60:
            indicator = "[CAUTION]"
        elif score >= 40:
            indicator = "[WARNING]"
        else:
            indicator = "[DANGER]"

        print(f"  Health Score: {score:.1f}/100 {indicator}")
        print(f"  Risk Level:   {risk}")
        print(f"  Factor of Safety: {status['factor_of_safety']:.2f}")
        print(f"  Phreatic Ratio:   {status['phreatic_ratio']:.1%}")
        print(f"  Movement Rate:    {status['movement_rate']:.2f} mm/day")
        print(f"  Pond Distance:   {status['pond_distance']:.1f} m")

        if status["factors"]:
            print("\n  Contributing Factors:")
            for factor in status["factors"]:
                print(f"    - {factor}")

        print("\n  Sensor Summary:")
        for sensor_id, data in list(sensors.items())[:5]:
            print(f"    {sensor_id}: {data['current']:.2f} (avg: {data['avg']:.2f})")

        print("=" * 60 + "\n")
