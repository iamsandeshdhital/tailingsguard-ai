"""
Sensor Health Monitor Module
Detects sensor failures, drift, and data quality issues before they cause false predictions.
"""

import time
import threading
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import deque

import numpy as np

# Use RLock to prevent deadlocks from nested lock acquisition
_lock = threading.RLock()


@dataclass
class SensorHealth:
    """Health status of a single sensor."""
    sensor_id: str
    sensor_type: str
    status: str  # "healthy", "degraded", "failed", "unknown"
    last_reading: float
    last_seen: float
    reading_count: int
    error_count: int
    drift_score: float
    battery_level: Optional[float] = None
    signal_strength: Optional[float] = None


@dataclass
class DataQualityReport:
    """Report on overall data quality."""
    timestamp: float
    total_sensors: int
    healthy_sensors: int
    degraded_sensors: int
    failed_sensors: int
    missing_data_percent: float
    overall_quality: float  # 0-1
    issues: List[str]


class SensorHealthMonitor:
    """Monitors sensor health and data quality."""

    def __init__(self, config: dict):
        self.config = config
        self.sensor_health: Dict[str, SensorHealth] = {}
        self._lock = _lock
        self._running = False

        # Thresholds
        self.max_silence_seconds = config.get("sensors", {}).get("max_silence_seconds", 300)
        self.drift_threshold = config.get("sensors", {}).get("drift_threshold", 0.3)
        self.min_readings_per_hour = config.get("sensors", {}).get("min_readings_per_hour", 10)

        # Historical data for drift detection
        self._history: Dict[str, deque] = {}
        self._baseline: Dict[str, Dict[str, float]] = {}

    def start(self):
        """Start sensor health monitoring."""
        self._running = True
        print("[SensorHealth] Started. Monitoring sensor health and data quality")

    def stop(self):
        """Stop sensor health monitoring."""
        self._running = False
        print("[SensorHealth] Stopped")

    def register_sensor(self, sensor_id: str, sensor_type: str):
        """Register a new sensor for monitoring."""
        with self._lock:
            self.sensor_health[sensor_id] = SensorHealth(
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                status="unknown",
                last_reading=0,
                last_seen=0,
                reading_count=0,
                error_count=0,
                drift_score=0.0
            )
            self._history[sensor_id] = deque(maxlen=1000)

    def update_reading(self, sensor_id: str, value: float, timestamp: Optional[float] = None):
        """Update with a new sensor reading."""
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            if sensor_id not in self.sensor_health:
                self.register_sensor(sensor_id, "unknown")

            health = self.sensor_health[sensor_id]
            health.last_reading = value
            health.last_seen = timestamp
            health.reading_count += 1

            # Add to history
            self._history[sensor_id].append({
                "timestamp": timestamp,
                "value": value
            })

            # Update baseline if needed
            if sensor_id not in self._baseline and len(self._history[sensor_id]) >= 100:
                self._calculate_baseline(sensor_id)

            # Check for drift
            if sensor_id in self._baseline:
                health.drift_score = self._calculate_drift(sensor_id)

            # Update status
            health.status = self._determine_status(sensor_id)

    def _calculate_baseline(self, sensor_id: str):
        """Calculate baseline statistics for a sensor."""
        values = [r["value"] for r in self._history[sensor_id]]
        self._baseline[sensor_id] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }

    def _calculate_drift(self, sensor_id: str) -> float:
        """Calculate drift score for a sensor."""
        if sensor_id not in self._baseline:
            return 0.0

        baseline = self._baseline[sensor_id]
        # Use only last 20 readings for drift calculation (much faster)
        values = [r["value"] for r in list(self._history[sensor_id])[-20:]]

        if not values:
            return 0.0

        current_mean = np.mean(values)
        current_std = np.std(values)

        # Drift = how much current stats differ from baseline
        mean_drift = abs(current_mean - baseline["mean"]) / max(baseline["std"], 1)
        std_drift = abs(current_std - baseline["std"]) / max(baseline["std"], 1)

        return min(1.0, (mean_drift + std_drift) / 2)

    def _determine_status(self, sensor_id: str) -> str:
        """Determine sensor status."""
        health = self.sensor_health[sensor_id]
        current_time = time.time()

        # Check if sensor is silent
        silence_duration = current_time - health.last_seen
        if silence_duration > self.max_silence_seconds:
            return "failed"

        # Check drift
        if health.drift_score > self.drift_threshold:
            return "degraded"

        # Check reading rate
        if health.reading_count > 0:
            time_span = current_time - (current_time - 3600)  # Last hour
            if time_span > 0:
                rate = health.reading_count / time_span * 3600
                if rate < self.min_readings_per_hour:
                    return "degraded"

        return "healthy"

    def get_health_report(self) -> DataQualityReport:
        """Generate overall data quality report."""
        with self._lock:
            current_time = time.time()
            total = len(self.sensor_health)
            healthy = sum(1 for h in self.sensor_health.values() if h.status == "healthy")
            degraded = sum(1 for h in self.sensor_health.values() if h.status == "degraded")
            failed = sum(1 for h in self.sensor_health.values() if h.status == "failed")

            # Calculate missing data
            expected_sensors = self.config.get("sensors", {}).get("piezometers", {}).get("count", 12)
            expected_sensors += self.config.get("sensors", {}).get("inclinometers", {}).get("count", 6)
            expected_sensors += 3  # water_level, weather, seismic

            missing = max(0, expected_sensors - total)
            missing_percent = missing / max(expected_sensors, 1) * 100

            # Overall quality score
            if total > 0:
                quality = (healthy * 1.0 + degraded * 0.5) / total
            else:
                quality = 0.0

            # Generate issues list
            issues = []
            for sensor_id, health in self.sensor_health.items():
                if health.status == "failed":
                    issues.append(f"FAILED: {sensor_id} - no data for {current_time - health.last_seen:.0f}s")
                elif health.status == "degraded":
                    issues.append(f"DEGRADED: {sensor_id} - drift score {health.drift_score:.2f}")

            return DataQualityReport(
                timestamp=current_time,
                total_sensors=total,
                healthy_sensors=healthy,
                degraded_sensors=degraded,
                failed_sensors=failed,
                missing_data_percent=missing_percent,
                overall_quality=quality,
                issues=issues
            )

    def get_sensor_status(self, sensor_id: str) -> Optional[SensorHealth]:
        """Get health status for a specific sensor."""
        return self.sensor_health.get(sensor_id)

    def display_health_report(self, report: DataQualityReport):
        """Display health report in console."""
        print("\n" + "-" * 50)
        print("  SENSOR HEALTH REPORT")
        print("-" * 50)
        print(f"  Total Sensors:    {report.total_sensors}")
        print(f"  Healthy:          {report.healthy_sensors}")
        print(f"  Degraded:         {report.degraded_sensors}")
        print(f"  Failed:           {report.failed_sensors}")
        print(f"  Missing Data:     {report.missing_data_percent:.1f}%")
        print(f"  Overall Quality:  {report.overall_quality:.0%}")

        if report.issues:
            print("\n  Issues:")
            for issue in report.issues:
                print(f"    - {issue}")
        else:
            print("\n  No issues detected.")

        print("-" * 50 + "\n")
