"""
Sensor Health Monitor Module
Detects sensor failures, drift, and data quality issues.
"""

import time
import threading
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import deque

import numpy as np

_lock = threading.RLock()


@dataclass
class SensorHealth:
    sensor_id: str
    sensor_type: str
    status: str
    last_reading: float
    last_seen: float
    reading_count: int
    error_count: int
    drift_score: float


@dataclass
class DataQualityReport:
    timestamp: float
    total_sensors: int
    healthy_sensors: int
    degraded_sensors: int
    failed_sensors: int
    missing_data_percent: float
    overall_quality: float
    issues: List[str]


class SensorHealthMonitor:
    def __init__(self, config: dict):
        self.config = config
        self.sensor_health: Dict[str, SensorHealth] = {}
        self._lock = _lock
        self._running = False
        self.max_silence_seconds = config.get("sensors", {}).get("max_silence_seconds", 300)
        self.drift_threshold = config.get("sensors", {}).get("drift_threshold", 0.3)
        self.min_readings_per_hour = config.get("sensors", {}).get("min_readings_per_hour", 10)
        self._history: Dict[str, deque] = {}
        self._baseline: Dict[str, Dict[str, float]] = {}

    def start(self):
        self._running = True
        print("[SensorHealth] Started")

    def stop(self):
        self._running = False

    def register_sensor(self, sensor_id: str, sensor_type: str):
        with self._lock:
            self.sensor_health[sensor_id] = SensorHealth(
                sensor_id=sensor_id, sensor_type=sensor_type, status="unknown",
                last_reading=0, last_seen=0, reading_count=0, error_count=0, drift_score=0.0
            )
            self._history[sensor_id] = deque(maxlen=1000)

    def update_reading(self, sensor_id: str, value: float, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            if sensor_id not in self.sensor_health:
                self.register_sensor(sensor_id, "unknown")
            health = self.sensor_health[sensor_id]
            health.last_reading = value
            health.last_seen = timestamp
            health.reading_count += 1
            self._history[sensor_id].append({"timestamp": timestamp, "value": value})
            if sensor_id not in self._baseline and len(self._history[sensor_id]) >= 100:
                self._calculate_baseline(sensor_id)
            if sensor_id in self._baseline:
                health.drift_score = self._calculate_drift(sensor_id)
            health.status = self._determine_status(sensor_id)

    def _calculate_baseline(self, sensor_id: str):
        values = [r["value"] for r in self._history[sensor_id]]
        self._baseline[sensor_id] = {"mean": np.mean(values), "std": np.std(values), "min": np.min(values), "max": np.max(values)}

    def _calculate_drift(self, sensor_id: str) -> float:
        if sensor_id not in self._baseline:
            return 0.0
        baseline = self._baseline[sensor_id]
        values = [r["value"] for r in list(self._history[sensor_id])[-20:]]
        if not values:
            return 0.0
        current_mean = np.mean(values)
        current_std = np.std(values)
        mean_drift = abs(current_mean - baseline["mean"]) / max(baseline["std"], 1)
        std_drift = abs(current_std - baseline["std"]) / max(baseline["std"], 1)
        return min(1.0, (mean_drift + std_drift) / 2)

    def _determine_status(self, sensor_id: str) -> str:
        health = self.sensor_health[sensor_id]
        current_time = time.time()
        if current_time - health.last_seen > self.max_silence_seconds:
            return "failed"
        if health.drift_score > self.drift_threshold:
            return "degraded"
        return "healthy"

    def get_health_report(self) -> DataQualityReport:
        with self._lock:
            current_time = time.time()
            total = len(self.sensor_health)
            healthy = sum(1 for h in self.sensor_health.values() if h.status == "healthy")
            degraded = sum(1 for h in self.sensor_health.values() if h.status == "degraded")
            failed = sum(1 for h in self.sensor_health.values() if h.status == "failed")
            expected = 21
            missing = max(0, expected - total)
            missing_percent = missing / max(expected, 1) * 100
            quality = (healthy * 1.0 + degraded * 0.5) / total if total > 0 else 0.0
            issues = []
            for sensor_id, health in self.sensor_health.items():
                if health.status == "failed":
                    issues.append(f"FAILED: {sensor_id}")
                elif health.status == "degraded":
                    issues.append(f"DEGRADED: {sensor_id}")
            return DataQualityReport(
                timestamp=current_time, total_sensors=total, healthy_sensors=healthy,
                degraded_sensors=degraded, failed_sensors=failed,
                missing_data_percent=missing_percent, overall_quality=quality, issues=issues
            )
