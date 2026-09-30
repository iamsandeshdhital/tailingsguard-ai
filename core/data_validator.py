"""
Data Validation Module
Validates sensor data quality before it reaches the ML model.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

import numpy as np


@dataclass
class ValidationResult:
    sensor_id: str
    is_valid: bool
    issues: List[str]
    corrected_value: Optional[float]
    confidence: float


@dataclass
class ValidationReport:
    timestamp: float
    total_readings: int
    valid_readings: int
    invalid_readings: int
    corrected_readings: int
    overall_quality: float
    issues: List[str]


class DataValidator:
    def __init__(self, config: dict):
        self.config = config
        self._lock = threading.Lock()
        self._running = False
        self.rules = {
            "piezometer": {"min": 0, "max": 500, "max_rate_of_change": 50, "max_std_dev": 100},
            "inclinometer": {"min": -100, "max": 100, "max_rate_of_change": 20, "max_std_dev": 50},
            "water_level": {"min": 0, "max": 100, "max_rate_of_change": 5, "max_std_dev": 10},
            "weather": {"min": 0, "max": 500, "max_rate_of_change": 200, "max_std_dev": 100},
            "seismic": {"min": 0, "max": 10, "max_rate_of_change": 5, "max_std_dev": 3}
        }
        self._history: Dict[str, deque] = {}
        self._last_values: Dict[str, float] = {}
        self._last_times: Dict[str, float] = {}
        self.total_validations = 0
        self.total_valid = 0
        self.total_corrected = 0
        self.total_invalid = 0

    def start(self):
        self._running = True
        print("[DataValidator] Started")

    def stop(self):
        self._running = False

    def validate_reading(self, sensor_id: str, sensor_type: str, value: float, timestamp: Optional[float] = None) -> ValidationResult:
        if timestamp is None:
            timestamp = time.time()
        issues = []
        corrected_value = None
        confidence = 1.0
        rules = self.rules.get(sensor_type, {})
        if "min" in rules and value < rules["min"]:
            issues.append(f"Value {value} below minimum {rules['min']}")
            corrected_value = rules["min"]
            confidence *= 0.5
        if "max" in rules and value > rules["max"]:
            issues.append(f"Value {value} above maximum {rules['max']}")
            corrected_value = rules["max"]
            confidence *= 0.5
        if sensor_id in self._last_values and sensor_id in self._last_times:
            time_delta = timestamp - self._last_times[sensor_id]
            if time_delta > 0:
                value_delta = abs(value - self._last_values[sensor_id])
                rate = value_delta / (time_delta / 60)
                if "max_rate_of_change" in rules and rate > rules["max_rate_of_change"]:
                    issues.append(f"Rate of change {rate:.1f}/min exceeds max {rules['max_rate_of_change']}/min")
                    confidence *= 0.7
        if sensor_id not in self._history:
            self._history[sensor_id] = deque(maxlen=100)
        history = self._history[sensor_id]
        if len(history) >= 10:
            values = [h["value"] for h in history]
            mean = np.mean(values)
            std = np.std(values)
            if std > 0:
                z_score = abs(value - mean) / std
                if z_score > 3:
                    issues.append(f"Statistical outlier: z-score = {z_score:.1f}")
                    confidence *= 0.6
                    if z_score > 5:
                        corrected_value = mean
        if len(history) >= 5:
            recent_values = [h["value"] for h in list(history)[-5:]]
            if all(v == recent_values[0] for v in recent_values) and value == recent_values[0]:
                issues.append("Sensor appears stuck")
                confidence *= 0.3
        history.append({"timestamp": timestamp, "value": value})
        self._last_values[sensor_id] = value
        self._last_times[sensor_id] = timestamp
        self.total_validations += 1
        if not issues:
            self.total_valid += 1
        elif corrected_value is not None:
            self.total_corrected += 1
        else:
            self.total_invalid += 1
        return ValidationResult(sensor_id=sensor_id, is_valid=len(issues) == 0, issues=issues, corrected_value=corrected_value, confidence=confidence)

    def validate_batch(self, readings: List[Tuple[str, str, float, float]]) -> ValidationReport:
        results = []
        for sensor_id, sensor_type, value, timestamp in readings:
            result = self.validate_reading(sensor_id, sensor_type, value, timestamp)
            results.append(result)
        valid_count = sum(1 for r in results if r.is_valid)
        corrected_count = sum(1 for r in results if r.corrected_value is not None)
        invalid_count = len(results) - valid_count
        all_issues = []
        for r in results:
            all_issues.extend([f"{r.sensor_id}: {issue}" for issue in r.issues])
        overall_quality = valid_count / max(len(results), 1)
        return ValidationReport(
            timestamp=time.time(), total_readings=len(results), valid_readings=valid_count,
            invalid_readings=invalid_count, corrected_readings=corrected_count,
            overall_quality=overall_quality, issues=all_issues
        )

    def get_validation_stats(self) -> dict:
        return {
            "total_validations": self.total_validations, "valid": self.total_valid,
            "corrected": self.total_corrected, "invalid": self.total_invalid,
            "accuracy": self.total_valid / max(self.total_validations, 1)
        }
