"""
Adaptive Alert System Module
Prevents "cry wolf" problem by learning from false alarms and adjusting thresholds.
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque

import numpy as np


@dataclass
class AlertFeedback:
    """Feedback on whether an alert was valid."""
    alert_id: str
    timestamp: float
    was_true_positive: bool
    user_notes: str = ""
    response_time_minutes: float = 0.0


@dataclass
class AdaptiveThreshold:
    """An adaptive threshold that learns from feedback."""
    parameter: str
    current_value: float
    original_value: float
    min_value: float
    max_value: float
    adjustment_rate: float = 0.05
    true_positives: int = 0
    false_positives: int = 0


class AdaptiveAlertSystem:
    """Alert system that learns from false alarms to prevent alert fatigue."""

    def __init__(self, config: dict):
        self.config = config
        self._lock = threading.Lock()
        self._running = False

        # Alert tracking
        self.active_alerts: Dict[str, dict] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.feedback_history: List[AlertFeedback] = []

        # Adaptive thresholds
        self.thresholds: Dict[str, AdaptiveThreshold] = {}
        self._init_thresholds()

        # Alert fatigue prevention
        self.max_alerts_per_hour = config.get("alerts", {}).get("max_alerts_per_hour", 5)
        self.cooldown_minutes = config.get("alerts", {}).get("cooldown_minutes", 30)
        self.recent_alerts: deque = deque(maxlen=100)

        # Statistics
        self.total_alerts = 0
        self.true_positives = 0
        self.false_positives = 0

    def _init_thresholds(self):
        """Initialize adaptive thresholds."""
        safety = self.config.get("safety", {})

        self.thresholds["factor_of_safety"] = AdaptiveThreshold(
            parameter="factor_of_safety",
            current_value=safety.get("fs_warning", 1.3),
            original_value=safety.get("fs_warning", 1.3),
            min_value=1.0,
            max_value=2.0
        )

        self.thresholds["phreatic_ratio"] = AdaptiveThreshold(
            parameter="phreatic_ratio",
            current_value=safety.get("phreatic_max_ratio", 0.7),
            original_value=safety.get("phreatic_max_ratio", 0.7),
            min_value=0.3,
            max_value=0.95
        )

        self.thresholds["movement_rate"] = AdaptiveThreshold(
            parameter="movement_rate",
            current_value=5.0,
            original_value=5.0,
            min_value=1.0,
            max_value=20.0
        )

        self.thresholds["pond_distance"] = AdaptiveThreshold(
            parameter="pond_distance",
            current_value=safety.get("min_beach_width_m", 50),
            original_value=safety.get("min_beach_width_m", 50),
            min_value=10.0,
            max_value=200.0
        )

    def start(self):
        """Start adaptive alert system."""
        self._running = True
        print("[AdaptiveAlerts] Started. Learning from feedback to prevent alert fatigue")

    def stop(self):
        """Stop adaptive alert system."""
        self._running = False
        print("[AdaptiveAlerts] Stopped")

    def should_alert(self, parameter: str, value: float, severity: str) -> bool:
        """
        Determine if an alert should be triggered.
        Implements cooldown and rate limiting.
        """
        current_time = time.time()

        # Check rate limiting
        recent_count = sum(
            1 for alert_time, _ in self.recent_alerts
            if current_time - alert_time < 3600  # Last hour
        )
        if recent_count >= self.max_alerts_per_hour:
            return False

        # Check cooldown for this parameter
        for alert_time, alert_param in reversed(self.recent_alerts):
            if alert_param == parameter:
                if current_time - alert_time < self.cooldown_minutes * 60:
                    return False
                break

        # Check threshold
        if parameter in self.thresholds:
            threshold = self.thresholds[parameter]
            if severity == "critical":
                return value < threshold.current_value * 0.8
            elif severity == "warning":
                return value < threshold.current_value

        return True

    def trigger_alert(self, parameter: str, value: float, severity: str, message: str) -> Optional[str]:
        """Trigger an alert if appropriate."""
        if not self.should_alert(parameter, value, severity):
            return None

        alert_id = f"ALT_{int(time.time())}_{parameter}"

        with self._lock:
            self.active_alerts[alert_id] = {
                "id": alert_id,
                "parameter": parameter,
                "value": value,
                "severity": severity,
                "message": message,
                "timestamp": time.time(),
                "acknowledged": False
            }
            self.recent_alerts.append((time.time(), parameter))
            self.total_alerts += 1

        print(f"[ADAPTIVE ALERT] {severity.upper()}: {message}")
        return alert_id

    def provide_feedback(self, alert_id: str, was_true_positive: bool, notes: str = ""):
        """Provide feedback on an alert to improve future predictions."""
        with self._lock:
            if alert_id not in self.active_alerts:
                return

            alert = self.active_alerts[alert_id]
            parameter = alert["parameter"]

            feedback = AlertFeedback(
                alert_id=alert_id,
                timestamp=time.time(),
                was_true_positive=was_true_positive,
                user_notes=notes
            )
            self.feedback_history.append(feedback)

            # Update statistics
            if was_true_positive:
                self.true_positives += 1
            else:
                self.false_positives += 1

            # Adjust threshold
            if parameter in self.thresholds:
                self._adjust_threshold(parameter, was_true_positive)

            # Remove from active alerts
            del self.active_alerts[alert_id]

    def _adjust_threshold(self, parameter: str, was_true_positive: bool):
        """Adjust threshold based on feedback."""
        threshold = self.thresholds[parameter]

        if was_true_positive:
            # True positive - threshold is good, maybe make slightly more sensitive
            threshold.current_value -= threshold.adjustment_rate * 0.1
        else:
            # False positive - make less sensitive
            threshold.current_value += threshold.adjustment_rate

        # Clamp to valid range
        threshold.current_value = max(threshold.min_value,
                                       min(threshold.max_value, threshold.current_value))

        # Update counts
        if was_true_positive:
            threshold.true_positives += 1
        else:
            threshold.false_positives += 1

    def get_alert_accuracy(self) -> float:
        """Get alert accuracy (true positives / total alerts)."""
        total = self.true_positives + self.false_positives
        if total == 0:
            return 1.0
        return self.true_positives / total

    def get_threshold_status(self) -> dict:
        """Get current threshold values."""
        return {
            param: {
                "current": t.current_value,
                "original": t.original_value,
                "true_positives": t.true_positives,
                "false_positives": t.false_positives
            }
            for param, t in self.thresholds.items()
        }

    def reset_thresholds(self):
        """Reset all thresholds to original values."""
        for param, threshold in self.thresholds.items():
            threshold.current_value = threshold.original_value
            threshold.true_positives = 0
            threshold.false_positives = 0
        print("[AdaptiveAlerts] All thresholds reset to original values")
