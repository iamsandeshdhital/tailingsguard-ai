"""
Failure Prediction Module
Uses ML to predict tailings dam failures before they happen.
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque

import numpy as np


@dataclass
class FailurePrediction:
    """Represents a failure prediction."""
    timestamp: float
    probability: float  # 0-1
    time_to_failure_hours: Optional[float]
    confidence: float
    risk_factors: List[str]
    recommended_actions: List[str]


class FailurePredictor:
    """Predicts tailings dam failures using sensor data patterns."""

    def __init__(self, config: dict):
        self.config = config
        self.prediction_horizon = config.get("models", {}).get("failure_predictor", {}).get("prediction_horizon_hours", 72)

        # Historical data for pattern matching
        self.sensor_history: Dict[str, deque] = {}
        self.failure_patterns: List[dict] = []
        self._lock = threading.Lock()
        self._running = False

        # Known failure precursors (from historical dam failures)
        self.failure_precursors = {
            "pore_pressure_spike": {
                "description": "Sudden increase in pore water pressure",
                "threshold": 1.5,  # 50% increase in 24h
                "weight": 0.3
            },
            "acceleration": {
                "description": "Accelerating deformation rate",
                "threshold": 2.0,  # Rate doubling
                "weight": 0.25
            },
            "pond_encroachment": {
                "description": "Pond moving toward crest",
                "threshold": 0.8,  # 80% of dam height
                "weight": 0.2
            },
            "heavy_rainfall": {
                "description": "Extreme rainfall event",
                "threshold": 100,  # mm/day
                "weight": 0.15
            },
            "seismic_activity": {
                "description": "Seismic event near dam",
                "threshold": 4.0,  # magnitude
                "weight": 0.1
            }
        }

    def start(self):
        """Start failure prediction."""
        self._running = True
        print("[FailurePredictor] Started. Prediction horizon: 72 hours")

    def stop(self):
        """Stop failure prediction."""
        self._running = False
        print("[FailurePredictor] Stopped")

    def predict(self, sensor_data: Dict[str, List[float]]) -> FailurePrediction:
        """
        Predict failure probability based on current sensor data.
        Returns FailurePrediction with probability and recommendations.
        """
        current_time = time.time()
        risk_factors = []
        risk_score = 0.0

        # Check each failure precursor
        for precursor_name, precursor in self.failure_precursors.items():
            score = self._check_precursor(precursor_name, precursor, sensor_data)
            if score > 0:
                risk_score += score * precursor["weight"]
                risk_factors.append(precursor["description"])

        # Apply sigmoid to get probability (steeper curve for better sensitivity)
        probability = 1 / (1 + np.exp(-8 * (risk_score - 0.3)))

        # Estimate time to failure
        time_to_failure = self._estimate_time_to_failure(probability, sensor_data)

        # Generate recommendations
        actions = self._generate_recommendations(probability, risk_factors)

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(sensor_data)

        return FailurePrediction(
            timestamp=current_time,
            probability=probability,
            time_to_failure_hours=time_to_failure,
            confidence=confidence,
            risk_factors=risk_factors,
            recommended_actions=actions
        )

    def _check_precursor(self, name: str, precursor: dict, sensor_data: Dict[str, List[float]]) -> float:
        """Check if a failure precursor is present. Returns 0-1 score."""
        if name == "pore_pressure_spike":
            return self._check_pore_pressure(sensor_data, precursor["threshold"])
        elif name == "acceleration":
            return self._check_acceleration(sensor_data, precursor["threshold"])
        elif name == "pond_encroachment":
            return self._check_pond_encroachment(sensor_data, precursor["threshold"])
        elif name == "heavy_rainfall":
            return self._check_rainfall(sensor_data, precursor["threshold"])
        elif name == "seismic_activity":
            return self._check_seismic(sensor_data, precursor["threshold"])
        return 0.0

    def _check_pore_pressure(self, sensor_data: Dict[str, List[float]], threshold: float) -> float:
        """Check for pore pressure spike."""
        pressures = []
        for key, values in sensor_data.items():
            if "piezometer" in key and len(values) >= 2:
                # Check for 50% increase
                recent = values[-10:]
                if len(recent) >= 2:
                    change = (recent[-1] - recent[0]) / max(recent[0], 1)
                    if change > threshold - 1:
                        pressures.append(min(1.0, change / threshold))

        return max(pressures) if pressures else 0.0

    def _check_acceleration(self, sensor_data: Dict[str, List[float]], threshold: float) -> float:
        """Check for accelerating deformation."""
        for key, values in sensor_data.items():
            if "inclinometer" in key and len(values) >= 3:
                recent = list(values)[-10:]
                if len(recent) >= 3:
                    # Calculate acceleration (second derivative)
                    velocities = np.diff(recent)
                    if len(velocities) >= 2:
                        accel = np.diff(velocities)
                        if np.mean(accel) > 0:
                            return min(1.0, np.mean(accel) / threshold)
        return 0.0

    def _check_pond_encroachment(self, sensor_data: Dict[str, List[float]], threshold: float) -> float:
        """Check if pond is too close to crest."""
        for key, values in sensor_data.items():
            if "water_level" in key and values:
                dam_height = self.config.get("dam", {}).get("height_m", 45)
                ratio = values[-1] / dam_height
                if ratio > threshold:
                    return min(1.0, ratio)
        return 0.0

    def _check_rainfall(self, sensor_data: Dict[str, List[float]], threshold: float) -> float:
        """Check for extreme rainfall."""
        for key, values in sensor_data.items():
            if "weather" in key and values:
                if values[-1] > threshold:
                    return min(1.0, values[-1] / (threshold * 2))
        return 0.0

    def _check_seismic(self, sensor_data: Dict[str, List[float]], threshold: float) -> float:
        """Check for seismic activity."""
        for key, values in sensor_data.items():
            if "seismic" in key and values:
                if values[-1] > threshold:
                    return min(1.0, values[-1] / 7.0)
        return 0.0

    def _estimate_time_to_failure(self, probability: float, sensor_data: Dict[str, List[float]]) -> Optional[float]:
        """Estimate time to failure in hours."""
        if probability < 0.3:
            return None  # No immediate risk

        # Higher probability = shorter time
        # This is a simplified model
        if probability > 0.9:
            return 1  # 1 hour
        elif probability > 0.7:
            return 6  # 6 hours
        elif probability > 0.5:
            return 24  # 24 hours
        else:
            return 72  # 72 hours

    def _generate_recommendations(self, probability: float, risk_factors: List[str]) -> List[str]:
        """Generate recommended actions based on risk level."""
        actions = []

        if probability > 0.9:
            actions = [
                "IMMEDIATE: Evacuate all personnel from downstream area",
                "IMMEDIATE: Notify emergency services",
                "IMMEDIATE: Activate emergency response plan",
                "URGENT: Lower pond level if possible",
                "URGENT: Continuous monitoring - 15 minute intervals"
            ]
        elif probability > 0.7:
            actions = [
                "URGENT: Increase monitoring frequency to 30 minutes",
                "URGENT: Notify mine manager and safety officer",
                "Prepare emergency response team",
                "Consider lowering pond level",
                "Restrict access to downstream areas"
            ]
        elif probability > 0.5:
            actions = [
                "Increase monitoring frequency to 1 hour",
                "Review drainage systems",
                "Check piezometer readings",
                "Notify safety officer"
            ]
        elif probability > 0.3:
            actions = [
                "Continue normal monitoring",
                "Review recent sensor trends",
                "Schedule inspection"
            ]
        else:
            actions = ["Continue normal operations"]

        return actions

    def _calculate_confidence(self, sensor_data: Dict[str, List[float]]) -> float:
        """Calculate prediction confidence based on data quality."""
        total_sensors = len(sensor_data)
        active_sensors = sum(1 for v in sensor_data.values() if len(v) > 0)

        if total_sensors == 0:
            return 0.0

        coverage = active_sensors / total_sensors

        # More data = higher confidence
        data_points = sum(len(v) for v in sensor_data.values())
        data_confidence = min(1.0, data_points / 100)

        return (coverage * 0.6) + (data_confidence * 0.4)
