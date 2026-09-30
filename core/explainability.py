"""
Explainability Module
Shows WHY the model predicts failure — critical for engineer trust.
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Explanation:
    """Represents an explanation for a prediction."""
    timestamp: float
    prediction_probability: float
    top_factors: List[dict]
    feature_importance: Dict[str, float]
    confidence_interval: tuple
    similar_historical_cases: List[dict]
    narrative_summary: str


class ExplainabilityEngine:
    """Generates human-readable explanations for ML predictions."""

    def __init__(self, config: dict):
        self.config = config
        self._lock = threading.Lock()
        self._running = False

        self.feature_descriptions = {
            "piezometer": "Pore water pressure inside the dam",
            "inclinometer": "Slope movement/deformation",
            "water_level": "Distance from pond to crest",
            "weather": "Rainfall intensity",
            "seismic": "Seismic activity near dam",
            "factor_of_safety": "Calculated stability margin",
            "phreatic_ratio": "Saturation level inside dam",
            "movement_rate": "Speed of deformation",
            "pond_distance": "Beach width (freeboard)"
        }

        self.risk_descriptions = {
            "low": "Dam is operating within normal parameters",
            "medium": "Some parameters elevated, increased monitoring recommended",
            "high": "Multiple risk factors present, action required",
            "critical": "Imminent failure risk, evacuate and respond immediately"
        }

    def start(self):
        self._running = True
        print("[Explainability] Started")

    def stop(self):
        self._running = False

    def explain_prediction(self, prediction, health_index, sensor_data: Dict[str, List[float]]) -> Explanation:
        current_time = time.time()
        importance = self._calculate_feature_importance(sensor_data, health_index)
        top_factors = self._identify_top_factors(importance, health_index)
        confidence = self._calculate_confidence_interval(prediction)
        similar_cases = self._find_similar_cases(prediction, sensor_data)
        narrative = self._generate_narrative(prediction, health_index, top_factors)

        return Explanation(
            timestamp=current_time,
            prediction_probability=prediction.probability,
            top_factors=top_factors,
            feature_importance=importance,
            confidence_interval=confidence,
            similar_historical_cases=similar_cases,
            narrative_summary=narrative
        )

    def _calculate_feature_importance(self, sensor_data: Dict[str, List[float]], health_index) -> Dict[str, float]:
        importance = {}
        if "piezometer" in sensor_data:
            pressures = sensor_data["piezometer"]
            max_pressure = max(pressures) if pressures else 0
            importance["piezometer"] = min(1.0, max_pressure / 250)
        if "inclinometer" in sensor_data:
            movements = sensor_data["inclinometer"]
            max_movement = max(abs(m) for m in movements) if movements else 0
            importance["inclinometer"] = min(1.0, max_movement / 30)
        if "water_level" in sensor_data:
            level = sensor_data["water_level"][-1] if sensor_data["water_level"] else 0
            dam_height = self.config.get("dam", {}).get("height_m", 45)
            importance["water_level"] = min(1.0, level / dam_height)
        if "weather" in sensor_data:
            rain = sensor_data["weather"][-1] if sensor_data["weather"] else 0
            importance["weather"] = min(1.0, rain / 150)
        if "seismic" in sensor_data:
            mag = sensor_data["seismic"][-1] if sensor_data["seismic"] else 0
            importance["seismic"] = min(1.0, mag / 5)
        importance["factor_of_safety"] = 1.0 - min(1.0, health_index.factor_of_safety / 3)
        importance["phreatic_ratio"] = health_index.phreatic_ratio
        importance["movement_rate"] = min(1.0, health_index.movement_rate / 30)
        importance["pond_distance"] = 1.0 - min(1.0, health_index.pond_distance_m / 100)
        return importance

    def _identify_top_factors(self, importance: Dict[str, float], health_index) -> List[dict]:
        sorted_factors = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_factors = []
        for feature, score in sorted_factors[:5]:
            if score > 0.1:
                top_factors.append({
                    "feature": feature,
                    "description": self.feature_descriptions.get(feature, feature),
                    "importance_score": score,
                    "current_value": self._get_current_value(feature, health_index),
                    "threshold": self._get_threshold(feature),
                    "status": "critical" if score > 0.7 else "warning" if score > 0.4 else "normal"
                })
        return top_factors

    def _get_current_value(self, feature: str, health_index) -> str:
        if feature == "factor_of_safety":
            return f"{health_index.factor_of_safety:.2f}"
        elif feature == "phreatic_ratio":
            return f"{health_index.phreatic_ratio:.1%}"
        elif feature == "movement_rate":
            return f"{health_index.movement_rate:.2f} mm/day"
        elif feature == "pond_distance":
            return f"{health_index.pond_distance_m:.1f} m"
        return "N/A"

    def _get_threshold(self, feature: str) -> str:
        thresholds = {
            "factor_of_safety": "> 1.5", "phreatic_ratio": "< 70%",
            "movement_rate": "< 5 mm/day", "pond_distance": "> 50 m",
            "piezometer": "< 150 kPa", "inclinometer": "< 5 mm/day",
            "water_level": "< 80% of dam height", "weather": "< 100 mm/day",
            "seismic": "< 4.0 magnitude"
        }
        return thresholds.get(feature, "N/A")

    def _calculate_confidence_interval(self, prediction) -> tuple:
        prob = prediction.probability
        confidence = prediction.confidence
        margin = (1 - confidence) * 0.3
        return (max(0, prob - margin), min(1, prob + margin))

    def _find_similar_cases(self, prediction, sensor_data) -> List[dict]:
        return []

    def _generate_narrative(self, prediction, health_index, top_factors) -> str:
        prob = prediction.probability
        risk = health_index.risk_level
        if prob > 0.7:
            opening = f"CRITICAL: The model predicts a {prob:.0%} probability of dam failure within 72 hours."
        elif prob > 0.4:
            opening = f"WARNING: The model predicts a {prob:.0%} probability of dam failure within 72 hours."
        else:
            opening = f"The model predicts a {prob:.0%} probability of dam failure within 72 hours. Dam appears stable."
        risk_desc = self.risk_descriptions.get(risk, "Unknown risk level.")
        if top_factors:
            factors_text = "Key factors: " + ", ".join([f"{f['description']} ({f['status']})" for f in top_factors[:3]])
        else:
            factors_text = "No single factor dominates."
        if prob > 0.7:
            recommendation = "IMMEDIATE ACTION REQUIRED: Evacuate downstream areas and activate emergency response."
        elif prob > 0.4:
            recommendation = "Increased monitoring and preparation for potential emergency response recommended."
        else:
            recommendation = "Continue normal operations with standard monitoring."
        return f"{opening}\n{risk_desc}\n{factors_text}\n{recommendation}"

    def display_explanation(self, explanation: Explanation):
        print("\n" + "=" * 60)
        print("  PREDICTION EXPLANATION")
        print("=" * 60)
        print(f"\n{explanation.narrative_summary}")
        print(f"\nConfidence Interval: {explanation.confidence_interval[0]:.0%} - {explanation.confidence_interval[1]:.0%}")
        if explanation.top_factors:
            print("\nTop Contributing Factors:")
            for i, factor in enumerate(explanation.top_factors, 1):
                status_icon = "[!]" if factor["status"] == "critical" else "[+]" if factor["status"] == "warning" else "[ok]"
                print(f"  {i}. {status_icon} {factor['description']}")
                print(f"     Current: {factor['current_value']} | Threshold: {factor['threshold']} | Importance: {factor['importance_score']:.0%}")
        print("=" * 60 + "\n")
