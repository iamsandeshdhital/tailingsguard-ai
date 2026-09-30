"""
Model Drift Detection Module
Detects when the ML model's predictions become unreliable.
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque

import numpy as np


@dataclass
class DriftReport:
    timestamp: float
    drift_detected: bool
    drift_score: float
    affected_features: List[str]
    recommendation: str
    retrain_recommended: bool


@dataclass
class PredictionRecord:
    timestamp: float
    predicted_probability: float
    actual_outcome: Optional[bool] = None
    sensor_values: Dict[str, float] = field(default_factory=dict)


class ModelDriftDetector:
    def __init__(self, config: dict):
        self.config = config
        self._lock = threading.Lock()
        self._running = False
        self.drift_threshold = config.get("models", {}).get("drift_threshold", 0.3)
        self.min_samples_for_drift = config.get("models", {}).get("min_samples_for_drift", 100)
        self.prediction_window = config.get("models", {}).get("prediction_window", 1000)
        self.predictions: deque = deque(maxlen=self.prediction_window)
        self.outcomes: deque = deque(maxlen=self.prediction_window)
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None
        self.baseline_features: Dict[str, Dict[str, float]] = {}
        self.drift_history: List[DriftReport] = []
        self.last_drift_check = 0
        self.drift_check_interval = 3600

    def start(self):
        self._running = True
        print("[ModelDrift] Started")

    def stop(self):
        self._running = False

    def record_prediction(self, probability: float, sensor_values: Dict[str, float]):
        with self._lock:
            self.predictions.append(PredictionRecord(timestamp=time.time(), predicted_probability=probability, sensor_values=sensor_values))
            if self.baseline_mean is None and len(self.predictions) >= 100:
                self._calculate_baseline()

    def record_outcome(self, prediction_index: int, actual_failure: bool):
        with self._lock:
            if prediction_index < len(self.predictions):
                self.predictions[prediction_index].actual_outcome = actual_failure
                self.outcomes.append({"timestamp": time.time(), "predicted": self.predictions[prediction_index].predicted_probability, "actual": actual_failure})

    def check_drift(self) -> DriftReport:
        current_time = time.time()
        if current_time - self.last_drift_check < self.drift_check_interval:
            return DriftReport(timestamp=current_time, drift_detected=False, drift_score=0.0, affected_features=[], recommendation="No check due yet", retrain_recommended=False)
        self.last_drift_check = current_time
        with self._lock:
            if len(self.predictions) < self.min_samples_for_drift:
                return DriftReport(timestamp=current_time, drift_detected=False, drift_score=0.0, affected_features=[], recommendation="Insufficient data", retrain_recommended=False)
            drift_score = self._calculate_drift_score()
            drift_detected = drift_score > self.drift_threshold
            affected = self._identify_affected_features() if drift_detected else []
            if drift_detected:
                recommendation = "Model drift detected. Retraining recommended."
                retrain = True
            else:
                recommendation = "Model performing within expected parameters."
                retrain = False
            report = DriftReport(timestamp=current_time, drift_detected=drift_detected, drift_score=drift_score, affected_features=affected, recommendation=recommendation, retrain_recommended=retrain)
            self.drift_history.append(report)
            return report

    def _calculate_baseline(self):
        probs = [p.predicted_probability for p in self.predictions]
        self.baseline_mean = np.mean(probs)
        self.baseline_std = np.std(probs)
        feature_values: Dict[str, List[float]] = {}
        for pred in self.predictions:
            for feature, value in pred.sensor_values.items():
                if feature not in feature_values:
                    feature_values[feature] = []
                feature_values[feature].append(value)
        for feature, values in feature_values.items():
            self.baseline_features[feature] = {"mean": np.mean(values), "std": np.std(values)}

    def _calculate_drift_score(self) -> float:
        if self.baseline_mean is None:
            return 0.0
        recent = list(self.predictions)[-100:]
        recent_probs = [p.predicted_probability for p in recent]
        recent_mean = np.mean(recent_probs)
        mean_drift = abs(recent_mean - self.baseline_mean) / max(self.baseline_std, 0.01)
        feature_drift = 0.0
        for feature, baseline in self.baseline_features.items():
            recent_values = [p.sensor_values.get(feature, 0) for p in recent if feature in p.sensor_values]
            if recent_values:
                recent_mean = np.mean(recent_values)
                feature_drift += abs(recent_mean - baseline["mean"]) / max(baseline["std"], 0.01)
        feature_drift /= max(len(self.baseline_features), 1)
        total_drift = (mean_drift * 0.5) + (feature_drift * 0.5)
        return min(1.0, total_drift / 3.0)

    def _identify_affected_features(self) -> List[str]:
        if not self.baseline_features:
            return []
        recent = list(self.predictions)[-100:]
        affected = []
        for feature, baseline in self.baseline_features.items():
            recent_values = [p.sensor_values.get(feature, 0) for p in recent if feature in p.sensor_values]
            if recent_values:
                recent_mean = np.mean(recent_values)
                drift = abs(recent_mean - baseline["mean"]) / max(baseline["std"], 0.01)
                if drift > 2.0:
                    affected.append(feature)
        return affected

    def get_model_performance(self) -> dict:
        with self._lock:
            known = [o for o in self.outcomes if o["actual"] is not None]
            if not known:
                return {"status": "insufficient_data"}
            true_positives = sum(1 for o in known if o["predicted"] > 0.5 and o["actual"])
            false_positives = sum(1 for o in known if o["predicted"] > 0.5 and not o["actual"])
            true_negatives = sum(1 for o in known if o["predicted"] <= 0.5 and not o["actual"])
            false_negatives = sum(1 for o in known if o["predicted"] <= 0.5 and o["actual"])
            precision = true_positives / max(true_positives + false_positives, 1)
            recall = true_positives / max(true_positives + false_negatives, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1)
            return {
                "total_predictions": len(self.predictions), "known_outcomes": len(known),
                "precision": precision, "recall": recall, "f1_score": f1,
                "false_negative_rate": false_negatives / max(false_negatives + true_positives, 1)
            }
