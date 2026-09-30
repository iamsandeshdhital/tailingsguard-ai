"""
Sensor Fusion Module
Combines data from multiple sensors to create a unified dam health picture.
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque

import numpy as np


@dataclass
class SensorReading:
    """Single sensor reading."""
    timestamp: float
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    location: str
    quality: float = 1.0  # 0-1 data quality indicator


@dataclass
class DamHealthIndex:
    """Overall dam health assessment."""
    timestamp: float
    overall_score: float  # 0-100, higher is safer
    factor_of_safety: float
    phreatic_ratio: float
    movement_rate: float
    pond_distance_m: float
    risk_level: str  # "low", "medium", "high", "critical"
    contributing_factors: List[str] = field(default_factory=list)


class SensorFusion:
    """Fuses multi-sensor data into actionable dam safety insights."""

    def __init__(self, config: dict):
        self.config = config
        self.sensor_data: Dict[str, deque] = {}
        self.health_history: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        self._running = False

        # Safety thresholds
        self.fs_minimum = config.get("safety", {}).get("fs_minimum", 1.5)
        self.fs_warning = config.get("safety", {}).get("fs_warning", 1.3)
        self.fs_critical = config.get("safety", {}).get("fs_critical", 1.1)
        self.phreatic_max = config.get("safety", {}).get("phreatic_max_ratio", 0.7)
        self.min_beach = config.get("safety", {}).get("min_beach_width_m", 50)

    def start(self):
        """Start sensor fusion."""
        self._running = True
        print("[SensorFusion] Started multi-sensor data fusion")

    def stop(self):
        """Stop sensor fusion."""
        self._running = False
        print("[SensorFusion] Stopped")

    def add_reading(self, reading: SensorReading):
        """Add a new sensor reading."""
        with self._lock:
            key = f"{reading.sensor_type}_{reading.sensor_id}"
            if key not in self.sensor_data:
                self.sensor_data[key] = deque(maxlen=1000)
            self.sensor_data[key].append(reading)

    def calculate_health_index(self) -> DamHealthIndex:
        """
        Calculate overall dam health index from all sensor data.
        Returns DamHealthIndex with risk assessment.
        """
        current_time = time.time()
        factors = []
        scores = []

        # 1. Factor of Safety (from piezometers - pore pressure)
        fs = self._calculate_factor_of_safety()
        scores.append(self._fs_to_score(fs))
        if fs < self.fs_critical:
            factors.append(f"CRITICAL: Factor of Safety = {fs:.2f} (below {self.fs_critical})")
        elif fs < self.fs_warning:
            factors.append(f"WARNING: Factor of Safety = {fs:.2f} (below {self.fs_warning})")

        # 2. Phreatic surface ratio
        phreatic = self._calculate_phreatic_ratio()
        scores.append(self._phreatic_to_score(phreatic))
        if phreatic > self.phreatic_max:
            factors.append(f"CRITICAL: Phreatic surface at {phreatic:.0%} of dam height")

        # 3. Movement rate (from inclinometers)
        movement = self._calculate_movement_rate()
        scores.append(self._movement_to_score(movement))
        if movement > 20:
            factors.append(f"CRITICAL: Movement rate {movement:.1f} mm/day")
        elif movement > 5:
            factors.append(f"WARNING: Movement rate {movement:.1f} mm/day")

        # 4. Pond distance (freeboard / beach width)
        pond_dist = self._calculate_pond_distance()
        scores.append(self._pond_to_score(pond_dist))
        if pond_dist < self.min_beach:
            factors.append(f"WARNING: Beach width only {pond_dist:.0f}m (min {self.min_beach}m)")

        # 5. Weather impact
        rain_score = self._assess_weather_impact()
        scores.append(rain_score)
        if rain_score < 50:
            factors.append("WARNING: Heavy rainfall increasing risk")

        # Calculate overall score
        overall = np.mean(scores) if scores else 50.0

        # Determine risk level using worst-factor-dominates approach
        # If any individual factor is critical (<30), overall risk is at least high
        min_score = min(scores) if scores else overall

        # Use weighted combination: 40% overall, 60% worst factor
        # This ensures critical factors are not diluted by healthy ones
        weighted_score = overall * 0.4 + min_score * 0.6

        if weighted_score >= 75:
            risk = "low"
        elif weighted_score >= 50:
            risk = "medium"
        elif weighted_score >= 30:
            risk = "high"
        else:
            risk = "critical"

        health = DamHealthIndex(
            timestamp=current_time,
            overall_score=overall,
            factor_of_safety=fs,
            phreatic_ratio=phreatic,
            movement_rate=movement,
            pond_distance_m=pond_dist,
            risk_level=risk,
            contributing_factors=factors
        )

        with self._lock:
            self.health_history.append(health)

        return health

    def _calculate_factor_of_safety(self) -> float:
        """Estimate Factor of Safety from piezometer data."""
        # Simplified Bishop's method approximation
        # FS = (c' + (sigma - u) * tan(phi')) / (gamma * H * sin(alpha) * cos(alpha))
        # We use pore pressure ratio as proxy

        pore_pressures = []
        for key, readings in self.sensor_data.items():
            if "piezometer" in key and readings:
                pore_pressures.append(readings[-1].value)

        if not pore_pressures:
            return 2.0  # Default safe value

        avg_pressure = np.mean(pore_pressures)
        max_pressure = max(pore_pressures)

        # Higher pore pressure = lower FS
        # Typical: 0 kPa -> FS ~2.5, 150 kPa -> FS ~1.5, 250 kPa -> FS ~1.0
        fs = 2.5 - (max_pressure / 250) * 1.5
        return max(0.5, min(3.0, fs))

    def _calculate_phreatic_ratio(self) -> float:
        """Calculate phreatic surface as ratio of dam height."""
        # Find highest piezometer reading relative to dam height
        dam_height = self.config.get("dam", {}).get("height_m", 45)

        highest_pressure = 0
        for key, readings in self.sensor_data.items():
            if "piezometer" in key and readings:
                # Convert pressure to head (kPa -> m of water)
                head = readings[-1].value / 9.81
                highest_pressure = max(highest_pressure, head)

        return min(1.0, highest_pressure / dam_height)

    def _calculate_movement_rate(self) -> float:
        """Calculate slope movement rate from inclinometer data."""
        max_rate = 0

        for key, readings in self.sensor_data.items():
            if "inclinometer" in key and len(readings) >= 2:
                # Calculate rate from last two readings
                recent = list(readings)[-10:]
                if len(recent) >= 2:
                    time_span = recent[-1].timestamp - recent[0].timestamp
                    if time_span > 0:
                        value_change = abs(recent[-1].value - recent[0].value)
                        rate = value_change / (time_span / 86400)  # mm per day
                        max_rate = max(max_rate, rate)

        return max_rate

    def _calculate_pond_distance(self) -> float:
        """Calculate distance from pond to crest."""
        for key, readings in self.sensor_data.items():
            if "water_level" in key and readings:
                return readings[-1].value
        return 100.0  # Default safe value

    def _assess_weather_impact(self) -> float:
        """Assess weather impact on dam safety."""
        for key, readings in self.sensor_data.items():
            if "weather" in key and readings:
                rain = readings[-1].value
                if rain > 100:
                    return 30
                elif rain > 50:
                    return 60
                else:
                    return 100
        return 100

    def _fs_to_score(self, fs: float) -> float:
        """Convert Factor of Safety to 0-100 score."""
        if fs >= 2.0:
            return 100
        elif fs >= 1.5:
            return 80
        elif fs >= 1.3:
            return 60
        elif fs >= 1.1:
            return 40
        else:
            return 20

    def _phreatic_to_score(self, ratio: float) -> float:
        """Convert phreatic ratio to 0-100 score."""
        if ratio < 0.3:
            return 100
        elif ratio < 0.5:
            return 80
        elif ratio < 0.7:
            return 50
        else:
            return 20

    def _movement_to_score(self, rate: float) -> float:
        """Convert movement rate to 0-100 score."""
        if rate < 1:
            return 100
        elif rate < 5:
            return 80
        elif rate < 20:
            return 50
        else:
            return 20

    def _pond_to_score(self, distance: float) -> float:
        """Convert pond distance to 0-100 score."""
        if distance > 100:
            return 100
        elif distance > 50:
            return 80
        elif distance > 20:
            return 50
        else:
            return 20

    def get_trend(self, hours: int = 24) -> dict:
        """Get health trend over specified hours."""
        with self._lock:
            cutoff = time.time() - (hours * 3600)
            recent = [h for h in self.health_history if h.timestamp > cutoff]

        if not recent:
            return {"trend": "stable", "change": 0}

        scores = [h.overall_score for h in recent]
        change = scores[-1] - scores[0]

        if change > 10:
            trend = "improving"
        elif change < -10:
            trend = "deteriorating"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "change": change,
            "current_score": scores[-1],
            "min_score": min(scores),
            "max_score": max(scores)
        }
