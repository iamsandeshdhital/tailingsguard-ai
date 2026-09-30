"""
Sensor Simulator Module
Simulates realistic sensor data for testing and demonstration.
"""

import time
import random
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class SimulatedSensor:
    """Configuration for a simulated sensor."""
    sensor_id: str
    sensor_type: str
    location: str
    normal_mean: float
    normal_std: float
    unit: str
    drift_rate: float = 0.0  # How much the value drifts over time


class SensorSimulator:
    """Simulates realistic tailings dam sensor data."""

    def __init__(self, config: dict):
        self.config = config
        self.sensors: Dict[str, SimulatedSensor] = {}
        self._running = False
        self._lock = threading.Lock()

        # Simulation state
        self._time = 0
        self._failure_mode = False
        self._failure_start_time = 0

        self._init_sensors()

    def _init_sensors(self):
        """Initialize simulated sensors based on config."""
        dam_config = self.config.get("dam", {})
        sensor_config = self.config.get("sensors", {})

        # Piezometers
        piezometer_config = sensor_config.get("piezometers", {})
        if piezometer_config.get("enabled", True):
            locations = piezometer_config.get("locations", ["crest", "downstream_slope", "foundation", "toe"])
            for i in range(piezometer_config.get("count", 12)):
                location = locations[i % len(locations)]
                self.sensors[f"piezo_{i}"] = SimulatedSensor(
                    sensor_id=f"piezo_{i}",
                    sensor_type="piezometer",
                    location=location,
                    normal_mean=80 + random.uniform(-20, 20),
                    normal_std=10,
                    unit="kPa",
                    drift_rate=0.01
                )

        # Inclinometers
        inclinometer_config = sensor_config.get("inclinometers", {})
        if inclinometer_config.get("enabled", True):
            for i in range(inclinometer_config.get("count", 6)):
                self.sensors[f"inclin_{i}"] = SimulatedSensor(
                    sensor_id=f"inclin_{i}",
                    sensor_type="inclinometer",
                    location="downstream_slope",
                    normal_mean=0,
                    normal_std=0.5,
                    unit="mm",
                    drift_rate=0.001
                )

        # Water level
        water_config = sensor_config.get("water_level", {})
        if water_config.get("enabled", True):
            self.sensors["water_level"] = SimulatedSensor(
                sensor_id="water_level",
                sensor_type="water_level",
                location="pond",
                normal_mean=dam_config.get("height_m", 45) * 0.5,
                normal_std=2,
                unit="m",
                drift_rate=0.005
            )

        # Weather
        weather_config = sensor_config.get("weather", {})
        if weather_config.get("enabled", True):
            self.sensors["weather"] = SimulatedSensor(
                sensor_id="weather",
                sensor_type="weather",
                location="site",
                normal_mean=5,
                normal_std=10,
                unit="mm/day",
                drift_rate=0
            )

        # Seismic
        seismic_config = sensor_config.get("seismic", {})
        if seismic_config.get("enabled", True):
            self.sensors["seismic"] = SimulatedSensor(
                sensor_id="seismic",
                sensor_type="seismic",
                location="site",
                normal_mean=0,
                normal_std=0.1,
                unit="magnitude",
                drift_rate=0
            )

    def start(self):
        """Start simulation."""
        self._running = True
        print(f"[Simulator] Started. {len(self.sensors)} sensors simulated")

    def stop(self):
        """Stop simulation."""
        self._running = False
        print("[Simulator] Stopped")

    def trigger_failure_scenario(self):
        """Trigger a simulated failure scenario for testing."""
        self._failure_mode = True
        self._failure_start_time = self._time
        print("[Simulator] FAILURE SCENARIO TRIGGERED")

    def reset_scenario(self):
        """Reset simulation to normal state."""
        self._failure_mode = False
        self._failure_start_time = 0
        print("[Simulator] Scenario reset to normal")

    def get_readings(self) -> Dict[str, float]:
        """Get current readings from all sensors."""
        self._time += 1
        readings = {}

        for sensor_id, sensor in self.sensors.items():
            # Base value with noise
            value = np.random.normal(sensor.normal_mean, sensor.normal_std)

            # Add drift
            value += sensor.drift_rate * self._time

            # Failure mode effects
            if self._failure_mode:
                failure_duration = self._time - self._failure_start_time

                if sensor.sensor_type == "piezometer":
                    # Pore pressure increases rapidly
                    value += failure_duration * 2.5
                elif sensor.sensor_type == "inclinometer":
                    # Movement accelerates
                    value += failure_duration * 0.5 * (1 + failure_duration * 0.1)
                elif sensor.sensor_type == "water_level":
                    # Pond rises
                    value += failure_duration * 0.3
                elif sensor.sensor_type == "weather":
                    # Heavy rain
                    value = max(value, 80 + failure_duration * 5)

            readings[sensor_id] = max(0, value)

        return readings

    def get_sensor_config(self) -> Dict[str, dict]:
        """Get sensor configuration for display."""
        return {
            sensor_id: {
                "type": sensor.sensor_type,
                "location": sensor.location,
                "unit": sensor.unit,
                "normal_range": f"{sensor.normal_mean - 2*sensor.normal_std:.1f} - {sensor.normal_mean + 2*sensor.normal_std:.1f}"
            }
            for sensor_id, sensor in self.sensors.items()
        }
