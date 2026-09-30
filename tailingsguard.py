"""
TailingsGuard AI - Tailings Dam Failure Prediction System
=========================================================
Predicts catastrophic tailings dam failures before they happen,
saving lives, preventing environmental disasters, and protecting mining companies.

Usage:
    python tailingsguard.py              # Start with simulation
    python tailingsguard.py --demo       # Run demo with failure scenario
    python tailingsguard.py --port 8050  # Custom port
"""

import os
import sys
import time
import signal
import argparse
import threading
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.sensor_fusion import SensorFusion, SensorReading
from core.failure_predictor import FailurePredictor
from core.alert_system import AlertSystem, AlertLevel
from core.dashboard import Dashboard
from core.data_logger import DataLogger
from core.explainability import ExplainabilityEngine
from core.sensor_health import SensorHealthMonitor
from core.adaptive_alerts import AdaptiveAlertSystem
from core.data_validator import DataValidator
from core.model_drift import ModelDriftDetector
from core.integration_gateway import IntegrationGateway, ConnectionConfig, ProtocolType
from core.human_in_loop import HumanInLoop
from core.simulator import SensorSimulator


class TailingsGuard:
    """Main TailingsGuard AI application."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.running = False
        self._shutdown_event = threading.Event()

        # Initialize modules
        print("=" * 60)
        print("  TailingsGuard AI v1.0")
        print("  Tailings Dam Failure Prediction System")
        print("=" * 60)

        self.sensor_fusion = SensorFusion(self.config)
        self.failure_predictor = FailurePredictor(self.config)
        self.alert_system = AlertSystem(self.config)
        self.dashboard = Dashboard(self.config)
        self.data_logger = DataLogger(self.config)

        # Risk mitigation modules
        self.explainability = ExplainabilityEngine(self.config)
        self.sensor_health = SensorHealthMonitor(self.config)
        self.adaptive_alerts = AdaptiveAlertSystem(self.config)
        self.data_validator = DataValidator(self.config)
        self.model_drift = ModelDriftDetector(self.config)
        self.integration = IntegrationGateway(self.config)
        self.human_in_loop = HumanInLoop(self.config)
        self.simulator = SensorSimulator(self.config)

        # Statistics
        self.scan_count = 0
        self.alerts_triggered = 0
        self.predictions_made = 0
        self.start_time = None

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            print(f"[TailingsGuard] Config not found: {config_path}, using defaults")
            return self._default_config()

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[TailingsGuard] Loaded configuration from {config_path}")
            return config
        except Exception as e:
            print(f"[TailingsGuard] Error loading config: {e}, using defaults")
            return self._default_config()

    def _default_config(self) -> dict:
        """Default configuration."""
        return {
            "dam": {"height_m": 45, "type": "upstream"},
            "safety": {"fs_minimum": 1.5, "fs_warning": 1.3, "fs_critical": 1.1},
            "sensors": {"piezometers": {"enabled": True, "count": 12}},
            "alerts": {"enabled": True, "escalation_minutes": 15},
            "dashboard": {"refresh_rate_seconds": 10},
            "database": {"path": "data/tailingsguard.db"}
        }

    def start(self, demo_mode: bool = False):
        """Start the system."""
        self.running = True
        self.start_time = time.time()

        # Start all modules
        self.sensor_fusion.start()
        self.failure_predictor.start()
        self.alert_system.start()
        self.dashboard.start()
        self.data_logger.start()
        self.simulator.start()

        # Start risk mitigation modules
        self.explainability.start()
        self.sensor_health.start()
        self.adaptive_alerts.start()
        self.data_validator.start()
        self.model_drift.start()
        self.integration.start()
        self.human_in_loop.start()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("\n[TailingsGuard] All modules started. Monitoring active.")
        print("[TailingsGuard] Press Ctrl+C to stop.\n")

        if demo_mode:
            print("[TailingsGuard] DEMO MODE: Will trigger failure scenario in 30 seconds\n")
            threading.Timer(30, self._trigger_demo_failure).start()

        # Main loop
        interval = self.config.get("dashboard", {}).get("refresh_rate_seconds", 10)

        try:
            while self.running and not self._shutdown_event.is_set():
                self._scan_cycle()
                self._shutdown_event.wait(interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _scan_cycle(self):
        """Execute one full scan cycle."""
        self.scan_count += 1

        # 1. Get sensor readings
        readings = self.simulator.get_readings()

        # 2. Feed data to sensor fusion
        sensor_data = {}
        for sensor_id, value in readings.items():
            # Determine sensor type from ID
            if "piezo" in sensor_id:
                sensor_type = "piezometer"
            elif "inclin" in sensor_id:
                sensor_type = "inclinometer"
            elif "water" in sensor_id:
                sensor_type = "water_level"
            elif "weather" in sensor_id:
                sensor_type = "weather"
            elif "seismic" in sensor_id:
                sensor_type = "seismic"
            else:
                sensor_type = "unknown"

            reading = SensorReading(
                timestamp=time.time(),
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                value=value,
                unit="",
                location=""
            )
            self.sensor_fusion.add_reading(reading)

            # Group by type for prediction
            if sensor_type not in sensor_data:
                sensor_data[sensor_type] = []
            sensor_data[sensor_type].append(value)

            # Log to database
            self.data_logger.log_sensor_reading(
                sensor_id, sensor_type, value, "", ""
            )

        # 3. Calculate health index
        health = self.sensor_fusion.calculate_health_index()
        self.dashboard.update_health(health)
        self.data_logger.log_health_index(health)

        # 4. Run failure prediction
        prediction = self.failure_predictor.predict(sensor_data)
        self.predictions_made += 1
        self.data_logger.log_prediction(prediction)

        # 5. Record prediction for drift detection
        self.model_drift.record_prediction(prediction.probability, {
            "piezometer": max(sensor_data.get("piezometer", [0])),
            "inclinometer": max(sensor_data.get("inclinometer", [0])),
            "water_level": max(sensor_data.get("water_level", [0])),
        })

        # 6. Generate explanation
        explanation = self.explainability.explain_prediction(prediction, health, sensor_data)

        # 7. Check for alerts
        self._check_alerts(health, prediction)

        # 8. Check model drift
        drift_report = self.model_drift.check_drift()
        if drift_report.drift_detected:
            print(f"[MODEL DRIFT] {drift_report.recommendation}")

        # 9. Check sensor health
        health_report = self.sensor_health.get_health_report()
        if health_report.overall_quality < 0.7:
            print(f"[SENSOR HEALTH] Quality: {health_report.overall_quality:.0%}")

        # 10. Check for timed-out human decisions
        self.human_in_loop.check_timeouts()

        # 11. Update dashboard
        if self.scan_count % 6 == 0:  # Every ~60 seconds
            self.dashboard.display_console()
            self.explainability.display_explanation(explanation)

    def _check_alerts(self, health, prediction):
        """Check and generate alerts based on health and prediction."""
        # Health-based alerts
        if health.risk_level == "critical":
            self._create_alert(
                AlertLevel.EMERGENCY,
                "CRITICAL DAM SAFETY ALERT",
                f"Health score: {health.overall_score:.1f}/100. Factor of Safety: {health.factor_of_safety:.2f}",
                "SensorFusion"
            )
        elif health.risk_level == "high":
            self._create_alert(
                AlertLevel.CRITICAL,
                "HIGH RISK - Dam Safety",
                f"Health score: {health.overall_score:.1f}/100. Factors: {', '.join(health.contributing_factors)}",
                "SensorFusion"
            )
        elif health.risk_level == "medium":
            self._create_alert(
                AlertLevel.WARNING,
                "Medium Risk - Dam Safety",
                f"Health score: {health.overall_score:.1f}/100",
                "SensorFusion"
            )

        # Prediction-based alerts
        if prediction.probability > 0.9:
            self._create_alert(
                AlertLevel.EMERGENCY,
                "IMMINENT FAILURE PREDICTED",
                f"Probability: {prediction.probability:.1%}. Time to failure: {prediction.time_to_failure_hours}h",
                "FailurePredictor"
            )
        elif prediction.probability > 0.7:
            self._create_alert(
                AlertLevel.CRITICAL,
                "Failure Risk Detected",
                f"Probability: {prediction.probability:.1%}. Time to failure: {prediction.time_to_failure_hours}h",
                "FailurePredictor"
            )
        elif prediction.probability > 0.5:
            self._create_alert(
                AlertLevel.HIGH,
                "Elevated Failure Risk",
                f"Probability: {prediction.probability:.1%}",
                "FailurePredictor"
            )

    def _create_alert(self, level: AlertLevel, title: str, message: str, source: str):
        """Create and dispatch an alert."""
        alert = self.alert_system.create_alert(level, title, message, source)
        self.data_logger.log_alert(alert)
        self.alerts_triggered += 1

    def _trigger_demo_failure(self):
        """Trigger demo failure scenario."""
        print("\n" + "!" * 60)
        print("  TRIGGERING FAILURE SCENARIO FOR DEMONSTRATION")
        print("!" * 60 + "\n")
        self.simulator.trigger_failure_scenario()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n[TailingsGuard] Shutdown signal received...")
        self.running = False
        self._shutdown_event.set()

    def stop(self):
        """Stop the system."""
        print("\n[TailingsGuard] Stopping all modules...")

        self.sensor_fusion.stop()
        self.failure_predictor.stop()
        self.alert_system.stop()
        self.dashboard.stop()
        self.data_logger.stop()
        self.simulator.stop()

        # Stop risk mitigation modules
        self.explainability.stop()
        self.sensor_health.stop()
        self.adaptive_alerts.stop()
        self.data_validator.stop()
        self.model_drift.stop()
        self.integration.stop()
        self.human_in_loop.stop()

        # Final summary
        uptime = time.time() - self.start_time if self.start_time else 0
        print("\n" + "=" * 60)
        print("  TailingsGuard AI - Session Summary")
        print("=" * 60)
        print(f"  Uptime:            {uptime:.0f} seconds")
        print(f"  Scans completed:   {self.scan_count}")
        print(f"  Predictions made:  {self.predictions_made}")
        print(f"  Alerts triggered:  {self.alerts_triggered}")

        # Export compliance report
        report_path = self.data_logger.export_compliance_report(days=1)
        print(f"  Compliance report: {report_path}")
        print("=" * 60)
        print("[TailingsGuard] Stay safe!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TailingsGuard AI - Tailings Dam Failure Prediction System"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run in demo mode with failure scenario"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8050,
        help="Dashboard port"
    )

    args = parser.parse_args()

    app = TailingsGuard(config_path=args.config)

    if args.demo:
        app.start(demo_mode=True)
    else:
        app.start()


if __name__ == "__main__":
    main()
