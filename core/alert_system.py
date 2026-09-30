"""
Alert System Module
Manages all alerts, notifications, and emergency responses.
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# Use RLock to prevent deadlocks from nested lock acquisition
_lock = threading.RLock()


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """Represents a security alert."""
    timestamp: float
    level: AlertLevel
    title: str
    message: str
    source: str
    acknowledged: bool = False
    acknowledged_by: str = ""
    acknowledged_at: float = 0.0


class AlertSystem:
    """Manages dam safety alerts and notifications."""

    def __init__(self, config: dict):
        self.config = config
        self.alerts: List[Alert] = []
        self._lock = _lock
        self._running = False

        # Alert configuration
        self.escalation_minutes = config.get("alerts", {}).get("escalation_minutes", 15)
        self.recipients = config.get("alerts", {}).get("recipients", [])

        # Alert history
        self.alert_history: List[Alert] = []

    def start(self):
        """Start alert system."""
        self._running = True
        print("[AlertSystem] Started. Monitoring for dam safety alerts")

    def stop(self):
        """Stop alert system."""
        self._running = False
        print("[AlertSystem] Stopped")

    def create_alert(self, level: AlertLevel, title: str, message: str, source: str) -> Alert:
        """Create and dispatch a new alert."""
        alert = Alert(
            timestamp=time.time(),
            level=level,
            title=title,
            message=message,
            source=source
        )

        with self._lock:
            self.alerts.append(alert)
            self.alert_history.append(alert)

        # Dispatch notifications
        self._dispatch_notifications(alert)

        return alert

    def _dispatch_notifications(self, alert: Alert):
        """Send notifications through all configured channels."""
        # Dashboard alert (always)
        self._dashboard_alert(alert)

        # Email for high/critical/emergency
        if alert.level in (AlertLevel.HIGH, AlertLevel.CRITICAL, AlertLevel.EMERGENCY):
            self._email_alert(alert)

        # SMS for critical/emergency
        if alert.level in (AlertLevel.CRITICAL, AlertLevel.EMERGENCY):
            self._sms_alert(alert)

        # Siren for emergency
        if alert.level == AlertLevel.EMERGENCY:
            self._siren_alert(alert)

    def _dashboard_alert(self, alert: Alert):
        """Display alert on dashboard."""
        level_str = alert.level.value.upper()
        print(f"[ALERT:{level_str}] {alert.title} - {alert.message}")

    def _email_alert(self, alert: Alert):
        """Send email notification."""
        # In production, this would use SMTP
        for recipient in self.recipients:
            print(f"[EMAIL] To: {recipient.get('email')} - {alert.title}")

    def _sms_alert(self, alert: Alert):
        """Send SMS notification."""
        # In production, this would use Twilio or similar
        for recipient in self.recipients:
            print(f"[SMS] To: {recipient.get('phone')} - {alert.title}")

    def _siren_alert(self, alert: Alert):
        """Trigger emergency siren."""
        print(f"[SIREN] EMERGENCY ALERT: {alert.title}")

    def acknowledge_alert(self, alert: Alert, user: str):
        """Acknowledge an alert."""
        alert.acknowledged = True
        alert.acknowledged_by = user
        alert.acknowledged_at = time.time()
        print(f"[ALERT] Acknowledged by {user}: {alert.title}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all unacknowledged alerts."""
        with self._lock:
            return [a for a in self.alerts if not a.acknowledged]

    def get_critical_alerts(self) -> List[Alert]:
        """Get all critical/emergency alerts."""
        with self._lock:
            return [a for a in self.alerts
                    if a.level in (AlertLevel.CRITICAL, AlertLevel.EMERGENCY)
                    and not a.acknowledged]

    def clear_alert(self, alert: Alert):
        """Clear/resolve an alert."""
        with self._lock:
            if alert in self.alerts:
                self.alerts.remove(alert)

    def get_alert_summary(self) -> dict:
        """Get alert summary statistics."""
        with self._lock:
            total = len(self.alert_history)
            active = len(self.get_active_alerts())
            critical = len(self.get_critical_alerts())

            by_level = {}
            for alert in self.alert_history:
                level = alert.level.value
                by_level[level] = by_level.get(level, 0) + 1

            return {
                "total_alerts": total,
                "active_alerts": active,
                "critical_alerts": critical,
                "by_level": by_level
            }
