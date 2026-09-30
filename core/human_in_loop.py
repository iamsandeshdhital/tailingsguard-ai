"""
Human-in-the-Loop Module
Ensures critical decisions require human approval — protects against liability.
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class DecisionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    TIMED_OUT = "timed_out"


@dataclass
class HumanDecision:
    decision_id: str
    timestamp: float
    action: str
    reason: str
    risk_level: str
    status: DecisionStatus
    requested_by: str
    approved_by: Optional[str] = None
    approved_at: Optional[float] = None
    comments: str = ""
    timeout_seconds: float = 300.0


class HumanInLoop:
    def __init__(self, config: dict):
        self.config = config
        self._lock = threading.Lock()
        self._running = False
        self.pending_decisions: Dict[str, HumanDecision] = {}
        self.decision_history: List[HumanDecision] = []
        self.require_approval_for = config.get("human_in_loop", {}).get("require_approval_for", ["evacuate", "lower_pond", "emergency_response", "shutdown"])
        self.default_timeout = config.get("human_in_loop", {}).get("default_timeout_seconds", 300)
        self.escalation_contacts = config.get("alerts", {}).get("recipients", [])
        self.total_decisions = 0
        self.approved_decisions = 0
        self.rejected_decisions = 0
        self.timed_out_decisions = 0

    def start(self):
        self._running = True
        print("[HumanInLoop] Started")

    def stop(self):
        self._running = False

    def request_decision(self, action: str, reason: str, risk_level: str, requested_by: str = "system") -> str:
        decision_id = f"DEC_{int(time.time())}_{action}"
        decision = HumanDecision(
            decision_id=decision_id, timestamp=time.time(), action=action,
            reason=reason, risk_level=risk_level, status=DecisionStatus.PENDING,
            requested_by=requested_by, timeout_seconds=self.default_timeout
        )
        with self._lock:
            self.pending_decisions[decision_id] = decision
            self.total_decisions += 1
        return decision_id

    def approve_decision(self, decision_id: str, approved_by: str, comments: str = "") -> bool:
        with self._lock:
            if decision_id not in self.pending_decisions:
                return False
            decision = self.pending_decisions[decision_id]
            decision.status = DecisionStatus.APPROVED
            decision.approved_by = approved_by
            decision.approved_at = time.time()
            decision.comments = comments
            self.approved_decisions += 1
            self.decision_history.append(decision)
            del self.pending_decisions[decision_id]
        return True

    def reject_decision(self, decision_id: str, rejected_by: str, comments: str = "") -> bool:
        with self._lock:
            if decision_id not in self.pending_decisions:
                return False
            decision = self.pending_decisions[decision_id]
            decision.status = DecisionStatus.REJECTED
            decision.approved_by = rejected_by
            decision.approved_at = time.time()
            decision.comments = comments
            self.rejected_decisions += 1
            self.decision_history.append(decision)
            del self.pending_decisions[decision_id]
        return True

    def check_timeouts(self):
        current_time = time.time()
        timed_out = []
        with self._lock:
            for decision_id, decision in list(self.pending_decisions.items()):
                if current_time - decision.timestamp > decision.timeout_seconds:
                    decision.status = DecisionStatus.TIMED_OUT
                    self.timed_out_decisions += 1
                    self.decision_history.append(decision)
                    del self.pending_decisions[decision_id]
                    timed_out.append(decision)
        for decision in timed_out:
            self._escalate_decision(decision)

    def _escalate_decision(self, decision: HumanDecision):
        print(f"[HumanInLoop] ESCALATING: {decision.action}")

    def get_pending_decisions(self) -> List[HumanDecision]:
        with self._lock:
            return list(self.pending_decisions.values())

    def get_decision_status(self, decision_id: str) -> Optional[DecisionStatus]:
        with self._lock:
            if decision_id in self.pending_decisions:
                return self.pending_decisions[decision_id].status
        for decision in self.decision_history:
            if decision.decision_id == decision_id:
                return decision.status
        return None

    def get_statistics(self) -> dict:
        return {
            "total_decisions": self.total_decisions, "approved": self.approved_decisions,
            "rejected": self.rejected_decisions, "timed_out": self.timed_out_decisions,
            "pending": len(self.pending_decisions)
        }
