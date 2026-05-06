from __future__ import annotations

from collections import Counter, deque
from statistics import mean
from typing import Deque, Dict, List


class PerceptionAgent:
    """Builds a structured, compact view of the current scene."""

    def run(self, scene_state: Dict) -> Dict:
        zones = scene_state.get("zones", [])
        all_detections: List[str] = []
        for z in zones:
            all_detections.extend(z.get("detections", []))
        counts = Counter(all_detections)
        emergency = bool(scene_state.get("mock_emergency") or any("emergency" in d.lower() or "ambulance" in d.lower() for d in all_detections))
        top = ", ".join([f"{k}:{v}" for k, v in counts.most_common(3)]) if counts else "no detections"
        return {
            "agent": "Perception Agent",
            "status": "ACTIVE",
            "confidence": 0.92 if counts else 0.5,
            "output": f"Parsed {len(zones)} zone(s), detections: {top}",
            "recommendation": "Scene state refreshed",
            "risk_level": "medium" if counts else "low",
            "explanation": "Aggregates detector outputs into structured zone-level state.",
            "emergency_detected": emergency,
            "detection_counts": dict(counts),
        }


class TrafficControlAgent:
    """Selects control policy from congestion and waiting conditions."""

    def run(self, scene_state: Dict) -> Dict:
        zones = scene_state.get("zones", [])
        if not zones:
            return {
                "agent": "Traffic Control Agent",
                "status": "STANDBY",
                "confidence": 0.4,
                "output": "No zone input",
                "recommendation": "KEEP_CYCLE",
                "risk_level": "low",
                "explanation": "No zones provided.",
                "action": "KEEP_CYCLE",
            }
        avg_cong = mean([float(z.get("congestion", 0.0)) for z in zones])
        avg_wait = mean([float(z.get("waiting_time", 0.0)) for z in zones])
        avg_crit = mean([float(z.get("criticality", 0.0)) for z in zones])
        if avg_cong >= 0.65 or avg_crit >= 75:
            action = "EXTEND_GREEN"
        elif avg_cong >= 0.35 or avg_wait >= 3.0:
            action = "ADAPTIVE_CYCLE"
        elif avg_cong <= 0.2 and avg_wait < 2.0:
            action = "REDUCE_GREEN"
        else:
            action = "KEEP_CYCLE"
        return {
            "agent": "Traffic Control Agent",
            "status": "ACTIVE",
            "confidence": min(0.98, 0.65 + avg_cong * 0.35),
            "output": f"avg congestion={avg_cong:.2f}, avg wait={avg_wait:.1f}m, avg criticality={avg_crit:.0f}",
            "recommendation": action,
            "risk_level": "high" if action == "EXTEND_GREEN" else "medium" if action == "ADAPTIVE_CYCLE" else "low",
            "explanation": "Maps congestion/criticality to signal strategy.",
            "action": action,
        }


class EmergencyAgent:
    """Overrides normal strategy when emergency vehicles are present."""

    def run(self, scene_state: Dict, perception: Dict) -> Dict:
        emergency = bool(perception.get("emergency_detected"))
        if emergency:
            return {
                "agent": "Emergency Agent",
                "status": "OVERRIDE",
                "confidence": 0.99,
                "output": "Emergency vehicle detected",
                "recommendation": "EMERGENCY_PRIORITY_MODE",
                "risk_level": "high",
                "explanation": "Opens green corridor for emergency lane and holds conflicting lanes.",
                "priority_mode": True,
                "traffic_light_action": "OPEN_GREEN_CORRIDOR",
            }
        return {
            "agent": "Emergency Agent",
            "status": "STANDBY",
            "confidence": 0.9,
            "output": "No emergency vehicle detected",
            "recommendation": "No override",
            "risk_level": "low",
            "explanation": "Emergency override not required.",
            "priority_mode": False,
            "traffic_light_action": "NONE",
        }


class ViolationAgent:
    """Computes violation likelihood from current context and detections."""

    def run(self, scene_state: Dict, perception: Dict) -> Dict:
        zones = scene_state.get("zones", [])
        avg_crit = mean([float(z.get("criticality", 0.0)) for z in zones]) if zones else 0.0
        labels = " ".join(perception.get("detection_counts", {}).keys()).lower()
        high_keywords = any(k in labels for k in ["wrong-way", "restricted", "stop-line", "red-cross"])
        if high_keywords or avg_crit >= 80:
            risk = "high"
        elif avg_crit >= 55:
            risk = "medium"
        else:
            risk = "low"
        msg = "Potential violation risk elevated" if risk != "low" else "No strong violation signal"
        return {
            "agent": "Violation Agent",
            "status": "ALERT" if risk == "high" else "ACTIVE",
            "confidence": 0.8 if risk != "low" else 0.65,
            "output": msg,
            "recommendation": "Mark zone as risky and increase monitoring" if risk != "low" else "Keep monitoring",
            "risk_level": risk,
            "explanation": "Returns risk level (not certainty) unless strong violation evidence exists.",
        }


class PredictionAgent:
    """Predicts short-term congestion trend from recent history."""

    def __init__(self, history_size: int = 30) -> None:
        self.history: Deque[float] = deque(maxlen=history_size)

    def run(self, scene_state: Dict) -> Dict:
        zones = scene_state.get("zones", [])
        cur = mean([float(z.get("congestion", 0.0)) for z in zones]) if zones else 0.0
        self.history.append(cur)
        trend = "STABLE"
        if len(self.history) >= 3:
            delta = self.history[-1] - self.history[-3]
            if self.history[-1] >= 0.55 and delta > 0.05:
                trend = "MEDIUM_TO_HIGH"
            elif self.history[-1] >= 0.25 and delta > 0.05:
                trend = "LOW_TO_MEDIUM"
            elif delta < -0.05:
                trend = "DECREASING"
            else:
                trend = "STABLE"
        return {
            "agent": "Prediction Agent",
            "status": "ACTIVE",
            "confidence": 0.7 if len(self.history) >= 3 else 0.55,
            "output": f"Predicted trend: {trend}",
            "recommendation": "Pre-emptive cycle adaptation" if trend in {"LOW_TO_MEDIUM", "MEDIUM_TO_HIGH"} else "Keep current monitoring",
            "risk_level": "high" if trend == "MEDIUM_TO_HIGH" else "medium" if trend == "LOW_TO_MEDIUM" else "low",
            "explanation": "Uses recent congestion trajectory from frame history.",
            "trend": trend,
        }


class ReroutingAgent:
    """Suggests rerouting policy based on congestion and emergency state."""

    def run(self, scene_state: Dict, emergency: Dict, traffic: Dict) -> Dict:
        zones = scene_state.get("zones", [])
        avg_cong = mean([float(z.get("congestion", 0.0)) for z in zones]) if zones else 0.0
        if emergency.get("priority_mode"):
            level = "mandatory rerouting"
        elif avg_cong >= 0.65 or traffic.get("action") == "EXTEND_GREEN":
            level = "recommended rerouting"
        elif avg_cong >= 0.35:
            level = "optional rerouting"
        else:
            level = "no rerouting"
        return {
            "agent": "Rerouting Agent",
            "status": "ACTIVE",
            "confidence": 0.84,
            "output": f"Rerouting policy: {level}",
            "recommendation": level,
            "risk_level": "high" if level == "mandatory rerouting" else "medium" if level == "recommended rerouting" else "low",
            "explanation": "Combines emergency state and congestion pressure.",
        }


class XAIAgent:
    """Explains the main factors that drove the final policy."""

    def run(self, scene_state: Dict, emergency: Dict) -> Dict:
        zones = scene_state.get("zones", [])
        avg_cong = mean([float(z.get("congestion", 0.0)) for z in zones]) if zones else 0.0
        avg_count = mean([float(z.get("vehicle_count", 0.0)) for z in zones]) if zones else 0.0
        avg_wait = mean([float(z.get("waiting_time", 0.0)) for z in zones]) if zones else 0.0
        avg_crit = mean([float(z.get("criticality", 0.0)) for z in zones]) if zones else 0.0
        emergency_factor = 100.0 if emergency.get("priority_mode") else 0.0
        return {
            "agent": "XAI Agent",
            "status": "ACTIVE",
            "confidence": 0.9,
            "output": "Decision factors computed",
            "recommendation": "Expose interpretable reasoning",
            "risk_level": "info",
            "explanation": "Ranks factors influencing final decision.",
            "factors": {
                "congestion": float(max(0.0, min(100.0, avg_cong * 100.0))),
                "vehicle_count": float(max(0.0, min(100.0, (avg_count / 20.0) * 100.0))),
                "waiting_time": float(max(0.0, min(100.0, (avg_wait / 10.0) * 100.0))),
                "criticality_score": float(max(0.0, min(100.0, avg_crit))),
                "emergency_presence": emergency_factor,
                "traffic_light_state": 55.0 if zones else 20.0,
            },
            "decision_chain": "Perception -> Emergency Check -> Traffic Decision -> Rerouting -> Supervisor Final Decision",
        }


class SupervisorAgent:
    """Applies priority policy and emits the final control decision."""

    def run(
        self,
        emergency: Dict,
        violation: Dict,
        traffic: Dict,
        prediction: Dict,
        rerouting: Dict,
    ) -> Dict:
        if emergency.get("priority_mode"):
            final_decision = "EMERGENCY_PRIORITY_MODE"
            final_reco = "Open green corridor, hold conflicting lanes, notify supervisors."
            status = "OVERRIDE"
            risk = "high"
            selected_action = "OPEN_GREEN_CORRIDOR"
        elif violation.get("risk_level") == "high":
            final_decision = "SAFETY_ENFORCEMENT_MODE"
            final_reco = "Keep adaptive cycle, increase enforcement and zone monitoring."
            status = "ALERT"
            risk = "high"
            selected_action = traffic.get("action", "ADAPTIVE_CYCLE")
        else:
            final_decision = traffic.get("action", "KEEP_CYCLE")
            final_reco = f"{traffic.get('action', 'KEEP_CYCLE')} with {rerouting.get('recommendation', 'no rerouting')}."
            status = "ACTIVE"
            risk = max(traffic.get("risk_level", "low"), prediction.get("risk_level", "low"), key=lambda x: {"low": 1, "medium": 2, "high": 3}.get(x, 0))
            selected_action = traffic.get("action", "KEEP_CYCLE")

        return {
            "agent": "Supervisor Agent",
            "status": status,
            "confidence": 0.93,
            "output": f"Final decision: {final_decision}",
            "recommendation": final_reco,
            "risk_level": risk,
            "explanation": "Resolves agent conflicts by priority: Emergency > Violation > Traffic > Prediction > Rerouting.",
            "final_decision": final_decision,
            "selected_action": selected_action,
            "final_recommendation": final_reco,
        }


class AgenticOrchestrator:
    """Container object so stateful agents (Prediction) keep history."""

    def __init__(self) -> None:
        self.perception = PerceptionAgent()
        self.traffic = TrafficControlAgent()
        self.emergency = EmergencyAgent()
        self.violation = ViolationAgent()
        self.prediction = PredictionAgent()
        self.rerouting = ReroutingAgent()
        self.xai = XAIAgent()
        self.supervisor = SupervisorAgent()

    def run_agentic_system(self, scene_state: Dict) -> Dict:
        perception_out = self.perception.run(scene_state)
        emergency_out = self.emergency.run(scene_state, perception_out)
        traffic_out = self.traffic.run(scene_state)
        violation_out = self.violation.run(scene_state, perception_out)
        prediction_out = self.prediction.run(scene_state)
        rerouting_out = self.rerouting.run(scene_state, emergency_out, traffic_out)
        xai_out = self.xai.run(scene_state, emergency_out)
        supervisor_out = self.supervisor.run(
            emergency=emergency_out,
            violation=violation_out,
            traffic=traffic_out,
            prediction=prediction_out,
            rerouting=rerouting_out,
        )

        agents = [
            perception_out,
            traffic_out,
            emergency_out,
            violation_out,
            prediction_out,
            rerouting_out,
            xai_out,
            supervisor_out,
        ]

        event_log = [
            "Perception Agent analyzed current frame",
            "Emergency Agent checked priority mode",
            f"Traffic Control Agent recommends {traffic_out.get('action', 'KEEP_CYCLE')}",
            f"Prediction Agent trend: {prediction_out.get('trend', 'STABLE')}",
            f"Rerouting Agent: {rerouting_out.get('recommendation', 'no rerouting')}",
            f"Supervisor selected: {supervisor_out.get('final_decision', 'KEEP_CYCLE')}",
        ]

        return {
            "agents": agents,
            "final_decision": supervisor_out.get("final_decision", "KEEP_CYCLE"),
            "final_recommendation": supervisor_out.get("final_recommendation", ""),
            "selected_action": supervisor_out.get("selected_action", "KEEP_CYCLE"),
            "emergency_mode": bool(emergency_out.get("priority_mode")),
            "risk_level": supervisor_out.get("risk_level", "low"),
            "xai": xai_out,
            "event_log": event_log,
            "traffic_light_recommendation": traffic_out.get("action", "KEEP_CYCLE"),
            "rerouting_recommendation": rerouting_out.get("recommendation", "no rerouting"),
            "violation_risk_level": violation_out.get("risk_level", "low"),
        }


def run_agentic_system(scene_state: Dict, orchestrator: AgenticOrchestrator | None = None) -> Dict:
    """Entry point used by UI; supports optional external orchestrator."""
    runner = orchestrator or AgenticOrchestrator()
    return runner.run_agentic_system(scene_state)

