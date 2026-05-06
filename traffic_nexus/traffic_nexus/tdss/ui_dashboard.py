from typing import Dict, List

import streamlit as st

from .decision_engine import ZoneDecision
from .xai import XAIExplanation


def inject_modern_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            background: #0b1220 !important;
        }
        .stApp { background: linear-gradient(135deg,#0b1220,#0a0f1a); color:#e5e7eb; }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            background: #0b1220 !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg,#0a1220,#0a0f1a) !important;
            border-right: 1px solid rgba(51,65,85,0.55);
        }
        [data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }
        [data-testid="stSidebar"] .stRadio > div,
        [data-testid="stSidebar"] .stSelectbox > div,
        [data-testid="stSidebar"] .stTextInput > div,
        [data-testid="stSidebar"] .stTextArea > div,
        [data-testid="stSidebar"] .stNumberInput > div {
            background: rgba(15,23,42,0.9) !important;
            border-radius: 10px !important;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            background: rgba(15,23,42,0.9) !important;
            color: #e5e7eb !important;
        }
        [data-testid="stSidebar"] .stButton button {
            background: linear-gradient(180deg,#111827,#0f172a) !important;
            color: #e5e7eb !important;
            border: 1px solid rgba(71,85,105,0.55) !important;
            border-radius: 10px !important;
        }
        [data-testid="stSidebar"] .stButton button:hover {
            border-color: rgba(96,165,250,0.65) !important;
            box-shadow: 0 0 0 1px rgba(96,165,250,0.2) inset;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: rgba(15,23,42,0.9) !important;
            border: 1px solid rgba(71,85,105,0.45) !important;
            border-radius: 10px !important;
        }
        [data-testid="stSidebar"] [data-baseweb="slider"] > div > div {
            background: #334155 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
            background: #38bdf8 !important;
            border-color: #38bdf8 !important;
        }
        /* Force custom canvas wrappers/iframes to dark background */
        [data-testid="stCustomComponentV1"],
        [data-testid="stCustomComponentV1"] > div,
        [data-testid="stCustomComponentV1"] iframe {
            background: #0b1220 !important;
            border-radius: 8px !important;
        }
        canvas {
            background: #0b1220 !important;
        }
        .stButton button {
            background: linear-gradient(180deg,#111827,#0f172a) !important;
            color: #e5e7eb !important;
            border: 1px solid rgba(71,85,105,0.55) !important;
            border-radius: 10px !important;
        }
        .stButton button:hover {
            border-color: rgba(96,165,250,0.65) !important;
            box-shadow: 0 0 0 1px rgba(96,165,250,0.2) inset;
        }
        .main-title { font-size: 2rem; font-weight: 800; color:#60a5fa; margin-bottom:0.2rem; }
        .subtitle { color:#94a3b8; margin-bottom:1rem; }
        .card {
            background: rgba(17,24,39,0.9);
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
            margin-bottom: 10px;
        }
        .badge { padding: 2px 8px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }
        .badge-low { background:#065f46; color:#d1fae5; }
        .badge-medium { background:#78350f; color:#fef3c7; }
        .badge-high { background:#7f1d1d; color:#fee2e2; }
        .kpi-title { color:#93c5fd; font-size:0.85rem; }
        .kpi-value { color:#22c55e; font-size:1.6rem; font-weight:800; }
        .alert-box {
            background:#3b0a0a; border:1px solid #ef4444; color:#fee2e2;
            border-radius:10px; padding:10px; margin:8px 0;
        }
        .timeline-item{
            border-left:2px solid #334155; padding-left:10px; margin:8px 0;
            color:#cbd5e1; font-size:0.9rem;
        }
        .agentic-shell{
            background: linear-gradient(135deg, rgba(18,31,52,0.78), rgba(11,20,37,0.72));
            border: 1px solid rgba(96,165,250,0.2);
            border-radius: 16px;
            padding: 14px;
            margin-top: 14px;
            backdrop-filter: blur(8px);
        }
        .agent-grid{
            display:grid;
            grid-template-columns: repeat(auto-fit, minmax(220px,1fr));
            gap:10px;
        }
        .agent-card{
            background: rgba(15,23,42,0.7);
            border:1px solid rgba(148,163,184,0.2);
            border-radius:12px;
            padding:10px;
        }
        .status-active{ color:#22c55e; font-weight:700; }
        .status-standby{ color:#94a3b8; font-weight:700; }
        .status-override{ color:#f97316; font-weight:700; }
        .status-alert{ color:#ef4444; font-weight:700; }
        .priority-banner{
            background: linear-gradient(90deg, rgba(127,29,29,0.95), rgba(239,68,68,0.5));
            border: 1px solid rgba(254,202,202,0.25);
            border-radius: 12px;
            padding: 10px;
            margin: 8px 0;
            color: #fee2e2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(level: str) -> str:
    css = f"badge-{level}"
    return f'<span class="badge {css}">{level.upper()}</span>'


def render_zone_decision(zone_idx: int, d: ZoneDecision, border_color: str) -> None:
    st.markdown(
        f"""
        <div class="card" style="border-left:4px solid {border_color};">
            <div><b>Zone {zone_idx + 1}</b> {badge(d.congestion_level)}</div>
            <div>Signal action: <b>{d.signal_action}</b></div>
            <div>Green/Red: <b>{d.green_time_s}s / {d.red_time_s}s</b></div>
            <div>Waiting time: <b>{d.estimated_waiting_time_min} min</b></div>
            <div>Criticality score: <b>{d.traffic_criticality_score}/100</b></div>
            <div>Rerouting: {d.rerouting_suggestion}</div>
            <div>Entry limit: {d.entry_limitation_recommendation}</div>
            <div><b>Final:</b> {d.final_recommendation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_global_overview(total_vehicles: int, avg_density: float, level: str, criticality: int) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="kpi-title">Global Traffic State</div>
            <div style="display:flex;justify-content:space-between;gap:8px;margin-top:8px;">
              <div><div class="kpi-title">Vehicles</div><div class="kpi-value">{total_vehicles}</div></div>
              <div><div class="kpi-title">Density</div><div class="kpi-value">{avg_density:.2f}</div></div>
              <div><div class="kpi-title">Congestion</div><div class="kpi-value">{level.upper()}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(max(0.0, min(1.0, criticality / 100.0)), text=f"Traffic criticality: {criticality}/100")


def render_zone_counts_table(rows: List[Dict]) -> None:
    st.markdown('<div class="card"><b>Vehicle Counting by Zone</b></div>', unsafe_allow_html=True)
    if not rows:
        st.info("No zone counts yet.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_class_counts_table(rows: List[Dict]) -> None:
    st.markdown('<div class="card"><b>Congestion Detections by Class</b></div>', unsafe_allow_html=True)
    if not rows:
        st.info("No detections yet.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_congestion_class_cards(rows: List[Dict]) -> None:
    st.markdown("#### Class distribution")
    if not rows:
        st.info("No detections yet.")
        return
    for row in rows:
        label = str(row.get("Class", "-"))
        count = int(row.get("Count", 0))
        color = str(row.get("Color", "#22c55e"))
        st.markdown(
            f"""
            <div class="card" style="border-left:4px solid {color};">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div><b>{label}</b></div>
                    <div style="font-size:1.2rem;font-weight:800;color:{color};">{count}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_xai(exp: XAIExplanation) -> None:
    st.markdown(f'<div class="card"><b>{exp.title}</b><br>{exp.message}</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Congestion", f"{exp.feature_congestion:.1f}%")
    cols[1].metric("Vehicle count", f"{exp.feature_vehicle_count:.0f}")
    cols[2].metric("Waiting (min)", f"{exp.feature_waiting_time:.1f}")
    cols[3].metric("Criticality", f"{exp.feature_criticality:.0f}/100")
    st.markdown("**Influence factors**")
    st.progress(min(1.0, exp.feature_congestion / 100.0), text=f"Congestion level: {exp.feature_congestion:.1f}%")
    st.progress(min(1.0, exp.feature_criticality / 100.0), text=f"Criticality score: {exp.feature_criticality:.0f}/100")
    st.progress(min(1.0, exp.feature_waiting_time / 15.0), text=f"Estimated waiting: {exp.feature_waiting_time:.1f} min")


def render_history(history: List[str]) -> None:
    st.markdown('<div class="card"><b>Decision History (latest)</b></div>', unsafe_allow_html=True)
    for line in history[-8:][::-1]:
        st.markdown(f'<div class="timeline-item">{line}</div>', unsafe_allow_html=True)


def render_alert_if_needed(level: str, alert_message: str) -> None:
    if level == "high":
        st.markdown(
            f'<div class="alert-box"><b>Driver Alert:</b> {alert_message}</div>',
            unsafe_allow_html=True,
        )


def render_agentic_control_center(agentic: Dict) -> None:
    st.markdown("### Agentic AI Control Center")
    st.markdown('<div class="agentic-shell">', unsafe_allow_html=True)
    by_name = {str(a.get("agent", "")): a for a in agentic.get("agents", [])}
    p = by_name.get("Perception Agent", {})
    v = by_name.get("Violation Agent", {})
    pr = by_name.get("Prediction Agent", {})
    x = by_name.get("XAI Agent", {})
    s = by_name.get("Supervisor Agent", {})
    t = by_name.get("Traffic Control Agent", {})
    e = by_name.get("Emergency Agent", {})
    r = by_name.get("Rerouting Agent", {})

    st.markdown("#### Sense")
    st.markdown(
        f"""
        <div class="card" style="border-left:4px solid #22d3ee;background:linear-gradient(135deg,rgba(8,47,73,0.55),rgba(15,23,42,0.9));">
            <div><b>👁️ Perception Agent — The Eyes of the System</b></div>
            <div style="margin-top:6px;color:#67e8f9;">Scene scanned • Objects detected • Visual context updated</div>
            <div style="margin-top:6px;">Status: <b>{p.get("status","STANDBY")}</b> | Confidence: <b>{float(p.get("confidence",0))*100:.0f}%</b></div>
            <div style="margin-top:6px;color:#cbd5e1;">{p.get("output","No scene state yet.")}</div>
            <div style="margin-top:8px;height:38px;border-radius:8px;border:1px solid rgba(34,211,238,0.25);background:
                linear-gradient(to right, rgba(34,211,238,0.06) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(34,211,238,0.06) 1px, transparent 1px);
                background-size:14px 14px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Understand")
    c_u1, c_u2, c_u3 = st.columns(3)
    with c_u1:
        st.markdown(
            f"""
            <div class="card" style="border-left:4px solid #f59e0b;">
                <div><b>⚠️ Violation Agent — Law Enforcer</b></div>
                <div style="margin-top:6px;">Evidence strength: <b>{float(v.get("confidence",0))*100:.0f}%</b></div>
                <div style="margin-top:6px;color:#cbd5e1;">{v.get("output","No violation signal.")}</div>
                <div style="margin-top:6px;color:#fde68a;">Reasoning: {v.get("explanation","Risk-based assessment only.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_u2:
        st.markdown(
            f"""
            <div class="card" style="border-left:4px solid #a78bfa;">
                <div><b>📈 Prediction Agent — Forecaster</b></div>
                <div style="margin-top:6px;color:#ddd6fe;">{pr.get("output","Trend unavailable.")}</div>
                <div style="margin-top:6px;">Confidence: <b>{float(pr.get("confidence",0))*100:.0f}%</b></div>
                <div style="margin-top:6px;color:#cbd5e1;">Preparation: {pr.get("recommendation","Monitor trend.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_u3:
        st.markdown(
            f"""
            <div class="card" style="border-left:4px solid #f472b6;">
                <div><b>🧠 XAI Agent — Explainer</b></div>
                <div style="margin-top:6px;color:#fbcfe8;">{x.get("output","Factor analysis available.")}</div>
                <div style="margin-top:6px;color:#cbd5e1;">{x.get("explanation","Top factors shown below.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Decide")
    st.markdown(
        f"""
        <div class="card" style="border-left:6px solid #60a5fa;background:linear-gradient(135deg,rgba(29,78,216,0.25),rgba(15,23,42,0.92));">
            <div><b>🛡️ Supervisor Agent — Final Decision Core</b></div>
            <div style="margin-top:8px;font-size:1.1rem;"><b>Final decision:</b> {agentic.get("final_decision","-")}</div>
            <div style="margin-top:4px;"><b>Selected action:</b> {agentic.get("selected_action","-")}</div>
            <div style="margin-top:4px;"><b>Risk level:</b> {str(agentic.get("risk_level","-")).upper()}</div>
            <div style="margin-top:6px;color:#cbd5e1;">{agentic.get("final_recommendation","")}</div>
            <div style="margin-top:8px;">Consensus confidence: <b>{float(s.get("confidence",0))*100:.0f}%</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Act")
    c_a1, c_a2, c_a3 = st.columns(3)
    with c_a1:
        st.markdown(
            f"""
            <div class="card" style="border-left:4px solid #22c55e;">
                <div><b>🚦 Traffic Control Agent — Signal Operator</b></div>
                <div style="margin-top:6px;color:#86efac;">Signal logic: {t.get("recommendation","KEEP_CYCLE")}</div>
                <div style="margin-top:6px;color:#cbd5e1;">{t.get("output","Signal phase stable.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_a2:
        panel_bg = "linear-gradient(90deg, rgba(127,29,29,0.95), rgba(239,68,68,0.5))" if str(e.get("status","")).upper() == "OVERRIDE" else "rgba(15,23,42,0.9)"
        st.markdown(
            f"""
            <div class="card" style="border-left:4px solid #ef4444;background:{panel_bg};">
                <div><b>🚑 Emergency Agent — Priority Guardian</b></div>
                <div style="margin-top:6px;"><b>Emergency detected:</b> {"YES" if agentic.get("emergency_mode") else "NO"}</div>
                <div style="margin-top:4px;"><b>Override active:</b> {"YES" if str(e.get("status","")).upper()=="OVERRIDE" else "NO"}</div>
                <div style="margin-top:6px;color:#fee2e2;">{e.get("output","Emergency standby.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_a3:
        st.markdown(
            f"""
            <div class="card" style="border-left:4px solid #2dd4bf;">
                <div><b>🧭 Rerouting Agent — Navigator</b></div>
                <div style="margin-top:6px;color:#99f6e4;">Route action: {r.get("recommendation","no rerouting")}</div>
                <div style="margin-top:6px;color:#cbd5e1;">{r.get("output","No rerouting needed.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Agent Reasoning / XAI")
    xai = agentic.get("xai", {}) or {}
    factors = xai.get("factors", {}) if isinstance(xai, dict) else {}
    st.markdown(f"Decision chain: `{xai.get('decision_chain', 'Perception -> Supervisor')}`")
    for k in ["congestion", "vehicle_count", "waiting_time", "criticality_score", "emergency_presence", "traffic_light_state"]:
        if k in factors:
            st.progress(min(1.0, float(factors[k]) / 100.0), text=f"{k.replace('_', ' ').title()}: {float(factors[k]):.1f}%")

    st.markdown("</div>", unsafe_allow_html=True)

