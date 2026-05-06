import React from "react";

type AgentStatus = "ACTIVE" | "STANDBY" | "OVERRIDE" | "ALERT";
type RiskLevel = "low" | "medium" | "high" | "info";

export type AgentPayload = {
  agent: string;
  status: AgentStatus;
  confidence: number; // 0..1
  output: string;
  recommendation: string;
  risk_level: RiskLevel | string;
  explanation?: string;
  [key: string]: unknown;
};

export type AgenticResult = {
  agents: AgentPayload[];
  final_decision: string;
  final_recommendation: string;
  selected_action: string;
  emergency_mode: boolean;
  risk_level: RiskLevel | string;
  traffic_light_recommendation?: string;
  rerouting_recommendation?: string;
  xai?: {
    factors?: Record<string, number>;
    decision_chain?: string;
  };
};

const statusPill: Record<AgentStatus, string> = {
  ACTIVE: "bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-400/40",
  STANDBY: "bg-slate-500/20 text-slate-300 ring-1 ring-slate-400/40",
  OVERRIDE: "bg-orange-500/20 text-orange-300 ring-1 ring-orange-400/40",
  ALERT: "bg-rose-500/20 text-rose-300 ring-1 ring-rose-400/40",
};

const riskClass = (risk: string) => {
  const r = risk.toLowerCase();
  if (r === "high") return "text-rose-300";
  if (r === "medium") return "text-amber-300";
  if (r === "low") return "text-emerald-300";
  return "text-cyan-300";
};

const asPct = (v: number) => `${Math.round(Math.max(0, Math.min(1, v)) * 100)}%`;

const AgentShell: React.FC<{
  title: string;
  icon: string;
  accent: string;
  children: React.ReactNode;
}> = ({ title, icon, accent, children }) => (
  <section
    className={`relative rounded-2xl border ${accent} bg-slate-950/70 backdrop-blur-xl p-4 shadow-[0_0_0_1px_rgba(255,255,255,0.03),0_14px_30px_rgba(2,6,23,0.5)]`}
  >
    <header className="mb-3 flex items-center justify-between">
      <h3 className="text-sm sm:text-base font-semibold text-slate-100 tracking-wide">
        <span className="mr-2">{icon}</span>
        {title}
      </h3>
    </header>
    {children}
  </section>
);

const PerceptionPanel: React.FC<{ a: AgentPayload }> = ({ a }) => {
  const conf = asPct(a.confidence);
  return (
    <AgentShell title="Perception Agent — The Eyes" icon="👁️" accent="border-cyan-400/30">
      <div className="mb-2 grid grid-cols-2 gap-2 text-xs">
        <div className="rounded-lg bg-cyan-500/10 p-2 ring-1 ring-cyan-400/20">
          <p className="text-cyan-300">Scene scanned</p>
          <p className="text-slate-200 font-semibold">{String(a.output || "Visual context updated")}</p>
        </div>
        <div className="rounded-lg bg-slate-900/80 p-2 ring-1 ring-slate-700">
          <p className="text-slate-400">Confidence</p>
          <p className="text-slate-100 font-semibold">{conf}</p>
        </div>
      </div>
      <div className="relative h-24 rounded-xl bg-gradient-to-br from-cyan-500/15 to-blue-600/10 ring-1 ring-cyan-400/20 overflow-hidden">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(34,211,238,0.08)_1px,transparent_1px),linear-gradient(to_bottom,rgba(34,211,238,0.08)_1px,transparent_1px)] bg-[size:14px_14px]" />
        <div className="absolute inset-x-0 top-1/2 h-px bg-cyan-300/50 animate-pulse" />
        <div className="absolute bottom-2 left-2 text-[11px] text-cyan-200">Objects detected • Visual context updated</div>
      </div>
    </AgentShell>
  );
};

const TrafficControlPanel: React.FC<{ a: AgentPayload }> = ({ a }) => (
  <AgentShell title="Traffic Control Agent — Signal Operator" icon="🚦" accent="border-emerald-400/30">
    <div className="grid grid-cols-3 gap-2 text-xs mb-2">
      <div className="rounded-lg bg-emerald-500/10 p-2 ring-1 ring-emerald-400/20">GREEN</div>
      <div className="rounded-lg bg-amber-500/10 p-2 ring-1 ring-amber-400/20">YELLOW</div>
      <div className="rounded-lg bg-rose-500/10 p-2 ring-1 ring-rose-400/20">RED</div>
    </div>
    <div className="rounded-xl bg-slate-900/80 p-3 ring-1 ring-slate-700 text-sm">
      <p className="text-slate-300">Signal logic: <span className="text-slate-100 font-semibold">{String(a.recommendation || a.output)}</span></p>
      <p className="text-slate-400 mt-1">Decision impact: adaptive cycle + queue balancing</p>
    </div>
  </AgentShell>
);

const EmergencyPanel: React.FC<{ a: AgentPayload }> = ({ a }) => {
  const override = a.status === "OVERRIDE";
  return (
    <AgentShell
      title="Emergency Agent — Priority Guardian"
      icon="🚑"
      accent={override ? "border-rose-400/60" : "border-slate-600/40"}
    >
      <div className={`rounded-xl p-3 ring-1 ${override ? "bg-rose-500/20 ring-rose-300/50 animate-pulse" : "bg-slate-900/80 ring-slate-700"}`}>
        <p className="text-sm font-semibold text-slate-100">Emergency detected: {override ? "YES" : "NO"}</p>
        <p className="text-xs text-slate-300 mt-1">Override active: {override ? "YES" : "NO"}</p>
        <p className="text-xs mt-2 text-slate-200">{String(a.output || "Emergency corridor status nominal")}</p>
      </div>
    </AgentShell>
  );
};

const ViolationPanel: React.FC<{ a: AgentPayload }> = ({ a }) => (
  <AgentShell title="Violation Agent — Law Enforcer" icon="⚠️" accent="border-amber-400/30">
    <div className="rounded-xl bg-slate-900/80 p-3 ring-1 ring-amber-300/20 text-xs">
      <p className="text-amber-200">Evidence inspection</p>
      <p className="text-slate-200 mt-1">{String(a.output || "No strong violation signal")}</p>
      <div className="mt-2 h-2 rounded bg-slate-700 overflow-hidden">
        <div className="h-full bg-amber-400/80" style={{ width: asPct(a.confidence) }} />
      </div>
      <p className="mt-1 text-slate-400">Confirmation threshold: probabilistic</p>
    </div>
  </AgentShell>
);

const PredictionPanel: React.FC<{ a: AgentPayload }> = ({ a }) => (
  <AgentShell title="Prediction Agent — Forecaster" icon="📈" accent="border-violet-400/30">
    <div className="rounded-xl bg-violet-500/10 p-3 ring-1 ring-violet-300/20">
      <p className="text-violet-200 text-sm font-medium">Trend forecast</p>
      <p className="text-slate-100 text-sm mt-1">{String(a.output || "STABLE")}</p>
      <div className="mt-3 grid grid-cols-3 gap-2 text-[11px] text-slate-300">
        <div className="rounded bg-slate-900/70 p-2">30s</div>
        <div className="rounded bg-slate-900/70 p-2">60s</div>
        <div className="rounded bg-slate-900/70 p-2">120s</div>
      </div>
    </div>
  </AgentShell>
);

const ReroutingPanel: React.FC<{ a: AgentPayload }> = ({ a }) => (
  <AgentShell title="Rerouting Agent — Navigator" icon="🧭" accent="border-teal-400/30">
    <div className="rounded-xl bg-slate-900/80 p-3 ring-1 ring-teal-300/20 text-xs">
      <p className="text-teal-200">Route graph status</p>
      <p className="mt-1 text-slate-100">{String(a.recommendation || "optional rerouting")}</p>
      <div className="mt-3 flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-teal-300" />
        <span className="h-0.5 w-10 bg-teal-300/70" />
        <span className="h-2 w-2 rounded-full bg-slate-300" />
        <span className="h-0.5 w-10 bg-slate-500" />
        <span className="h-2 w-2 rounded-full bg-amber-300" />
      </div>
    </div>
  </AgentShell>
);

const XAIPanel: React.FC<{ a: AgentPayload; xai?: AgenticResult["xai"] }> = ({ a, xai }) => {
  const factors = xai?.factors || {};
  const top = Object.entries(factors).sort((x, y) => (y[1] ?? 0) - (x[1] ?? 0)).slice(0, 3);
  return (
    <AgentShell title="XAI Agent — Explainer" icon="🧠" accent="border-pink-400/30">
      <p className="text-xs text-pink-200 mb-2">Cause → Effect reasoning</p>
      <div className="space-y-2">
        {top.map(([k, v]) => (
          <div key={k}>
            <div className="flex justify-between text-[11px] text-slate-300">
              <span>{k.replace(/_/g, " ")}</span>
              <span>{Math.round(v)}%</span>
            </div>
            <div className="h-1.5 rounded bg-slate-700 overflow-hidden">
              <div className="h-full bg-pink-400/80" style={{ width: `${Math.min(100, Math.max(0, v))}%` }} />
            </div>
          </div>
        ))}
      </div>
      <p className="mt-2 text-xs text-slate-300">{String(a.output || a.explanation || "Human-readable rationale available.")}</p>
    </AgentShell>
  );
};

const SupervisorPanel: React.FC<{ agentic: AgenticResult; a: AgentPayload | undefined }> = ({ agentic, a }) => (
  <section className="relative rounded-3xl border border-blue-300/30 bg-gradient-to-br from-slate-900 via-blue-950/70 to-slate-900 p-5 shadow-[0_0_30px_rgba(59,130,246,0.18)]">
    <div className="flex items-start justify-between gap-3">
      <div>
        <h2 className="text-lg font-bold text-slate-100">🛡️ Supervisor Agent — Final Decision Core</h2>
        <p className="text-sm text-blue-200 mt-1">Central command module</p>
      </div>
      {a && (
        <span className={`px-2 py-1 rounded-full text-xs ${statusPill[a.status] || statusPill.STANDBY}`}>
          {a.status}
        </span>
      )}
    </div>
    <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
      <div className="rounded-xl bg-slate-900/80 p-3 ring-1 ring-blue-300/20">
        <p className="text-xs text-slate-400">Final decision</p>
        <p className="text-lg font-semibold text-slate-100">{agentic.final_decision}</p>
      </div>
      <div className="rounded-xl bg-slate-900/80 p-3 ring-1 ring-blue-300/20">
        <p className="text-xs text-slate-400">Selected action</p>
        <p className="text-lg font-semibold text-slate-100">{agentic.selected_action}</p>
      </div>
      <div className="rounded-xl bg-slate-900/80 p-3 ring-1 ring-blue-300/20">
        <p className="text-xs text-slate-400">Global risk level</p>
        <p className={`text-lg font-semibold ${riskClass(String(agentic.risk_level))}`}>{String(agentic.risk_level).toUpperCase()}</p>
      </div>
      <div className="rounded-xl bg-slate-900/80 p-3 ring-1 ring-blue-300/20">
        <p className="text-xs text-slate-400">Consensus / rationale</p>
        <p className="text-sm text-slate-100">{agentic.final_recommendation}</p>
      </div>
    </div>
  </section>
);

const pickAgent = (agents: AgentPayload[], name: string) => agents.find((a) => a.agent === name);

export const AgenticAIControlCenter: React.FC<{ data: AgenticResult }> = ({ data }) => {
  const agents = data.agents || [];
  const perception = pickAgent(agents, "Perception Agent");
  const traffic = pickAgent(agents, "Traffic Control Agent");
  const emergency = pickAgent(agents, "Emergency Agent");
  const violation = pickAgent(agents, "Violation Agent");
  const prediction = pickAgent(agents, "Prediction Agent");
  const rerouting = pickAgent(agents, "Rerouting Agent");
  const xai = pickAgent(agents, "XAI Agent");
  const supervisor = pickAgent(agents, "Supervisor Agent");

  return (
    <div className="w-full rounded-3xl border border-slate-700/60 bg-slate-950/70 p-4 sm:p-6 backdrop-blur-xl">
      <div className="mb-5">
        <h1 className="text-xl sm:text-2xl font-bold text-slate-100">Agentic AI Control Center</h1>
        <p className="text-slate-400 text-sm mt-1">Smart City multi-agent command flow: Sense → Understand → Decide → Act</p>
      </div>

      <div className="mb-6">
        <p className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Sense</p>
        {perception && <PerceptionPanel a={perception} />}
      </div>

      <div className="my-4 text-center text-slate-500">↓ information flow ↓</div>

      <div className="mb-6">
        <p className="text-xs uppercase tracking-wider text-violet-300 mb-2">Understand</p>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {violation && <ViolationPanel a={violation} />}
          {prediction && <PredictionPanel a={prediction} />}
          {xai && <XAIPanel a={xai} xai={data.xai} />}
        </div>
      </div>

      <div className="my-4 text-center text-slate-500">↓ decision core ↓</div>

      <div className="mb-6">
        <SupervisorPanel agentic={data} a={supervisor} />
      </div>

      <div className="my-4 text-center text-slate-500">↓ execution layer ↓</div>

      <div>
        <p className="text-xs uppercase tracking-wider text-emerald-300 mb-2">Act</p>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {traffic && <TrafficControlPanel a={traffic} />}
          {emergency && <EmergencyPanel a={emergency} />}
          {rerouting && <ReroutingPanel a={rerouting} />}
        </div>
      </div>
    </div>
  );
};

export default AgenticAIControlCenter;

