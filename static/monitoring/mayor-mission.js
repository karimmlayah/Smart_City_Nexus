/**
 * Mission Mayor: Save the City — interactive decision game
 */
const root = document.getElementById("mmMayorRoot");
if (!root) throw new Error("Mayor mission root missing");

const cfg = {
  challengesUrl: root.dataset.challengesUrl,
  classicUrl: root.dataset.classicUrl,
  medinamindUrl: root.dataset.medinamindUrl,
  finishUrl: root.dataset.finishUrl,
  certificateUrl: root.dataset.certificateUrl,
  totalRounds: parseInt(root.dataset.totalRounds || "5", 10),
  roundSeconds: parseInt(root.dataset.roundSeconds || "55", 10),
};

const game = {
  state: "welcome",
  challenges: [],
  roundOrder: [],
  roundIndex: 0,
  currentChallenge: null,
  score: 50,
  cityHealth: 70,
  citizensProtected: 0,
  citizensImpacted: 0,
  damageAvoidedTND: 0,
  damageCostTND: 0,
  responseTimeSaved: 0,
  modulesUsed: [],
  decisionHistory: [],
  avatarDataUrl: null,
  hasFace: false,
  stream: null,
  faceDetector: null,
  muted: false,
  audioReady: false,
  busy: false,
  roundTimeLeft: cfg.roundSeconds,
  timerActive: false,
  timerInterval: null,
};

const els = {
  hud: document.getElementById("mmMayorHud"),
  steps: document.getElementById("mmMayorSteps"),
  game: document.getElementById("mmMayorGame"),
  history: document.getElementById("mmMayorHistory"),
  historyList: document.getElementById("mmHistoryList"),
  startBtn: document.getElementById("mmMayorStartBtn"),
  muteBtn: document.getElementById("mmMayorMuteBtn"),
  hudRound: document.getElementById("mmHudRound"),
  hudTimer: document.getElementById("mmHudTimer"),
  hudTimerWrap: document.getElementById("mmHudTimerWrap"),
  hudScore: document.getElementById("mmHudScore"),
  hudHealth: document.getElementById("mmHudHealth"),
  hudStars: document.getElementById("mmHudStars"),
  sideHealth: document.getElementById("mmSideHealth"),
  sideScore: document.getElementById("mmSideScore"),
  sideStars: document.getElementById("mmSideStars"),
  sideTimer: document.getElementById("mmSideTimer"),
  healthGauge: document.getElementById("mmHealthGauge"),
  sideCitizens: document.getElementById("mmSideCitizens"),
  sideImpacted: document.getElementById("mmSideImpacted"),
  sideDamage: document.getElementById("mmSideDamage"),
  sideDamageCost: document.getElementById("mmSideDamageCost"),
  sideTime: document.getElementById("mmSideTime"),
  video: document.getElementById("mmMayorVideo"),
  scanPlaceholder: document.getElementById("mmMayorScanPlaceholder"),
  scanSection: document.getElementById("mmMayorScanSection"),
  scanBtn: document.getElementById("mmMayorScanBtn"),
  skipFaceBtn: document.getElementById("mmMayorSkipFaceBtn"),
  retakeBtn: document.getElementById("mmMayorRetakeBtn"),
  scanStatus: document.getElementById("mmMayorScanStatus"),
  avatarFaceWrap: document.getElementById("mmAvatarFaceWrap"),
  avatarFallbackWrap: document.getElementById("mmAvatarFallbackWrap"),
  avatarImg: document.getElementById("mmMayorAvatarImg"),
  avatarFallback: document.getElementById("mmMayorAvatarFallback"),
  identityBadge: document.getElementById("mmMayorIdentityBadge"),
  nameInput: document.getElementById("mmMayorNameInput"),
  crisisAlert: document.getElementById("mmCrisisAlert"),
  crisisImg: document.getElementById("mmCrisisImg"),
  crisisPlaceholder: document.getElementById("mmCrisisPlaceholder"),
  damageOverlay: document.getElementById("mmDamageOverlay"),
  crackOverlay: document.getElementById("mmCrackOverlay"),
  aiOverlay: document.getElementById("mmAiOverlay"),
  crisisScan: document.getElementById("mmCrisisScan"),
  aiResultPanel: document.getElementById("mmAiResultPanel"),
  aiModule: document.getElementById("mmAiModule"),
  aiDetected: document.getElementById("mmAiDetected"),
  aiConfidence: document.getElementById("mmAiConfidence"),
  aiSeverity: document.getElementById("mmAiSeverity"),
  aiRecommendation: document.getElementById("mmAiRecommendation"),
  aiFallback: document.getElementById("mmAiFallback"),
  classicResultPanel: document.getElementById("mmClassicResultPanel"),
  classicDelay: document.getElementById("mmClassicDelay"),
  classicCitizens: document.getElementById("mmClassicCitizens"),
  classicDamage: document.getElementById("mmClassicDamage"),
  mapMarkers: document.getElementById("mmMayorMapMarkers"),
  challengeTitle: document.getElementById("mmChallengeTitle"),
  challengeStory: document.getElementById("mmChallengeStory"),
  challengeZone: document.getElementById("mmChallengeZone"),
  challengeSeverity: document.getElementById("mmChallengeSeverity"),
  classicBtn: document.getElementById("mmClassicBtn"),
  aiBtn: document.getElementById("mmAiBtn"),
  roundSummary: document.getElementById("mmRoundSummary"),
  roundSummaryList: document.getElementById("mmRoundSummaryList"),
  nextRoundBtn: document.getElementById("mmNextRoundBtn"),
  overlay: document.getElementById("mmConsequenceOverlay"),
  overlayTitle: document.getElementById("mmOverlayTitle"),
  overlayLines: document.getElementById("mmOverlayLines"),
  overlayMetrics: document.getElementById("mmOverlayMetrics"),
  finalScreen: document.getElementById("mmFinalScreen"),
  confetti: document.getElementById("mmConfetti"),
  trophy: document.getElementById("mmTrophy"),
  finalAvatar: document.getElementById("mmFinalAvatar"),
  finalTitle: document.getElementById("mmFinalTitle"),
  finalSubtitle: document.getElementById("mmFinalSubtitle"),
  finalStars: document.getElementById("mmFinalStars"),
  finalMetrics: document.getElementById("mmFinalMetrics"),
  certBtn: document.getElementById("mmCertBtn"),
  restartBtn: document.getElementById("mmRestartBtn"),
};

const UI_STEPS = [
  "welcome", "face_scan", "mayor_ready", "challenge_intro", "decision", "consequence", "final_result",
];

const audioCtx = typeof AudioContext !== "undefined" ? new AudioContext() : null;

function ensureAudio() {
  if (!audioCtx || game.muted) return;
  if (audioCtx.state === "suspended") audioCtx.resume();
  game.audioReady = true;
}

function playTone(freq, dur, type = "sine", gain = 0.15) {
  if (!audioCtx || game.muted || !game.audioReady) return;
  const osc = audioCtx.createOscillator();
  const g = audioCtx.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  g.gain.setValueAtTime(gain, audioCtx.currentTime);
  g.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
  osc.connect(g);
  g.connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + dur);
}

function playAlert() {
  playTone(880, 0.12, "square", 0.12);
  setTimeout(() => playTone(660, 0.15, "square", 0.1), 140);
  setTimeout(() => playTone(880, 0.2, "square", 0.12), 320);
}

function playFailure() {
  playTone(220, 0.3, "sawtooth", 0.14);
  setTimeout(() => playTone(165, 0.4, "sawtooth", 0.12), 200);
}

function playSuccess() {
  [523, 659, 784, 1047].forEach((f, i) => setTimeout(() => playTone(f, 0.18, "sine", 0.1), i * 90));
}

function playVictory() {
  [392, 523, 659, 784, 1047].forEach((f, i) => setTimeout(() => playTone(f, 0.25, "triangle", 0.12), i * 120));
}

function playWarning() {
  playTone(440, 0.15, "square", 0.14);
  setTimeout(() => playTone(330, 0.2, "square", 0.12), 120);
}

function setState(next) {
  game.state = next;
  const uiStep = { simulating: "decision", next_round: "challenge_intro" }[next] || next;
  const idx = UI_STEPS.indexOf(uiStep);
  els.steps.querySelectorAll("li").forEach((li) => {
    const si = UI_STEPS.indexOf(li.dataset.step);
    li.classList.toggle("is-active", li.dataset.step === uiStep);
    li.classList.toggle("is-done", si >= 0 && si < idx);
  });
  document.querySelectorAll("[data-screen]").forEach((el) => {
    const screens = el.dataset.screen.split(" ");
    el.hidden = !screens.some((s) => s === next || s === uiStep);
  });
  if (next !== "welcome" && next !== "final_result") {
    els.hud.hidden = false;
    els.game.hidden = false;
    els.history.hidden = false;
  }
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function starsFromScore(score) {
  if (score >= 90) return 5;
  if (score >= 75) return 4;
  if (score >= 60) return 3;
  if (score >= 40) return 2;
  return 1;
}

function renderStars(el, count) {
  el.textContent = "★".repeat(count) + "☆".repeat(5 - count);
}

function formatTnd(n) {
  const v = Number(n || 0);
  if (v >= 1_000_000) return `≈ ${(v / 1_000_000).toFixed(2)} million TND`;
  return `${Math.round(v).toLocaleString("fr-FR")} TND`;
}

function animateCounter(el, from, to, formatter = (v) => String(Math.round(v))) {
  const start = performance.now();
  const dur = 600;
  function tick(now) {
    const t = Math.min(1, (now - start) / dur);
    const val = from + (to - from) * t;
    el.textContent = formatter(val);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function updateAvatarDisplay() {
  if (game.hasFace && game.avatarDataUrl) {
    els.avatarFaceWrap.hidden = false;
    els.avatarFallbackWrap.hidden = true;
    els.avatarImg.src = game.avatarDataUrl;
  } else {
    els.avatarFaceWrap.hidden = true;
    els.avatarFallbackWrap.hidden = false;
  }
}

function updateHud(animate = false) {
  const roundLabel = `${Math.min(game.roundIndex + 1, cfg.totalRounds)} / ${cfg.totalRounds}`;
  els.hudRound.textContent = roundLabel;
  els.hudTimer.textContent = game.roundTimeLeft;
  els.sideTimer.textContent = `${game.roundTimeLeft}s`;

  els.hudTimerWrap.classList.toggle("is-low", game.timerActive && game.roundTimeLeft <= 10 && game.roundTimeLeft > 5);
  els.hudTimerWrap.classList.toggle("is-critical", game.timerActive && game.roundTimeLeft <= 5);

  if (animate) {
    animateCounter(els.hudScore, parseInt(els.hudScore.textContent, 10) || game.score, game.score);
    animateCounter(els.sideScore, parseInt(els.sideScore.textContent, 10) || game.score, game.score);
    animateCounter(els.hudHealth, parseInt(els.hudHealth.textContent, 10) || game.cityHealth, game.cityHealth);
    animateCounter(els.sideHealth, parseInt(els.sideHealth.textContent, 10) || game.cityHealth, game.cityHealth, (v) => `${Math.round(v)}`);
  } else {
    els.hudScore.textContent = game.score;
    els.sideScore.textContent = game.score;
    els.hudHealth.textContent = game.cityHealth;
    els.sideHealth.textContent = game.cityHealth;
  }

  els.healthGauge.style.width = `${game.cityHealth}%`;
  els.sideCitizens.textContent = game.citizensProtected.toLocaleString();
  els.sideImpacted.textContent = game.citizensImpacted.toLocaleString();
  els.sideDamage.textContent = formatTnd(game.damageAvoidedTND);
  els.sideDamageCost.textContent = formatTnd(game.damageCostTND);
  els.sideTime.textContent = `${game.responseTimeSaved} min`;
  renderStars(els.hudStars, starsFromScore(game.score));
  renderStars(els.sideStars, starsFromScore(game.score));
}

function getCsrf() {
  const m = document.cookie.match(/csrftoken=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : "";
}

/* ── Timer ── */
function stopRoundTimer() {
  game.timerActive = false;
  if (game.timerInterval) {
    clearInterval(game.timerInterval);
    game.timerInterval = null;
  }
  els.hudTimerWrap.classList.remove("is-low", "is-critical");
}

function startRoundTimer() {
  stopRoundTimer();
  game.roundTimeLeft = cfg.roundSeconds;
  game.timerActive = true;
  updateHud();

  game.timerInterval = setInterval(() => {
    if (!game.timerActive || game.busy || game.state !== "decision") return;
    game.roundTimeLeft -= 1;
    updateHud();
    if (game.roundTimeLeft === 10) playWarning();
    if (game.roundTimeLeft <= 0) {
      stopRoundTimer();
      makeDecision("classic", { timedOut: true });
    }
  }, 1000);
}

/* ── Challenges ── */
async function loadChallenges() {
  const res = await fetch(cfg.challengesUrl);
  const data = await res.json();
  game.challenges = data.challenges || [];
  shuffleRoundOrder();
}

function shuffleRoundOrder() {
  const ids = game.challenges.map((c) => c.id);
  for (let i = ids.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [ids[i], ids[j]] = [ids[j], ids[i]];
  }
  game.roundOrder = ids.slice(0, cfg.totalRounds);
  game.roundIndex = 0;
}

function currentChallenge() {
  const id = game.roundOrder[game.roundIndex];
  return game.challenges.find((c) => c.id === id) || game.challenges[0];
}

const markerIcons = {
  traffic: "🚦", fire: "🔥", emergency: "🚑", road: "🛣️",
  uav: "🛸", waste: "🗑️", citizens: "👥", alert: "⚠️",
};

function renderMapMarkers(challenge) {
  els.mapMarkers.innerHTML = "";
  const markers = (challenge && challenge.map_markers) || [];
  const positions = [[18, 22], [72, 30], [45, 65], [80, 70], [30, 78], [60, 18]];
  markers.forEach((m, i) => {
    const el = document.createElement("span");
    el.className = "mm-mayor-marker";
    el.textContent = markerIcons[m] || "📍";
    el.style.left = `${positions[i % positions.length][0]}%`;
    el.style.top = `${positions[i % positions.length][1]}%`;
    els.mapMarkers.appendChild(el);
  });
}

function resetCrisisMedia() {
  els.damageOverlay.hidden = true;
  els.crackOverlay.hidden = true;
  els.aiOverlay.hidden = true;
  els.crisisScan.hidden = true;
  els.aiResultPanel.hidden = true;
  els.classicResultPanel.hidden = true;
  els.crisisImg.classList.remove("is-worse", "is-ai-result");
  root.classList.remove("mm-mayor--shake");
}

function showChallengeIntro() {
  resetCrisisMedia();
  stopRoundTimer();
  const c = currentChallenge();
  game.currentChallenge = c;
  if (!c) return;

  els.challengeTitle.textContent = c.title;
  els.challengeStory.textContent = c.story;
  els.challengeZone.textContent = c.zone;
  els.challengeSeverity.textContent = c.severity;

  els.crisisImg.src = c.problem_image_url;
  els.crisisImg.hidden = false;
  els.crisisPlaceholder.hidden = true;
  els.crisisAlert.hidden = false;

  renderMapMarkers(c);
  els.classicBtn.disabled = false;
  els.aiBtn.disabled = false;
  els.roundSummary.hidden = true;

  setState("challenge_intro");
  playAlert();
  setTimeout(() => {
    setState("decision");
    startRoundTimer();
  }, 1200);
}

/* ── Face scan ── */
async function startCamera() {
  try {
    game.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    els.video.srcObject = game.stream;
    els.video.hidden = false;
    els.scanPlaceholder.hidden = true;
    els.scanStatus.textContent = "Camera ready — align your face";
    if ("FaceDetector" in window) {
      try { game.faceDetector = new FaceDetector({ fastMode: true, maxDetectedFaces: 1 }); }
      catch { game.faceDetector = null; }
    }
  } catch {
    els.scanStatus.textContent = "Camera denied — use generic avatar";
  }
}

async function captureFace() {
  const video = els.video;
  if (!video.videoWidth) {
    els.scanStatus.textContent = "Start camera first or skip";
    return;
  }
  els.scanStatus.textContent = "Scanning…";
  const canvas = document.createElement("canvas");
  const w = video.videoWidth, h = video.videoHeight;
  canvas.width = w; canvas.height = h;
  canvas.getContext("2d").drawImage(video, 0, 0, w, h);

  let box = null;
  if (game.faceDetector) {
    try {
      const faces = await game.faceDetector.detect(video);
      if (faces && faces.length) box = faces[0].boundingBox;
    } catch { box = null; }
  }

  const size = Math.min(w, h) * (box ? Math.max(box.width, box.height) / Math.min(w, h) * 1.35 : 0.55);
  const cx = box ? box.x + box.width / 2 : w / 2;
  const cy = box ? box.y + box.height / 2 : h / 2;
  const crop = document.createElement("canvas");
  crop.width = crop.height = 240;
  const cctx = crop.getContext("2d");
  cctx.beginPath();
  cctx.arc(120, 120, 120, 0, Math.PI * 2);
  cctx.closePath();
  cctx.clip();
  cctx.drawImage(canvas, cx - size / 2, cy - size / 2, size, size, 0, 0, 240, 240);

  game.avatarDataUrl = crop.toDataURL("image/jpeg", 0.92);
  game.hasFace = true;
  updateAvatarDisplay();
  els.identityBadge.hidden = false;
  els.retakeBtn.hidden = false;
  els.scanSection.hidden = true;
  els.scanStatus.textContent = "Mayor identity confirmed";
  setState("mayor_ready");
  setTimeout(beginFirstRound, 800);
}

function skipFace() {
  game.avatarDataUrl = null;
  game.hasFace = false;
  updateAvatarDisplay();
  els.identityBadge.hidden = true;
  els.scanSection.hidden = true;
  setState("mayor_ready");
  setTimeout(beginFirstRound, 600);
}

function beginFirstRound() {
  showChallengeIntro();
}

/* ── Decision ── */
async function makeDecision(mode, opts = {}) {
  if (game.busy) return;
  if (game.state !== "decision" && !opts.timedOut) return;

  stopRoundTimer();
  game.busy = true;
  ensureAudio();
  els.classicBtn.disabled = true;
  els.aiBtn.disabled = true;
  els.classicBtn.classList.toggle("is-selected", mode === "classic");
  els.aiBtn.classList.toggle("is-selected", mode === "medinamind");
  setState("simulating");

  const c = game.currentChallenge;
  const url = mode === "classic" ? cfg.classicUrl : cfg.medinamindUrl;
  const payload = { challenge_id: c.id, inputs: {} };
  if (opts.timedOut) payload.timed_out = true;

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRFToken": getCsrf() },
      body: JSON.stringify(payload),
    });
    const result = await res.json();
    if (!result.success) {
      alert(result.error || "Simulation failed");
      game.busy = false;
      els.classicBtn.disabled = false;
      els.aiBtn.disabled = false;
      startRoundTimer();
      setState("decision");
      return;
    }
    applyResult(mode, result);
    showConsequenceOverlay(mode, result);
  } catch {
    alert("Network error — try again");
    game.busy = false;
    els.classicBtn.disabled = false;
    els.aiBtn.disabled = false;
    startRoundTimer();
    setState("decision");
  }
}

function applyResult(mode, result) {
  const scoreDelta = result.score_delta || 0;
  const healthDelta = result.city_health_delta || 0;
  game.score = clamp(game.score + scoreDelta, 0, 100);
  game.cityHealth = clamp(game.cityHealth + healthDelta, 0, 100);

  if (mode === "medinamind") {
    game.citizensProtected += result.citizens_protected || 0;
    game.damageAvoidedTND += result.damage_avoided_tnd || 0;
    game.responseTimeSaved += result.response_time_saved_min || 0;
    if (result.module_used && !game.modulesUsed.includes(result.module_label || result.module_used)) {
      game.modulesUsed.push(result.module_label || result.module_used);
    }
    showAiResult(result);
    if (result.annotated_image_url) {
      els.crisisImg.src = result.annotated_image_url;
      els.crisisImg.classList.add("is-ai-result");
    }
    els.aiOverlay.hidden = false;
    els.crisisScan.hidden = false;
    els.mapMarkers.querySelectorAll(".mm-mayor-marker").forEach((m) => m.classList.add("is-resolved"));
  } else {
    game.citizensImpacted += result.citizens_impacted || 0;
    game.damageCostTND += result.damage_increase_tnd || result.damage_cost_tnd || 0;
    showClassicResult(result);
    els.crisisImg.classList.add("is-worse");
    els.damageOverlay.hidden = false;
    els.crackOverlay.hidden = false;
    root.classList.add("mm-mayor--shake");
    els.mapMarkers.querySelectorAll(".mm-mayor-marker").forEach((m) => m.classList.add("is-danger"));
  }

  game.decisionHistory.push({
    round: game.roundIndex + 1,
    challenge: game.currentChallenge.title,
    mode,
    score_delta: scoreDelta,
  });
  renderHistory();
  updateHud(true);
}

function showAiResult(r) {
  els.aiResultPanel.hidden = false;
  els.aiModule.textContent = r.module_label || r.module_used || "—";
  els.aiDetected.textContent = r.detected_issue || "—";
  els.aiConfidence.textContent = r.confidence != null ? `${Math.round(r.confidence * 100)}%` : "—";
  els.aiSeverity.textContent = r.severity || "—";
  els.aiRecommendation.textContent = r.recommendation || "—";
  if (r.fallback_used) {
    els.aiFallback.hidden = false;
    els.aiFallback.textContent = r.fallback_message || "Real model unavailable — demo fallback used";
  } else {
    els.aiFallback.hidden = true;
  }
}

function showClassicResult(r) {
  els.classicResultPanel.hidden = false;
  els.classicDelay.textContent = `+${r.response_delay_min || 0} min`;
  els.classicCitizens.textContent = (r.citizens_impacted || 0).toLocaleString();
  els.classicDamage.textContent = formatTnd(r.damage_increase_tnd || r.damage_cost_tnd || 0);
}

function showConsequenceOverlay(mode, result) {
  setState("consequence");
  const isNeg = mode === "classic";
  if (isNeg) playFailure(); else playSuccess();

  els.overlay.className = `mm-mayor-overlay mm-mayor-overlay--${isNeg ? "negative" : "positive"}`;
  els.overlayTitle.textContent = result.headline || (isNeg ? "Delayed Response" : "AI Intervention Successful");
  els.overlayLines.innerHTML = (result.sublines || []).map((s) => `<li>${s}</li>`).join("");
  const metric = isNeg
    ? `<span>Score ${result.score_delta}</span><span>Health ${result.city_health_delta}</span>`
    : `<span>Score +${result.score_delta}</span><span>Health +${result.city_health_delta}</span>`;
  els.overlayMetrics.innerHTML = metric;
  els.overlay.hidden = false;

  setTimeout(() => {
    els.overlay.hidden = true;
    game.busy = false;
    showRoundSummary(mode, result);
  }, 2800);
}

function showRoundSummary(mode, result) {
  els.roundSummary.hidden = false;
  const items = mode === "classic"
    ? [
      `Delay: +${result.response_delay_min} min`,
      `Citizens impacted: ${(result.citizens_impacted || 0).toLocaleString()}`,
      `Damage cost: ${formatTnd(result.damage_increase_tnd || result.damage_cost_tnd || 0)}`,
      `Score: ${result.score_delta}`,
    ]
    : [
      `Detected: ${result.detected_issue}`,
      `Citizens protected: ${(result.citizens_protected || 0).toLocaleString()}`,
      `Damage avoided: ${formatTnd(result.damage_avoided_tnd || 0)}`,
      `Score: +${result.score_delta}`,
    ];
  els.roundSummaryList.innerHTML = items.map((t) => `<li>${t}</li>`).join("");
  els.nextRoundBtn.textContent = game.roundIndex + 1 >= cfg.totalRounds ? "View Final Result" : "Next Crisis";
  setState("next_round");
}

function renderHistory() {
  els.historyList.innerHTML = game.decisionHistory.map((d) => {
    const cls = d.mode === "classic" ? "is-classic" : "is-ai";
    const label = d.mode === "classic" ? "Classic" : "MedinaMind";
    return `<li class="${cls}">R${d.round}: ${label} (${d.score_delta > 0 ? "+" : ""}${d.score_delta})</li>`;
  }).join("");
}

function nextRound() {
  game.roundIndex += 1;
  if (game.roundIndex >= cfg.totalRounds) {
    finishGame();
    return;
  }
  els.classicBtn.classList.remove("is-selected");
  els.aiBtn.classList.remove("is-selected");
  showChallengeIntro();
}

async function finishGame() {
  stopRoundTimer();
  setState("final_result");
  els.game.hidden = true;

  const payload = {
    score: game.score,
    city_health: game.cityHealth,
    citizens_protected: game.citizensProtected,
    citizens_impacted: game.citizensImpacted,
    damage_avoided_tnd: game.damageAvoidedTND,
    damage_cost_tnd: game.damageCostTND,
    response_time_saved: game.responseTimeSaved,
    modules_used: game.modulesUsed,
    decision_history: game.decisionHistory,
    rounds_played: cfg.totalRounds,
  };

  let finish = {};
  try {
    const res = await fetch(cfg.finishUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRFToken": getCsrf() },
      body: JSON.stringify(payload),
    });
    finish = await res.json();
  } catch {
    finish = { win: game.score >= 70, title: game.score >= 70 ? "CITY SAVED!" : "CITY AT RISK" };
  }

  ensureAudio();
  if (finish.win) playVictory();

  els.finalTitle.textContent = finish.title || (finish.win ? "CITY SAVED!" : "CITY AT RISK");
  els.finalSubtitle.textContent = finish.subtitle ||
    (finish.win ? `Congratulations, ${mayorName()}!` : "Try MedinaMind AI to improve city safety.");
  renderStars(els.finalStars, finish.stars || starsFromScore(game.score));

  if (game.hasFace && game.avatarDataUrl) {
    els.finalAvatar.src = game.avatarDataUrl;
    els.finalAvatar.hidden = false;
  } else {
    els.finalAvatar.hidden = true;
  }
  els.trophy.hidden = !finish.win;

  els.finalMetrics.innerHTML = [
    ["Final Score", `${game.score} / 100`],
    ["City Health", `${game.cityHealth}%`],
    ["Citizens Protected", game.citizensProtected.toLocaleString()],
    ["Damage Avoided", formatTnd(game.damageAvoidedTND)],
    ["Time Saved", `${game.responseTimeSaved} min`],
    ["AI Modules Used", game.modulesUsed.join(", ") || "—"],
  ].map(([k, v]) => `<li><span>${k}</span><strong>${v}</strong></li>`).join("");

  els.finalScreen.hidden = false;
  if (finish.win) launchConfetti();
}

function mayorName() {
  return els.nameInput.value.trim() || "Mayor";
}

function launchConfetti() {
  els.confetti.innerHTML = "";
  for (let i = 0; i < 60; i++) {
    const p = document.createElement("span");
    p.style.left = `${Math.random() * 100}%`;
    p.style.animationDelay = `${Math.random() * 0.8}s`;
    p.style.background = ["#00aeda", "#f59e0b", "#22c55e", "#a78bfa"][i % 4];
    els.confetti.appendChild(p);
  }
}

async function downloadCertificate() {
  const payload = {
    game_state: {
      score: game.score,
      city_health: game.cityHealth,
      citizens_protected: game.citizensProtected,
      damage_avoided_tnd: game.damageAvoidedTND,
      response_time_saved: game.responseTimeSaved,
      modules_used: game.modulesUsed,
    },
    mayor_name: mayorName(),
  };
  const res = await fetch(cfg.certificateUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-CSRFToken": getCsrf() },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!data.ok) return;
  const cert = data.certificate;
  const text = [
    cert.title,
    `Mayor: ${cert.mayor_name}`,
    `Final Score: ${cert.final_score}/100`,
    `City Health: ${cert.city_health}%`,
    `Stars: ${cert.stars}/5`,
    `Citizens Protected: ${cert.citizens_protected}`,
    `Damage Avoided: ${formatTnd(cert.damage_avoided_tnd || 0)}`,
    `Time Saved: ${cert.response_time_saved_min} min`,
    `Modules: ${(cert.modules_used || []).join(", ")}`,
    `Certificate ID: ${cert.certificate_id}`,
  ].join("\n");
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `medinamind-mayor-certificate-${cert.certificate_id}.txt`;
  a.click();
}

function restartMission() {
  stopRoundTimer();
  game.score = 50;
  game.cityHealth = 70;
  game.citizensProtected = 0;
  game.citizensImpacted = 0;
  game.damageAvoidedTND = 0;
  game.damageCostTND = 0;
  game.responseTimeSaved = 0;
  game.modulesUsed = [];
  game.decisionHistory = [];
  game.roundIndex = 0;
  game.busy = false;
  game.avatarDataUrl = null;
  game.hasFace = false;
  game.roundTimeLeft = cfg.roundSeconds;
  shuffleRoundOrder();
  updateAvatarDisplay();
  updateHud();

  els.finalScreen.hidden = true;
  els.game.hidden = true;
  els.hud.hidden = true;
  els.history.hidden = true;
  els.historyList.innerHTML = "";
  els.identityBadge.hidden = true;
  els.scanSection.hidden = false;
  els.retakeBtn.hidden = true;
  els.classicBtn.classList.remove("is-selected");
  els.aiBtn.classList.remove("is-selected");
  resetCrisisMedia();

  document.querySelector('[data-screen="welcome"]').classList.add("is-visible");
  setState("welcome");
}

els.startBtn.addEventListener("click", async () => {
  ensureAudio();
  document.querySelector('[data-screen="welcome"]').classList.remove("is-visible");
  await loadChallenges();
  updateHud();
  setState("face_scan");
  startCamera();
});

els.scanBtn.addEventListener("click", () => captureFace());
els.skipFaceBtn.addEventListener("click", () => skipFace());
els.retakeBtn.addEventListener("click", () => {
  game.hasFace = false;
  game.avatarDataUrl = null;
  updateAvatarDisplay();
  els.identityBadge.hidden = true;
  els.scanSection.hidden = false;
  setState("face_scan");
});
els.classicBtn.addEventListener("click", () => makeDecision("classic"));
els.aiBtn.addEventListener("click", () => makeDecision("medinamind"));
els.nextRoundBtn.addEventListener("click", () => nextRound());
els.certBtn.addEventListener("click", () => downloadCertificate());
els.restartBtn.addEventListener("click", () => restartMission());
els.muteBtn.addEventListener("click", () => {
  game.muted = !game.muted;
  els.muteBtn.setAttribute("aria-pressed", String(game.muted));
  els.muteBtn.textContent = game.muted ? "🔇" : "🔊";
});

updateAvatarDisplay();
loadChallenges();
