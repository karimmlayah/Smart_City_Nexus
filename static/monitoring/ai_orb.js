/**
 * MedinaMind AI Orb — invisible until clap / "Hello MedinaMind", opens centered
 */
(function () {
  "use strict";

  var cfg = window.MedinaMindOrbConfig || {};
  var STATES = ["sleep", "idle", "wake", "listening", "thinking", "speaking", "error"];
  var IDLE_TIMEOUT_MS = 120000;
  var TRANSCRIPT_DEBOUNCE_MS = 800;
  var DEFAULT_RECOG_LANGS = [
    "en-US", "fr-FR", "ar-SA", "ar-TN", "ar-MA", "es-ES", "de-DE", "it-IT", "pt-BR", "tr-TR",
  ];
  var WAKE_MIN_CONFIDENCE = 0.72;
  var WAKE_PHRASE = /\b(?:hello|hey|hi|bonjour|salut)\s+medina\s*mind\b|\bmedina\s*mind\b|\bmedinamind\b|\bmadina\s*mind\b/i;
  var CHIP_PROMPTS = {
    traffic_control: "Open traffic control",
    road_analysis: "Open road analysis",
    generate_report: "Generate a report",
    summarize_incident: "Summarize the current incident",
  };

  var layer = document.getElementById("mmAiOrbLayer");
  if (!layer) return;

  var panel = document.getElementById("mmAiOrbPanel");
  var backdrop = document.getElementById("mmAiOrbBackdrop");
  var visual = document.getElementById("mmAiOrbVisual");
  var statusEl = document.getElementById("mmAiOrbStatus");
  var transcriptEl = document.getElementById("mmAiOrbTranscript");
  var replyEl = document.getElementById("mmAiOrbReply");
  var micBtn = document.getElementById("mmAiOrbMicBtn");
  var micLabel = document.getElementById("mmAiOrbMicLabel");
  var closeBtn = document.getElementById("mmAiOrbClose");
  var minBtn = document.getElementById("mmAiOrbMinimize");
  var ttsBtn = document.getElementById("mmAiOrbTtsToggle");
  var enableClapBtn = document.getElementById("mmAiOrbEnableClap");
  var armBtn = document.getElementById("mmAiOrbArm");

  var state = "sleep";
  var active = false;
  var ttsEnabled = true;
  var clapDetector = null;
  var recognition = null;
  var wakeRecognition = null;
  var wakeWanted = false;
  var wakeRestartTimer = 0;
  var history = [];
  var idleTimer = 0;
  var speaking = false;
  var micArmed = false;
  var recognitionWanted = false;
  var recognitionStarting = false;
  var processingGroq = false;
  var speakGeneration = 0;
  var restartTimer = 0;
  var lastFinalText = "";
  var lastFinalAt = 0;
  var listenSessionId = 0;
  var lastProcessedResultIndex = -1;
  var recogLangs = [];
  var recogLangIndex = 0;
  var wakeLangIndex = 0;
  var sessionRecogLang = "";
  var preferredVoice = null;
  var preferredVoiceLang = "";

  function csrfToken() {
    var inp = document.getElementById("mmAiOrbCsrf");
    if (inp && inp.value) return inp.value;
    var m = document.cookie.match(/csrftoken=([^;]+)/);
    return m ? decodeURIComponent(m[1]) : "";
  }

  function setState(next) {
    if (STATES.indexOf(next) < 0) next = "idle";
    state = next;
    layer.setAttribute("data-state", next);
    if (visual) visual.setAttribute("data-orb-state", next === "sleep" ? "idle" : next);
    resetIdleTimer();
  }

  function setStatus(text) {
    if (statusEl) statusEl.textContent = text;
  }

  function updateMicUi(on) {
    if (!micBtn || !micLabel) return;
    micBtn.classList.toggle("is-active", !!on);
    if (speaking) micLabel.textContent = "Stop speaking";
    else if (on) micLabel.textContent = "Mic active";
    else micLabel.textContent = "Tap to speak";
  }

  function resetIdleTimer() {
    clearTimeout(idleTimer);
    if (!active || state === "sleep") return;
    idleTimer = setTimeout(function () {
      sleepOrb(true);
    }, IDLE_TIMEOUT_MS);
  }

  function showPanel(show) {
    if (!panel || !backdrop) return;
    panel.hidden = !show;
    backdrop.hidden = !show;
    layer.classList.toggle("mm-ai-orb-layer--open", !!show);
    layer.setAttribute("aria-hidden", show ? "false" : "true");
  }

  function updateArmUi(armed, msg) {
    if (!armBtn) return;
    if (active) {
      armBtn.classList.add("is-hidden");
      return;
    }
    var textEl = armBtn.querySelector(".mm-ai-orb-arm__text");
    if (armed) {
      armBtn.classList.add("is-ready");
      armBtn.classList.remove("is-hidden");
      if (textEl) {
        textEl.textContent = msg || "Listening — double clap or say \"Hello MedinaMind\"";
      }
      setTimeout(function () {
        if (micArmed && !active && armBtn) armBtn.classList.add("is-hidden");
      }, 4000);
    } else {
      armBtn.classList.remove("is-ready", "is-hidden");
      if (textEl) {
        textEl.textContent = msg || "Voice assistant · double clap or say \"Hello MedinaMind\"";
      }
    }
  }

  function buildRecogLangs() {
    var fromCfg = cfg.recognitionLangs || cfg.recognition_langs || [];
    var fromNav = navigator.languages ? Array.prototype.slice.call(navigator.languages) : [];
    var nav = navigator.language ? [navigator.language] : [];
    var merged = [].concat(fromCfg, fromNav, nav, DEFAULT_RECOG_LANGS);
    var out = [];
    var seen = {};
    for (var i = 0; i < merged.length; i++) {
      var lang = (merged[i] || "").trim();
      if (!lang) continue;
      var key = lang.toLowerCase();
      if (seen[key]) continue;
      seen[key] = true;
      out.push(lang);
    }
    return out.length ? out : DEFAULT_RECOG_LANGS.slice();
  }

  function ensureRecogLangs() {
    if (!recogLangs.length) recogLangs = buildRecogLangs();
  }

  function pickBestLangForBase(base) {
    ensureRecogLangs();
    var b = (base || "en").toLowerCase();
    for (var i = 0; i < recogLangs.length; i++) {
      if (recogLangs[i].toLowerCase().indexOf(b) === 0) return recogLangs[i];
    }
    var defaults = { en: "en-US", fr: "fr-FR", ar: "ar-SA", es: "es-ES", de: "de-DE", it: "it-IT", pt: "pt-BR", tr: "tr-TR" };
    return defaults[b] || "en-US";
  }

  function getRecognitionLang() {
    ensureRecogLangs();
    if (sessionRecogLang) return sessionRecogLang;
    return recogLangs[recogLangIndex % recogLangs.length] || "en-US";
  }

  function rotateRecognitionLang() {
    ensureRecogLangs();
    if (sessionRecogLang) return sessionRecogLang;
    recogLangIndex = (recogLangIndex + 1) % recogLangs.length;
    return getRecognitionLang();
  }

  function getWakeLang() {
    ensureRecogLangs();
    return recogLangs[0] || navigator.language || "en-US";
  }

  function matchesWakePhrase(text) {
    var t = (text || "").toLowerCase().replace(/[^a-z0-9\s\u0600-\u06FF-]/gi, " ").replace(/\s+/g, " ").trim();
    if (!t) return false;
    return WAKE_PHRASE.test(t);
  }

  function wakePhraseConfidence(result) {
    if (!result || !result.length) return 0;
    var alt = result[0];
    if (alt && typeof alt.confidence === "number" && alt.confidence > 0) return alt.confidence;
    return 1;
  }

  function lockSessionLangFromText(text) {
    var t = text || "";
    if (/[\u0600-\u06FF]/.test(t)) sessionRecogLang = pickBestLangForBase("ar");
    else if (/[àâäéèêëïîôùûüç]/i.test(t)) sessionRecogLang = pickBestLangForBase("fr");
    else if (/[ñ¿¡]/i.test(t)) sessionRecogLang = pickBestLangForBase("es");
    else if (/\b(bonjour|salut|merci|ouvre|ouvrir|affiche)\b/i.test(t)) sessionRecogLang = pickBestLangForBase("fr");
    else sessionRecogLang = pickBestLangForBase("en");
    preferredVoice = null;
    preferredVoiceLang = "";
  }

  function pickVoice() {
    if (!window.speechSynthesis) return null;
    var targetLang = sessionRecogLang || getRecognitionLang() || "en-US";
    if (preferredVoice && preferredVoiceLang === targetLang) return preferredVoice;
    var voices = window.speechSynthesis.getVoices() || [];
    var base = targetLang.split("-")[0].toLowerCase();
    preferredVoice =
      voices.filter(function (v) {
        return v.lang === targetLang;
      })[0] ||
      voices.filter(function (v) {
        return v.lang && v.lang.toLowerCase().indexOf(base) === 0;
      })[0] ||
      voices.filter(function (v) {
        return /en-US/i.test(v.lang);
      })[0] ||
      voices[0] ||
      null;
    preferredVoiceLang = targetLang;
    return preferredVoice;
  }

  if (window.speechSynthesis) {
    window.speechSynthesis.onvoiceschanged = pickVoice;
    pickVoice();
  }

  function stopRecognitionEngine() {
    clearTimeout(restartTimer);
    if (recognition) {
      try {
        recognition.onend = null;
        recognition.onerror = null;
        recognition.stop();
      } catch (e0) {}
      recognition = null;
    }
    recognitionStarting = false;
  }

  function pauseRecognition() {
    recognitionWanted = false;
    stopRecognitionEngine();
    updateMicUi(false);
  }

  function beginListening() {
    if (!active) return;
    listenSessionId += 1;
    var session = listenSessionId;
    recognitionWanted = true;
    speaking = false;
    processingGroq = false;
    lastProcessedResultIndex = -1;
    stopRecognitionEngine();
    setState("listening");
    setStatus("Listening — speak in any language");
    updateMicUi(false);
    clearTimeout(restartTimer);
    restartTimer = setTimeout(function () {
      if (session !== listenSessionId || !active || !recognitionWanted) return;
      startRecognitionEngine();
    }, 550);
  }

  function pauseClapDetector() {
    if (clapDetector) clapDetector.pause();
  }

  function resumeClapDetector() {
    if (!micArmed || active || !clapDetector) return;
    clapDetector.resume().catch(function () {});
  }

  function stopWakePhraseLoop() {
    wakeWanted = false;
    clearTimeout(wakeRestartTimer);
    if (wakeRecognition) {
      try {
        wakeRecognition.onend = null;
        wakeRecognition.abort();
      } catch (e0) {}
      wakeRecognition = null;
    }
  }

  function listenWakePhraseOnce() {
    if (!wakeWanted || active || !micArmed) return;

    var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      wakeRestartTimer = setTimeout(listenWakePhraseOnce, 2000);
      return;
    }

    if (wakeRecognition) return;

    wakeRecognition = new SR();
    wakeRecognition.lang = getWakeLang();
    wakeRecognition.continuous = false;
    wakeRecognition.interimResults = false;
    wakeRecognition.maxAlternatives = 1;

    wakeRecognition.onresult = function (ev) {
      for (var i = ev.resultIndex; i < ev.results.length; i++) {
        if (!ev.results[i].isFinal) continue;
        var conf = wakePhraseConfidence(ev.results[i]);
        if (conf < WAKE_MIN_CONFIDENCE) continue;
        var text = ev.results[i][0].transcript || "";
        if (matchesWakePhrase(text)) {
          stopWakePhraseLoop();
          wakeOrb("phrase");
          return;
        }
      }
    };

    wakeRecognition.onerror = function (ev) {
      if (ev.error === "aborted") return;
      if (ev.error === "not-allowed") {
        wakeWanted = false;
        micArmed = false;
        updateArmUi(false, "Microphone denied — click to retry");
      }
    };

    wakeRecognition.onend = function () {
      wakeRecognition = null;
      if (wakeWanted && !active) {
        wakeRestartTimer = setTimeout(listenWakePhraseOnce, 1800);
      }
    };

    try {
      wakeRecognition.start();
    } catch (e1) {
      wakeRecognition = null;
      wakeRestartTimer = setTimeout(listenWakePhraseOnce, 1000);
    }
  }

  function startSleepListening() {
    if (!micArmed || active) return;
    wakeWanted = true;
    resumeClapDetector();
    listenWakePhraseOnce();
  }

  function wakeOrb(source) {
    if (active) return;
    active = true;
    sessionRecogLang = "";
    recogLangIndex = 0;
    preferredVoice = null;
    preferredVoiceLang = "";
    stopWakePhraseLoop();
    pauseClapDetector();
    updateArmUi(true);
    if (enableClapBtn) enableClapBtn.hidden = true;

    showPanel(true);
    layer.classList.add("mm-ai-orb-layer--open");
    setState("wake");

    if (source === "clap") {
      setStatus("Double clap detected");
    } else if (source === "phrase") {
      setStatus("Hello MedinaMind");
    } else {
      setStatus("Assistant activated");
    }

    setTimeout(function () {
      lastFinalText = "";
      lastFinalAt = 0;
      beginListening();
    }, 900);
  }

  function sleepOrb(silent) {
    active = false;
    processingGroq = false;
    lastFinalText = "";
    lastFinalAt = 0;
    lastProcessedResultIndex = -1;
    sessionRecogLang = "";
    recogLangIndex = 0;
    preferredVoice = null;
    preferredVoiceLang = "";
    listenSessionId += 1;
    stopSpeaking();
    stopRecognitionLoop();
    showPanel(false);
    layer.classList.remove("mm-ai-orb-layer--open");
    setState("sleep");
    updateArmUi(micArmed);
    if (micArmed) setTimeout(startSleepListening, 500);
  }

  function localIntent(text) {
    var t = (text || "").toLowerCase();
    var routes = cfg.modules || {};
    var checks = [
      { re: /\b(traffic control|traffic nexus|trafic|contrôle traffic)\b/, key: "traffic_control", say: "Opening Traffic Control." },
      { re: /\b(smart traffic|traffic light|signal control)\b/, key: "traffic_signal", say: "Opening Smart Traffic Lights." },
      { re: /\b(red light|violation)\b/, key: "traffic_violation", say: "Opening Red Light Detection." },
      { re: /\b(road analysis|road damage|pothole|route|chaussée|nid de poule)\b/, key: "road_analysis", say: "Opening Road Analysis." },
      { re: /\b(city monitoring|surveillance|monitoring ville)\b/, key: "city_monitoring", say: "Opening City Monitoring." },
      { re: /\b(fire|smoke|feu|fumée|incendie)\b/, key: "fire_smoke", say: "Opening Fire and Smoke." },
      { re: /\b(waste|garbage|déchet|ordures)\b/, key: "street_waste", say: "Opening Street Waste." },
      { re: /\b(drone|uav)\b/, key: "drone_monitoring", say: "Opening Drone Monitoring." },
      { re: /\b(camera settings|smart camera|caméra)\b/, key: "camera_settings", say: "Opening Camera Settings." },
      { re: /\b(ai dashboard|dashboard|command center|tableau de bord)\b/, key: "ai_dashboard", say: "Opening the AI Dashboard." },
      { re: /\b(platform settings|settings|paramètres)\b/, key: "platform_settings", say: "Opening Platform Settings." },
      { re: /\b(home|homepage|accueil)\b/, key: "home", say: "Opening the homepage." },
      { re: /\b(generate|create)\b.*\breport\b/, key: "generate_report", say: "Opening reports." },
    ];
    for (var i = 0; i < checks.length; i++) {
      if (/\b(open|go to|show|launch|navigate|ouvre|ouvrir|affiche|afficher)\b/.test(t) && checks[i].re.test(t)) {
        return { url: routes[checks[i].key], say: checks[i].say, intent: checks[i].key };
      }
    }
    return null;
  }

  function navigate(url, delay) {
    if (!url) return;
    setTimeout(function () {
      window.location.href = url;
    }, delay || 1200);
  }

  function stopSpeaking() {
    speakGeneration++;
    speaking = false;
    if (window.speechSynthesis) {
      try {
        window.speechSynthesis.cancel();
      } catch (e0) {}
    }
  }

  function interruptAndListen() {
    stopSpeaking();
    beginListening();
  }

  function resumeListening() {
    beginListening();
  }

  function speak(text, onEnd) {
    if (!ttsEnabled || !window.speechSynthesis) {
      speaking = false;
      if (onEnd) onEnd();
      return;
    }
    recognitionWanted = false;
    stopRecognitionEngine();
    stopSpeaking();
    var gen = ++speakGeneration;
    var finished = false;
    var pollTimer = 0;
    var fallbackTimer = 0;
    var ttsStarted = false;
    var ttsWasAudible = false;

    function done() {
      if (finished || gen !== speakGeneration) return;
      finished = true;
      clearInterval(pollTimer);
      clearTimeout(fallbackTimer);
      speaking = false;
      if (onEnd) onEnd();
    }

    var u = new SpeechSynthesisUtterance(text);
    var voice = pickVoice();
    if (voice) u.voice = voice;
    u.lang = voice && voice.lang ? voice.lang : sessionRecogLang || getRecognitionLang() || "en-US";
    u.rate = 0.96;
    u.pitch = 1;
    u.onstart = function () {
      if (gen !== speakGeneration) return;
      ttsStarted = true;
      ttsWasAudible = true;
      speaking = true;
      setState("speaking");
      setStatus("Speaking…");
    };
    u.onend = done;
    u.onerror = done;

    var wordCount = (text || "").split(/\s+/).length;
    fallbackTimer = setTimeout(done, Math.max(5000, wordCount * 520 + 1200));

    if (window.speechSynthesis.paused) {
      try {
        window.speechSynthesis.resume();
      } catch (e0) {}
    }
    window.speechSynthesis.speak(u);

    pollTimer = setInterval(function () {
      if (gen !== speakGeneration) {
        done();
        return;
      }
      if (window.speechSynthesis.speaking) ttsWasAudible = true;
      if (ttsStarted && ttsWasAudible && !window.speechSynthesis.speaking) done();
    }, 400);
  }

  function handleReply(data, userText) {
    var reply = (data && data.reply) || "I didn't understand that.";
    if (replyEl) replyEl.textContent = reply;
    history.push({ role: "user", content: userText });
    history.push({ role: "assistant", content: reply });
    if (history.length > 12) history = history.slice(-12);

    speak(reply, function () {
      if (data && data.action === "navigate" && data.url) {
        navigate(data.url, 400);
        return;
      }
      resumeListening();
    });
  }

  function callGroq(message) {
    processingGroq = true;
    recognitionWanted = false;
    stopRecognitionEngine();
    setState("thinking");
    setStatus("Thinking…");

    var local = localIntent(message);
    if (local && local.url) {
      processingGroq = false;
      handleReply({ reply: local.say, action: "navigate", url: local.url, intent: local.intent }, message);
      return;
    }

    fetch(cfg.apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": csrfToken(),
        "X-Requested-With": "XMLHttpRequest",
      },
      body: JSON.stringify({ message: message, history: history }),
    })
      .then(function (r) {
        return r.json().then(function (d) {
          return { ok: r.ok, data: d };
        });
      })
      .then(function (res) {
        processingGroq = false;
        if (!res.ok) throw new Error((res.data && res.data.error) || "Assistant unavailable.");
        handleReply(res.data, message);
      })
      .catch(function (err) {
        processingGroq = false;
        if (replyEl) replyEl.textContent = String(err.message || err);
        resumeListening();
      });
  }

  function onTranscript(text) {
    var msg = (text || "").trim();
    if (!msg || msg.length < 2) return;
    if (processingGroq) return;

    var now = Date.now();
    if (msg === lastFinalText && now - lastFinalAt < TRANSCRIPT_DEBOUNCE_MS) return;
    lastFinalText = msg;
    lastFinalAt = now;
    lockSessionLangFromText(msg);

    if (transcriptEl) transcriptEl.textContent = msg;
    recognitionWanted = false;
    stopRecognitionEngine();
    callGroq(msg);
  }

  function shouldAcceptTranscript(text) {
    if (!text || text.length < 2) return false;
    if (/^(ok|uh|um|ah|oh|hm+|mm+|yes|no|yeah|okay)$/.test(text.toLowerCase())) return false;
    return true;
  }

  function handleRecognitionResult(ev) {
    if (!active || !recognitionWanted || speaking || processingGroq) return;

    for (var i = ev.resultIndex; i < ev.results.length; i++) {
      var result = ev.results[i];
      var chunk = (result[0] && result[0].transcript) || "";

      if (!result.isFinal) {
        if (transcriptEl && chunk.trim()) transcriptEl.textContent = chunk.trim() + "…";
        continue;
      }

      if (i <= lastProcessedResultIndex) continue;

      var text = chunk.trim();
      if (!shouldAcceptTranscript(text)) continue;

      lastProcessedResultIndex = i;
      onTranscript(text);
      return;
    }
  }

  function scheduleRecognitionRestart(delay) {
    clearTimeout(restartTimer);
    if (!recognitionWanted || !active || speaking || processingGroq) return;
    restartTimer = setTimeout(startRecognitionEngine, delay || 500);
  }

  function initRecognition() {
    var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return null;
    var rec = new SR();
    rec.lang = getRecognitionLang();
    rec.interimResults = true;
    rec.maxAlternatives = 1;
    rec.continuous = false;
    rec.onresult = handleRecognitionResult;
    rec.onerror = function (ev) {
      if (ev.error === "aborted" || !recognitionWanted || !active) return;
      if (ev.error === "not-allowed") {
        setStatus("Microphone denied");
        recognitionWanted = false;
        return;
      }
      if (ev.error === "no-speech") rotateRecognitionLang();
      scheduleRecognitionRestart(ev.error === "no-speech" ? 700 : 900);
    };
    rec.onend = function () {
      recognitionStarting = false;
      updateMicUi(false);
      if (recognitionWanted && active && !speaking && !processingGroq) {
        scheduleRecognitionRestart(600);
      }
    };
    rec.onstart = function () {
      recognitionStarting = false;
      updateMicUi(true);
    };
    return rec;
  }

  function startRecognitionEngine() {
    if (!recognitionWanted || !active || recognitionStarting || speaking || processingGroq) return;
    stopRecognitionEngine();
    recognition = initRecognition();
    if (!recognition) return;
    recognitionStarting = true;
    try {
      recognition.start();
    } catch (e1) {
      recognitionStarting = false;
      recognition = null;
      scheduleRecognitionRestart(900);
      return;
    }
    setTimeout(function () {
      if (recognitionStarting) {
        recognitionStarting = false;
        if (recognitionWanted && active && !speaking && !processingGroq) {
          scheduleRecognitionRestart(700);
        }
      }
    }, 1800);
  }

  function ensureRecognition(want) {
    if (want) beginListening();
    else pauseRecognition();
  }

  function stopRecognitionLoop() {
    listenSessionId += 1;
    pauseRecognition();
  }

  function enableClapDetection() {
    if (!window.MedinaMindClapDetector) {
      return Promise.reject(new Error("Clap detector unavailable"));
    }

    if (!clapDetector) {
      clapDetector = new window.MedinaMindClapDetector({
        threshold: 0.42,
        minAbsolutePeak: 0.36,
        minSharpness: 2.8,
        minClapGap: 180,
        cooldownMs: 3500,
        onClap: function () {},
        onDoubleClap: function () {
          if (active) return;
          wakeOrb("clap");
        },
        onError: function () {},
      });
    }

    return clapDetector.start().then(function () {
      micArmed = true;
      if (enableClapBtn) enableClapBtn.hidden = true;
      updateArmUi(true);
      if (!active) startSleepListening();
    });
  }

  function armMicrophone() {
    if (micArmed) return Promise.resolve();
    if (armBtn) armBtn.classList.remove("is-hidden");
    return enableClapDetection().catch(function () {
      updateArmUi(false, "Microphone denied — click to retry");
      if (enableClapBtn) {
        enableClapBtn.hidden = false;
        enableClapBtn.textContent = "Allow microphone";
      }
    });
  }

  function bindEvents() {
    if (micBtn) {
      micBtn.addEventListener("click", function () {
        if (speaking) interruptAndListen();
        else if (processingGroq) return;
        else beginListening();
      });
    }

    if (closeBtn) closeBtn.addEventListener("click", function () {
      sleepOrb(false);
    });

    if (minBtn) minBtn.addEventListener("click", function () {
      sleepOrb(false);
    });

    if (backdrop) {
      backdrop.addEventListener("click", function (ev) {
        ev.stopPropagation();
      });
    }

    if (ttsBtn) {
      ttsBtn.addEventListener("click", function () {
        ttsEnabled = !ttsEnabled;
        ttsBtn.setAttribute("aria-pressed", ttsEnabled ? "true" : "false");
        if (!ttsEnabled) stopSpeaking();
      });
    }

    if (enableClapBtn) {
      enableClapBtn.addEventListener("click", function () {
        armMicrophone().catch(function () {});
      });
    }

    if (armBtn) {
      armBtn.addEventListener("click", function () {
        armMicrophone().catch(function () {});
      });
    }

    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && active) {
        ev.preventDefault();
        if (speaking) interruptAndListen();
        else sleepOrb(false);
        return;
      }
      if (ev.ctrlKey && ev.shiftKey && (ev.key === "M" || ev.key === "m")) {
        ev.preventDefault();
        if (!micArmed) {
          armMicrophone().then(function () {
            wakeOrb("shortcut");
          });
        } else {
          wakeOrb("shortcut");
        }
      }
    });

    document.querySelectorAll(".mm-ai-orb-chip").forEach(function (chip) {
      chip.addEventListener("click", function () {
        var intent = chip.getAttribute("data-intent");
        var prompt = CHIP_PROMPTS[intent] || chip.textContent;
        if (transcriptEl) transcriptEl.textContent = prompt;
        callGroq(prompt);
      });
    });
  }

  window.MedinaMindOrb = {
    wake: function () {
      wakeOrb("api");
    },
    arm: armMicrophone,
    sleep: function () {
      sleepOrb(false);
    },
    isArmed: function () {
      return micArmed;
    },
    isActive: function () {
      return active;
    },
  };

  bindEvents();
  setState("sleep");
  showPanel(false);
  updateArmUi(false);
})();
