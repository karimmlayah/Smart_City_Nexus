/**
 * MedinaMind AR Control — webcam and simplified gestures
 */
(function () {
  "use strict";

  console.log("AR gesture script loaded");

  var MODULE_KEYS = ["traffic", "energy", "pollution", "security"];
  var METRIC_BY_MODULE = {
    traffic: "traffic",
    energy: "energy",
    pollution: "air-quality",
    security: "safety",
  };
  var MODULE_INFO = {
    traffic: {
      title: "Traffic Monitoring",
      badge: "Traffic Monitoring",
      centerLabel: "Traffic Flow",
      text: "Live road flow and congestion insights.",
    },
    energy: {
      title: "Energy Network",
      badge: "Energy Network",
      centerLabel: "Energy Usage",
      text: "Track city energy usage in real time.",
    },
    pollution: {
      title: "Air Quality Monitoring",
      badge: "Air Quality Monitoring",
      centerLabel: "Air Quality",
      text: "Monitor pollution levels and environmental health.",
    },
    security: {
      title: "Safety Monitoring",
      badge: "Safety Monitoring",
      centerLabel: "Safety Status",
      text: "Support faster response to risky situations.",
    },
  };

  var GESTURE_COOLDOWN = 1000;
  var DEMO_CYCLE_MS = 3000;
  var HAND_COLORS = ["#00D5FF", "#A78BFA"];
  var ZOOM_MIN = 0.8;
  var ZOOM_MAX = 2.0;
  var ZOOM_SPREAD_SENSITIVITY = 3.5;
  var ZOOM_SPREAD_MIN_DELTA = 0.005;

  var video = document.getElementById("camVideo");
  var canvas = document.getElementById("camCanvas");
  var ctx = canvas ? canvas.getContext("2d") : null;
  var noCam = document.getElementById("noCam");
  var retryBtn = document.getElementById("retryBtn");
  var holoCityWrap = document.getElementById("holoCityWrap");
  var demoBtn = document.getElementById("demoBtn");
  var toastEl = document.getElementById("arToast");
  var moduleFloatCard = document.getElementById("moduleFloatCard");
  var moduleFloatTitle = document.getElementById("moduleFloatTitle");
  var moduleFloatText = document.getElementById("moduleFloatText");
  var zoomVal = document.getElementById("zoomVal");
  var arHudRoot = document.getElementById("arHudRoot");
  var cameraStage = document.getElementById("arCameraStage");

  var btnPrev = document.getElementById("btnPrev");
  var btnNext = document.getElementById("btnNext");
  var btnZoomIn = document.getElementById("btnZoomIn");
  var btnZoomOut = document.getElementById("btnZoomOut");
  var btnFullscreen = document.getElementById("btnFullscreen");

  var modules = [];
  var statCards = MODULE_KEYS.map(function (k) {
    return document.getElementById(
      "holoCard" + k.charAt(0).toUpperCase() + k.slice(1)
    );
  });

  var handsModel = null;
  var mpCamera = null;
  var cameraReady = false;
  var handTrackingReady = false;
  var cameraError = false;
  var handInitInProgress = false;
  var demoMode = false;
  var demoTimer = null;
  var handDetected = false;
  var arOverlayVisible = false;
  var zoomLevel = 1.0;

  var selectedModuleIndex = 0;
  var currentGesture = "Waiting";
  var lastGesture = null;
  var lastGestureTime = 0;
  var lastHandSpread = null;
  var twoHandZooming = false;

  var currentTargetX = null;
  var currentTargetY = null;
  var currentTargetScale = null;
  var SMOOTHING = 0.12;
  var holoEnterTimer = null;
  var holoHideTimer = null;
  var parallaxRaf = null;
  var parallaxPending = null;

  function setHandText(selector, value) {
    document.querySelectorAll(selector).forEach(function (el) {
      el.textContent = value;
    });
  }

  function showToast(message) {
    if (!toastEl) return;
    toastEl.textContent = message;
    toastEl.classList.add("visible");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(function () {
      toastEl.classList.remove("visible");
    }, 2200);
  }

  function updateHandUI(status, points, gesture) {
    setHandText("[data-hand='status']", status);
    setHandText("[data-hand='points']", points);
    setHandText("[data-hand='gesture']", gesture);
  }

  function pulseElement(el) {
    if (!el) return;
    el.classList.remove("hud-pulse");
    void el.offsetWidth;
    el.classList.add("hud-pulse");
    setTimeout(function () {
      el.classList.remove("hud-pulse");
    }, 700);
  }

  function refreshModules() {
    modules = Array.from(document.querySelectorAll(".hud-module"));
    modules.forEach(function (mod, i) {
      mod.onclick = function () {
        selectModule(i);
        showToast("Module selected: " + getModuleName(i));
      };
    });
    if (modules.length) setModule(selectedModuleIndex, true);
  }

  function getModuleKey(index) {
    var mod = modules[index];
    return (mod && mod.getAttribute("data-mod")) || MODULE_KEYS[index] || MODULE_KEYS[0];
  }

  function getModuleName(index) {
    var mod = modules[index];
    if (mod) {
      var nameEl = mod.querySelector(".hud-mod-name");
      if (nameEl) return nameEl.textContent;
    }
    var key = getModuleKey(index);
    return (MODULE_INFO[key] && MODULE_INFO[key].title) || "Module";
  }

  function updateModuleFloatCard(key) {
    var info = MODULE_INFO[key] || {};
    if (moduleFloatTitle) moduleFloatTitle.textContent = info.title || key;
    if (moduleFloatText) moduleFloatText.textContent = info.text || "";
    var holoMod = document.getElementById("holoActiveModule");
    if (holoMod) holoMod.textContent = info.badge || info.title || getModuleName(selectedModuleIndex);
    var centerLabel = document.querySelector('[data-holo="center-label"]');
    if (centerLabel) centerLabel.textContent = info.centerLabel || info.title || "Smart City";
    var holoPanel = document.getElementById("holoCityHub");
    if (holoPanel) holoPanel.setAttribute("data-module", key);
    if (moduleFloatCard && arOverlayVisible) moduleFloatCard.classList.add("visible");
  }

  function highlightHoloMetric(key) {
    document.querySelectorAll(".hologram-stat-card").forEach(function (el) {
      el.classList.remove("active-module", "holo-selected");
    });
    var metricKey = METRIC_BY_MODULE[key];
    if (!metricKey) return;
    var pill = document.querySelector(
      '.hologram-stat-card[data-metric="' + metricKey + '"]'
    );
    if (pill) pill.classList.add("active-module");
  }

  function setModule(index, silent) {
    if (!modules.length) return;
    index = ((index % modules.length) + modules.length) % modules.length;
    selectedModuleIndex = index;
    var key = getModuleKey(index);

    modules.forEach(function (mod, i) {
      mod.classList.toggle("active", i === index);
    });
    highlightHoloMetric(key);
    updateModuleFloatCard(key);
    if (!silent) pulseElement(modules[index]);
  }

  function moduleToastLabel(index) {
    var key = getModuleKey(index);
    var labels = {
      traffic: "Traffic module",
      energy: "Energy module",
      pollution: "Pollution module",
      security: "Security module",
    };
    return labels[key] || "Module changed";
  }

  function selectNextModule() {
    setModule(selectedModuleIndex + 1);
    showToast(moduleToastLabel(selectedModuleIndex));
  }

  function selectPreviousModule() {
    setModule(selectedModuleIndex - 1);
    showToast(moduleToastLabel(selectedModuleIndex));
  }

  function activateCurrentModule() {
    setModule(selectedModuleIndex);
    var key = getModuleKey(selectedModuleIndex);
    var metricKey = METRIC_BY_MODULE[key];
    document.querySelectorAll(".hologram-stat-card").forEach(function (el) {
      el.classList.remove("holo-selected");
    });
    var pill = metricKey
      ? document.querySelector('.hologram-stat-card[data-metric="' + metricKey + '"]')
      : null;
    if (pill) {
      pill.classList.add("holo-selected");
      pulseElement(pill);
    }
    showToast("Module selected");
  }

  function selectModule(index) {
    setModule(index);
  }

  function showAROverlay(force) {
    if (!holoCityWrap) return;
    if (arOverlayVisible && !force) return;
    arOverlayVisible = true;
    if (holoHideTimer) clearTimeout(holoHideTimer);
    holoCityWrap.classList.remove("is-hiding");
    holoCityWrap.classList.add("visible", "is-entering", "is-revealing");
    if (holoEnterTimer) clearTimeout(holoEnterTimer);
    holoEnterTimer = setTimeout(function () {
      if (holoCityWrap) {
        holoCityWrap.classList.remove("is-entering", "is-revealing");
      }
    }, 720);
    applyHoloTransform();
    setModule(selectedModuleIndex, true);
    updateModuleFloatCard(getModuleKey(selectedModuleIndex));
    if (moduleFloatCard) moduleFloatCard.classList.add("visible");
    if (!force) showToast("Smart city hologram activated");
  }

  function initControls() {
    btnPrev = document.getElementById("btnPrev");
    btnNext = document.getElementById("btnNext");
    btnZoomIn = document.getElementById("btnZoomIn");
    btnZoomOut = document.getElementById("btnZoomOut");
    btnFullscreen = document.getElementById("btnFullscreen");
    demoBtn = document.getElementById("demoBtn");

    if (demoBtn) demoBtn.onclick = toggleDemoMode;
    if (btnPrev) btnPrev.onclick = selectPreviousModule;
    if (btnNext) btnNext.onclick = selectNextModule;
    if (btnZoomIn) {
      btnZoomIn.onclick = function () {
        zoomLevel += 0.1;
        applyZoom();
      };
    }
    if (btnZoomOut) {
      btnZoomOut.onclick = function () {
        zoomLevel -= 0.1;
        applyZoom();
      };
    }
    if (btnFullscreen) {
      btnFullscreen.onclick = function () {
        var el = cameraStage || arHudRoot || document.documentElement;
        if (!document.fullscreenElement) {
          (el.requestFullscreen || el.webkitRequestFullscreen).call(el);
        } else {
          (document.exitFullscreen || document.webkitExitFullscreen).call(document);
        }
      };
    }
  }

  function hideAROverlay() {
    if (!arOverlayVisible || !holoCityWrap) return;
    if (demoMode) return;
    arOverlayVisible = false;
    holoCityWrap.classList.remove("is-revealing", "is-entering");
    holoCityWrap.classList.add("is-hiding");
    if (moduleFloatCard) moduleFloatCard.classList.remove("visible");
    currentTargetX = null;
    currentTargetY = null;
    currentTargetScale = null;
    if (holoHideTimer) clearTimeout(holoHideTimer);
    holoHideTimer = setTimeout(function () {
      if (holoCityWrap) {
        holoCityWrap.classList.remove("visible", "is-hiding");
      }
    }, 420);
    showToast("Smart city hologram hidden");
  }

  function initHoloParallax() {
    if (!cameraStage) return;
    cameraStage.addEventListener("mousemove", function (e) {
      if (!arOverlayVisible) return;
      parallaxPending = e;
      if (parallaxRaf) return;
      parallaxRaf = requestAnimationFrame(function () {
        parallaxRaf = null;
        if (!parallaxPending || !arOverlayVisible) return;
        var rect = cameraStage.getBoundingClientRect();
        var x = (parallaxPending.clientX - rect.left) / rect.width - 0.5;
        var y = (parallaxPending.clientY - rect.top) / rect.height - 0.5;
        var city = document.getElementById("hologramCity");
        if (city) {
          city.style.setProperty("--px", x * 8 + "px");
          city.style.setProperty("--py", y * 5 + "px");
        }
        parallaxPending = null;
      });
    });
  }

  function applyZoom() {
    zoomLevel = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoomLevel));
    if (zoomVal) zoomVal.textContent = zoomLevel.toFixed(1) + "x";
    applyHoloTransform();
  }

  function applyHoloTransform() {
    if (!holoCityWrap) return;
    var stage = cameraStage;
    var w = stage ? stage.clientWidth : window.innerWidth;
    var h = stage ? stage.clientHeight : window.innerHeight;
    var tX = currentTargetX !== null ? currentTargetX - w / 2 : 0;
    var tY = currentTargetY !== null ? currentTargetY - h * 0.48 : 24;
    var baseScale =
      currentTargetScale !== null ? currentTargetScale : 0.88;
    var finalScale = baseScale * zoomLevel;

    holoCityWrap.style.setProperty("--holo-pos-x", tX + "px");
    holoCityWrap.style.setProperty("--holo-pos-y", tY + "px");
    holoCityWrap.style.setProperty("--holo-scale", String(finalScale));
  }

  function setDemoMode(on) {
    demoMode = on;
    if (demoBtn) {
      demoBtn.textContent = on ? "Demo Mode: On" : "Demo Mode: Off";
      demoBtn.classList.toggle("demo-active", on);
    }
    if (on) {
      showAROverlay();
      currentTargetX = null;
      applyHoloTransform();
      if (demoTimer) clearInterval(demoTimer);
      demoTimer = setInterval(function () {
        setModule(selectedModuleIndex + 1, true);
      }, DEMO_CYCLE_MS);
      showToast("Demo mode enabled");
    } else {
      if (demoTimer) clearInterval(demoTimer);
      demoTimer = null;
      if (!handDetected) hideAROverlay();
      showToast("Demo mode disabled");
    }
  }

  function toggleDemoMode() {
    setDemoMode(!demoMode);
  }

  function isFingerExtended(lm, tip, mcp) {
    return lm[tip].y < lm[mcp].y - 0.03;
  }

  function countExtended(lm) {
    var thumbExt = Math.abs(lm[4].x - lm[3].x) > 0.05;
    return [
      thumbExt,
      isFingerExtended(lm, 8, 5),
      isFingerExtended(lm, 12, 9),
      isFingerExtended(lm, 16, 13),
      isFingerExtended(lm, 20, 17),
    ];
  }

  function detectOpenHand(extended) {
    return extended.filter(Boolean).length >= 4;
  }

  function detectFist(extended) {
    return extended.filter(Boolean).length <= 1;
  }

  function detectPinch(lm) {
    var dx = lm[4].x - lm[8].x;
    var dy = lm[4].y - lm[8].y;
    return Math.sqrt(dx * dx + dy * dy) < 0.05;
  }

  function detectPeaceSign(extended) {
    return extended[1] && extended[2] && !extended[3] && !extended[4];
  }

  function getHandednessLabel(results, index) {
    if (!results.multiHandedness || !results.multiHandedness[index]) return "";
    var entry = results.multiHandedness[index];
    return (entry.label || entry.categoryName || "").toLowerCase();
  }

  function detectSingleHandGesture(lm, extended) {
    if (detectPinch(lm)) return "pinch";
    if (detectPeaceSign(extended)) return "peace";
    if (detectFist(extended)) return "fist";
    if (detectOpenHand(extended)) return "open_hand";
    return "hand_detected";
  }

  function detectMultiHandGesture(results) {
    var hands = results.multiHandLandmarks;
    var count = hands.length;
    if (!count) return { gesture: null, label: "Waiting" };

    if (count === 1) {
      var ext1 = countExtended(hands[0]);
      var g1 = detectSingleHandGesture(hands[0], ext1);
      if (g1 === "peace") {
        var side1 = getHandednessLabel(results, 0);
        if (side1 === "left") g1 = "peace_left";
        else if (side1 === "right") g1 = "peace_right";
      }
      return {
        gesture: g1,
        label: gestureLabel(g1),
        points: "21/21",
        status: "1 hand detected",
      };
    }

    var extList = hands.map(function (lm) {
      return countExtended(lm);
    });
    var gestures = hands.map(function (lm, i) {
      return detectSingleHandGesture(lm, extList[i]);
    });
    var openCount = extList.filter(function (ext) {
      return detectOpenHand(ext);
    }).length;
    var fistCount = extList.filter(function (ext) {
      return detectFist(ext);
    }).length;

    if (openCount === 2) {
      return {
        gesture: "two_hands_open",
        label: "Two Open Hands",
        points: "42/42",
        status: "2 hands detected",
      };
    }

    if (fistCount === 2) {
      return {
        gesture: "two_hands_fist",
        label: "Two Fists",
        points: "42/42",
        status: "2 hands detected",
      };
    }

    for (var i = 0; i < count; i++) {
      if (gestures[i] === "peace") {
        var side = getHandednessLabel(results, i);
        if (side === "left") {
          return {
            gesture: "peace_left",
            label: "Peace — Left Hand",
            points: "42/42",
            status: "2 hands detected",
          };
        }
        if (side === "right") {
          return {
            gesture: "peace_right",
            label: "Peace — Right Hand",
            points: "42/42",
            status: "2 hands detected",
          };
        }
      }
    }

    for (var j = 0; j < count; j++) {
      if (gestures[j] === "pinch") {
        return {
          gesture: "pinch",
          label: "Pinch",
          points: "42/42",
          status: "2 hands detected",
        };
      }
    }

    if (openCount === 1) {
      return {
        gesture: "open_hand",
        label: "Open Hand",
        points: "42/42",
        status: "2 hands detected",
      };
    }

    return {
      gesture: "two_hands_detected",
      label: "Two Hands",
      points: "42/42",
      status: "2 hands detected",
    };
  }

  function gestureLabel(key) {
    var map = {
      Waiting: "Waiting",
      hand_detected: "Hand detected",
      open_hand: "Open Hand",
      two_hands_open: "Two Open Hands",
      two_hands_fist: "Two Fists",
      two_hands_detected: "Two Hands",
      two_hand_zoom: "Two-Hand Zoom",
      fist: "Fist",
      pinch: "Pinch",
      peace: "Peace Sign",
      peace_left: "Peace — Left Hand",
      peace_right: "Peace — Right Hand",
    };
    return map[key] || key;
  }

  function canTriggerGesture() {
    return performance.now() - lastGestureTime >= GESTURE_COOLDOWN;
  }

  function handleGesture(gesture) {
    switch (gesture) {
      case "open_hand":
        if (canTriggerGesture()) {
          lastGestureTime = performance.now();
          lastGesture = gesture;
          updateHandUI("Gesture recognized", "21/21", "Open Hand");
          showAROverlay();
          pulseElement(modules[selectedModuleIndex]);
        }
        break;
      case "fist":
        if (canTriggerGesture()) {
          lastGestureTime = performance.now();
          lastGesture = gesture;
          updateHandUI("Gesture recognized", "21/21", "Fist");
          hideAROverlay();
        }
        break;
      case "pinch":
        if (canTriggerGesture()) {
          lastGestureTime = performance.now();
          lastGesture = gesture;
          updateHandUI("Gesture recognized", "21/21", "Pinch");
          showAROverlay();
          activateCurrentModule();
        }
        break;
      case "two_hands_open":
        if (canTriggerGesture()) {
          lastGestureTime = performance.now();
          lastGesture = gesture;
          updateHandUI("Gesture recognized", "42/42", "Two Open Hands");
          showAROverlay();
          pulseElement(modules[selectedModuleIndex]);
        }
        break;
      case "two_hands_fist":
        if (canTriggerGesture()) {
          lastGestureTime = performance.now();
          lastGesture = gesture;
          updateHandUI("Gesture recognized", "42/42", "Two Fists");
          hideAROverlay();
        }
        break;
      case "peace_left":
        if (canTriggerGesture()) {
          lastGestureTime = performance.now();
          lastGesture = gesture;
          updateHandUI("Gesture recognized", "42/42", "Peace — Left Hand");
          selectPreviousModule();
        }
        break;
      case "peace_right":
        if (canTriggerGesture()) {
          lastGestureTime = performance.now();
          lastGesture = gesture;
          updateHandUI("Gesture recognized", "42/42", "Peace — Right Hand");
          selectNextModule();
        }
        break;
      default:
        break;
    }
  }

  function drawLandmarks(lm, colorIndex) {
    if (!ctx || !canvas) return;
    var w = canvas.width;
    var h = canvas.height;
    var color = HAND_COLORS[colorIndex] || HAND_COLORS[0];
    var connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [5, 9], [9, 10], [10, 11], [11, 12],
      [9, 13], [13, 14], [14, 15], [15, 16],
      [13, 17], [17, 18], [18, 19], [19, 20],
      [0, 17],
    ];
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.75;
    ctx.lineWidth = 2;
    connections.forEach(function (pair) {
      ctx.beginPath();
      ctx.moveTo(lm[pair[0]].x * w, lm[pair[0]].y * h);
      ctx.lineTo(lm[pair[1]].x * w, lm[pair[1]].y * h);
      ctx.stroke();
    });
    for (var i = 0; i < lm.length; i++) {
      var x = lm[i].x * w;
      var y = lm[i].y * h;
      var isTip = [4, 8, 12, 16, 20].indexOf(i) >= 0;
      ctx.beginPath();
      ctx.arc(x, y, isTip ? 5 : 3, 0, Math.PI * 2);
      ctx.fillStyle = isTip ? color : "rgba(47,128,237,0.9)";
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function getPalmCenter(lm) {
    return {
      x: (lm[0].x + lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 5,
      y: (lm[0].y + lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 5,
    };
  }

  function getTwoHandSpread(hands) {
    if (!hands || hands.length < 2) return null;
    var a = getPalmCenter(hands[0]);
    var b = getPalmCenter(hands[1]);
    var dx = a.x - b.x;
    var dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function updateTwoHandZoom(hands) {
    if (!hands || hands.length < 2) {
      lastHandSpread = null;
      twoHandZooming = false;
      return false;
    }

    var spread = getTwoHandSpread(hands);
    var zooming = false;

    if (lastHandSpread !== null) {
      var delta = spread - lastHandSpread;
      if (Math.abs(delta) > ZOOM_SPREAD_MIN_DELTA) {
        zoomLevel += delta * ZOOM_SPREAD_SENSITIVITY;
        applyZoom();
        zooming = true;
      }
    }

    lastHandSpread = spread;
    twoHandZooming = zooming;
    return zooming;
  }

  function getHologramAnchor(results) {
    var hands = results.multiHandLandmarks || [];
    if (hands.length >= 2) {
      var p1 = getPalmCenter(hands[0]);
      var p2 = getPalmCenter(hands[1]);
      return { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
    }
    if (hands.length === 1) return getPalmCenter(hands[0]);
    return null;
  }

  function onResults(results) {
    if (!canvas || !ctx || !video) return;
    canvas.width = video.videoWidth || canvas.offsetWidth;
    canvas.height = video.videoHeight || canvas.offsetHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    var handCount = results.multiHandLandmarks ? results.multiHandLandmarks.length : 0;
    handDetected = handCount > 0;

    if (!handDetected) {
      updateHandUI("Searching", "0/42", "Waiting");
      lastHandSpread = null;
      twoHandZooming = false;
      currentTargetX = null;
      if (!demoMode) hideAROverlay();
      return;
    }

    results.multiHandLandmarks.forEach(function (lm, i) {
      drawLandmarks(lm, i);
    });

    var isZooming = false;
    if (handCount >= 2) {
      isZooming = updateTwoHandZoom(results.multiHandLandmarks);
    } else {
      lastHandSpread = null;
      twoHandZooming = false;
    }

    var analysis = detectMultiHandGesture(results);
    var gesture = analysis.gesture;
    var points = analysis.points || (handCount === 2 ? "42/42" : "21/21");
    var status = analysis.status || (handCount === 2 ? "2 hands detected" : "1 hand detected");
    var gestureLabelText = analysis.label;

    if (isZooming) {
      gestureLabelText = "Two-Hand Zoom " + zoomLevel.toFixed(1) + "x";
      status = "2 hands detected";
    }

    if (gesture === "hand_detected" || gesture === "two_hands_detected") {
      updateHandUI(status, points, gestureLabelText);
    } else if (gesture && !isZooming) {
      updateHandUI("Gesture recognized", points, gestureLabelText);
      handleGesture(gesture);
    } else if (isZooming) {
      updateHandUI(status, points, gestureLabelText);
    } else {
      updateHandUI(status, points, gestureLabelText);
    }

    if ((gesture === "fist" || gesture === "two_hands_fist") && !demoMode) {
      hideAROverlay();
    }

    if (arOverlayVisible && holoCityWrap && !demoMode) {
      var anchor = getHologramAnchor(results);
      if (anchor) positionHologram(anchor);
    }
  }

  function positionHologram(palm) {
    var stage = cameraStage;
    var w = stage ? stage.clientWidth : window.innerWidth;
    var h = stage ? stage.clientHeight : window.innerHeight;
    var targetX = (1 - palm.x) * w;
    var targetY = palm.y * h * 0.55;
    var targetScale = Math.max(0.72, Math.min(0.65 + (0.55 - palm.y) * 0.6, 1.05));

    targetX = Math.max(w * 0.2, Math.min(w * 0.8, targetX));
    targetY = Math.max(h * 0.22, Math.min(h * 0.62, targetY));

      if (currentTargetX === null) {
        currentTargetX = targetX;
        currentTargetY = targetY;
        currentTargetScale = targetScale;
      } else {
        currentTargetX += (targetX - currentTargetX) * SMOOTHING;
        currentTargetY += (targetY - currentTargetY) * SMOOTHING;
        currentTargetScale += (targetScale - currentTargetScale) * SMOOTHING;
      }

    applyHoloTransform();
  }

  function stopCameraStream() {
    if (mpCamera) {
      try {
        mpCamera.stop();
      } catch (e) {
        console.warn("Camera stop error:", e);
      }
      mpCamera = null;
    }
    if (!video || !video.srcObject) return;
    video.srcObject.getTracks().forEach(function (track) {
      track.stop();
    });
    video.srcObject = null;
  }

  function waitForVideoFrames() {
    return new Promise(function (resolve) {
      if (!video) {
        resolve(false);
        return;
      }
      if (video.readyState >= 2 && video.videoWidth > 0) {
        resolve(true);
        return;
      }
      var done = false;
      function finish(ok) {
        if (done) return;
        done = true;
        video.removeEventListener("loadeddata", onReady);
        video.removeEventListener("playing", onReady);
        resolve(ok);
      }
      function onReady() {
        finish(video.videoWidth > 0);
      }
      video.addEventListener("loadeddata", onReady);
      video.addEventListener("playing", onReady);
      setTimeout(function () {
        finish(video.videoWidth > 0);
      }, 8000);
    });
  }

  function initCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      cameraError = true;
      if (noCam) noCam.classList.remove("hidden");
      updateHandUI("Camera not supported", "0/21", "Waiting");
      return Promise.resolve(false);
    }

    stopCameraStream();
    updateHandUI("Starting camera...", "0/21", "Waiting");

    return navigator.mediaDevices
      .getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      })
      .then(function (stream) {
        if (!video) return false;
        video.srcObject = stream;
        return video.play().then(function () {
          return waitForVideoFrames();
        });
      })
      .then(function (hasFrames) {
        if (!hasFrames) {
          throw new Error("Camera produced no frames");
        }
        cameraReady = true;
        cameraError = false;
        if (noCam) noCam.classList.add("hidden");
        return true;
      })
      .catch(function (err) {
        console.error("Camera init failed:", err);
        cameraReady = false;
        cameraError = true;
        if (noCam) noCam.classList.remove("hidden");
        updateHandUI("Camera access denied", "0/21", "Waiting");
        return false;
      });
  }

  function startHandLoop() {
    var busy = false;
    var loop = function () {
      if (!cameraError && handsModel && video && video.readyState >= 2 && video.videoWidth > 0) {
        if (!busy) {
          busy = true;
          handsModel
            .send({ image: video })
            .catch(function (err) {
              console.warn("MediaPipe send error:", err);
            })
            .finally(function () {
              busy = false;
            });
        }
      }
      requestAnimationFrame(loop);
    };
    loop();
  }

  function startMpCamera() {
    if (typeof Camera === "undefined" || !video) {
      return initCamera().then(function (camOk) {
        if (camOk) startHandLoop();
        return camOk;
      });
    }

    if (mpCamera) {
      try {
        mpCamera.stop();
      } catch (e) {
        console.warn("Camera stop error:", e);
      }
      mpCamera = null;
    }

    mpCamera = new Camera(video, {
      onFrame: function () {
        if (!handsModel || cameraError) return Promise.resolve();
        return handsModel.send({ image: video });
      },
      width: 1280,
      height: 720,
    });

    return mpCamera
      .start()
      .then(function () {
          cameraReady = true;
          cameraError = false;
        if (noCam) noCam.classList.add("hidden");
        return waitForVideoFrames().then(function (hasFrames) {
          if (!hasFrames) throw new Error("Camera produced no frames");
          return true;
        });
      })
      .catch(function (err) {
        console.warn("MediaPipe camera helper failed, using fallback:", err);
        if (mpCamera) {
          try {
            mpCamera.stop();
          } catch (e) {
            console.warn("Camera stop error:", e);
          }
          mpCamera = null;
        }
        return initCamera().then(function (camOk) {
          if (camOk) startHandLoop();
          return camOk;
        });
      });
  }

  function initHandTracking() {
    if (handInitInProgress) return;
    if (typeof Hands === "undefined") {
      updateHandUI("Hand tracking unavailable", "0/21", "Waiting");
      return;
    }

    handInitInProgress = true;
    handTrackingReady = false;
    updateHandUI("Loading hand model...", "0/21", "Waiting");

    if (handsModel) {
      try {
        handsModel.close();
      } catch (e) {
        console.warn("Hands close error:", e);
      }
      handsModel = null;
    }

    handsModel = new Hands({
      locateFile: function (file) {
        return "https://cdn.jsdelivr.net/npm/@mediapipe/hands/" + file;
      },
    });

    handsModel.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5,
    });

    handsModel.onResults(function (results) {
      if (!handTrackingReady) {
        handTrackingReady = true;
        updateHandUI("Searching", "0/21", "Waiting");
      }
      onResults(results);
    });

    var initPromise = handsModel.initialize
      ? handsModel.initialize()
      : Promise.resolve();

    initPromise
      .then(function () {
        updateHandUI("Starting camera...", "0/21", "Waiting");
        if (typeof Camera !== "undefined") {
          return startMpCamera();
        }
        return initCamera().then(function (camOk) {
          if (camOk) startHandLoop();
          return camOk;
        });
      })
      .then(function (camOk) {
        if (camOk === false) return;
        setTimeout(function () {
          if (!handTrackingReady && cameraReady && !cameraError) {
            updateHandUI("Hand model loading — please wait", "0/21", "Waiting");
          }
        }, 12000);
      })
      .catch(function (err) {
        console.error("Hand tracking init failed:", err);
        updateHandUI("Hand tracking failed — click Retry", "0/21", "Waiting");
        cameraError = true;
        if (noCam) noCam.classList.remove("hidden");
      })
      .finally(function () {
        handInitInProgress = false;
      });
  }

  if (retryBtn) {
    retryBtn.addEventListener("click", function () {
    cameraError = false;
      handTrackingReady = false;
      if (noCam) noCam.classList.add("hidden");
      initHandTracking();
    });
  }

  document.addEventListener("ar:modules-loaded", function () {
    refreshModules();
  });

  function flashHoloChip(id) {
    var chip = document.getElementById(id);
    if (!chip) return;
    chip.classList.remove("val-updated");
    void chip.offsetWidth;
    chip.classList.add("val-updated");
    setTimeout(function () {
      chip.classList.remove("val-updated");
    }, 600);
  }

  function boot() {
    if (!video || !canvas) {
      updateHandUI("Camera elements missing", "0/21", "Waiting");
      return;
    }
    initControls();
    if (holoCityWrap) holoCityWrap.classList.remove("visible");
    applyZoom();
    applyHoloTransform();
    refreshModules();
    setModule(0, true);
    initHoloParallax();
    initHandTracking();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
