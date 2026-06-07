/**
 * Traffic Nexus — Smooth vision stream engine
 * Separates display rAF loop from throttled AI inference scheduling.
 */
(function (global) {
  "use strict";

  var INFERENCE_WIDTH = 640;
  var SMOOTH = 0.75;

  var CLASS_COLORS_HEX = {
    ambulance: "#ffb428",
    car: "#50dc50",
    motorcycle: "#ff9f43",
    bus: "#ffc14d",
    truck: "#6eb8ff",
    van: "#7eb8ff",
    bicycle: "#52d6ba",
    person: "#c9a0ff",
    congestion: "#ff5050",
    congested: "#ff5050",
    heavy: "#ff5050",
    high: "#ff5050",
    jam: "#ff5050",
    faible: "#50dc50",
    low: "#50dc50",
    light: "#50dc50",
    low_congestion: "#50dc50",
    congestion_faible: "#50dc50",
    normal: "#50dc50",
    free: "#50dc50",
    medium: "#ffd700",
    moderate: "#ffd700",
  };

  var FALLBACK_CLASS_COLORS = [
    "#50dc50", "#ff5050", "#ffd700", "#6eb8ff", "#c9a0ff",
  ];

  function _isAmbulanceLabel(label) {
    var lb = String(label || "").trim().toLowerCase();
    if (!lb) return false;
    if (lb.indexOf("ambul") >= 0) return true;
    if (lb.indexOf("emergency") >= 0 || lb === "ems") return true;
    return false;
  }

  function formatClassLabel(label) {
    var raw = String(label || "vehicle").trim();
    var lb = raw.toLowerCase();
    if (_isAmbulanceLabel(raw)) return "Ambulance";
    if (lb.indexOf("ambulance:") === 0) return "Ambulance";
    if (lb.indexOf("bus") >= 0) return "Bus";
    if (lb.indexOf("truck") >= 0 || lb.indexOf("lorry") >= 0) return "Truck";
    if (lb.indexOf("motor") >= 0 || lb.indexOf("bike") >= 0) return "Motorcycle";
    if (lb.indexOf("bicycle") >= 0 || lb === "bike") return "Bicycle";
    if (lb.indexOf("car") >= 0 || lb.indexOf("vehicle") >= 0) return "Car";
    if (lb.indexOf("van") >= 0) return "Van";
    if (lb.indexOf("person") >= 0 || lb === "pedestrian") return "Person";
    if (raw.length <= 3 && /^[a-z0-9_-]+$/i.test(raw)) {
      return raw.charAt(0).toUpperCase() + raw.slice(1);
    }
    return raw.charAt(0).toUpperCase() + raw.slice(1);
  }

  function colorForClass(label, clsId) {
    if (_isAmbulanceLabel(label)) return CLASS_COLORS_HEX.ambulance;
    var cid = Number(clsId);
    if (!isNaN(cid)) {
      if (cid === 1) return "#50dc50";
      if (cid === 0) return "#ff5050";
    }
    var key = String(label || "").trim().toLowerCase();
    if (key.indexOf("ambulance:") === 0) return CLASS_COLORS_HEX.ambulance;
    if (CLASS_COLORS_HEX[key]) return CLASS_COLORS_HEX[key];
    var base = key.split(":")[0].split(" ")[0];
    if (CLASS_COLORS_HEX[base]) return CLASS_COLORS_HEX[base];
    var idx = !isNaN(cid) ? cid : 0;
    for (var i = 0; i < key.length; i++) idx += key.charCodeAt(i);
    return FALLBACK_CLASS_COLORS[Math.abs(idx) % FALLBACK_CLASS_COLORS.length];
  }

  var deps = null;
  var displayRafId = null;
  var running = false;
  var liveInferenceEnabled = false;
  var inferenceRunning = false;
  var lastInferenceTime = 0;
  var aiFps = 10;
  var inferenceCount = 0;
  var inferenceWindowStart = 0;
  var measuredAiFps = 0;

  var frameImage = null;
  var frameReady = false;
  var pendingB64 = null;
  var uploadVideoMode = false;
  var useClientOverlay = true;
  var displayMode = "normal_detection";

  var detections = [];
  var smoothed = {};
  var hiddenCanvas = null;
  var hiddenCtx = null;

  function init(options) {
    deps = options || {};
    frameImage = new Image();
    frameImage.onload = function () {
      frameReady = true;
      pendingB64 = null;
    };
    hiddenCanvas = document.createElement("canvas");
    hiddenCtx = hiddenCanvas.getContext("2d", { alpha: false });
    if (deps.aiFps) setAiFps(deps.aiFps);
  }

  function setAiFps(fps) {
    var n = Number(fps);
    if (!isNaN(n) && n >= 5 && n <= 20) aiFps = n;
    updateIndicators();
  }

  function getAiIntervalMs() {
    return 1000 / aiFps;
  }

  function setUploadVideoMode(on) {
    uploadVideoMode = !!on;
  }

  function setClientOverlay(on) {
    useClientOverlay = !!on;
  }

  function setDisplayMode(mode) {
    displayMode = mode || "normal_detection";
    useClientOverlay = displayMode !== "congestion_heatmap";
  }

  function setLiveInferenceEnabled(on) {
    liveInferenceEnabled = !!on;
    if (on) lastInferenceTime = 0;
  }

  function start() {
    if (running) return;
    running = true;
    inferenceWindowStart = performance.now();
    inferenceCount = 0;
    displayLoop(performance.now());
  }

  function stop() {
    running = false;
    liveInferenceEnabled = false;
    if (displayRafId) {
      cancelAnimationFrame(displayRafId);
      displayRafId = null;
    }
    inferenceRunning = false;
  }

  function isInferenceRunning() {
    return inferenceRunning;
  }

  function syncDetOverlaySize() {
    if (!deps || !deps.detOverlay || !deps.canvas) return;
    var oc = deps.detOverlay;
    var c = deps.canvas;
    var wrap = c.parentElement;
    if (oc.width !== c.width || oc.height !== c.height) {
      oc.width = c.width;
      oc.height = c.height;
    }
    if (!wrap) return;
    var wrapRect = wrap.getBoundingClientRect();
    var rect = c.getBoundingClientRect();
    oc.style.position = "absolute";
    oc.style.left = (rect.left - wrapRect.left) + "px";
    oc.style.top = (rect.top - wrapRect.top) + "px";
    oc.style.width = rect.width + "px";
    oc.style.height = rect.height + "px";
  }

  function drawUploadVideo() {
    if (!deps || !deps.uploadVideo || !deps.canvas || !deps.ctx) return false;
    var v = deps.uploadVideo;
    if (!v.src || v.readyState < 2 || !v.videoWidth) return false;
    var c = deps.canvas;
    var ctx = deps.ctx;
    if (c.width !== v.videoWidth || c.height !== v.videoHeight) {
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      if (deps.onCanvasResize) deps.onCanvasResize(c.width, c.height);
    }
    ctx.drawImage(v, 0, 0);
    c.style.display = "block";
    if (deps.hidePlaceholder) deps.hidePlaceholder();
    return true;
  }

  function drawCachedFrame() {
    if (!deps || !deps.canvas || !deps.ctx || !frameReady || !frameImage) return;
    var c = deps.canvas;
    if (frameImage.naturalWidth && (c.width !== frameImage.naturalWidth || c.height !== frameImage.naturalHeight)) {
      c.width = frameImage.naturalWidth;
      c.height = frameImage.naturalHeight;
      if (deps.onCanvasResize) deps.onCanvasResize(c.width, c.height);
    }
    deps.ctx.drawImage(frameImage, 0, 0);
    c.style.display = "block";
    if (deps.hidePlaceholder) deps.hidePlaceholder();
  }

  function smoothBox(key, box) {
    var prev = smoothed[key];
    if (!prev) {
      smoothed[key] = { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 };
      return smoothed[key];
    }
    prev.x1 = prev.x1 * SMOOTH + box.x1 * (1 - SMOOTH);
    prev.y1 = prev.y1 * SMOOTH + box.y1 * (1 - SMOOTH);
    prev.x2 = prev.x2 * SMOOTH + box.x2 * (1 - SMOOTH);
    prev.y2 = prev.y2 * SMOOTH + box.y2 * (1 - SMOOTH);
    return prev;
  }

  function drawDetections() {
    if (!deps || !deps.detOverlay) return;
    syncDetOverlaySize();
    var oc = deps.detOverlay;
    var octx = oc.getContext("2d");
    octx.clearRect(0, 0, oc.width, oc.height);
    if (!useClientOverlay || displayMode === "congestion_heatmap") return;
    if (!detections.length) return;

    var showConf = typeof deps.showConfidence === "function"
      ? deps.showConfidence()
      : deps.showConfidence !== false;
    detections.forEach(function (d, i) {
      var key = String(d.track_id != null ? d.track_id : i);
      var b = smoothBox(key, {
        x1: Number(d.x1) || 0,
        y1: Number(d.y1) || 0,
        x2: Number(d.x2) || 0,
        y2: Number(d.y2) || 0,
      });
      var classLabel = formatClassLabel(d.label);
      var stroke = colorForClass(d.label, d.cls_id);
      octx.strokeStyle = stroke;
      octx.lineWidth = _isAmbulanceLabel(d.label) ? 3 : 2;
      octx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
      var lbl = classLabel;
      if (showConf) lbl += " " + (Number(d.conf) || 0).toFixed(2);
      if (typeof d.zone_idx === "number" && d.zone_idx >= 0) {
        lbl += " · Z" + (d.zone_idx + 1);
      }
      octx.font = "600 11px Inter, sans-serif";
      var tw = octx.measureText(lbl).width + 8;
      octx.fillStyle = "rgba(0,0,0,0.55)";
      octx.fillRect(b.x1, Math.max(0, b.y1 - 16), tw, 16);
      octx.fillStyle = stroke;
      octx.fillText(lbl, b.x1 + 4, Math.max(12, b.y1 - 4));
    });
  }

  function displayLoop(ts) {
    if (!running) return;

    if (uploadVideoMode && liveInferenceEnabled) {
      drawUploadVideo();
    } else if (frameReady) {
      drawCachedFrame();
    }

    if (deps && deps.drawRoi) deps.drawRoi();
    drawDetections();
    if (deps && deps.drawGestureOverlay) deps.drawGestureOverlay();
    syncDetOverlaySize();
    if (deps && deps.syncGestureOverlay) deps.syncGestureOverlay();

    if (
      liveInferenceEnabled &&
      deps &&
      deps.runInference &&
      !inferenceRunning &&
      document.visibilityState === "visible" &&
      ts - lastInferenceTime >= getAiIntervalMs()
    ) {
      lastInferenceTime = ts;
      triggerInference();
    }

    var elapsed = ts - inferenceWindowStart;
    if (elapsed >= 2000) {
      measuredAiFps = Math.round((inferenceCount * 1000) / elapsed);
      inferenceCount = 0;
      inferenceWindowStart = ts;
      updateIndicators();
    }

    displayRafId = requestAnimationFrame(displayLoop);
  }

  function triggerInference() {
    if (inferenceRunning || !deps || !deps.runInference) return;
    inferenceRunning = true;
    Promise.resolve(deps.runInference())
      .then(function () {
        inferenceCount++;
      })
      .catch(function () {})
      .finally(function () {
        inferenceRunning = false;
      });
  }

  function setFrameFromBase64(b64, opts) {
    if (!b64) return;
    opts = opts || {};
    pendingB64 = b64;
    if (opts.storeOnly) {
      if (deps && deps.onImageBase64) deps.onImageBase64(b64);
      return;
    }
    frameImage.onload = function () {
      frameReady = true;
      pendingB64 = null;
      if (deps && deps.onImageBase64) deps.onImageBase64(b64);
      if (deps && deps.onCanvasResize && frameImage.naturalWidth) {
        deps.onCanvasResize(frameImage.naturalWidth, frameImage.naturalHeight);
      }
    };
    frameImage.src = "data:image/jpeg;base64," + b64;
  }

  function updateDetections(list) {
    detections = (list || []).slice();
    var active = {};
    detections.forEach(function (d, i) {
      active[String(d.track_id != null ? d.track_id : i)] = true;
    });
    Object.keys(smoothed).forEach(function (k) {
      if (!active[k]) delete smoothed[k];
    });
  }

  function captureInferenceBase64() {
    if (!deps || !deps.canvas) return null;
    if (uploadVideoMode && deps.uploadVideo && deps.uploadVideo.readyState >= 2) {
      var v = deps.uploadVideo;
      var h = Math.round((v.videoHeight / v.videoWidth) * INFERENCE_WIDTH);
      hiddenCanvas.width = INFERENCE_WIDTH;
      hiddenCanvas.height = Math.max(1, h);
      hiddenCtx.drawImage(v, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
      return hiddenCanvas.toDataURL("image/jpeg", 0.82).split(",")[1];
    }
    if (!frameReady && !deps.canvas.width) return deps.getImageBase64 ? deps.getImageBase64() : null;
    var c = deps.canvas;
    var scale = INFERENCE_WIDTH / Math.max(1, c.width);
    hiddenCanvas.width = INFERENCE_WIDTH;
    hiddenCanvas.height = Math.max(1, Math.round(c.height * scale));
    hiddenCtx.drawImage(c, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
    return hiddenCanvas.toDataURL("image/jpeg", 0.82).split(",")[1];
  }

  function updateIndicators() {
    if (!deps || !deps.indicators) return;
    var el = deps.indicators;
    if (el.video) el.video.textContent = "Video: Smooth";
    if (el.ai) el.ai.textContent = "AI: " + (measuredAiFps || aiFps) + " FPS";
    if (el.gesture) {
      var gestActive = typeof deps.gestureActive === "function"
        ? deps.gestureActive()
        : !!deps.gestureActive;
      el.gesture.textContent = gestActive ? "Gesture: Active" : "Gesture: Off";
    }
    if (el.source && deps.getSourceLabel) el.source.textContent = "Source: " + deps.getSourceLabel();
  }

  function redrawRoiOnly() {
    if (uploadVideoMode && liveInferenceEnabled) drawUploadVideo();
    else if (frameReady) drawCachedFrame();
    if (deps && deps.drawRoi) deps.drawRoi();
    drawDetections();
    if (deps && deps.drawGestureOverlay) deps.drawGestureOverlay();
  }

  global.TnVisionStream = {
    init: init,
    start: start,
    stop: stop,
    isRunning: function () { return running; },
    setAiFps: setAiFps,
    getAiFps: function () { return aiFps; },
    formatClassLabel: formatClassLabel,
    colorForClass: colorForClass,
    setUploadVideoMode: setUploadVideoMode,
    setClientOverlay: setClientOverlay,
    setDisplayMode: setDisplayMode,
    setLiveInferenceEnabled: setLiveInferenceEnabled,
    isInferenceRunning: isInferenceRunning,
    setFrameFromBase64: setFrameFromBase64,
    updateDetections: updateDetections,
    captureInferenceBase64: captureInferenceBase64,
    drawDetections: drawDetections,
    syncDetOverlaySize: syncDetOverlaySize,
    redrawRoiOnly: redrawRoiOnly,
    updateIndicators: updateIndicators,
  };
})(window);
