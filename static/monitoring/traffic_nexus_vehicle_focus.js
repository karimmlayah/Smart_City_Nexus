/**
 * Traffic Nexus — Gesture Vehicle Focus
 * Hand pointer + vehicle selection + speed estimation overlay
 */
(function (global) {
  "use strict";

  var MP_VERSION = "0.10.14";
  var MP_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@" + MP_VERSION;
  var MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

  var LM = { INDEX_TIP: 8 };

  var SELECT_HOLD_MS = 200;
  var MIN_SELECT_LOCK_MS = 1500;
  var SELECTION_MISS_GRACE_MS = 1800;
  var HIT_RADIUS_PX = 60;
  var SMOOTH_PREV = 0.75;
  var SMOOTH_CUR = 0.25;
  var SPEED_HISTORY_MAX = 18;
  var SPEED_MIN_WINDOW_MS = 700;
  var SPEED_MIN_SAMPLE_MS = 150;
  var SPEED_SMOOTH = 0.9;
  var SPEED_MAX_KMH = 130;
  var SPEED_UI_MIN_DELTA = 2;
  var SPEED_UI_MIN_INTERVAL_MS = 450;
  var HAND_TARGET_FPS = 15;
  var HAND_INTERVAL_MS = 1000 / HAND_TARGET_FPS;

  var deps = null;
  var enabled = false;
  var running = false;
  var rafId = null;
  var stream = null;
  var landmarker = null;
  var loadError = null;

  var handDetected = false;
  var cursor = { x: 0, y: 0, rawX: 0, rawY: 0, visible: false };
  var pulsePhase = 0;

  var detections = [];
  var trackHistory = {};
  var trackFirstSeen = {};
  var smoothCentroid = {};
  var smoothedSpeedOut = {};
  var metersPerPixel = 0.05;

  var pendingTrackId = null;
  var pendingSince = 0;
  var selectedTrackId = null;
  var selectedSince = 0;
  var selectedVehicle = null;
  var selectedMissingSince = null;
  var lastHandTrackTime = 0;
  var lastStatusText = "";
  var lastStatusKind = "idle";
  var lastInsightUpdateMs = 0;
  var lastDisplayedSpeed = null;

  function setStatus(text, kind) {
    lastStatusText = text || "";
    lastStatusKind = kind || "idle";
    if (deps && deps.setStatus) deps.setStatus(lastStatusText, lastStatusKind);
  }

  function setStatusIfChanged(text, kind) {
    var k = kind || "idle";
    if (text === lastStatusText && k === lastStatusKind) return;
    setStatus(text, k);
  }

  function updateCamStatus(detected) {
    if (deps && deps.updateCamStatus) deps.updateCamStatus(detected);
  }

  function formatType(label) {
    var lb = String(label || "vehicle").toLowerCase();
    if (lb.indexOf("bus") >= 0) return "Bus";
    if (lb.indexOf("truck") >= 0 || lb.indexOf("lorry") >= 0) return "Truck";
    if (lb.indexOf("motor") >= 0 || lb.indexOf("bike") >= 0) return "Motorcycle";
    if (lb.indexOf("ambul") >= 0) return "Ambulance";
    if (lb.indexOf("car") >= 0 || lb.indexOf("vehicle") >= 0) return "Car";
    return label ? label.charAt(0).toUpperCase() + label.slice(1) : "Vehicle";
  }

  function bboxCenter(d) {
    return {
      x: (d.x1 + d.x2) / 2,
      y: (d.y1 + d.y2) / 2,
    };
  }

  function pointInBBox(px, py, d) {
    return px >= d.x1 && px <= d.x2 && py >= d.y1 && py <= d.y2;
  }

  function distToCenter(px, py, d) {
    var c = bboxCenter(d);
    return Math.hypot(px - c.x, py - c.y);
  }

  function hitRadiusPx() {
    var c = deps && deps.canvas;
    if (!c || !c.width) return 80;
    return Math.max(50, Math.min(140, c.width * 0.05));
  }

  function normTrackId(id) {
    var n = Number(id);
    return isNaN(n) ? id : n;
  }

  function findVehicleAt(px, py) {
    var best = null;
    var bestDist = Infinity;
    var radius = hitRadiusPx();
    detections.forEach(function (d) {
      var inside = pointInBBox(px, py, d);
      var dist = distToCenter(px, py, d);
      if (inside || dist <= radius) {
        var score = inside ? dist : dist + 1000;
        if (score < bestDist) {
          bestDist = score;
          best = d;
        }
      }
    });
    return best;
  }

  function smoothCenter(tid, cx, cy) {
    var key = String(tid);
    var prev = smoothCentroid[key];
    if (!prev) {
      smoothCentroid[key] = { cx: cx, cy: cy };
      return smoothCentroid[key];
    }
    prev.cx = prev.cx * 0.8 + cx * 0.2;
    prev.cy = prev.cy * 0.8 + cy * 0.2;
    return prev;
  }

  function median(values) {
    if (!values.length) return null;
    var sorted = values.slice().sort(function (a, b) { return a - b; });
    var mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function inferenceFpsHint() {
    if (global.TnVisionStream && global.TnVisionStream.getAiFps) {
      return Math.max(5, global.TnVisionStream.getAiFps());
    }
    return 10;
  }

  function speedFromDetectionPx(d) {
    var px = Number(d && d.speed_px);
    if (!px || px <= 0 || isNaN(px)) return null;
    return Math.min(SPEED_MAX_KMH, px * inferenceFpsHint() * metersPerPixel * 3.6);
  }

  function recordTrackSample(d, nowMs) {
    var tid = d.track_id;
    var c = bboxCenter(d);
    var sm = smoothCenter(tid, c.x, c.y);
    if (!trackFirstSeen[tid]) trackFirstSeen[tid] = nowMs;
    var hist = trackHistory[tid] || [];
    if (hist.length && (nowMs - hist[hist.length - 1].t) < SPEED_MIN_SAMPLE_MS) {
      hist[hist.length - 1] = { cx: sm.cx, cy: sm.cy, t: nowMs };
    } else {
      hist.push({ cx: sm.cx, cy: sm.cy, t: nowMs });
    }
    if (hist.length > SPEED_HISTORY_MAX) hist.shift();
    trackHistory[tid] = hist;
  }

  function speedFromHistory(tid) {
    var hist = trackHistory[tid];
    if (!hist || hist.length < 2) return null;

    var newest = hist[hist.length - 1];
    var windowStart = newest.t - SPEED_MIN_WINDOW_MS;
    var startIdx = 0;
    for (var i = hist.length - 2; i >= 0; i--) {
      if (hist[i].t <= windowStart) {
        startIdx = i;
        break;
      }
    }

    var start = hist[startIdx];
    var dt = (newest.t - start.t) / 1000;
    if (dt < 0.4) return null;

    var windowSpeed = (Math.hypot(newest.cx - start.cx, newest.cy - start.cy) / dt) * metersPerPixel * 3.6;

    var segSpeeds = [];
    for (var j = startIdx + 1; j < hist.length; j++) {
      var segDt = (hist[j].t - hist[j - 1].t) / 1000;
      if (segDt < 0.1) continue;
      var segDist = Math.hypot(hist[j].cx - hist[j - 1].cx, hist[j].cy - hist[j - 1].cy);
      var segSpd = (segDist / segDt) * metersPerPixel * 3.6;
      if (segSpd > 0 && segSpd <= SPEED_MAX_KMH) segSpeeds.push(segSpd);
    }

    var raw = windowSpeed;
    if (segSpeeds.length >= 2) {
      var med = median(segSpeeds);
      if (med != null) {
        var filtered = segSpeeds.filter(function (v) {
          return v >= med * 0.35 && v <= med * 2.0;
        });
        if (filtered.length) {
          raw = filtered.reduce(function (a, b) { return a + b; }, 0) / filtered.length;
        } else {
          raw = med;
        }
      }
    }

    return Math.max(0, Math.min(SPEED_MAX_KMH, raw));
  }

  function computeSpeedKmh(tid, det) {
    var histSpeed = speedFromHistory(tid);
    var detSpeed = det ? speedFromDetectionPx(det) : null;
    var raw = null;
    if (histSpeed != null && detSpeed != null) {
      raw = histSpeed * 0.7 + detSpeed * 0.3;
    } else if (histSpeed != null) {
      raw = histSpeed;
    } else if (detSpeed != null) {
      raw = detSpeed;
    }
    if (raw == null || isNaN(raw)) {
      var prev = smoothedSpeedOut[String(tid)];
      return prev != null && !isNaN(prev) ? prev : null;
    }

    raw = Math.max(0, Math.min(SPEED_MAX_KMH, raw));
    var key = String(tid);
    var prevOut = smoothedSpeedOut[key];
    if (prevOut == null || isNaN(prevOut)) {
      smoothedSpeedOut[key] = raw;
    } else {
      smoothedSpeedOut[key] = prevOut * SPEED_SMOOTH + raw * (1 - SPEED_SMOOTH);
    }
    return smoothedSpeedOut[key];
  }

  function computeDirection(tid) {
    var hist = trackHistory[tid];
    if (!hist || hist.length < 2) return "Stable direction";
    var a = hist[hist.length - 2];
    var b = hist[hist.length - 1];
    var dx = b.cx - a.cx;
    var dy = b.cy - a.cy;
    if (Math.hypot(dx, dy) < 2) return "Stable direction";
    if (Math.abs(dx) > Math.abs(dy)) {
      return dx > 0 ? "Moving right" : "Moving left";
    }
    return dy > 0 ? "Moving down" : "Moving up";
  }

  function statusFromSpeed(kmh) {
    if (kmh === null || isNaN(kmh)) return { status: "Unknown", risk: "Low" };
    if (kmh < 3) return { status: "Stopped", risk: "Medium" };
    if (kmh < 15) return { status: "Slow", risk: "Medium" };
    if (kmh <= 60) return { status: "Normal", risk: "Low" };
    return { status: "Fast", risk: "High" };
  }

  function smartNote(status, direction, zoneName) {
    if (status === "Stopped") return "Vehicle appears stationary — monitor for queue buildup.";
    if (status === "Fast") return "Higher-than-normal speed detected in the monitored corridor.";
    if (direction.indexOf("left") >= 0 || direction.indexOf("right") >= 0) {
      return "Lateral movement suggests lane change or turning behavior.";
    }
    if (zoneName) return "Vehicle is moving normally inside " + zoneName + ".";
    return "Vehicle is moving normally inside the monitored corridor.";
  }

  function buildInsight(d, nowMs) {
    var tid = d.track_id;
    var speed = computeSpeedKmh(tid, d);
    var dir = computeDirection(tid);
    var st = statusFromSpeed(speed);
    var zoneIdx = typeof d.zone_idx === "number" ? d.zone_idx : 0;
    var zoneName = zoneIdx >= 0 ? "Zone " + (zoneIdx + 1) : null;
    if (deps && deps.getZoneName) {
      var zn = deps.getZoneName(zoneIdx);
      if (zn) zoneName = zn;
    }
    var trackedSec = trackFirstSeen[tid]
      ? Math.max(0, (nowMs - trackFirstSeen[tid]) / 1000)
      : 0;
    return {
      track_id: tid,
      vehicle_id: "Vehicle #" + String(tid).padStart(2, "0"),
      type: formatType(d.label),
      speed_kmh: speed,
      direction: dir,
      status: st.status,
      risk: st.risk,
      zone: zoneName || "Full frame",
      confidence: Math.round((d.conf || 0) * 100),
      time_tracked_s: trackedSec,
      smart_note: smartNote(st.status, dir, zoneName),
      bbox: { x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 },
    };
  }

  function confirmSelection(d) {
    var nowMs = Date.now();
    selectedTrackId = normTrackId(d.track_id);
    selectedSince = nowMs;
    lastDisplayedSpeed = null;
    lastInsightUpdateMs = 0;
    publishInsightIfNeeded(buildInsight(d, nowMs), nowMs);
    setStatus("Vehicle selected", "success");
  }

  function clearSelection() {
    selectedTrackId = null;
    selectedVehicle = null;
    pendingTrackId = null;
    lastDisplayedSpeed = null;
    lastInsightUpdateMs = 0;
    if (deps && deps.onSelectionChange) deps.onSelectionChange(null);
  }

  function publishInsightIfNeeded(insight, nowMs) {
    var speed = insight.speed_kmh;
    var prevSpeed = lastDisplayedSpeed;
    var speedDelta = (speed == null || prevSpeed == null)
      ? 999
      : Math.abs(speed - prevSpeed);
    var statusChanged = !selectedVehicle || selectedVehicle.status !== insight.status;
    var typeChanged = !selectedVehicle || selectedVehicle.type !== insight.type;
    var dueTime = (nowMs - lastInsightUpdateMs) >= SPEED_UI_MIN_INTERVAL_MS;
    if (speedDelta >= SPEED_UI_MIN_DELTA || statusChanged || typeChanged || dueTime) {
      selectedVehicle = insight;
      lastDisplayedSpeed = speed;
      lastInsightUpdateMs = nowMs;
      if (deps && deps.onSelectionChange) deps.onSelectionChange(selectedVehicle);
      return;
    }
    if (selectedVehicle) {
      selectedVehicle.bbox = insight.bbox;
      selectedVehicle.direction = insight.direction;
      selectedVehicle.time_tracked_s = insight.time_tracked_s;
    }
  }

  function processPointer(px, py, immediate) {
    if (!detections.length) {
      pendingTrackId = null;
      if (enabled) setStatus("Run analysis first — no vehicles detected yet", "warn");
      return;
    }
    var hit = findVehicleAt(px, py);
    var now = Date.now();
    if (!hit) {
      pendingTrackId = null;
      return;
    }
    var hitId = normTrackId(hit.track_id);
    if (immediate) {
      confirmSelection(hit);
      pendingTrackId = hitId;
      pendingSince = now;
      return;
    }
    if (hitId !== pendingTrackId) {
      pendingTrackId = hitId;
      pendingSince = now;
      setStatus("Hold on vehicle…", "active");
      return;
    }
    if (now - pendingSince >= SELECT_HOLD_MS) {
      if (selectedTrackId === null || selectedTrackId === hitId) {
        confirmSelection(hit);
      } else if (now - selectedSince >= MIN_SELECT_LOCK_MS) {
        confirmSelection(hit);
      }
    }
  }

  function mapToCanvas(lm, canvas) {
    var tip = lm[LM.INDEX_TIP];
    var x = (1 - tip.x) * canvas.width;
    var y = tip.y * canvas.height;
    return [
      Math.max(0, Math.min(canvas.width - 1, x)),
      Math.max(0, Math.min(canvas.height - 1, y)),
    ];
  }

  function smoothCursor(rawX, rawY) {
    if (!cursor.visible) {
      cursor.x = rawX;
      cursor.y = rawY;
    } else {
      cursor.x = cursor.x * SMOOTH_PREV + rawX * SMOOTH_CUR;
      cursor.y = cursor.y * SMOOTH_PREV + rawY * SMOOTH_CUR;
    }
    cursor.rawX = rawX;
    cursor.rawY = rawY;
  }

  function syncOverlaySize() {
    if (!deps || !deps.overlayCanvas || !deps.canvas) return;
    var oc = deps.overlayCanvas;
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
    oc.style.right = "auto";
    oc.style.bottom = "auto";
  }

  function drawDetectionHints(octx) {
    if (!detections.length) return;
    var colorFn = global.TnVisionStream && global.TnVisionStream.colorForClass
      ? function (d) { return global.TnVisionStream.colorForClass(d.label, d.cls_id); }
      : function () { return "#00d5ff"; };
    detections.forEach(function (d) {
      var tid = normTrackId(d.track_id);
      var pending = pendingTrackId === tid && selectedTrackId !== tid;
      var stroke = colorFn(d);
      octx.save();
      octx.strokeStyle = stroke;
      octx.globalAlpha = pending ? 0.85 : 0.24;
      octx.lineWidth = pending ? 2 : 1;
      octx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
      octx.restore();
    });
  }

  function drawSelectedVehicle(octx) {
    if (!selectedVehicle || !selectedVehicle.bbox) return;
    var b = selectedVehicle.bbox;
    var pad = 4;
    pulsePhase += 0.08;
    var glow = 8 + Math.sin(pulsePhase) * 4;
    octx.save();
    octx.strokeStyle = "#00e5ff";
    octx.lineWidth = 3;
    octx.shadowColor = "#00e5ff";
    octx.shadowBlur = glow;
    octx.strokeRect(b.x1 - pad, b.y1 - pad, b.x2 - b.x1 + pad * 2, b.y2 - b.y1 + pad * 2);
    octx.shadowBlur = 0;
    octx.font = "bold 11px Inter, sans-serif";
    octx.fillStyle = "rgba(0, 229, 255, 0.92)";
    octx.fillText("SELECTED VEHICLE", b.x1, Math.max(14, b.y1 - 8));
    var cx = (b.x1 + b.x2) / 2;
    var cy = (b.y1 + b.y2) / 2;
    if (cursor.visible) {
      octx.beginPath();
      octx.moveTo(cx, cy);
      octx.lineTo(cursor.x, cursor.y);
      octx.strokeStyle = "rgba(0, 213, 255, 0.35)";
      octx.lineWidth = 1.5;
      octx.setLineDash([6, 4]);
      octx.stroke();
      octx.setLineDash([]);
    }
    octx.restore();
  }

  function drawPointer(octx) {
    if (!cursor.visible) return;
    pulsePhase += 0.06;
    var r = 12 + Math.sin(pulsePhase) * 3;
    octx.save();
    octx.beginPath();
    octx.arc(cursor.x, cursor.y, r, 0, Math.PI * 2);
    octx.strokeStyle = "rgba(0, 229, 255, 0.35)";
    octx.lineWidth = 2;
    octx.stroke();
    octx.beginPath();
    octx.arc(cursor.x, cursor.y, 5, 0, Math.PI * 2);
    octx.fillStyle = "#00e5ff";
    octx.shadowColor = "#00e5ff";
    octx.shadowBlur = 14;
    octx.fill();
    octx.shadowBlur = 0;
    octx.font = "600 10px Inter, sans-serif";
    octx.fillStyle = "rgba(0, 229, 255, 0.9)";
    octx.fillText("Pointer", cursor.x + 10, cursor.y - 10);
    octx.restore();
  }

  function drawOverlay() {
    if (!deps || !deps.overlayCanvas) return;
    syncOverlaySize();
    var oc = deps.overlayCanvas;
    var octx = oc.getContext("2d");
    octx.clearRect(0, 0, oc.width, oc.height);
    if (!enabled && !selectedVehicle && !detections.length) return;
    drawDetectionHints(octx);
    drawSelectedVehicle(octx);
    if (enabled) drawPointer(octx);
  }

  function tick(ts) {
    if (!enabled || !running) return;
    rafId = global.requestAnimationFrame(tick);

    var video = deps && deps.videoEl;
    var canvas = deps && deps.canvas;
    var nowTs = ts || performance.now();
    var shouldDetect = (nowTs - lastHandTrackTime) >= HAND_INTERVAL_MS;
    var streamDraws = global.TnVisionStream && global.TnVisionStream.isRunning && global.TnVisionStream.isRunning();

    if (!video || !canvas || !landmarker || video.readyState < 2) {
      if (shouldDetect) {
        handDetected = false;
        cursor.visible = false;
        updateCamStatus(false);
        setStatusIfChanged("Waiting for hand", "wait");
      }
      if (!streamDraws) drawOverlay();
      return;
    }

    if (shouldDetect) {
      lastHandTrackTime = nowTs;
      try {
        var results = landmarker.detectForVideo(video, nowTs);
        if (results.landmarks && results.landmarks.length > 0) {
          handDetected = true;
          updateCamStatus(true);
          var lm = results.landmarks[0];
          var xy = mapToCanvas(lm, canvas);
          smoothCursor(xy[0], xy[1]);
          cursor.visible = true;
          setStatusIfChanged("Point at a vehicle (" + detections.length + " detected)", "active");
          processPointer(cursor.x, cursor.y, false);
          refreshSelectedInsight();
        } else {
          handDetected = false;
          cursor.visible = false;
          updateCamStatus(false);
          setStatusIfChanged("Waiting for hand", "wait");
        }
      } catch (e) {
        handDetected = false;
        cursor.visible = false;
        updateCamStatus(false);
      }
    }
    if (!streamDraws) drawOverlay();
  }

  function refreshSelectedInsight() {
    if (selectedTrackId === null) return;
    var d = null;
    detections.forEach(function (v) {
      if (normTrackId(v.track_id) === selectedTrackId) d = v;
    });
    if (!d) return;
    var nowMs = Date.now();
    publishInsightIfNeeded(buildInsight(d, nowMs), nowMs);
  }

  function stopStream() {
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    if (deps && deps.videoEl) deps.videoEl.srcObject = null;
  }

  function stopLoop() {
    running = false;
    if (rafId) {
      global.cancelAnimationFrame(rafId);
      rafId = null;
    }
    cursor.visible = false;
    handDetected = false;
    pendingTrackId = null;
    updateCamStatus(false);
    drawOverlay();
  }

  async function ensureLandmarker() {
    if (landmarker) return landmarker;
    if (loadError) throw loadError;
    try {
      var mod = await import(MP_CDN);
      var vision = await mod.FilesetResolver.forVisionTasks(MP_CDN + "/wasm");
      try {
        landmarker = await mod.HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numHands: 1,
        });
      } catch (gpuErr) {
        landmarker = await mod.HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
          runningMode: "VIDEO",
          numHands: 1,
        });
      }
      return landmarker;
    } catch (e) {
      loadError = e;
      throw e;
    }
  }

  async function startWebcam() {
    await ensureLandmarker();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    var video = deps.videoEl;
    video.srcObject = stream;
    await video.play();
    running = true;
    setStatus("Gesture Vehicle Focus active", "active");
    tick();
  }

  async function setEnabled(on) {
    enabled = !!on;
    if (!enabled) {
      stopLoop();
      stopStream();
      setStatus("Gesture focus off", "idle");
      if (deps && deps.onModeChange) deps.onModeChange(false);
      drawOverlay();
      return;
    }
    if (deps && deps.onModeChange) deps.onModeChange(true);
    setStatus("Starting gesture camera…", "wait");
    try {
      await startWebcam();
    } catch (e) {
      enabled = false;
      stopStream();
      var msg = "Gesture camera unavailable. You can still inspect vehicles with mouse hover/click.";
      if (e && e.name === "NotAllowedError") msg = "Webcam permission denied. Use mouse hover/click on vehicles.";
      else if (loadError) msg = "MediaPipe failed to load. Check network connection.";
      setStatus(msg, "error");
      if (deps && deps.showMsg) deps.showMsg(msg, true);
      if (deps && deps.onModeChange) deps.onModeChange(false);
    }
  }

  function updateDetections(list, timestampMs) {
    var nowMs = timestampMs || Date.now();
    detections = (list || []).map(function (d) {
      return {
        track_id: normTrackId(d.track_id),
        label: String(d.label || "vehicle"),
        conf: Number(d.conf) || 0,
        x1: Number(d.x1) || 0,
        y1: Number(d.y1) || 0,
        x2: Number(d.x2) || 0,
        y2: Number(d.y2) || 0,
        zone_idx: Number(d.zone_idx) || 0,
        speed_px: Number(d.speed_px) || 0,
        speed_status: String(d.speed_status || "Medium"),
      };
    });
    detections.forEach(function (d) {
      recordTrackSample(d, nowMs);
    });
    if (selectedTrackId !== null) {
      var still = detections.some(function (d) { return normTrackId(d.track_id) === selectedTrackId; });
      if (still) {
        selectedMissingSince = null;
        publishInsightIfNeeded(buildInsight(
          detections.filter(function (d) { return normTrackId(d.track_id) === selectedTrackId; })[0],
          nowMs
        ), nowMs);
      } else if (!selectedMissingSince) {
        selectedMissingSince = nowMs;
      } else if (nowMs - selectedMissingSince >= SELECTION_MISS_GRACE_MS) {
        selectedTrackId = null;
        selectedVehicle = null;
        selectedMissingSince = null;
        if (deps && deps.onSelectionChange) deps.onSelectionChange(null);
      }
    }
    if (deps && deps.onDetectionCount) deps.onDetectionCount(detections.length);
    drawOverlay();
  }

  function setMetersPerPixel(v) {
    var n = parseFloat(v);
    if (!isNaN(n) && n > 0) metersPerPixel = n;
  }

  function pointerAtCanvasCoords(px, py, immediate) {
    processPointer(px, py, !!immediate);
    drawOverlay();
  }

  function init(options) {
    deps = options || {};
    if (deps.metersPerPixel) setMetersPerPixel(deps.metersPerPixel);
  }

  function isEnabled() {
    return enabled;
  }

  function isHandDetected() {
    return handDetected;
  }

  function getCursorPosition() {
    if (!cursor.visible) return null;
    return [Math.round(cursor.x), Math.round(cursor.y)];
  }

  function getSelectedVehicle() {
    return selectedVehicle;
  }

  function destroy() {
    setEnabled(false);
    clearSelection();
    landmarker = null;
    detections = [];
    trackHistory = {};
    trackFirstSeen = {};
    smoothCentroid = {};
    smoothedSpeedOut = {};
    deps = null;
  }

  global.TnVehicleFocus = {
    init: init,
    setEnabled: setEnabled,
    isEnabled: isEnabled,
    isHandDetected: isHandDetected,
    getCursorPosition: getCursorPosition,
    getSelectedVehicle: getSelectedVehicle,
    updateDetections: updateDetections,
    setMetersPerPixel: setMetersPerPixel,
    pointerAtCanvasCoords: pointerAtCanvasCoords,
    clearSelection: clearSelection,
    destroy: destroy,
    drawOverlay: drawOverlay,
    syncOverlaySize: syncOverlaySize,
  };
})(window);
