/* ==========================================================
   MedinaMind — Full-Screen AR HUD + Gestures
   Hologram is HIDDEN by default. Shown only by gesture or demo.
   ========================================================== */

(function () {
  "use strict";

  /* ---- DOM refs ------------------------------------------ */
  const video        = document.getElementById("camVideo");
  const canvas       = document.getElementById("camCanvas");
  const ctx          = canvas.getContext("2d");
  const noCam        = document.getElementById("noCam");
  const retryBtn     = document.getElementById("retryBtn");
  const holoCityWrap = document.getElementById("holoCityWrap");
  const demoBtn      = document.getElementById("demoBtn");

  const gestureState   = document.getElementById("gestureState");
  const pointsDetected = document.getElementById("pointsDetected");
  const holoStatusTxt  = document.getElementById("holoStatusTxt");

  const modules = Array.from(document.querySelectorAll(".hud-module"));
  const statCards = [
    document.getElementById("holoCardTraffic"),
    document.getElementById("holoCardEnergy"),
    document.getElementById("holoCardPollution"),
    document.getElementById("holoCardSecurity"),
    document.getElementById("holoCardParking")
  ];

  /* ---- state --------------------------------------------- */
  let handsModel = null;

  let cameraReady       = false;
  let handTrackingReady = false;
  let cameraError       = false;
  let demoMode          = false;
  let handDetected      = false;
  let hologramVisible   = false;

  /* AR Smoothing */
  let currentTargetX     = null;
  let currentTargetY     = null;
  let currentTargetScale = null;
  const SMOOTHING = 0.15;

  /* Gesture State */
  let currentModuleIndex = 0;
  let lastPinchTs  = 0;
  let lastSwipeTs  = 0;
  let palmXHistory = [];

  /* ---- helper: set hologram status text ------------------- */
  function updateHoloStatus() {
    if (holoStatusTxt) {
      holoStatusTxt.textContent = hologramVisible ? "Actif ✦" : "Masqué";
      holoStatusTxt.style.color = hologramVisible ? "#10B981" : "";
    }
  }

  /* ---- module selection ---------------------------------- */
  function setModule(index) {
    modules.forEach((mod, i) => {
      if (i === index) mod.classList.add("active");
      else mod.classList.remove("active");
    });
    statCards.forEach((card, i) => {
      if (card) {
        if (i === index) card.classList.add("active-module");
        else card.classList.remove("active-module");
      }
    });
  }

  /* ---- finger extension detection ------------------------ */
  function isFingerExtended(lm, tip, mcp) {
    return lm[tip].y < lm[mcp].y - 0.03;
  }

  function countExtended(lm) {
    const thumbExt = Math.abs(lm[4].x - lm[3].x) > 0.05;
    const idx  = isFingerExtended(lm, 8,  5);
    const mid  = isFingerExtended(lm, 12, 9);
    const ring = isFingerExtended(lm, 16, 13);
    const pnk  = isFingerExtended(lm, 20, 17);
    return [thumbExt, idx, mid, ring, pnk];
  }

  function isOpenHand(extended) {
    return extended.filter(Boolean).length >= 4;
  }

  function isFist(extended) {
    return extended.filter(Boolean).length <= 1;
  }

  function detectPinch(lm) {
    const dx = lm[4].x - lm[8].x;
    const dy = lm[4].y - lm[8].y;
    return Math.sqrt(dx * dx + dy * dy) < 0.05;
  }

  /* ---- hologram show / hide ------------------------------ */
  function showHologram() {
    if (hologramVisible) return;
    hologramVisible = true;
    holoCityWrap.classList.add("visible");
    setModule(currentModuleIndex);
    updateHoloStatus();
    console.log("Hologram shown");
  }

  function hideHologram() {
    if (!hologramVisible) return;
    if (demoMode) return;  // Never hide while demo is on
    hologramVisible = false;
    holoCityWrap.classList.remove("visible");
    currentTargetX = null;
    updateHoloStatus();
    console.log("Hologram hidden");
  }

  /* ---- draw hand landmarks ------------------------------- */
  function drawLandmarks(lm) {
    const w = canvas.width;
    const h = canvas.height;

    const connections = window.HAND_CONNECTIONS || [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [5,9],[9,10],[10,11],[11,12],
      [9,13],[13,14],[14,15],[15,16],
      [13,17],[17,18],[18,19],[19,20],
      [0,17]
    ];

    // Connections
    ctx.strokeStyle = "rgba(6,182,212,0.55)";
    ctx.lineWidth = 2;
    for (const [a, b] of connections) {
      ctx.beginPath();
      ctx.moveTo(lm[a].x * w, lm[a].y * h);
      ctx.lineTo(lm[b].x * w, lm[b].y * h);
      ctx.stroke();
    }

    // Landmark dots
    for (let i = 0; i < lm.length; i++) {
      const x = lm[i].x * w;
      const y = lm[i].y * h;
      const isTip = [4, 8, 12, 16, 20].includes(i);

      ctx.beginPath();
      ctx.arc(x, y, isTip ? 6 : 3.5, 0, Math.PI * 2);
      ctx.fillStyle = isTip ? "#22D3EE" : "rgba(37,99,235,0.9)";
      ctx.fill();

      if (isTip) {
        ctx.beginPath();
        ctx.arc(x, y, 11, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(34,211,238,0.25)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
  }

  function getPalmCenter(lm) {
    return {
      x: (lm[0].x + lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 5,
      y: (lm[0].y + lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 5
    };
  }

  /* ---- MediaPipe results callback ------------------------ */
  function onResults(results) {
    canvas.width  = video.videoWidth  || canvas.offsetWidth;
    canvas.height = video.videoHeight || canvas.offsetHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const detected = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
    handDetected = detected;

    /* ---- No hands detected ---- */
    if (!detected) {
      if (cameraReady && handTrackingReady) {
        gestureState.textContent = "Recherche...";
      }
      pointsDetected.textContent = "0/21";
      currentTargetX = null;
      palmXHistory = [];

      // Hide hologram if no demo
      if (!demoMode) hideHologram();
      return;
    }

    /* ---- Hands detected — analyze ---- */
    let totalPoints = 0;
    let primaryHandLm = null;
    let openHandsCount = 0;

    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const lm = results.multiHandLandmarks[i];
      totalPoints += lm.length;
      drawLandmarks(lm);
      if (i === 0) primaryHandLm = lm;

      const ext = countExtended(lm);
      if (isOpenHand(ext)) openHandsCount++;
    }

    pointsDetected.textContent = `${totalPoints}/${results.multiHandLandmarks.length * 21}`;

    const extended = countExtended(primaryHandLm);
    const palm = getPalmCenter(primaryHandLm);
    const now = performance.now();

    /* ---- Swipe detection ---- */
    palmXHistory.push({ x: palm.x, t: now });
    if (palmXHistory.length > 10) palmXHistory.shift();

    if (palmXHistory.length > 5 && (now - lastSwipeTs > 800)) {
      const old = palmXHistory[0];
      const dx = palm.x - old.x;
      if (dx > 0.15) {
        currentModuleIndex = (currentModuleIndex + 1) % modules.length;
        setModule(currentModuleIndex);
        lastSwipeTs = now;
      } else if (dx < -0.15) {
        currentModuleIndex = (currentModuleIndex - 1 + modules.length) % modules.length;
        setModule(currentModuleIndex);
        lastSwipeTs = now;
      }
    }

    /* ---- Determine gesture & whether to show hologram ---- */
    let gestureName = "Main détectée";
    let shouldShow = false;

    if (results.multiHandLandmarks.length === 2 && openHandsCount === 2) {
      gestureName = "Deux mains — Mode AR";
      shouldShow = true;
    } else if (isFist(extended)) {
      gestureName = "Poing";
      shouldShow = false;
      if (!demoMode) hideHologram();
    } else if (detectPinch(primaryHandLm)) {
      gestureName = "Pinch (Sélection)";
      shouldShow = true;
      if (now - lastPinchTs > 1000) {
        const card = statCards[currentModuleIndex];
        if (card) {
          card.style.transform = "scale(1.25)";
          setTimeout(() => { card.style.transform = ""; }, 300);
        }
        lastPinchTs = now;
      }
    } else if (isOpenHand(extended)) {
      gestureName = "Main ouverte";
      shouldShow = true;
    }

    gestureState.textContent = gestureName;

    /* ---- Show or hide hologram based on gesture ---- */
    if (shouldShow && !demoMode) {
      showHologram();
    } else if (!shouldShow && !demoMode) {
      hideHologram();
    }

    /* ---- Position hologram to follow hands ---- */
    if (hologramVisible && !demoMode) {
      const w = window.innerWidth;
      const h = window.innerHeight;
      let targetX, targetY, targetScale;

      if (results.multiHandLandmarks.length === 2 && openHandsCount === 2) {
        const p1 = getPalmCenter(results.multiHandLandmarks[0]);
        const p2 = getPalmCenter(results.multiHandLandmarks[1]);
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        targetX = (1 - midX) * w;
        targetY = midY * h - 80;

        const dist = Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
        targetScale = Math.max(0.6, Math.min(dist * 3.0, 2.2));
      } else {
        targetX = (1 - palm.x) * w;
        targetY = palm.y * h - 180;

        const yOffset = 0.5 - palm.y;
        targetScale = 0.7 + yOffset * 1.2;
        targetScale = Math.max(0.5, Math.min(targetScale, 1.3));
      }

      // Apply smoothing
      if (currentTargetX === null) {
        currentTargetX = targetX;
        currentTargetY = targetY;
        currentTargetScale = targetScale;
      } else {
        currentTargetX += (targetX - currentTargetX) * SMOOTHING;
        currentTargetY += (targetY - currentTargetY) * SMOOTHING;
        currentTargetScale += (targetScale - currentTargetScale) * SMOOTHING;
      }

      const tX = currentTargetX - (w / 2);
      const tY = currentTargetY - (h * 0.55);
      holoCityWrap.style.transform = `translateX(calc(-50% + ${tX}px)) translateY(${tY}px) scale(${currentTargetScale})`;
    }
  }

  /* ---- initialization ------------------------------------ */
  async function initCamera() {
    console.log("Starting camera initialization...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 1280, height: 720 },
        audio: false
      });
      console.log("Camera stream started");

      video.srcObject = stream;
      video.play();

      video.addEventListener("loadedmetadata", () => {
        console.log("Video metadata loaded");
      }, { once: true });

      video.addEventListener("canplay", () => {
        if (!cameraReady) {
          console.log("Camera ready");
          cameraReady = true;
          cameraError = false;
          noCam.classList.add("hidden");

          if (!handTrackingReady) {
            gestureState.textContent = "Caméra active — initialisation suivi...";
          } else {
            gestureState.textContent = "Recherche...";
          }
        }
      }, { once: true });

      return true;
    } catch (err) {
      console.error("Camera error:", err);
      cameraError = true;
      cameraReady = false;
      noCam.classList.remove("hidden");
      gestureState.textContent = "Accès caméra refusé";
      return false;
    }
  }

  async function initHandTracking() {
    if (typeof Hands === "undefined") {
      console.error("MediaPipe Hands is not loaded.");
      gestureState.textContent = "Erreur MediaPipe";
      return;
    }

    handsModel = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    handsModel.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.6,
    });

    handsModel.onResults((results) => {
      if (!handTrackingReady) {
        console.log("Hand tracking ready");
        handTrackingReady = true;
        if (cameraReady) {
          gestureState.textContent = "Recherche...";
        }
      }
      onResults(results);
    });

    const camSuccess = await initCamera();

    if (camSuccess) {
      const loop = async () => {
        if (video.readyState >= 2 && !cameraError) {
          try {
            await handsModel.send({ image: video });
          } catch (e) {
            console.warn("MediaPipe send error:", e);
          }
        }
        requestAnimationFrame(loop);
      };
      video.addEventListener("loadeddata", loop, { once: true });
    }
  }

  /* ---- Retry button -------------------------------------- */
  retryBtn.addEventListener("click", () => {
    console.log("Retrying camera connection...");
    cameraError = false;
    cameraReady = false;
    noCam.classList.add("hidden");
    gestureState.textContent = "Tentative de connexion...";

    if (handsModel) {
      initCamera().then((success) => {
        if (success) {
          const loop = async () => {
            if (video.readyState >= 2 && !cameraError) {
              try { await handsModel.send({ image: video }); } catch(e) {}
            }
            requestAnimationFrame(loop);
          };
          video.addEventListener("loadeddata", loop, { once: true });
        }
      });
    } else {
      initHandTracking();
    }
  });

  /* ---- Demo button --------------------------------------- */
  demoBtn.addEventListener("click", () => {
    demoMode = !demoMode;

    if (demoMode) {
      demoBtn.textContent = "✦ Démo active — Cliquer pour désactiver";
      demoBtn.classList.add("demo-active");
      // Reset transform to centered default
      holoCityWrap.style.transform = "";
      showHologram();
      gestureState.textContent = "Mode Démo";
    } else {
      demoBtn.textContent = "Mode démo hologramme";
      demoBtn.classList.remove("demo-active");
      currentTargetX = null;
      // Only keep hologram if a hand gesture currently warrants it
      if (!handDetected) {
        hologramVisible = true; // force so hideHologram will work
        hideHologram();
      }
      gestureState.textContent = (cameraReady && handTrackingReady)
        ? "Recherche..."
        : "Initialisation...";
    }
  });

  /* ---- Boot ---------------------------------------------- */
  // Ensure hologram is hidden on load
  holoCityWrap.classList.remove("visible");
  updateHoloStatus();

  setTimeout(initHandTracking, 800);
  setModule(0);

})();
