/**
 * Fusion Hub — UI behaviours (timeline strip, modal, tabs, analyse overlay).
 * La logique d’inférence caméra / YouTube demeure dans le script inline fusion_hub.html.
 */
(function () {
  function pad(n) {
    return (n < 10 ? "0" : "") + n;
  }

  function tickClock() {
    var el = document.getElementById("fhClock");
    if (!el) return;
    var d = new Date();
    el.textContent =
      pad(d.getHours()) +
      ":" +
      pad(d.getMinutes()) +
      ":" +
      pad(d.getSeconds());
  }

  /* ——— Tabs (in-page panes only; links like Reports / Settings have no data-pane) ——— */
  document.querySelectorAll(".fh-tab[data-pane]").forEach(function (tab) {
    tab.addEventListener("click", function () {
      if (tab.getAttribute("aria-disabled") === "true") return;
      var pane = tab.getAttribute("data-pane");
      if (!pane) return;
      document.querySelectorAll(".fh-tab[data-pane]").forEach(function (t) {
        t.classList.toggle("is-active", t === tab);
      });
      document.querySelectorAll(".fh-pane").forEach(function (p) {
        p.classList.toggle("is-active", p.id === pane);
      });
    });
  });

  /* ——— Form analyse → loading overlay ——— */
  var form = document.getElementById("fusionBatchForm");
  var overlay = document.getElementById("fhLoadingOverlay");
  var submitAnalyze = document.getElementById("fhAnalyzeBtn");
  if (form && overlay) {
    form.addEventListener("submit", function () {
      if (submitAnalyze && submitAnalyze.closest("fieldset[disabled]"))
        return;
      if (
        form.getAttribute("data-fusion-progressive") === "1" &&
        form.getAttribute("data-fusion-force-block") !== "1"
      )
        return;
      overlay.classList.add("is-on");
      overlay.setAttribute("aria-busy", "true");
    });
  }

  /* ——— Threat cards modal ——— */
  var cardsEl = document.getElementById("fusion-threat-cards");
  var backdrop = document.getElementById("fhEventBackdrop");
  var modalBody = document.getElementById("fhEventModalBody");

  function escapeHtml(s) {
    var d = document.createElement("div");
    d.textContent = String(s == null ? "" : s);
    return d.innerHTML;
  }

  function openEventModal(id) {
    if (!backdrop || !modalBody || !cardsEl) return;
    var list = [];
    try {
      list = JSON.parse(cardsEl.textContent);
    } catch (e) {
      return;
    }
    var c = list.find(function (x) {
      return String(x.id) === String(id);
    });
    if (!c) return;
    var imgs = "";
    (c.crop_urls || []).forEach(function (u) {
      imgs +=
        '<img src="' + escapeHtml(u) + '" style="max-height:140px;border-radius:8px;margin-right:6px;margin-bottom:6px;border:1px solid rgba(120,190,255,0.3)" />';
    });
    var fm = "";
    (c.face_matches || []).forEach(function (x) {
      fm +=
        "<li><strong>" +
        escapeHtml(x.display_name || "") +
        "</strong>" +
        (x.person_code ? " · ID " + escapeHtml(x.person_code) : "") +
        " · dist " +
        escapeHtml(x.distance) +
        "</li>";
    });

    modalBody.innerHTML =
      "<p><strong>Type</strong> · " +
      escapeHtml(c.title || "") +
      "</p>" +
      "<p><strong>Time</strong> · " +
      escapeHtml(c.time_label || "") +
      "</p>" +
      "<p><strong>Confidence</strong> · " +
      escapeHtml(c.confidence_display) +
      "%</p>" +
      (imgs
        ? "<p><strong>Captured crops</strong></p><div>" + imgs + "</div>"
        : "<p>No crop thumbnail for this timeline point.</p>") +
      (fm
        ? "<p><strong>Face registry matches</strong></p><ul>" + fm + "</ul>"
        : "<p>No face matches correlated for this card.</p>") +
      "<p class=\"fh-muted\"><strong>AI summary</strong> · Fusion pipeline signal only; operator must verify context and legal compliance.</p>";
    backdrop.classList.add("is-open");
    backdrop.style.display = "flex";
  }

  function closeEventModal() {
    if (!backdrop) return;
    backdrop.classList.remove("is-open");
    backdrop.style.display = "none";
  }

  var threatScroll = document.getElementById("fhThreatScroll");
  if (threatScroll) {
    threatScroll.addEventListener("click", function (ev) {
      var card = ev.target && ev.target.closest
        ? ev.target.closest(".fh-threat-card[data-event-id]")
        : null;
      if (!card) return;
      openEventModal(card.getAttribute("data-event-id"));
    });
  }
  var evClose = document.getElementById("fhEventClose");
  if (evClose) evClose.addEventListener("click", closeEventModal);
  if (backdrop)
    backdrop.addEventListener("click", function (e) {
      if (e.target === backdrop) closeEventModal();
    });

  /* ——— Report modal ——— */
  var rd = document.getElementById("fusion-report-payload");
  var rbBack = document.getElementById("fhReportBackdrop");
  var rbBody = document.getElementById("fhReportModalBody");
  function openReport() {
    if (!rbBack || !rbBody || !rd) return;
    var payload = {};
    try {
      payload = JSON.parse(rd.textContent || "{}");
    } catch (e) {}
    var threats = "";
    (payload.highlights || []).forEach(function (h) {
      threats +=
        '<tr class="fusion-report-table__row"><td>' +
        escapeHtml(h.title) +
        "</td><td>" +
        escapeHtml(h.time_label) +
        "</td><td>" +
        escapeHtml(h.confidence_display) +
        "%</td></tr>";
    });
    var facesBlock = escapeHtml(payload.faces_summary || "");
    rbBody.innerHTML =
      '<article class="fusion-report-sheet" aria-label="Incident report preview">' +
      '<div class="fusion-report-sheet__glow" aria-hidden="true"></div>' +
      '<h2 class="fusion-report-sheet__title">Incident report (preview)</h2>' +
      '<dl class="fusion-report-meta">' +
      '<div class="fusion-report-meta__row"><dt class="fusion-report-kicker">Run</dt><dd class="fusion-report-value">' +
      escapeHtml(payload.title || "—") +
      "</dd></div>" +
      '<div class="fusion-report-meta__row"><dt class="fusion-report-kicker">Timestamp</dt><dd class="fusion-report-value">' +
      escapeHtml(payload.ts || "—") +
      "</dd></div>" +
      '<div class="fusion-report-meta__row"><dt class="fusion-report-kicker">Verdict</dt><dd class="fusion-report-value fusion-report-value--accent">' +
      escapeHtml(payload.verdict || "—") +
      "</dd></div>" +
      '<div class="fusion-report-meta__row"><dt class="fusion-report-kicker">Mean P(fight)</dt><dd class="fusion-report-value">' +
      escapeHtml(payload.avg_p_fight != null ? String(payload.avg_p_fight) : "—") +
      "</dd></div></dl>" +
      '<div class="fusion-report-table-wrap">' +
      '<table class="fusion-report-table"><thead><tr><th scope="col">Signal</th><th scope="col">Time</th><th scope="col">Conf.</th></tr></thead><tbody>' +
      (threats ||
        '<tr class="fusion-report-table__empty"><td colspan="3">No flagged rows in payload.</td></tr>') +
      "</tbody></table></div>" +
      (facesBlock
        ? '<section class="fusion-report-ident"><h3 class="fusion-report-kicker fusion-report-ident__title">Identities</h3><p class="fusion-report-ident__body">' +
          facesBlock +
          "</p></section>"
        : "") +
      '<p class="fusion-report-action"><span class="fusion-report-kicker fusion-report-action__label">Recommended action</span><span class="fusion-report-action__text">Assess scene, notify security protocol if threat confirmed, preserve chain of custody for media.</span></p>' +
      "</article>";
    rbBack.style.display = "flex";
    rbBack.classList.add("is-open");
  }
  function closeReport() {
    if (!rbBack) return;
    rbBack.style.display = "none";
    rbBack.classList.remove("is-open");
  }
  var qr = document.getElementById("fhQuickReport");
  if (qr) qr.addEventListener("click", openReport);
  var qc = document.getElementById("fhReportClose");
  if (qc) qc.addEventListener("click", closeReport);
  if (rbBack)
    rbBack.addEventListener("click", function (e) {
      if (e.target === rbBack) closeReport();
    });
  var rp = document.getElementById("fhReportPrint");
  if (rp)
    rp.addEventListener("click", function () {
      window.print();
    });

  /* ——— Timeline thumbnails from JSON ——— */
  function buildTimelineStrip() {
    var el = document.getElementById("fhTimelineInner");
    var dataEl = document.getElementById("fight-timeline-data");
    var threatJson = document.getElementById("fusionThreatCardsJson");
    if (!el || !dataEl) return;
    var pts = [];
    try {
      pts = JSON.parse(dataEl.textContent);
    } catch (e) {}
    var thumbsByT = {};
    if (threatJson) {
      try {
        var tc = JSON.parse(threatJson.textContent);
        tc.forEach(function (c) {
          if (
            c.time_label &&
            c.thumb_url &&
            typeof c.t_sec === "number"
          ) {
            thumbsByT[Math.round(c.t_sec * 100) / 100] = c.thumb_url;
          }
        });
      } catch (e2) {}
    }

    function clsFor(pt) {
      if (pt.dual_highlight) return "fh-tl-marker--dual";
      var wm = typeof pt.weapon_max === "number" ? pt.weapon_max : 0;
      if (wm >= 0.38) return "fh-tl-marker--weapon";
      if (pt.label === "fight" || (typeof pt.p_fight === "number" && pt.p_fight >= 0.52))
        return "fh-tl-marker--fight";
      return "";
    }

    pts.forEach(function (pt, ix) {
      if (ix % Math.max(1, Math.ceil(pts.length / 72)) !== 0) return;
      var pack = document.createElement("button");
      pack.type = "button";
      pack.className = "fh-tl-thumb";
      pack.style.border = "none";
      pack.style.padding = "0";
      pack.style.cursor = "pointer";
      var t = pt.t != null ? pt.t : ix;
      var thumb = thumbsByT[Math.round(Number(t) * 100) / 100];
      if (thumb) {
        pack.innerHTML =
          '<img src="' +
          thumb.replace(/"/g, "&quot;") +
          "\" alt=\"t\"/><div style=\"font-size:10px;color:#8ba4c9;padding:2px\">t≈" +
          Number(t).toFixed(1) +
          "s</div>";
      } else {
        pack.innerHTML =
          '<div class="fh-tl-marker ' +
          escapeHtml(clsFor(pt)) +
          "\" style=\"margin:8px auto\"></div>" +
          '<div style=\"font-size:10px;color:#8ba4c9\">t≈' +
          Number(t).toFixed(1) +
          "s</div>";
      }
      pack.addEventListener("click", function () {
        el.querySelectorAll(".fh-tl-thumb").forEach(function (x) {
          x.classList.remove("is-current");
        });
        pack.classList.add("is-current");
        seekMediaSeconds(Number(t));
      });
      el.appendChild(pack);
    });
  }

  function seekMediaSeconds(tSec) {
    var v = document.getElementById("fightLocalVideo");
    if (v && typeof v.currentTime === "number") {
      v.currentTime = Math.max(0, tSec);
      return;
    }
    var yt = document.querySelector("#fightYoutubePlayer iframe");
    if (
      yt &&
      typeof yt.postMessage === "function" &&
      window.YT &&
      window.YT.Player
    ) {
      try {
        var api = yt.ytReadyPlayer;
        if (api && api.seekTo) api.seekTo(tSec, true);
      } catch (e3) {}
    }
  }

  buildTimelineStrip();
  setInterval(tickClock, 1000);
  tickClock();
})();

/**
 * Fusion Hub — cinematic biometric scanner: 0→100%, cycling status, reveal match/unknown.
 */
window.FusionHubBioScanner = (function () {
  var MESSAGES = [
    "Capturing face...",
    "Generating embedding...",
    "Searching database...",
  ];
  var RING_LEN = 113.1;

  function setRingProgress(ringFill, pct) {
    if (!ringFill) return;
    var p = Math.min(100, Math.max(0, pct));
    ringFill.style.strokeDashoffset = String(RING_LEN * (1 - p / 100));
  }

  function statusForPct(p) {
    if (p < 34) return MESSAGES[0];
    if (p < 67) return MESSAGES[1];
    return MESSAGES[2];
  }

  function easeOutQuad(t) {
    return 1 - (1 - t) * (1 - t);
  }

  function finishScan(card) {
    card.classList.remove("fh-bio-phase");
    var dock = card.querySelector(".bio-scanner-dock");
    if (dock) {
      window.setTimeout(function () {
        dock.classList.add("bio-scanner-dock--done");
      }, 400);
    }
    card.querySelectorAll(".fh-bio-reveal-hidden").forEach(function (el) {
      el.classList.remove("fh-bio-reveal-hidden");
      el.classList.add("fh-bio-reveal-visible");
    });
  }

  function runCard(card, index) {
    if (card.getAttribute("data-fusion-bio-init") === "1") return;
    card.setAttribute("data-fusion-bio-init", "1");

    var dock = card.querySelector(".bio-scanner-dock");
    if (!dock) {
      window.setTimeout(function () {
        finishScan(card);
      }, 900 + index * 240);
      return;
    }

    var pctEl = card.querySelector(".bio-scanner-pct");
    var statusEl = card.querySelector(".bio-scanner-status");
    var ringFill = card.querySelector(".bio-scanner-ring-fill");
    var duration = Math.min(4000, 2600 + index * 220);
    var t0 = typeof performance !== "undefined" ? performance.now() : Date.now();

    function loop(now) {
      var tn =
        typeof performance !== "undefined" ? performance.now() : Date.now();
      var raw = Math.min(1, (tn - t0) / duration);
      var eased = easeOutQuad(raw);
      var pRounded = Math.round(eased * 100);
      setRingProgress(ringFill, pRounded);
      if (pctEl) pctEl.textContent = pRounded + "%";
      if (statusEl) statusEl.textContent = statusForPct(pRounded);
      if (raw < 1) {
        requestAnimationFrame(loop);
      } else {
        setRingProgress(ringFill, 100);
        if (pctEl) pctEl.textContent = "100%";
        if (statusEl) statusEl.textContent = MESSAGES[2];
        finishScan(card);
      }
    }

    requestAnimationFrame(loop);
  }

  function initCards(root) {
    var scope = root && root.querySelectorAll ? root : document;
    var nodes = scope.querySelectorAll(".fh-bio-scanner-card.fh-bio-phase");
    if (!nodes.length) return;
    nodes.forEach(function (card, ix) {
      window.setTimeout(function () {
        runCard(card, ix);
      }, 40 + ix * 100);
    });
  }

  return { initCards: initCards };
})();
