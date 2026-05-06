/**
 * AI Dashboard — JSON-driven KPIs, Chart.js, table, activity (Smart City semantic colors).
 */
(function () {
  "use strict";

  var charts = { line: null, bar: null, pie: null };
  var datesInitialized = false;

  function $(sel, root) {
    return (root || document).querySelector(sel);
  }

  function formatInt(n) {
    return typeof n === "number" ? n.toLocaleString(undefined, { maximumFractionDigits: 0 }) : "—";
  }

  function formatKpiValue(kpis, card) {
    if (!kpis || !card || !card.kpi) return "—";
    var v = kpis[card.kpi];
    if (v == null || typeof v === "undefined") return "—";
    var f = card.format || "int";
    if (f === "pct") return (typeof v === "number" ? v.toFixed(2) : v) + "%";
    if (f === "ms") return formatInt(v) + " ms";
    return formatInt(v);
  }

  function trendHtmlCard(card, trends) {
    var key = card.trend_key;
    var tg = card.trend_good || "up";
    if (tg === "neutral") {
      return '<span class="ai-kpi-trend ai-kpi-trend--muted">steady baseline</span>';
    }
    var t = trends && trends[key];
    if (!t || typeof t.delta_pct !== "number") {
      return '<span class="ai-kpi-trend ai-kpi-trend--muted">—</span>';
    }
    var up = !!t.up;
    var arrow = up ? "↑" : "↓";
    var good =
      tg === "up" ? up : tg === "down" ? !up : up;
    var cls = good ? "ai-kpi-trend--up" : "ai-kpi-trend--down";
    var sign = t.delta_pct >= 0 ? "+" : "";
    return (
      '<div class="ai-kpi-trend ' +
      cls +
      '">' +
      arrow +
      " " +
      sign +
      t.delta_pct +
      "% vs last period</div>"
    );
  }

  function chartTextColors(root) {
    var light = root.getAttribute("data-ai-theme") === "light";
    return {
      light: light,
      tick: light ? "#64748b" : "#94a3b8",
      grid: light ? "rgba(15,23,42,0.08)" : "rgba(255,255,255,0.06)",
      tooltipBorder: light ? "rgba(14, 165, 233, 0.35)" : "rgba(14, 165, 233, 0.45)",
    };
  }

  function paletteFillForDomain(domain, semantic_palette) {
    var pal = semantic_palette || {};
    var d = domain && pal[domain] ? domain : "vision";
    var row = pal[d] || pal.vision || {};
    return row.fill || "rgba(56, 189, 248, 0.78)";
  }

  function destroyCharts() {
    ["line", "bar", "pie"].forEach(function (k) {
      if (charts[k]) {
        charts[k].destroy();
        charts[k] = null;
      }
    });
  }

  function buildCharts(root, data) {
    if (typeof Chart === "undefined") return;
    destroyCharts();
    var pal = chartTextColors(root);
    var ch = data.charts || {};
    var line = ch.requests_over_time || { labels: [], values: [] };
    var bar = ch.usage_per_model || { labels: [], values: [] };
    var pie = ch.distribution || [];
    var sem = data.semantic_palette || {};
    var ct = data.chart_theme || {};
    var lineTheme = ct.requests_line || {};
    var lineBorder = lineTheme.borderColor || "rgba(14, 165, 233, 0.92)";
    var lineFill = lineTheme.backgroundColor || "rgba(14, 165, 233, 0.12)";
    var barDomains = bar.domains || [];

    var lineCtx = $("#aiChartLine");
    if (lineCtx) {
      charts.line = new Chart(lineCtx, {
        type: "line",
        data: {
          labels: line.labels,
          datasets: [
            {
              label: lineTheme.label || "Throughput",
              data: line.values,
              fill: true,
              tension: 0.35,
              borderWidth: 2,
              pointRadius: 0,
              pointHoverRadius: 4,
              borderColor: lineBorder,
              backgroundColor: lineFill,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { intersect: false, mode: "index" },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: pal.light ? "rgba(255,255,255,0.96)" : "rgba(15,23,42,0.94)",
              titleColor: pal.light ? "#0f172a" : "#f1f5f9",
              bodyColor: pal.light ? "#334155" : "#cbd5e1",
              borderColor: pal.tooltipBorder,
              borderWidth: 1,
            },
          },
          scales: {
            x: {
              ticks: { color: pal.tick, maxRotation: 45, minRotation: 0 },
              grid: { color: pal.grid },
            },
            y: {
              ticks: { color: pal.tick },
              grid: { color: pal.grid },
              beginAtZero: true,
            },
          },
        },
      });
    }

    var barCtx = $("#aiChartBar");
    if (barCtx) {
      var vals = bar.values || [];
      charts.bar = new Chart(barCtx, {
        type: "bar",
        data: {
          labels: bar.labels,
          datasets: [
            {
              label: "Load index",
              data: vals,
              borderRadius: 8,
              borderSkipped: false,
              backgroundColor: vals.map(function (_, i) {
                return paletteFillForDomain(barDomains[i] || "vision", sem);
              }),
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: pal.light ? "rgba(255,255,255,0.96)" : "rgba(15,23,42,0.94)",
              titleColor: pal.light ? "#0f172a" : "#f1f5f9",
              bodyColor: pal.light ? "#334155" : "#cbd5e1",
              borderColor: pal.tooltipBorder,
              borderWidth: 1,
            },
          },
          scales: {
            x: {
              ticks: { color: pal.tick, autoSkip: true, maxTicksLimit: 8 },
              grid: { display: false },
            },
            y: {
              ticks: { color: pal.tick },
              grid: { color: pal.grid },
              beginAtZero: true,
            },
          },
        },
      });
    }

    var pieCtx = $("#aiChartPie");
    if (pieCtx && pie.length) {
      charts.pie = new Chart(pieCtx, {
        type: "pie",
        data: {
          labels: pie.map(function (p) {
            return p.label;
          }),
          datasets: [
            {
              data: pie.map(function (p) {
                return p.value;
              }),
              backgroundColor: pie.map(function (p) {
                return paletteFillForDomain(p.domain || "vision", sem);
              }),
              borderWidth: 2,
              borderColor: pal.light ? "#fff" : "rgba(15,23,42,0.85)",
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "bottom",
              labels: {
                color: pal.tick,
                boxWidth: 12,
                font: { size: 10 },
              },
            },
            tooltip: {
              backgroundColor: pal.light ? "rgba(255,255,255,0.96)" : "rgba(15,23,42,0.94)",
              titleColor: pal.light ? "#0f172a" : "#f1f5f9",
              bodyColor: pal.light ? "#334155" : "#cbd5e1",
              borderColor: pal.tooltipBorder,
              borderWidth: 1,
            },
          },
        },
      });
    }
  }

  function fallbackKpiCards() {
    return [
      {
        kpi: "total_models",
        label: "Registered endpoints",
        hint: "",
        icon: "◎",
        format: "int",
        trend_key: "total_models",
        trend_good: "neutral",
      },
      {
        kpi: "total_requests",
        label: "Inference load index",
        hint: "",
        icon: "📡",
        format: "int",
        trend_key: "total_requests",
        trend_good: "up",
      },
      {
        kpi: "success_rate",
        label: "Fleet reliability",
        hint: "",
        icon: "✓",
        format: "pct",
        trend_key: "success_rate",
        trend_good: "up",
      },
      {
        kpi: "avg_response_ms",
        label: "Latency index",
        hint: "",
        icon: "⏱",
        format: "ms",
        trend_key: "avg_response_ms",
        trend_good: "down",
      },
      {
        kpi: "online_services",
        label: "Online services",
        hint: "",
        icon: "●",
        format: "int",
        trend_key: "online_services",
        trend_good: "up",
      },
      {
        kpi: "offline_services",
        label: "Needs attention",
        hint: "",
        icon: "⚠",
        format: "int",
        trend_key: "offline_services",
        trend_good: "down",
      },
    ];
  }

  function renderKpis(root, data) {
    var kpis = data.kpis || {};
    var trends = data.kpi_trends || {};
    var cards = data.kpi_cards && data.kpi_cards.length ? data.kpi_cards : fallbackKpiCards();

    var html = cards
      .map(function (c) {
        var dom = String(c.kpi || "").replace(/[^a-z0-9_-]/gi, "") || "metric";
        var hint = c.hint
          ? '<div class="ai-kpi-hint">' + escapeHtml(c.hint) + "</div>"
          : "";
        return (
          '<article class="ai-kpi-card ai-kpi-card--domain-' +
          escapeHtml(dom) +
          '" data-kpi-key="' +
          escapeHtml(c.kpi) +
          '">' +
          '<div class="ai-kpi-card-inner">' +
          '<div class="ai-kpi-icon" aria-hidden="true">' +
          (c.icon || "◆") +
          "</div>" +
          '<div class="ai-kpi-label">' +
          escapeHtml(c.label || c.kpi) +
          "</div>" +
          '<div class="ai-kpi-value">' +
          formatKpiValue(kpis, c) +
          "</div>" +
          hint +
          trendHtmlCard(c, trends) +
          "</div></article>"
        );
      })
      .join("");

    var mount = $("#aiKpiGrid");
    if (mount) mount.innerHTML = html;
  }

  var sortState = { key: "requests", dir: -1 };

  function compareModels(a, b, key) {
    var dir = sortState.dir;
    var va, vb;
    if (key === "name" || key === "provider") {
      va = String(a[key] || "").toLowerCase();
      vb = String(b[key] || "").toLowerCase();
      return va < vb ? -dir : va > vb ? dir : 0;
    }
    if (key === "status") {
      va = a.status === "active" ? 1 : 0;
      vb = b.status === "active" ? 1 : 0;
    } else {
      va = Number(a[key]);
      vb = Number(b[key]);
    }
    return va < vb ? -dir : va > vb ? dir : 0;
  }

  function renderTable(models, ui) {
    var tbody = $("#aiModelTableBody");
    if (!tbody) return;
    var st = (ui && ui.status_labels) || { active: "Online", down: "Offline" };
    var rows = (models || []).slice().sort(function (a, b) {
      return compareModels(a, b, sortState.key);
    });
    tbody.innerHTML = rows
      .map(function (m) {
        var ok = m.status === "active";
        var dom = String(m.domain || "vision").replace(/[^a-z0-9_-]/gi, "") || "vision";
        var label = ok ? st.active : st.down;
        return (
          "<tr data-model-id=\"" +
          String(m.id).replace(/"/g, "&quot;") +
          '">' +
          "<td><span class=\"ai-domain-dot ai-domain-dot--" +
          escapeHtml(dom) +
          '" title="' +
          escapeHtml(dom) +
          '" aria-hidden="true"></span><strong>' +
          escapeHtml(m.name) +
          "</strong></td>" +
          "<td>" +
          escapeHtml(m.provider) +
          "</td>" +
          "<td>" +
          formatInt(m.requests) +
          "</td>" +
          "<td>" +
          formatInt(m.avg_ms) +
          " ms</td>" +
          "<td>" +
          (m.error_rate != null ? m.error_rate.toFixed(2) : "—") +
          "%</td>" +
          "<td>" +
          '<span class="ai-badge ' +
          (ok ? "ai-badge--ok" : "ai-badge--bad") +
          '">' +
          '<span class="ai-status-dot ' +
          (ok ? "ai-status-dot--live" : "") +
          '" style="' +
          (ok ? "" : "background:#ef4444;box-shadow:none") +
          '"></span>' +
          escapeHtml(label) +
          "</span></td></tr>"
        );
      })
      .join("");
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  var sortDelegationBound = false;

  function wireTableSortOnce() {
    if (sortDelegationBound) return;
    var table = $("#aiModelTable");
    if (!table || !table.tHead) return;
    sortDelegationBound = true;
    table.tHead.addEventListener("click", function (e) {
      var th = e.target.closest("th[data-sort]");
      if (!th || !table.contains(th)) return;
      var key = th.getAttribute("data-sort");
      if (!key) return;
      if (sortState.key === key) sortState.dir = -sortState.dir;
      else {
        sortState.key = key;
        sortState.dir = key === "name" || key === "provider" ? 1 : -1;
      }
      renderTable(modelsRef.current, uiRef.current);
    });
  }

  var uiRef = { current: null };
  var modelsRef = { current: [] };

  function renderFeed(items) {
    var el = $("#aiActivityFeed");
    if (!el) return;
    el.innerHTML = (items || [])
      .map(function (it) {
        var sev = it.severity || "info";
        var rowClass = "ai-feed-item";
        if (sev === "warn") rowClass += " ai-feed-item--warn";
        if (sev === "error") rowClass += " ai-feed-item--error";
        return (
          '<div class="' +
          rowClass +
          '">' +
          '<div class="ai-feed-meta"><span>' +
          escapeHtml(it.time) +
          '</span><span>·</span><span>' +
          escapeHtml(it.type) +
          "</span></div>" +
          "<div>" +
          escapeHtml(it.message) +
          "</div></div>"
        );
      })
      .join("");
  }

  function fillModelFilter(data, ui) {
    var sel = $("#aiFilterModel");
    if (!sel) return;
    var cur = sel.value;
    var models = data.filter_options || data.models || [];
    var allLabel =
      (ui && ui.toolbar && ui.toolbar.model_all) || "All endpoints";
    var opts =
      '<option value="all">' + escapeHtml(allLabel) + "</option>" +
      models
        .map(function (m) {
          return (
            '<option value="' +
            escapeHtml(m.id) +
            '">' +
            escapeHtml(m.name) +
            "</option>"
          );
        })
        .join("");
    sel.innerHTML = opts;
    if (cur && Array.from(sel.options).some(function (o) { return o.value === cur; })) {
      sel.value = cur;
    }
  }

  function applyUi(ui) {
    if (!ui) return;
    uiRef.current = ui;
    var tb = ui.toolbar || {};
    var el;

    el = $("#aiNavTag");
    if (el && ui.nav_tag) el.textContent = ui.nav_tag;
    el = $("#aiHeroTitle");
    if (el && ui.hero_title) el.textContent = ui.hero_title;
    el = $("#aiHeroSub");
    if (el && ui.hero_subtitle) el.textContent = ui.hero_subtitle;

    el = $("#aiLblModel");
    if (el && tb.model_label) el.textContent = tb.model_label;
    el = $("#aiLblFrom");
    if (el && tb.from_label) el.textContent = tb.from_label;
    el = $("#aiLblTo");
    if (el && tb.to_label) el.textContent = tb.to_label;
    el = $("#aiLblTheme");
    if (el && tb.theme) el.textContent = tb.theme;
    el = $("#aiLblRefresh");
    if (el && tb.refresh) el.textContent = tb.refresh;
    el = $("#aiLblAutoRefresh");
    if (el && tb.auto_refresh != null) {
      el.textContent = "Auto-refresh (" + tb.auto_refresh + "s)";
    }

    var pan = ui.panels || {};
    function setPanel(prefix, p) {
      if (!p) return;
      var k = $(prefix + "Kicker");
      var t = $(prefix + "Title");
      if (k && p.kicker) k.textContent = p.kicker;
      if (t && p.title) t.textContent = p.title;
    }
    setPanel("#aiPanelLine", pan.line);
    setPanel("#aiPanelBar", pan.bar);
    setPanel("#aiPanelPie", pan.pie);
    setPanel("#aiPanelTable", pan.table);
    setPanel("#aiPanelFeed", pan.feed);

    var th = ui.table_headers || {};
    var map = [
      ["aiThName", th.name],
      ["aiThProvider", th.provider],
      ["aiThRequests", th.requests],
      ["aiThLatency", th.avg_ms],
      ["aiThErr", th.error_rate],
      ["aiThStatus", th.status],
    ];
    map.forEach(function (pair) {
      var node = $("#" + pair[0]);
      if (node && pair[1]) node.textContent = pair[1];
    });
  }

  function applyFilterDates(data) {
    var f = data.filters || {};
    if (datesInitialized) return;
    var dff = f.default_date_from;
    var dtt = f.default_date_to;
    var i1 = $("#aiDateFrom");
    var i2 = $("#aiDateTo");
    if (i1 && dff) i1.value = String(dff).slice(0, 10);
    if (i2 && dtt) i2.value = String(dtt).slice(0, 10);
    datesInitialized = true;
  }

  function buildQuery(root) {
    var params = new URLSearchParams();
    var m = $("#aiFilterModel");
    var df = $("#aiDateFrom");
    var dt = $("#aiDateTo");
    if (m && m.value && m.value !== "all") params.set("model", m.value);
    if (df && df.value) params.set("date_from", df.value);
    if (dt && dt.value) params.set("date_to", dt.value);
    var qs = params.toString();
    return qs ? "?" + qs : "";
  }

  function setLoading(root, on) {
    if (on) root.classList.add("is-loading");
    else root.classList.remove("is-loading");
  }

  function showSkeleton(root) {
    var mount = $("#aiKpiGrid");
    if (!mount) return;
    mount.innerHTML = "";
    for (var i = 0; i < 6; i++) {
      mount.innerHTML +=
        '<article class="ai-kpi-card"><div class="ai-kpi-card-inner">' +
        '<span class="ai-skeleton" style="width:40%"></span>' +
        '<span class="ai-skeleton ai-skeleton--lg" style="width:72%"></span>' +
        '<span class="ai-skeleton" style="width:55%;margin-top:8px"></span>' +
        "</div></article>";
    }
  }

  function applyDashboard(root, apiUrl, data) {
    modelsRef.current = data.models || [];
    applyUi(data.ui);
    applyFilterDates(data);
    fillModelFilter(data, data.ui);
    renderKpis(root, data);
    renderTable(modelsRef.current, data.ui);
    wireTableSortOnce();
    renderFeed(data.activity);
    buildCharts(root, data);
    var hint = $("#aiGeneratedAt");
    var gl = (data.ui && data.ui.generated_label) || "Updated";
    if (hint && data.generated_at) {
      hint.textContent = gl + " · " + data.generated_at;
    }
  }

  function fetchSummary(root, apiUrl) {
    setLoading(root, true);
    showSkeleton(root);
    var url = apiUrl + buildQuery(root);
    fetch(url, { credentials: "same-origin", headers: { Accept: "application/json" } })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        applyDashboard(root, apiUrl, data);
      })
      .catch(function () {
        console.warn("[AI Dashboard] fetch failed — using bootstrap payload");
        var boot = window.__AI_DASHBOARD_BOOTSTRAP;
        if (boot && typeof boot === "object") applyDashboard(root, apiUrl, boot);
      })
      .finally(function () {
        setLoading(root, false);
      });
  }

  function initClock() {
    function pad(n) {
      return (n < 10 ? "0" : "") + n;
    }
    function tick() {
      var el = document.getElementById("fhClock");
      if (!el) return;
      var d = new Date();
      el.textContent =
        pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
    }
    setInterval(tick, 1000);
    tick();
  }

  function init() {
    var root = $("#aiDashRoot");
    if (!root) return;
    var apiUrl = root.getAttribute("data-api-url") || "";
    initClock();

    $("#aiBtnRefresh") &&
      $("#aiBtnRefresh").addEventListener("click", function () {
        fetchSummary(root, apiUrl);
      });

    var themeBtn = $("#aiBtnTheme");
    if (themeBtn) {
      themeBtn.addEventListener("click", function () {
        var t = root.getAttribute("data-ai-theme") === "light" ? "dark" : "light";
        root.setAttribute("data-ai-theme", t);
        fetchSummary(root, apiUrl);
      });
    }

    $("#aiFilterModel") &&
      $("#aiFilterModel").addEventListener("change", function () {
        fetchSummary(root, apiUrl);
      });
    $("#aiDateFrom") &&
      $("#aiDateFrom").addEventListener("change", function () {
        fetchSummary(root, apiUrl);
      });
    $("#aiDateTo") &&
      $("#aiDateTo").addEventListener("change", function () {
        fetchSummary(root, apiUrl);
      });

    var auto = $("#aiAutoRefresh");
    var timer = null;
    function refreshIntervalMs() {
      var sec = 45;
      try {
        var u = uiRef.current && uiRef.current.toolbar && uiRef.current.toolbar.auto_refresh;
        if (u != null) sec = parseInt(u, 10) || 45;
      } catch (e1) {}
      return Math.max(15000, sec * 1000);
    }
    function armTimer() {
      if (timer) clearInterval(timer);
      timer = null;
      if (auto && auto.checked) {
        timer = setInterval(function () {
          fetchSummary(root, apiUrl);
        }, refreshIntervalMs());
      }
    }
    if (auto) {
      auto.addEventListener("change", armTimer);
    }

    fetchSummary(root, apiUrl);
    if (auto) armTimer();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
