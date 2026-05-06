/**
 * AI Dashboard — loads summary JSON, renders KPIs, Chart.js charts, sortable table, activity feed.
 */
(function () {
  "use strict";

  var charts = { line: null, bar: null, pie: null };

  function $(sel, root) {
    return (root || document).querySelector(sel);
  }

  function formatInt(n) {
    return typeof n === "number" ? n.toLocaleString(undefined, { maximumFractionDigits: 0 }) : "—";
  }

  function formatMoney(n) {
    if (typeof n !== "number") return "—";
    return (
      "$" +
      n.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })
    );
  }

  function trendHtml(key, trends) {
    var t = trends && trends[key];
    if (!t || typeof t.delta_pct !== "number") {
      return '<span class="ai-kpi-trend ai-kpi-trend--muted">—</span>';
    }
    var up = !!t.up;
    var arrow = up ? "↑" : "↓";
    var good =
      key === "avg_response_ms" ? !up : key === "success_rate" ? up : up;
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
    };
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

    var barPalette = [
      "rgba(232, 121, 249, 0.75)",
      "rgba(168, 85, 247, 0.75)",
      "rgba(56, 189, 248, 0.75)",
      "rgba(244, 114, 182, 0.75)",
      "rgba(129, 140, 248, 0.75)",
      "rgba(34, 211, 238, 0.75)",
      "rgba(251, 191, 36, 0.72)",
      "rgba(248, 113, 113, 0.72)",
    ];

    var lineCtx = $("#aiChartLine");
    if (lineCtx) {
      charts.line = new Chart(lineCtx, {
        type: "line",
        data: {
          labels: line.labels,
          datasets: [
            {
              label: "Requests",
              data: line.values,
              fill: true,
              tension: 0.35,
              borderWidth: 2,
              pointRadius: 0,
              pointHoverRadius: 4,
              borderColor: "rgba(232, 121, 249, 0.95)",
              backgroundColor: "rgba(168, 85, 247, 0.12)",
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
              borderColor: "rgba(168, 85, 247, 0.35)",
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
      charts.bar = new Chart(barCtx, {
        type: "bar",
        data: {
          labels: bar.labels,
          datasets: [
            {
              label: "Requests",
              data: bar.values,
              borderRadius: 8,
              borderSkipped: false,
              backgroundColor: bar.values.map(function (_, i) {
                return barPalette[i % barPalette.length];
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
              borderColor: "rgba(168, 85, 247, 0.35)",
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
      var colors = [
        "rgba(232, 121, 249, 0.85)",
        "rgba(168, 85, 247, 0.85)",
        "rgba(56, 189, 248, 0.85)",
        "rgba(244, 114, 182, 0.85)",
        "rgba(129, 140, 248, 0.85)",
        "rgba(34, 211, 238, 0.85)",
        "rgba(251, 191, 36, 0.85)",
        "rgba(248, 113, 113, 0.85)",
      ];
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
              backgroundColor: pie.map(function (_, i) {
                return colors[i % colors.length];
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
              borderColor: "rgba(168, 85, 247, 0.35)",
              borderWidth: 1,
            },
          },
        },
      });
    }
  }

  function renderKpis(root, data) {
    var kpis = data.kpis || {};
    var trends = data.kpi_trends || {};
    var cards = [
      {
        icon: "🧠",
        label: "Total models used",
        key: "total_models",
        val: formatInt(kpis.total_models),
        trendKey: "total_models",
      },
      {
        icon: "📡",
        label: "Total requests",
        key: "total_requests",
        val: formatInt(kpis.total_requests),
        trendKey: "total_requests",
      },
      {
        icon: "✅",
        label: "Success rate",
        key: "success_rate",
        val: (kpis.success_rate != null ? kpis.success_rate.toFixed(2) : "—") + "%",
        trendKey: "success_rate",
      },
      {
        icon: "⏱",
        label: "Avg response time",
        key: "avg_response_ms",
        val: formatInt(kpis.avg_response_ms) + " ms",
        trendKey: "avg_response_ms",
      },
      {
        icon: "👥",
        label: "Active users",
        key: "active_users",
        val: formatInt(kpis.active_users),
        trendKey: "active_users",
      },
      {
        icon: "💰",
        label: "Revenue (est.)",
        key: "revenue_usd",
        val: formatMoney(kpis.revenue_usd),
        trendKey: "revenue_usd",
      },
    ];

    var html = cards
      .map(function (c) {
        return (
          '<article class="ai-kpi-card" data-kpi-key="' +
          c.key +
          '">' +
          '<div class="ai-kpi-card-inner">' +
          '<div class="ai-kpi-icon" aria-hidden="true">' +
          c.icon +
          "</div>" +
          '<div class="ai-kpi-label">' +
          c.label +
          "</div>" +
          '<div class="ai-kpi-value">' +
          c.val +
          "</div>" +
          trendHtml(c.trendKey, trends) +
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

  function renderTable(models) {
    var tbody = $("#aiModelTableBody");
    if (!tbody) return;
    var rows = (models || []).slice().sort(function (a, b) {
      return compareModels(a, b, sortState.key);
    });
    tbody.innerHTML = rows
      .map(function (m) {
        var ok = m.status === "active";
        return (
          "<tr data-model-id=\"" +
          String(m.id).replace(/"/g, "&quot;") +
          '">' +
          "<td><strong>" +
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
          (ok ? "Active" : "Down") +
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
      renderTable(modelsRef.current);
    });
  }

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

  function fillModelFilter(data) {
    var sel = $("#aiFilterModel");
    if (!sel) return;
    var cur = sel.value;
    var models = data.filter_options || data.models || [];
    var opts =
      '<option value="all">All models</option>' +
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

  function buildQuery(root) {
    var params = new URLSearchParams();
    var m = $("#aiFilterModel");
    var df = $('input[name="date_from"]');
    var dt = $('input[name="date_to"]');
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

  var modelsRef = { current: [] };

  function applyDashboard(root, apiUrl, data) {
    modelsRef.current = data.models || [];
    fillModelFilter(data);
    renderKpis(root, data);
    renderTable(modelsRef.current);
    wireTableSortOnce();
    renderFeed(data.activity);
    buildCharts(root, data);
    var hint = $("#aiGeneratedAt");
    if (hint && data.generated_at) {
      hint.textContent = "Updated · " + data.generated_at;
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
    $('input[name="date_from"]') &&
      $('input[name="date_from"]').addEventListener("change", function () {
        fetchSummary(root, apiUrl);
      });
    $('input[name="date_to"]') &&
      $('input[name="date_to"]').addEventListener("change", function () {
        fetchSummary(root, apiUrl);
      });

    var auto = $("#aiAutoRefresh");
    var refreshMs = 45000;
    var timer = null;
    function armTimer() {
      if (timer) clearInterval(timer);
      timer = null;
      if (auto && auto.checked) {
        timer = setInterval(function () {
          fetchSummary(root, apiUrl);
        }, refreshMs);
      }
    }
    if (auto) {
      auto.addEventListener("change", armTimer);
      armTimer();
    }

    fetchSummary(root, apiUrl);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
