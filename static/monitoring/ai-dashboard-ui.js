/**
 * MedinaMind AI Dashboard — UI interactions (sidebar, topbar, theme)
 */
(function () {
  "use strict";

  function pad(n) {
    return (n < 10 ? "0" : "") + n;
  }

  function setSidebarCollapsed(collapsed) {
    var root = document.documentElement;
    var body = document.body;
    var sidebar = document.getElementById("mmSidebar");
    var toggle = document.getElementById("mmSidebarToggle");

    if (collapsed) {
      root.classList.add("mm-sidebar-collapsed");
      if (body) body.classList.add("mm-sidebar-collapsed");
      if (sidebar) sidebar.classList.add("mm-sidebar--collapsed");
      if (toggle) {
        toggle.setAttribute("aria-expanded", "false");
        toggle.title = "Expand navigation";
      }
    } else {
      root.classList.remove("mm-sidebar-collapsed");
      if (body) body.classList.remove("mm-sidebar-collapsed");
      if (sidebar) sidebar.classList.remove("mm-sidebar--collapsed");
      if (toggle) {
        toggle.setAttribute("aria-expanded", "true");
        toggle.title = "Collapse navigation";
      }
    }
  }

  function initSidebarCollapse() {
    var toggle = document.getElementById("mmSidebarToggle");
    var sidebar = document.getElementById("mmSidebar");
    if (!toggle || !sidebar) return;

    var stored = null;
    try {
      stored = localStorage.getItem("mm-sidebar-collapsed");
    } catch (e0) {}

    var collapsed = stored === "1" || document.documentElement.classList.contains("mm-sidebar-collapsed");
    setSidebarCollapsed(collapsed);

    toggle.addEventListener("click", function () {
      var next = !document.documentElement.classList.contains("mm-sidebar-collapsed");
      setSidebarCollapsed(next);
      try {
        localStorage.setItem("mm-sidebar-collapsed", next ? "1" : "0");
      } catch (e1) {}
    });
  }

  function initClock() {
    var el = document.getElementById("fhClock");
    if (!el) return;

    function tick() {
      var d = new Date();
      el.textContent =
        pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
    }

    tick();
    window.setInterval(tick, 1000);
  }

  function initRefreshSpin() {
    var btn = document.getElementById("aiBtnRefresh");
    if (!btn) return;

    btn.addEventListener("click", function () {
      btn.classList.add("mm-topbar-btn--spinning");
      window.setTimeout(function () {
        btn.classList.remove("mm-topbar-btn--spinning");
      }, 900);

      if (!document.getElementById("aiDashRoot") && !document.getElementById("tnDashRoot")) {
        var fcSnap = document.getElementById("fcBtnRefreshSnap");
        if (fcSnap) {
          fcSnap.click();
          return;
        }
        window.location.reload();
      }
    });
  }

  function fullscreenTarget() {
    return (
      document.getElementById("aiDashRoot") ||
      document.getElementById("tnDashRoot") ||
      document.getElementById("fcCameraRoot") ||
      document.querySelector(".app-shell") ||
      document.documentElement
    );
  }

  function initFullscreen() {
    var btn = document.getElementById("aiBtnFullscreen");
    var lbl = document.getElementById("aiLblFullscreen");
    if (!btn) return;

    if (document.getElementById("aiDashRoot")) return;

    var target = fullscreenTarget();

    function sync() {
      var active =
        document.fullscreenElement === target ||
        document.webkitFullscreenElement === target ||
        document.msFullscreenElement === target;
      btn.setAttribute("aria-pressed", active ? "true" : "false");
      if (lbl) lbl.textContent = active ? "Exit Fullscreen" : "Fullscreen";
      btn.title = active ? "Exit fullscreen (Esc)" : "Enter fullscreen";
      if (target && target.classList) {
        target.classList.toggle("mm-app-shell--fullscreen", !!active);
      }
    }

    btn.addEventListener("click", function () {
      var fs =
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.msFullscreenElement;
      if (!fs) {
        var req =
          target.requestFullscreen ||
          target.webkitRequestFullscreen ||
          target.msRequestFullscreen;
        if (req) {
          try {
            var p = req.call(target);
            if (p && typeof p.catch === "function") p.catch(function () {});
          } catch (e1) {}
        }
      } else {
        var exit =
          document.exitFullscreen ||
          document.webkitExitFullscreen ||
          document.msExitFullscreen;
        if (exit) {
          try {
            exit.call(document);
          } catch (e2) {}
        }
      }
    });

    document.addEventListener("fullscreenchange", sync);
    document.addEventListener("webkitfullscreenchange", sync);
    sync();
  }

  function init() {
    initSidebarCollapse();
    initClock();
    initRefreshSpin();
    initFullscreen();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();