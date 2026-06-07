/**

 * MedinaMind — 3-theme cycle: blue-dark → light → dark

 * Priority: localStorage > cookie > server default (set in theme_boot_script.html)

 */

(function () {

  "use strict";



  var STORAGE_KEY = "medinamind-theme";

  var COOKIE_KEY = "medinamind-platform-theme";

  var THEMES = ["blue-dark", "light", "dark"];

  var THEME_LABELS = {

    "blue-dark": "Blue Dark Mode",

    light: "Light Mode",

    dark: "Classic Dark Mode",

  };



  function resolveSystem() {

    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "blue-dark";

  }



  function normalize(raw) {

    var v = (raw || "").trim().toLowerCase();

    if (v === "system") return resolveSystem();

    return THEMES.indexOf(v) >= 0 ? v : null;

  }



  function getCookie(name) {

    var m = document.cookie.match(

      new RegExp("(?:^|; )" + name.replace(/([.$?*|{}()[\]\\/+^])/g, "\\$1") + "=([^;]*)")

    );

    return m ? decodeURIComponent(m[1]) : "";

  }



  function setCookie(name, value, days) {

    var maxAge = (days || 365) * 24 * 3600;

    document.cookie =

      name + "=" + encodeURIComponent(value) + "; path=/; max-age=" + maxAge + "; SameSite=Lax";

  }



  function getTheme() {

    var t = document.documentElement.getAttribute("data-theme");

    return THEMES.indexOf(t) >= 0 ? t : "blue-dark";

  }



  function syncAiDashCompat(theme) {

    var root = document.getElementById("aiDashRoot");

    if (root) {

      root.setAttribute("data-ai-theme", theme === "light" ? "light" : "dark");

    }

  }



  function updateButtons(theme) {

    var title = THEME_LABELS[theme] || "Switch theme";

    document.querySelectorAll("[data-theme-toggle]").forEach(function (btn) {

      btn.setAttribute("title", title);

      btn.setAttribute("aria-label", title);

    });

  }



  function applyTheme(theme, silent) {

    if (THEMES.indexOf(theme) < 0) theme = "blue-dark";

    document.documentElement.setAttribute("data-theme", theme);

    try {

      localStorage.setItem(STORAGE_KEY, theme);

    } catch (e0) {}

    setCookie(COOKIE_KEY, theme);

    syncAiDashCompat(theme);

    updateButtons(theme);

    if (!silent) {

      try {

        document.dispatchEvent(

          new CustomEvent("medinamind-theme-change", { detail: { theme: theme } })

        );

      } catch (e1) {}

    }

  }



  function resolveInitialTheme() {

    var theme = null;

    try {

      theme = normalize(localStorage.getItem(STORAGE_KEY));

    } catch (e2) {}

    if (!theme) {

      theme = normalize(getCookie(COOKIE_KEY));

    }

    if (!theme) {

      theme = normalize(document.documentElement.getAttribute("data-theme"));

    }

    return theme || "blue-dark";

  }



  function nextTheme(current) {

    var i = THEMES.indexOf(current);

    return THEMES[(i + 1) % THEMES.length];

  }



  function init() {

    applyTheme(resolveInitialTheme(), true);



    document.querySelectorAll("[data-theme-toggle]").forEach(function (btn) {

      if (btn._mmThemeBound) return;

      btn._mmThemeBound = true;

      btn.addEventListener("click", function () {

        applyTheme(nextTheme(getTheme()), false);

      });

    });



    try {

      window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", function () {

        var platformMode = document.documentElement.getAttribute("data-platform-theme");

        if (platformMode === "system" && !localStorage.getItem(STORAGE_KEY)) {

          applyTheme(resolveSystem(), true);

        }

      });

    } catch (e3) {}

  }



  window.MedinaMindTheme = {

    get: getTheme,

    set: function (t) {

      applyTheme(t, false);

    },

    cycle: function () {

      applyTheme(nextTheme(getTheme()), false);

    },

    resolveSystem: resolveSystem,

  };



  if (document.readyState === "loading") {

    document.addEventListener("DOMContentLoaded", init);

  } else {

    init();

  }

})();

