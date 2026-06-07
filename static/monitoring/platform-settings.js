(function () {
  "use strict";

  var navItems = document.querySelectorAll(".ps-nav__item[data-section]");
  var panels = document.querySelectorAll(".ps-panel[data-section]");
  var activeInput = document.getElementById("psActiveSection");

  function showSection(id) {
    navItems.forEach(function (el) {
      el.classList.toggle("is-active", el.getAttribute("data-section") === id);
    });
    panels.forEach(function (el) {
      el.classList.toggle("is-active", el.getAttribute("data-section") === id);
    });
    if (activeInput) activeInput.value = id;
    if (history.replaceState) {
      history.replaceState(null, "", "?section=" + encodeURIComponent(id));
    }
  }

  navItems.forEach(function (el) {
    el.addEventListener("click", function (ev) {
      ev.preventDefault();
      showSection(el.getAttribute("data-section"));
    });
  });

  document.querySelectorAll("[data-test]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var url = btn.getAttribute("data-test-url");
      var out = document.getElementById(btn.getAttribute("data-test-out"));
      if (!url) return;
      btn.disabled = true;
      if (out) {
        out.textContent = "Testing…";
        out.className = "ps-test-result";
      }
      var csrf = document.querySelector("[name=csrfmiddlewaretoken]");
      fetch(url, {
        method: "POST",
        headers: {
          "X-CSRFToken": csrf ? csrf.value : "",
          "X-Requested-With": "XMLHttpRequest",
        },
      })
        .then(function (r) {
          return r.json();
        })
        .then(function (data) {
          if (out) {
            out.textContent = data.message || (data.ok ? "OK" : "Failed");
            out.className = "ps-test-result " + (data.ok ? "ok" : "err");
          }
        })
        .catch(function (err) {
          if (out) {
            out.textContent = String(err);
            out.className = "ps-test-result err";
          }
        })
        .finally(function () {
          btn.disabled = false;
        });
    });
  });

  var settingsForm = document.querySelector(".ps-form");
  if (settingsForm) {
    settingsForm.addEventListener("submit", function () {
      var themeField = settingsForm.querySelector('[name="theme_mode"]');
      if (!themeField) return;
      var mode = (themeField.value || "").trim().toLowerCase();
      try {
        if (mode === "system") {
          localStorage.removeItem("medinamind-theme");
        } else if (mode === "light" || mode === "dark" || mode === "blue-dark") {
          localStorage.setItem("medinamind-theme", mode);
        }
      } catch (e) {}
    });
  }
})();
