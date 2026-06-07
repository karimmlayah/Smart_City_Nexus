/**
 * MedinaMind AR — live API data fetching and UI binding
 */
console.log("AR live data script loaded");

(function () {
  "use strict";

  var AR_ENDPOINTS = {
    stats: "/api/ar/live-stats/",
    info: "/api/ar/live-info/",
    modules: "/api/ar/modules/",
  };

  var lastStats = null;
  var lastInfo = null;
  var lastWeatherKey = null;
  var TUNIS_TZ = "Africa/Tunis";
  var WEATHER_REFRESH_MS = 15 * 60 * 1000;

  function setText(selector, value) {
    var els = document.querySelectorAll(selector);
    if (!els.length) {
      console.warn("Missing selector:", selector);
      return;
    }
    var text =
      value !== undefined && value !== null && value !== "" ? String(value) : "—";
    els.forEach(function (el) {
      el.textContent = text;
    });
  }

  function setApiStatus(text, mode) {
    var el = document.querySelector("[data-api-status]");
    if (!el) return;
    el.textContent = text;
    el.classList.remove("online", "offline", "loading");
    if (mode) el.classList.add(mode);
  }

  function flashStats() {
    document.querySelectorAll("[data-stat]").forEach(function (el) {
      el.classList.remove("value-updated");
      void el.offsetWidth;
      el.classList.add("value-updated");
      setTimeout(function () {
        el.classList.remove("value-updated");
      }, 500);
    });
  }

  function fetchJSON(url) {
    return fetch(url, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
      credentials: "same-origin",
    }).then(function (response) {
      if (!response.ok) {
        return response.text().then(function (text) {
          console.error("API error:", url, response.status, text);
          throw new Error("API error " + response.status);
        });
      }
      var contentType = response.headers.get("content-type") || "";
      if (contentType.indexOf("application/json") === -1) {
        return response.text().then(function (text) {
          console.error("API did not return JSON:", url, text.slice(0, 200));
          throw new Error("API did not return JSON");
        });
      }
      return response.json();
    });
  }

  function setHoloHub(key, value) {
    var el = document.querySelector("[data-holo='" + key + "']");
    if (el && value != null && value !== "") el.textContent = String(value);
  }

  function flashHoloMetric(metricKey) {
    var card = document.querySelector(
      '.hologram-stat-card[data-metric="' + metricKey + '"]'
    );
    if (card) {
      card.classList.remove("val-updated");
      void card.offsetWidth;
      card.classList.add("val-updated");
      setTimeout(function () {
        card.classList.remove("val-updated");
      }, 600);
    }
    var val = document.querySelector(
      '.hologram-stat-card[data-metric="' + metricKey + '"] .holo-value'
    );
    if (val) {
      val.classList.remove("value-updated");
      void val.offsetWidth;
      val.classList.add("value-updated");
      setTimeout(function () {
        val.classList.remove("value-updated");
      }, 600);
    }
  }

  function updateHoloField(key, value, metricKey) {
    var el = document.querySelector("[data-holo='" + key + "']");
    if (!el) return;
    var next = value != null && value !== "" ? String(value) : "—";
    if (el.textContent === next) return;
    el.textContent = next;
    if (metricKey) {
      flashHoloMetric(metricKey);
      return;
    }
    if (el.classList.contains("holo-value")) {
      el.classList.remove("value-updated");
      void el.offsetWidth;
      el.classList.add("value-updated");
      setTimeout(function () {
        el.classList.remove("value-updated");
      }, 600);
    }
  }

  function applyHoloStats(data) {
    var trafficText =
      data.traffic ||
      (data.traffic_status && data.traffic_flow
        ? data.traffic_status + " · " + data.traffic_flow
        : data.traffic_status || data.traffic_flow || "—");
    updateHoloField("city", (data.city || "Tunis") + ", Tunisia");
    updateHoloField("traffic", trafficText, "traffic");
    updateHoloField("energy", data.energy_usage, "energy");
    updateHoloField("air-quality", data.air_quality, "air-quality");
    updateHoloField("safety", data.safety, "safety");
    updateHoloField(
      "parking",
      data.parking_spots != null ? String(data.parking_spots) : "—",
      "parking"
    );
    updateHoloField("population", data.population || "2.8M");
    updateHoloField("updated", data.last_updated);
    updateHoloField("feed-status", "Online");

    document.dispatchEvent(
      new CustomEvent("ar:live-stats", { detail: data })
    );
  }

  function applyHoloWeather(data) {
    if (data.city) updateHoloField("city", data.city + ", Tunisia");
    if (data.weather) {
      updateHoloField(
        "weather",
        (data.weather.temperature || "—") +
          " · " +
          (data.weather.condition || "—"),
        "weather"
      );
    }
    document.dispatchEvent(
      new CustomEvent("ar:live-weather", { detail: data })
    );
  }

  function applyStats(data) {
    setText("[data-stat='last-updated']", data.last_updated);
    setText("[data-stat='city']", data.city || "Tunis");
    setText("[data-stat='population']", data.population);
    setText("[data-stat='energy-usage']", data.energy_usage);
    setText("[data-stat='air-quality']", data.air_quality);
    setText("[data-stat='safety']", data.safety);
    setText("[data-stat='parking-spots']", data.parking_spots);
    setText(
      "[data-stat='traffic']",
      data.traffic || data.traffic_status || data.traffic_flow || "—"
    );

    applyHoloStats(data);
    flashStats();
  }

  function updateLiveStats() {
    setApiStatus("Updating...", "loading");

    return fetchJSON(AR_ENDPOINTS.stats)
      .then(function (data) {
        if (!data.success) throw new Error("Invalid stats response");
        lastStats = data;
        applyStats(data);
        setApiStatus("Tunis Live", "online");
      })
      .catch(function (error) {
        console.error("Live stats failed:", error);
        setApiStatus("API Offline", "offline");
        setHoloHub("feed-status", "—");
        if (lastStats) applyStats(lastStats);
      });
  }

  function tunisDateParts() {
    var now = new Date();
    var timeFmt = new Intl.DateTimeFormat("en-GB", {
      timeZone: TUNIS_TZ,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
    var dateFmt = new Intl.DateTimeFormat("en-GB", {
      timeZone: TUNIS_TZ,
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
    });
    var weekdayFmt = new Intl.DateTimeFormat("en-GB", {
      timeZone: TUNIS_TZ,
      weekday: "long",
    });

    var timeParts = {};
    timeFmt.formatToParts(now).forEach(function (p) {
      if (p.type !== "literal") timeParts[p.type] = p.value;
    });
    var dateParts = {};
    dateFmt.formatToParts(now).forEach(function (p) {
      if (p.type !== "literal") dateParts[p.type] = p.value;
    });

    return {
      time:
        timeParts.hour + ":" + timeParts.minute + ":" + timeParts.second,
      date: dateParts.day + "/" + dateParts.month + "/" + dateParts.year,
      weekday: weekdayFmt.format(now),
    };
  }

  function tickTunisClock() {
    var parts = tunisDateParts();
    setText("[data-live='time']", parts.time);
    setText("[data-live='date']", parts.date);
    setText("[data-live='weekday']", parts.weekday);
    setHoloHub("time", parts.time);
  }

  function applyWeather(weather, force) {
    if (!weather) return;
    var key =
      weather.temperature +
      "|" +
      weather.condition +
      "|" +
      weather.humidity +
      "|" +
      weather.wind;
    if (!force && key === lastWeatherKey) return;
    lastWeatherKey = key;

    setText("[data-weather='temperature']", weather.temperature);
    setText("[data-weather='condition']", weather.condition);
    setText("[data-weather='humidity']", weather.humidity);
    setText("[data-weather='wind']", weather.wind);
    setText("[data-weather='feels-like']", weather.feels_like);
  }

  function applyInfo(data, forceWeather) {
    setText("[data-live='city']", data.city || "Tunis, Tunisia");
    setText("[data-live='timezone']", data.timezone || "Africa/Tunis");
    tickTunisClock();
    applyWeather(data.weather, forceWeather);
    applyHoloWeather(data);
  }

  function updateLiveWeather() {
    return fetchJSON(AR_ENDPOINTS.info)
      .then(function (data) {
        if (!data.success) throw new Error("Invalid info response");
        lastInfo = data;
        applyInfo(data, true);
      })
      .catch(function (error) {
        console.error("Live weather failed:", error);
        if (lastInfo) applyInfo(lastInfo, false);
      });
  }

  function loadModules() {
    return fetchJSON(AR_ENDPOINTS.modules)
      .then(function (data) {
        if (!data.success || !data.modules) return data;
        var list = document.getElementById("moduleList");
        if (!list) return data;

        list.innerHTML = "";
        data.modules.forEach(function (mod, i) {
          var el = document.createElement("div");
          el.className = "hud-module" + (i === 0 ? " active" : "");
          el.setAttribute("data-mod", mod.id);
          el.innerHTML =
            '<span class="hud-mod-icon">' +
            (mod.icon || "") +
            '</span> <span class="hud-mod-name">' +
            mod.name +
            '</span> <span class="hud-mod-arrow">›</span>';
          list.appendChild(el);
        });

        document.dispatchEvent(
          new CustomEvent("ar:modules-loaded", { detail: data.modules })
        );
        return data;
      })
      .catch(function (error) {
        console.error("Modules failed:", error);
        return null;
      });
  }

  function init() {
    console.log("AR live data initialized");
    tickTunisClock();
    updateLiveStats();
    updateLiveWeather();
    loadModules();
    setInterval(updateLiveStats, 30000);
    setInterval(updateLiveWeather, WEATHER_REFRESH_MS);
    setInterval(tickTunisClock, 1000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  window.ARLiveData = {
    refreshStats: updateLiveStats,
    refreshInfo: updateLiveWeather,
    reloadModules: loadModules,
  };
})();
