/**
 * MedinaMind Homepage — navbar scroll, mobile menu, active links
 */
(function () {
  "use strict";

  function initNav() {
    var nav = document.getElementById("mmHomeNav");
    if (!nav) return;

    var burger = document.getElementById("mmHomeBurger");
    var links = nav.querySelectorAll(".mm-home-nav__link");

    function onScroll() {
      nav.classList.toggle("is-scrolled", window.scrollY > 12);
    }

    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    if (burger) {
      burger.addEventListener("click", function () {
        var open = nav.classList.toggle("is-open");
        burger.setAttribute("aria-expanded", open ? "true" : "false");
      });
    }

    links.forEach(function (link) {
      link.addEventListener("click", function () {
        nav.classList.remove("is-open");
        if (burger) burger.setAttribute("aria-expanded", "false");
      });
    });

    var sections = ["features", "modules", "why", "about"];
    var observer = null;
    if ("IntersectionObserver" in window) {
      observer = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (!entry.isIntersecting) return;
            var id = entry.target.id;
            links.forEach(function (lnk) {
              var href = lnk.getAttribute("href") || "";
              lnk.classList.toggle("is-active", href === "#" + id);
            });
          });
        },
        { rootMargin: "-40% 0px -50% 0px", threshold: 0 }
      );
      sections.forEach(function (id) {
        var el = document.getElementById(id);
        if (el) observer.observe(el);
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initNav);
  } else {
    initNav();
  }
})();
