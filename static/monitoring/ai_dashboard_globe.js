/**
 * AI Dashboard — WebGL Earth (Three.js). Auto-rotation, parallax léger, pause si onglet inactif.
 */
(function () {
  "use strict";

  var THREE = typeof window !== "undefined" ? window.THREE : null;

  function prefersReducedMotion() {
    try {
      return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch (e) {
      return false;
    }
  }

  function init() {
    if (!THREE) return;
    var canvas = document.getElementById("aiGlobeCanvas");
    var mount = document.getElementById("aiGlobeMount");
    if (!canvas || !mount) return;

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
    camera.position.set(0, 0, 2.75);

    var renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    var earthGroup = new THREE.Group();
    scene.add(earthGroup);

    var earthGeo = new THREE.SphereGeometry(1, 72, 72);
    var earthMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.82,
      metalness: 0.06,
    });
    var earthMesh = new THREE.Mesh(earthGeo, earthMat);
    earthGroup.add(earthMesh);

    var atmGeo = new THREE.SphereGeometry(1.06, 48, 48);
    var atmMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.14,
      side: THREE.BackSide,
      depthWrite: false,
    });
    earthGroup.add(new THREE.Mesh(atmGeo, atmMat));

    var rimGeo = new THREE.SphereGeometry(1.12, 48, 48);
    var rimMat = new THREE.MeshBasicMaterial({
      color: 0x60a5fa,
      transparent: true,
      opacity: 0.06,
      side: THREE.BackSide,
    });
    earthGroup.add(new THREE.Mesh(rimGeo, rimMat));

    var amb = new THREE.AmbientLight(0x445566, 0.55);
    var sun = new THREE.DirectionalLight(0xffffff, 1.35);
    sun.position.set(5.5, 2.2, 4);
    scene.add(amb, sun);
    var fill = new THREE.DirectionalLight(0x4488cc, 0.35);
    fill.position.set(-4, -1, -2);
    scene.add(fill);

    var starCount = 900;
    var starPos = new Float32Array(starCount * 3);
    for (var s = 0; s < starCount; s++) {
      var rad = 14 + Math.random() * 36;
      var u = Math.random();
      var v = Math.random();
      var th = u * Math.PI * 2;
      var ph = Math.acos(2 * v - 1);
      starPos[s * 3] = rad * Math.sin(ph) * Math.cos(th);
      starPos[s * 3 + 1] = rad * Math.sin(ph) * Math.sin(th);
      starPos[s * 3 + 2] = rad * Math.cos(ph);
    }
    var starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
    var stars = new THREE.Points(
      starGeo,
      new THREE.PointsMaterial({
        color: 0xaaddff,
        size: 0.045,
        transparent: true,
        opacity: 0.75,
        sizeAttenuation: true,
        depthWrite: false,
      })
    );
    scene.add(stars);

    /* Holographic city data nodes orbiting the globe */
    var nodeCount = 48;
    var nodeGeo = new THREE.BufferGeometry();
    var nodePos = new Float32Array(nodeCount * 3);
    var nodeColors = new Float32Array(nodeCount * 3);
    for (var n = 0; n < nodeCount; n++) {
      var nr = 1.18 + Math.random() * 0.35;
      var nu = Math.random();
      var nv = Math.random();
      var nth = nu * Math.PI * 2;
      var nph = Math.acos(2 * nv - 1);
      nodePos[n * 3] = nr * Math.sin(nph) * Math.cos(nth);
      nodePos[n * 3 + 1] = nr * Math.sin(nph) * Math.sin(nth);
      nodePos[n * 3 + 2] = nr * Math.cos(nph);
      var cyan = n % 3 === 0;
      nodeColors[n * 3] = cyan ? 0 : 0.15;
      nodeColors[n * 3 + 1] = cyan ? 0.84 : 0.39;
      nodeColors[n * 3 + 2] = cyan ? 1 : 0.92;
    }
    nodeGeo.setAttribute("position", new THREE.BufferAttribute(nodePos, 3));
    nodeGeo.setAttribute("color", new THREE.BufferAttribute(nodeColors, 3));
    var dataNodes = new THREE.Points(
      nodeGeo,
      new THREE.PointsMaterial({
        size: 0.055,
        vertexColors: true,
        transparent: true,
        opacity: 0.88,
        sizeAttenuation: true,
        depthWrite: false,
      })
    );
    earthGroup.add(dataNodes);

    /* Wireframe city grid ring */
    var gridRing = new THREE.Mesh(
      new THREE.TorusGeometry(1.22, 0.004, 8, 96),
      new THREE.MeshBasicMaterial({
        color: 0x00d5ff,
        transparent: true,
        opacity: 0.35,
        depthWrite: false,
      })
    );
    gridRing.rotation.x = Math.PI * 0.42;
    earthGroup.add(gridRing);

    var gridRing2 = gridRing.clone();
    gridRing2.rotation.x = Math.PI * 0.62;
    gridRing2.rotation.z = Math.PI * 0.25;
    gridRing2.material = gridRing.material.clone();
    gridRing2.material.opacity = 0.2;
    earthGroup.add(gridRing2);

    var texBase = "https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg";
    var loader = new THREE.TextureLoader();
    loader.setCrossOrigin("anonymous");
    loader.load(
      texBase,
      function (tex) {
        if (THREE.SRGBColorSpace !== undefined) {
          tex.colorSpace = THREE.SRGBColorSpace;
        } else if (THREE.sRGBEncoding !== undefined && tex.encoding !== undefined) {
          tex.encoding = THREE.sRGBEncoding;
        }
        tex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
        earthMat.map = tex;
        earthMat.needsUpdate = true;
      },
      undefined,
      function () {
        earthMat.color.setHex(0x0d3d5c);
      }
    );

    var mouse = { tx: 0, ty: 0, x: 0, y: 0 };
    var onMove = function (e) {
      var r = mount.getBoundingClientRect();
      if (r.width < 1) return;
      mouse.tx = ((e.clientX - r.left) / r.width - 0.5) * 2;
      mouse.ty = ((e.clientY - r.top) / r.height - 0.5) * 2;
    };
    mount.addEventListener("mousemove", onMove, { passive: true });
    mount.addEventListener(
      "touchmove",
      function (e) {
        if (!e.touches || !e.touches[0]) return;
        var r = mount.getBoundingClientRect();
        if (r.width < 1) return;
        mouse.tx = ((e.touches[0].clientX - r.left) / r.width - 0.5) * 2;
        mouse.ty = ((e.touches[0].clientY - r.top) / r.height - 0.5) * 2;
      },
      { passive: true }
    );

    function setSize() {
      var w = mount.clientWidth;
      var h = mount.clientHeight;
      if (w < 2 || h < 2) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    }

    var ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(setSize) : null;
    if (ro) ro.observe(mount);
    setSize();
    window.addEventListener("resize", setSize);

    var spin = prefersReducedMotion() ? 0 : 0.0011;
    var animId = null;

    function tick() {
      animId = requestAnimationFrame(tick);
      if (document.hidden) return;

      earthMesh.rotation.y += spin;
      stars.rotation.y += spin * 0.08;
      dataNodes.rotation.y += spin * 1.4;
      gridRing.rotation.z += spin * 0.6;
      gridRing2.rotation.z -= spin * 0.45;

      mouse.x += (mouse.tx - mouse.x) * 0.055;
      mouse.y += (mouse.ty - mouse.y) * 0.055;
      camera.position.x = mouse.x * 0.42;
      camera.position.y = -mouse.y * 0.32;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    }

    mount.classList.add("ai-globe-hero__stage--ready");

    if (prefersReducedMotion()) {
      renderer.render(scene, camera);
      return;
    }

    tick();

    window.addEventListener(
      "beforeunload",
      function () {
        if (animId) cancelAnimationFrame(animId);
        mount.removeEventListener("mousemove", onMove);
        if (ro) ro.disconnect();
        window.removeEventListener("resize", setSize);
        renderer.dispose();
        earthGeo.dispose();
        earthMat.dispose();
        atmGeo.dispose();
        atmMat.dispose();
        rimGeo.dispose();
        rimMat.dispose();
        starGeo.dispose();
        nodeGeo.dispose();
      },
      { once: true }
    );
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
