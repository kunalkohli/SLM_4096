/*
 * SLM 4096 - cursor-reactive particle field.
 * Thousands of dots rest in the shape of "SLM", drift gently, and scatter
 * away from the cursor, then ease back. Vanilla canvas, no libraries.
 */
(function () {
  const canvas = document.getElementById("particles");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const INK = "#1a1a1a";
  const ACCENT = "#c15f3c";
  const SAMPLE_STEP = 4;     // px between sampled dots (smaller = denser)
  const REPEL_RADIUS = 95;   // px around cursor that pushes dots away
  const RETURN_K = 0.045;    // spring strength back to home
  const DAMPING = 0.86;      // velocity damping

  let dpr = Math.max(1, window.devicePixelRatio || 1);
  let W = 0, H = 0;
  let particles = [];
  const mouse = { x: -9999, y: -9999, active: false };

  function resize() {
    const rect = canvas.getBoundingClientRect();
    W = rect.width;
    H = rect.height;
    dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.floor(W * dpr);
    canvas.height = Math.floor(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    buildTargets();
  }

  function buildTargets() {
    if (W === 0 || H === 0) return;
    const off = document.createElement("canvas");
    off.width = Math.floor(W);
    off.height = Math.floor(H);
    const octx = off.getContext("2d");

    // fit "SLM" to ~78% of the panel width
    const text = "SLM";
    let fontSize = Math.floor(H * 0.42);
    octx.font = `700 ${fontSize}px Arial, sans-serif`;
    let tw = octx.measureText(text).width;
    const targetW = W * 0.78;
    fontSize = Math.floor(fontSize * (targetW / tw));
    octx.font = `700 ${fontSize}px Arial, sans-serif`;
    octx.textAlign = "center";
    octx.textBaseline = "middle";
    octx.fillStyle = "#000";
    octx.fillText(text, W / 2, H / 2);

    const img = octx.getImageData(0, 0, off.width, off.height).data;
    const targets = [];
    for (let y = 0; y < off.height; y += SAMPLE_STEP) {
      for (let x = 0; x < off.width; x += SAMPLE_STEP) {
        const alpha = img[(y * off.width + x) * 4 + 3];
        if (alpha > 128) targets.push({ x, y });
      }
    }

    // reuse / create particles to match target count
    const next = [];
    for (let i = 0; i < targets.length; i++) {
      const t = targets[i];
      const prev = particles[i];
      next.push({
        x: prev ? prev.x : Math.random() * W,
        y: prev ? prev.y : Math.random() * H,
        vx: 0,
        vy: 0,
        hx: t.x,
        hy: t.y,
        accent: Math.random() < 0.06,
        size: Math.random() < 0.18 ? 2 : 1.4,
      });
    }
    particles = next;
  }

  function tick() {
    ctx.clearRect(0, 0, W, H);
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];

      // spring home
      p.vx += (p.hx - p.x) * RETURN_K;
      p.vy += (p.hy - p.y) * RETURN_K;

      // cursor repulsion
      if (mouse.active) {
        const dx = p.x - mouse.x;
        const dy = p.y - mouse.y;
        const dist = Math.hypot(dx, dy);
        if (dist < REPEL_RADIUS && dist > 0.01) {
          const force = (1 - dist / REPEL_RADIUS) * 5.5;
          p.vx += (dx / dist) * force;
          p.vy += (dy / dist) * force;
        }
      }

      p.vx *= DAMPING;
      p.vy *= DAMPING;
      p.x += p.vx;
      p.y += p.vy;

      ctx.fillStyle = p.accent ? ACCENT : INK;
      ctx.globalAlpha = p.accent ? 0.9 : 0.78;
      ctx.fillRect(p.x, p.y, p.size, p.size);
    }
    ctx.globalAlpha = 1;
    requestAnimationFrame(tick);
  }

  function setMouseFromEvent(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    mouse.x = clientX - rect.left;
    mouse.y = clientY - rect.top;
    mouse.active = true;
  }

  window.addEventListener("mousemove", (e) => setMouseFromEvent(e.clientX, e.clientY));
  window.addEventListener("mouseout", () => { mouse.active = false; mouse.x = -9999; mouse.y = -9999; });
  canvas.addEventListener("touchmove", (e) => {
    if (e.touches[0]) setMouseFromEvent(e.touches[0].clientX, e.touches[0].clientY);
  }, { passive: true });
  canvas.addEventListener("touchend", () => { mouse.active = false; });

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 150);
  });

  // wait a frame so layout is settled, then go
  requestAnimationFrame(() => { resize(); tick(); });
})();
