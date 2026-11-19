// Hero: the real SADL-trained four-task latents (raw z[:, :2]) drawn as particles.
// Intro motion (shared point -> learned glyph) is decorative; final positions are the data.
const TASKS = ['pusht', 'cube', 'reacher', 'tworoom'];
const COLORS = { pusht: [220, 73, 96], cube: [69, 163, 138], reacher: [63, 124, 196], tworoom: [213, 166, 53] };

export async function initHero(canvas) {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ctx = canvas.getContext('2d');
  let data;
  try { data = await (await fetch('assets/data/sadl-glyph.json')).json(); } catch { return; }

  const pts = [];
  for (const t of TASKS) {
    const a = data.tasks[t];
    for (let i = 0; i < a.length; i += 2) pts.push({ x: a[i], y: -a[i + 1], c: COLORS[t], seed: Math.random() });
  }
  // bounds of the real data
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const p of pts) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
  // centre on a blend of the bounding box and the point mass: the thin tail of the L
  // otherwise pulls the bounding-box centre right and the glyph reads off-centre
  const mx = pts.reduce((t, p) => t + p.x, 0) / pts.length;
  const cx = .2 * (x0 + x1) / 2 + .8 * mx, cy = (y0 + y1) / 2, span = x1 - x0;

  let W = 0, H = 0, dpr = 1, scale = 1, ox = 0, oy = 0;
  const pointer = { x: -1e4, y: -1e4 };
  function resize() {
    dpr = Math.min(Math.max(devicePixelRatio || 1, 1.5), 3);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    const glyphW = Math.min(W * (W < 720 ? .9 : .6), 900);
    scale = glyphW / span;
    const gh = (y1 - y0) * scale;
    canvas.parentElement.style.setProperty('--glyph-h', Math.round(gh) + 'px');
    ox = W / 2; oy = 84 + gh / 2;
  }
  resize();
  addEventListener('resize', () => { resize(); kick(); });
  new ResizeObserver(() => { resize(); kick(); }).observe(canvas);
  canvas.parentElement.addEventListener('pointermove', e => { const r = canvas.getBoundingClientRect(); pointer.x = e.clientX - r.left; pointer.y = e.clientY - r.top; });
  canvas.parentElement.addEventListener('pointerleave', () => { pointer.x = pointer.y = -1e4; });

  // every particle starts at (nearly) the same point: the shared component
  for (const p of pts) { p.px = (Math.random() - .5) * .02; p.py = (Math.random() - .5) * .02; p.vx = 0; p.vy = 0; }
  const t0 = performance.now();
  let visible = true, running = false;
  function kick() { if (!running) { running = true; requestAnimationFrame(frame); } }
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) kick(); }).observe(canvas);

  function frame(now) {
    running = false;
    const t = (now - t0) / 1000;
    const k = reduce ? 1 : Math.min(1, Math.max(0, (t - .35) / 2.4));
    const ease = 1 - Math.pow(1 - k, 3);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const r = W < 720 ? 1.3 : 1.7;
    for (const p of pts) {
      const tx = (p.x - cx) * ease, ty = (p.y - cy) * ease;
      const breathe = reduce ? 0 : Math.sin(t * .8 + p.seed * 6.28) * .004;
      p.px = tx + breathe; p.py = ty + breathe;
      let sx = ox + p.px * scale, sy = oy + p.py * scale;
      const dx = sx - pointer.x, dy = sy - pointer.y, d2 = dx * dx + dy * dy;
      if (d2 < 9000) { const f = (1 - d2 / 9000) * 18; const d = Math.sqrt(d2) || 1; p.vx += dx / d * f * .08; p.vy += dy / d * f * .08; }
      p.vx *= .86; p.vy *= .86; sx += p.vx * 6; sy += p.vy * 6;
      const [R, G, B] = p.c;
      ctx.fillStyle = `rgba(${R},${G},${B},${.55 + .4 * ease})`;
      ctx.beginPath(); ctx.arc(sx, sy, r * .6, 0, 6.2832); ctx.fill();
    }
    if (visible && (!reduce || k < 1)) kick();
  }
  kick();
}
