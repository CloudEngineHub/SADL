// Toy illustration: a shared shift of H is invisible to Z under a train-mode BatchNorm projector.
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent?.appendChild(e); return e; };

export function initNullSpace() {
  const svg = document.getElementById('ns-svg'); if (!svg) return;
  const shiftIn = document.getElementById('ns-shift'), spreadIn = document.getElementById('ns-spread');
  let norm = 'bn';
  let s = 11; const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
  const gauss = () => { const u = rnd() || 1e-9, v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.2832 * v); };
  const eps = Array.from({ length: 40 }, () => [Math.max(-2.2, Math.min(2.2, gauss())), Math.max(-2.2, Math.min(2.2, gauss())) * .6]);
  const Wm = [[1, .35], [-.25, .8]], m0 = [.45, .25], u = [.8, .6];
  const colors = ['#dc4960', '#45a38a', '#3f7cc4', '#d5a635'];
  const H = (shift, spread) => eps.map(([a, b]) => [m0[0] + shift * u[0] + spread * a * .3, m0[1] + shift * u[1] + spread * b * .3]);
  const lin = h => h.map(([a, b]) => [Wm[0][0] * a + Wm[0][1] * b, Wm[1][0] * a + Wm[1][1] * b]);
  const bn = z => {
    const mu = [0, 1].map(j => z.reduce((t, r) => t + r[j], 0) / z.length);
    const sd = [0, 1].map(j => Math.sqrt(z.reduce((t, r) => t + (r[j] - mu[j]) ** 2, 0) / z.length) || 1);
    return z.map(r => [(r[0] - mu[0]) / sd[0], (r[1] - mu[1]) / sd[1]]);
  };
  const Z = h => norm === 'bn' ? bn(lin(h)) : lin(h);

  // layout: two panels, each clipped so no dot ever touches a border
  const PW = 340, PH = 360, PY = 20, panels = [{ x: 20 }, { x: 400 }];
  svg.innerHTML = '';
  const defs = el('defs', {}, svg);
  panels.forEach((p, i) => { const c = el('clipPath', { id: 'nsclip' + i }, defs); el('rect', { x: p.x + 10, y: PY + 52, width: PW - 20, height: PH - (i ? 104 : 62), rx: 8 }, c); });
  const root = el('g', { 'font-family': 'Inter,Segoe UI,sans-serif' }, svg);
  const titles = [['Encoder representation', 'H'], ['Projector output', 'Z']];
  panels.forEach((p, i) => {
    el('rect', { x: p.x, y: PY, width: PW, height: PH, rx: 14, fill: 'rgba(255,255,255,.025)', stroke: 'rgba(255,255,255,.09)' }, root);
    const t = el('text', { x: p.x + 18, y: PY + 32, fill: '#eef2f8', 'font-size': 17, 'font-weight': 700 }, root);
    t.textContent = titles[i][0] + ' ';
    const m = el('tspan', { 'font-family': 'Cambria Math,Times New Roman,serif', 'font-style': 'italic', 'font-weight': 400, 'font-size': 19 }, t); m.textContent = titles[i][1];
  });
  const arrow = el('g', {}, root);
  el('path', { d: 'M366 200h28', stroke: '#b4bfd2', 'stroke-width': 1.6 }, arrow);
  el('path', { d: 'M390 195l6 5-6 5', fill: 'none', stroke: '#b4bfd2', 'stroke-width': 1.6 }, arrow);
  const layers = panels.map((p, i) => el('g', { 'clip-path': `url(#nsclip${i})` }, root));
  // H panel axes through the origin
  const HO = { x: panels[0].x + 92, y: PY + 292 }, HS = 158;
  el('line', { x1: panels[0].x + 14, x2: panels[0].x + PW - 14, y1: HO.y, y2: HO.y, stroke: 'rgba(255,255,255,.07)' }, layers[0]);
  el('line', { x1: HO.x, x2: HO.x, y1: PY + 64, y2: PY + PH - 12, stroke: 'rgba(255,255,255,.07)' }, layers[0]);
  const meanLine = el('line', { x1: HO.x, y1: HO.y, stroke: '#d4735f', 'stroke-width': 2.2, 'stroke-linecap': 'round' }, layers[0]);
  el('circle', { cx: HO.x, cy: HO.y, r: 3, fill: '#7f8ba1' }, layers[0]);
  const meanTag = el('text', { fill: '#d4735f', 'font-size': 21, 'font-weight': 700 }, layers[0]); meanTag.textContent = 'shared component';
  const ZO = { x: panels[1].x + PW / 2, y: PY + 52 + (PH - 104) / 2 }, ZS = 44;
  el('line', { x1: panels[1].x + 14, x2: panels[1].x + PW - 14, y1: ZO.y, y2: ZO.y, stroke: 'rgba(255,255,255,.07)' }, layers[1]);
  el('line', { x1: ZO.x, x2: ZO.x, y1: PY + 64, y2: PY + PH - 12, stroke: 'rgba(255,255,255,.07)' }, layers[1]);
  const dot = (g, i) => el('circle', { r: 4, fill: colors[i % 4], stroke: '#172033', 'stroke-width': 1.4 }, g);
  const hd = eps.map((_, i) => dot(layers[0], i)), zd = eps.map((_, i) => dot(layers[1], i));
  const badge = el('g', {}, root);
  const badgeRect = el('rect', { x: panels[1].x + PW - 132, y: PY + PH - 40, width: 116, height: 26, rx: 13 }, badge);
  const badgeText = el('text', { x: panels[1].x + PW - 74, y: PY + PH - 23, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 700 }, badge);

  let zCenter = [0, 0];
  function draw() {
    const shift = +shiftIn.value, spread = +spreadIn.value;
    document.getElementById('ns-shift-v').textContent = shift.toFixed(2);
    document.getElementById('ns-spread-v').textContent = spread.toFixed(2);
    const h = H(shift, spread), z = Z(h), z0 = Z(H(0, spread));
    if (norm !== 'bn') { const c = z0.reduce((t, r) => [t[0] + r[0] / z0.length, t[1] + r[1] / z0.length], [0, 0]); zCenter = c; } else zCenter = [0, 0];
    const zScale = norm === 'bn' ? ZS : 120;
    h.forEach(([a, b], i) => { hd[i].setAttribute('cx', HO.x + a * HS); hd[i].setAttribute('cy', HO.y - b * HS); });
    z.forEach(([a, b], i) => { zd[i].setAttribute('cx', ZO.x + (a - zCenter[0]) * zScale); zd[i].setAttribute('cy', ZO.y - (b - zCenter[1]) * zScale); });
    const m = [m0[0] + shift * u[0], m0[1] + shift * u[1]];
    meanLine.setAttribute('x2', HO.x + m[0] * HS); meanLine.setAttribute('y2', HO.y - m[1] * HS);
    meanTag.setAttribute('x', HO.x + m[0] * HS * .5 + 12); meanTag.setAttribute('y', HO.y - m[1] * HS * .5 + 26);
    const dz = z.reduce((t, r, i) => t + Math.hypot(r[0] - z0[i][0], r[1] - z0[i][1]), 0) / z.length;
    const still = dz < 1e-6;
    badgeRect.setAttribute('fill', still ? 'rgba(79,143,209,.16)' : 'rgba(212,115,95,.16)');
    badgeRect.setAttribute('stroke', still ? '#4f8fd1' : '#d4735f');
    badgeText.setAttribute('fill', still ? '#4f8fd1' : '#d4735f');
    badgeText.textContent = still ? 'Z unchanged' : 'Z moves';
    document.getElementById('ns-dz').textContent = dz.toFixed(3);
    const varE = eps.reduce((t, [a, b]) => t + (spread * .3) ** 2 * (a * a + b * b), 0) / eps.length;
    document.getElementById('ns-cef').textContent = Math.round(100 * varE / (varE + m[0] ** 2 + m[1] ** 2)) + '%';
  }
  shiftIn.addEventListener('input', draw); spreadIn.addEventListener('input', draw);
  document.querySelectorAll('[data-norm]').forEach(b => b.addEventListener('click', () => {
    norm = b.dataset.norm; document.querySelectorAll('[data-norm]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); draw();
  }));
  draw();
}
