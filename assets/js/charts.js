// Interactive figures: null-space toy, one-sided penalty curve, control dumbbell chart.
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent?.appendChild(e); return e; };

// ---------- one-sided penalty ----------
export function initGate() {
  const svg = document.getElementById('gate-plot'); if (!svg) return;
  const qIn = document.getElementById('g-q'), kIn = document.getElementById('g-k');
  const P = { l: 46, r: 500, t: 18, b: 222 }, YMAX = 3;
  const X = k => P.l + k * (P.r - P.l), Y = v => P.b - Math.min(v, YMAX) / YMAX * (P.b - P.t);
  const S = (q, k) => Math.max(0, Math.log(q + 1e-6) - Math.log(k + 1e-6)) ** 2;
  function draw() {
    const q = +qIn.value, k = +kIn.value;
    document.getElementById('g-q-v').textContent = q.toFixed(2);
    document.getElementById('g-k-v').textContent = k.toFixed(2);
    svg.innerHTML = '';
    const g = el('g', { 'font-family': 'Inter,Segoe UI,sans-serif', 'font-size': 11, fill: '#7f8ba1' }, svg);
    for (let v = 0; v <= YMAX; v++) { el('line', { x1: P.l, x2: P.r, y1: Y(v), y2: Y(v), stroke: 'rgba(255,255,255,.07)' }, g); el('text', { x: P.l - 8, y: Y(v) + 4, 'text-anchor': 'end' }, g).textContent = v; }
    for (const t of [0, .25, .5, .75, 1]) el('text', { x: X(t), y: P.b + 18, 'text-anchor': 'middle' }, g).textContent = t;
    el('text', { x: P.r, y: P.b + 36, 'text-anchor': 'end', fill: '#b4bfd2' }, g).textContent = 'supply k';
    el('text', { x: P.l, y: P.t - 4, fill: '#b4bfd2' }, g).textContent = 'penalty S';
    let d = ''; for (let i = 1; i <= 200; i++) { const kk = i / 200; d += (i === 1 ? 'M' : 'L') + X(kk).toFixed(1) + ' ' + Y(S(q, kk)).toFixed(1); }
    el('path', { d: d + `L${X(1)} ${P.b}L${X(.005)} ${P.b}Z`, fill: 'rgba(79,143,209,.10)' }, g);
    el('path', { d, fill: 'none', stroke: '#4f8fd1', 'stroke-width': 2 }, g);
    el('line', { x1: X(q), x2: X(q), y1: P.t, y2: P.b, stroke: '#d4735f', 'stroke-dasharray': '4 4', 'stroke-width': 1.5 }, g);
    el('text', { x: X(q) + 6, y: P.t + 12, fill: '#d4735f' }, g).textContent = 'demand q';
    el('circle', { cx: X(k), cy: Y(S(q, k)), r: 6, fill: '#eef2f8', stroke: '#172033', 'stroke-width': 2 }, g);
    const v = S(q, k);
    document.getElementById('gate-read').textContent = v > 0 ? `S = ${v.toFixed(3)} · supply is short of demand, the correction is active.` : 'S = 0 · supply meets demand, SADL is silent.';
  }
  qIn.addEventListener('input', draw); kIn.addEventListener('input', draw); draw();
}
