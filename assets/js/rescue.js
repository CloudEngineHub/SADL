// Post-collapse rescue at the default Supply Regularizer weight (lambda_S = .01), paper Table 16(b).
// Website shows the clearly rescued checkpoints; the full table (incl. non-rescued seeds) is in the paper.
// Each row resumes the same near-collapsed checkpoint; macro success over four tasks x eval seeds {0,1,42}.
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent?.appendChild(e); return e; };

export const RESCUE = [
  { model: 'INTACT', seed: '42', span: 'Ep. 1 → 2', note: 'machine A', base: 18.917, sadl: 68.333 },
  { model: 'INTACT', seed: '42', span: 'Ep. 1 → 2', note: 'replicate, machine B', base: 23.083, sadl: 69.083 },
  { model: 'INTACT', seed: '3072', span: 'Ep. 1 → 2', note: '', base: 24.417, sadl: 53.917 },
  { model: 'LeWM', seed: '21', span: 'Ep. 2 → 4', note: '', base: 25.250, sadl: 46.917 },
  { model: 'LeWM', seed: '7', span: 'Ep. 2 → 4', note: '', base: 61.000, sadl: 72.167 }
];

export function initRescue() {
  const svg = document.getElementById('rescue-svg'); if (!svg) return;
  const tip = document.getElementById('rescue-tip'), card = document.getElementById('rescue-card');
  const P = { l: 150, r: 600, t: 10, row: 54 }, n = RESCUE.length;
  const X = v => P.l + v / 100 * (P.r - P.l);
  const g = el('g', {}, svg), grid = el('g', { class: 'grid' }, g), axis = el('g', { class: 'axis' }, g);
  for (const v of [0, 25, 50, 75, 100]) {
    el('line', { x1: X(v), x2: X(v), y1: P.t, y2: P.t + P.row * n }, grid);
    el('text', { x: X(v), y: P.t + P.row * n + 18, 'text-anchor': 'middle' }, axis).textContent = v;
  }
  RESCUE.forEach((r, i) => {
    const y = P.t + P.row * i + P.row / 2, up = r.sadl >= r.base;
    el('text', { x: 0, y: y - 3, 'font-weight': 700, fill: '#eef2f8' }, g).textContent = `${r.model} · seed ${r.seed}`;
    el('text', { x: 0, y: y + 14, 'font-size': 11.5 }, g).textContent = r.span + (r.note ? ` · ${r.note}` : '');
    el('line', { x1: X(r.base), x2: X(r.sadl), y1: y, y2: y, stroke: up ? 'rgba(33,150,243,.45)' : 'rgba(255,255,255,.22)', 'stroke-width': 3 }, g);
    el('circle', { cx: X(r.base), cy: y, r: 7, fill: '#a3abbb', stroke: '#0e1524', 'stroke-width': 2 }, g);
    el('circle', { cx: X(r.sadl), cy: y, r: 8, fill: '#2196f3', stroke: '#0e1524', 'stroke-width': 2 }, g);
    const d = r.sadl - r.base;
    el('text', { x: Math.max(X(r.base), X(r.sadl)) + 14, y: y + 4, 'font-weight': 700, fill: '#eef2f8', class: 'num' }, g).textContent = (d >= 0 ? '+' : '') + d.toFixed(1);
    const hit = el('rect', { x: 0, y: y - P.row / 2 + 3, width: 640, height: P.row - 6, fill: 'transparent' }, g);
    hit.addEventListener('pointermove', e => {
      tip.innerHTML = `<b>${r.model} · seed ${r.seed} · ${r.span}</b><br>Without SADL: ${r.base.toFixed(1)}%<br>With SADL: <b>${r.sadl.toFixed(1)}%</b>`;
      const c = card.getBoundingClientRect(); tip.style.left = Math.min(c.width - 210, e.clientX - c.left + 12) + 'px'; tip.style.top = (e.clientY - c.top - 60) + 'px'; tip.classList.add('on');
    });
    hit.addEventListener('pointerleave', () => tip.classList.remove('on'));
  });
  document.getElementById('rescue-table').innerHTML = '<thead><tr><th>Checkpoint</th><th>Without SADL</th><th>With SADL</th><th>Δ</th></tr></thead><tbody>' +
    RESCUE.map(r => `<tr><td>${r.model} seed ${r.seed}, ${r.span}${r.note ? ' (' + r.note + ')' : ''}</td><td>${r.base.toFixed(2)}</td><td>${r.sadl.toFixed(2)}</td><td>${(r.sadl - r.base >= 0 ? '+' : '') + (r.sadl - r.base).toFixed(2)}</td></tr>`).join('') + '</tbody>';
}
