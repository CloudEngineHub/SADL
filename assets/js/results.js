// Control success: SIGReg baseline vs SIGReg + SADL (3 training seeds, mean and sample SD).
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent?.appendChild(e); return e; };

// values[epoch] = [baseline mean, baseline SD, SADL mean, SADL SD]
export const CONTROL = [
  { model: 'INTACT', backend: 'MATH', values: { 1: [40.5, 8.5, 94.9, 0.4], 2: [70.0, 6.1, 96.3, 0.6] } },
  { model: 'INTACT', backend: 'Flash', values: { 1: [45.8, 2.6, 95.4, 0.2], 2: [66.1, 34.5, 96.3, 0.1] } },
  { model: 'LeWM', backend: 'MATH', values: { 1: [30.9, 0.8, 63.9, 1.2], 2: [46.0, 6.0, 70.4, 1.6] } },
  { model: 'LeWM', backend: 'Flash', values: { 1: [32.1, 4.9, 63.4, 5.7], 2: [31.4, 15.8, 69.1, 2.9] } }
];

export function initControl() {
  const svg = document.getElementById('ctrl-svg'); if (!svg) return;
  const tip = document.getElementById('ctrl-tip'), card = document.getElementById('ctrl-card');
  let epoch = 1;
  const P = { l: 128, r: 620, t: 16, row: 64 };
  const X = v => P.l + Math.max(0, Math.min(100, v)) / 100 * (P.r - P.l);
  function show(e, html) {
    tip.innerHTML = html; tip.classList.add('on');
    const r = card.getBoundingClientRect();
    tip.style.left = Math.min(r.width - 200, e.clientX - r.left + 12) + 'px';
    tip.style.top = (e.clientY - r.top - 46) + 'px';
  }
  const hide = () => tip.classList.remove('on');
  function draw() {
    svg.innerHTML = '';
    const g = el('g', {}, svg);
    const grid = el('g', { class: 'grid' }, g), axis = el('g', { class: 'axis' }, g);
    for (const v of [0, 25, 50, 75, 100]) {
      el('line', { x1: X(v), x2: X(v), y1: P.t, y2: P.t + P.row * 4 }, grid);
      el('text', { x: X(v), y: P.t + P.row * 4 + 18, 'text-anchor': 'middle' }, axis).textContent = v;
    }
    CONTROL.forEach((row, i) => {
      const [bm, bs, sm, ss] = row.values[epoch];
      const y = P.t + P.row * i + P.row / 2;
      el('text', { x: 0, y: y - 4, 'font-weight': 700, fill: '#eef2f8' }, g).textContent = row.model;
      el('text', { x: 0, y: y + 14, 'font-size': 12 }, g).textContent = row.backend + ' attention';
      for (const [m, s, c] of [[bm, bs, '#a3abbb'], [sm, ss, '#2196f3']]) el('line', { x1: X(m - s), x2: X(m + s), y1: y, y2: y, stroke: c, 'stroke-width': 6, 'stroke-opacity': .25, 'stroke-linecap': 'round' }, g);
      el('line', { x1: X(bm), x2: X(sm), y1: y, y2: y, stroke: 'rgba(255,255,255,.22)', 'stroke-width': 2 }, g);
      const b = el('circle', { cx: X(bm), cy: y, r: 7, fill: '#a3abbb', stroke: '#0e1524', 'stroke-width': 2 }, g);
      const s = el('circle', { cx: X(sm), cy: y, r: 8, fill: '#2196f3', stroke: '#0e1524', 'stroke-width': 2 }, g);
      el('text', { x: X(sm) + 14, y: y + 4, 'font-weight': 700, fill: '#eef2f8', class: 'num' }, g).textContent = sm.toFixed(1);
      el('text', { x: X(bm) - 12, y: y + 4, 'text-anchor': 'end', class: 'num' }, g).textContent = bm.toFixed(1);
      const hit = el('rect', { x: P.l - 6, y: y - P.row / 2 + 4, width: P.r - P.l + 12, height: P.row - 8, fill: 'transparent' }, g);
      const html = `<b>${row.model} · ${row.backend} · epoch ${epoch}</b><br>SIGReg only: ${bm.toFixed(1)} ± ${bs.toFixed(1)}<br>+ SADL: <b>${sm.toFixed(1)} ± ${ss.toFixed(1)}</b>`;
      hit.addEventListener('pointermove', e => show(e, html)); hit.addEventListener('pointerleave', hide);
      void b; void s;
    });
    const tbl = document.getElementById('ctrl-table');
    tbl.innerHTML = '<thead><tr><th>Model · backend</th><th>Epoch</th><th>SIGReg only</th><th>+ SADL</th></tr></thead><tbody>' +
      CONTROL.flatMap(r => [1, 2].map(ep => { const [bm, bs, sm, ss] = r.values[ep]; return `<tr><td>${r.model} · ${r.backend}</td><td>${ep}</td><td>${bm.toFixed(1)} ± ${bs.toFixed(1)}</td><td>${sm.toFixed(1)} ± ${ss.toFixed(1)}</td></tr>`; })).join('') + '</tbody>';
  }
  document.querySelectorAll('[data-epoch]').forEach(btn => btn.addEventListener('click', () => {
    epoch = +btn.dataset.epoch; document.querySelectorAll('[data-epoch]').forEach(x => x.setAttribute('aria-pressed', String(x === btn))); draw();
  }));
  draw();
}
