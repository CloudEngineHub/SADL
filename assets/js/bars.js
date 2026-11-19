// Epoch-1 official macro success, D-only vs + SADL, three seeds (paper Table 4: tab:single / tab:joint).
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent?.appendChild(e); return e; };

// [model, regularizer, D-only mean, sd, + SADL mean, sd]
export const EPOCH1 = {
  single: [
    ['LeWM', 'SIGReg', 45.25, 6.50, 72.19, 0.97], ['LeWM', 'VISReg', 55.67, 5.46, 76.31, 1.23], ['LeWM', 'QQWorld', 51.19, 6.98, 75.28, 1.73],
    ['INTACT', 'SIGReg', 58.64, 8.14, 95.33, 0.44], ['INTACT', 'VISReg', 71.97, 12.79, 95.75, 0.08], ['INTACT', 'QQWorld', 61.56, 8.61, 95.47, 0.21]
  ],
  joint: [
    ['LeWM', 'SIGReg', 37.2, 8.3, 63.0, 1.3], ['LeWM', 'VISReg', 47.7, 5.1, 71.6, 3.0], ['LeWM', 'QQWorld', 32.9, 8.3, 68.4, 4.2],
    ['INTACT', 'SIGReg', 28.2, 6.4, 95.6, 0.2], ['INTACT', 'VISReg', 54.4, 2.9, 95.4, 0.6], ['INTACT', 'QQWorld', 38.6, 12.8, 95.6, 0.3]
  ]
};

export function initBars() {
  const svg = document.getElementById('bars-svg'); if (!svg) return;
  const tip = document.getElementById('bars-tip'), card = document.getElementById('bars-card');
  let mode = 'joint';
  const W = 1080, Hh = 380, P = { l: 46, r: 12, t: 16, b: 330 };
  const Y = v => P.b - v / 100 * (P.b - P.t);
  function draw() {
    svg.innerHTML = '';
    const g = el('g', { 'font-family': 'Inter,Segoe UI,sans-serif', 'font-size': 13 }, svg);
    for (const v of [0, 25, 50, 75, 100]) {
      el('line', { x1: P.l, x2: W - P.r, y1: Y(v), y2: Y(v), stroke: 'rgba(255,255,255,.07)' }, g);
      el('text', { x: P.l - 10, y: Y(v) + 4, 'text-anchor': 'end', fill: '#7f8ba1', 'font-size': 12 }, g).textContent = v;
    }
    const rows = EPOCH1[mode], gw = (W - P.l - P.r) / 7, bw = gw * .3;
    rows.forEach((r, i) => {
      const [model, reg, bm, bs, sm, ss] = r;
      const cx = P.l + gw * (i + (i >= 3 ? 1.5 : .5)) + gw * .1;
      for (const [k, m, s, col] of [[0, bm, bs, '#b9c0cc'], [1, sm, ss, '#4f8fd1']]) {
        const x = cx + (k ? 2 : -bw - 2);
        const bar = el('rect', { x, y: Y(m), width: bw, height: P.b - Y(m), rx: 4, fill: col }, g);
        bar.style.transformOrigin = `${x}px ${P.b}px`;
        el('line', { x1: x + bw / 2, x2: x + bw / 2, y1: Y(Math.min(100, m + s)), y2: Y(Math.max(0, m - s)), stroke: '#eef2f8', 'stroke-opacity': .55, 'stroke-width': 1.5 }, g);
        el('text', { x: x + bw / 2, y: Y(Math.min(100, m + s)) - 8, 'text-anchor': 'middle', fill: k ? '#eef2f8' : '#b4bfd2', 'font-weight': k ? 700 : 500, 'font-size': 13 }, g).textContent = m.toFixed(1);
      }
      el('text', { x: cx, y: P.b + 22, 'text-anchor': 'middle', fill: '#b4bfd2', 'font-weight': 600 }, g).textContent = reg;
      const hit = el('rect', { x: cx - gw / 2, y: P.t, width: gw, height: P.b - P.t, fill: 'transparent' }, g);
      hit.addEventListener('pointermove', e => {
        tip.innerHTML = `<b>${model} · ${reg} · epoch 1</b><br>${reg} only: ${bm.toFixed(1)} ± ${bs.toFixed(1)}<br>+ SADL: <b>${sm.toFixed(1)} ± ${ss.toFixed(1)}</b><br>Gain +${(sm - bm).toFixed(1)} pts`;
        const c = card.getBoundingClientRect(); tip.style.left = Math.min(c.width - 200, e.clientX - c.left + 12) + 'px'; tip.style.top = (e.clientY - c.top - 80) + 'px'; tip.classList.add('on');
      });
      hit.addEventListener('pointerleave', () => tip.classList.remove('on'));
    });
    for (const [label, i0] of [['LeWM · Pure-CEM', 0], ['INTACT · search-free Direct', 3]]) {
      const x0 = P.l + gw * (i0 + (i0 ? 1.5 : .5)) + gw * .1, x1 = x0 + gw * 2;
      el('line', { x1: x0 - gw * .4, x2: x1 + gw * .4, y1: P.b + 36, y2: P.b + 36, stroke: 'rgba(255,255,255,.18)' }, g);
      el('text', { x: (x0 + x1) / 2, y: P.b + 56, 'text-anchor': 'middle', fill: '#eef2f8', 'font-weight': 700, 'font-size': 15 }, g).textContent = label;
    }
    document.getElementById('bars-table').innerHTML = '<thead><tr><th>Model · regularizer</th><th>D-only</th><th>+ SADL</th></tr></thead><tbody>' +
      rows.map(r => `<tr><td>${r[0]} · ${r[1]}</td><td>${r[2].toFixed(1)} ± ${r[3].toFixed(1)}</td><td>${r[4].toFixed(1)} ± ${r[5].toFixed(1)}</td></tr>`).join('') + '</tbody>';
  }
  document.querySelectorAll('[data-bars]').forEach(b => b.addEventListener('click', () => {
    mode = b.dataset.bars; document.querySelectorAll('[data-bars]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); draw();
  }));
  draw();
}
