// Image classification curves (paired baseline vs SADL), drawn from the paper figure's own data.
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent?.appendChild(e); return e; };
const NAMES = { 'ImageNet-LT': 'ImageNet-LT', 'Galaxy10': 'Galaxy10', 'CIFAR-10': 'CIFAR-10', 'Imagenette': 'Imagenette' };

function niceTicks(lo, hi, n = 4) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

export async function initImageCharts() {
  const accSvg = document.getElementById('acc-svg'), gainSvg = document.getElementById('gain-svg');
  if (!accSvg) return;
  const data = (await (await fetch('assets/data/image-curves.json')).json()).datasets;
  const tip = document.getElementById('img-tip'), card = document.getElementById('img-card');
  let current = 'Imagenette';
  const W = 720, P = { l: 48, r: 18 };

  function draw() {
    const d = data[current], E = d.epochs, n = E.length;
    const X = e => P.l + (e - E[0]) / (E[n - 1] - E[0]) * (W - P.l - P.r);
    // accuracy chart
    const lo = Math.min(...d.baseline.map(v => v[0] - v[1]), ...d.sadl.map(v => v[0] - v[1]));
    const hi = Math.max(...d.baseline.map(v => v[0] + v[1]), ...d.sadl.map(v => v[0] + v[1]));
    const A = { t: 14, b: 214 }, ya = v => A.b - (v - lo) / (hi - lo || 1) * (A.b - A.t);
    accSvg.innerHTML = '';
    const ga = el('g', { 'font-family': 'Inter,Segoe UI,sans-serif', 'font-size': 11 }, accSvg);
    for (const v of niceTicks(lo, hi)) { el('line', { x1: P.l, x2: W - P.r, y1: ya(v), y2: ya(v), stroke: 'rgba(255,255,255,.07)' }, ga); el('text', { x: P.l - 8, y: ya(v) + 4, 'text-anchor': 'end', fill: '#7f8ba1' }, ga).textContent = v; }
    for (const [key, color] of [['baseline', '#a3abbb'], ['sadl', '#2196f3']]) {
      const s = d[key];
      let band = ''; s.forEach(([m, sd], i) => { band += (i ? 'L' : 'M') + X(E[i]).toFixed(1) + ' ' + ya(m + sd).toFixed(1); });
      for (let i = n - 1; i >= 0; i--) band += 'L' + X(E[i]).toFixed(1) + ' ' + ya(s[i][0] - s[i][1]).toFixed(1);
      el('path', { d: band + 'Z', fill: color, 'fill-opacity': .14 }, ga);
      el('path', { d: s.map(([m], i) => (i ? 'L' : 'M') + X(E[i]).toFixed(1) + ' ' + ya(m).toFixed(1)).join(''), fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round' }, ga);
    }
    // gain chart (single series, SADL minus paired baseline)
    const glo = Math.min(0, ...d.gain.map(v => v[0] - v[1])), ghi = Math.max(...d.gain.map(v => v[0] + v[1]));
    const G = { t: 12, b: 132 }, yg = v => G.b - (v - glo) / (ghi - glo || 1) * (G.b - G.t);
    gainSvg.innerHTML = '';
    const gg = el('g', { 'font-family': 'Inter,Segoe UI,sans-serif', 'font-size': 11 }, gainSvg);
    for (const v of niceTicks(glo, ghi, 3)) { el('line', { x1: P.l, x2: W - P.r, y1: yg(v), y2: yg(v), stroke: v === 0 ? 'rgba(255,255,255,.28)' : 'rgba(255,255,255,.07)' }, gg); el('text', { x: P.l - 8, y: yg(v) + 4, 'text-anchor': 'end', fill: '#7f8ba1' }, gg).textContent = (v > 0 ? '+' : '') + v; }
    let area = `M${X(E[0])} ${yg(0)}`; d.gain.forEach(([m], i) => { area += `L${X(E[i]).toFixed(1)} ${yg(m).toFixed(1)}`; }); area += `L${X(E[n - 1])} ${yg(0)}Z`;
    el('path', { d: area, fill: '#2196f3', 'fill-opacity': .18 }, gg);
    el('path', { d: d.gain.map(([m], i) => (i ? 'L' : 'M') + X(E[i]).toFixed(1) + ' ' + yg(m).toFixed(1)).join(''), fill: 'none', stroke: '#2196f3', 'stroke-width': 1.6 }, gg);
    for (const t of niceTicks(E[0], E[n - 1], 4)) el('text', { x: X(t), y: G.b + 18, 'text-anchor': 'middle', fill: '#7f8ba1' }, gg).textContent = t;
    el('text', { x: W - P.r, y: G.b + 34, 'text-anchor': 'end', fill: '#b4bfd2' }, gg).textContent = 'epoch';
    for (const c of d.checkpoints) {
      const i = E.indexOf(c.epoch); if (i < 0) continue;
      el('circle', { cx: X(c.epoch), cy: yg(c.gain), r: 4.5, fill: '#eef2f8', stroke: '#0e1524', 'stroke-width': 2 }, gg);
    }
    // checkpoint tiles
    const [early, late] = d.checkpoints;
    document.getElementById('ck-early').innerHTML = `<div class="v num">+${early.gain.toFixed(2)}<small> pts</small></div><p>at epoch ${early.epoch} · ${early.baseline.toFixed(1)} → ${early.sadl.toFixed(1)}%</p>`;
    document.getElementById('ck-late').innerHTML = `<div class="v num">+${late.gain.toFixed(2)}<small> pts</small></div><p>at the fixed endpoint, epoch ${late.epoch} · ${late.baseline.toFixed(1)} → ${late.sadl.toFixed(1)}%</p>`;
    document.getElementById('img-seeds').textContent = `${d.seeds} paired training seeds · mean ± SD`;
    // hover crosshair shared by both charts
    const cross = [accSvg, gainSvg].map(svg => el('line', { y1: 0, y2: 1000, stroke: 'rgba(255,255,255,.35)', 'stroke-width': 1, opacity: 0 }, svg));
    const move = ev => {
      const r = accSvg.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * W;
      let i = Math.round((x - P.l) / (W - P.l - P.r) * (n - 1)); i = Math.max(0, Math.min(n - 1, i));
      cross.forEach(c => { c.setAttribute('x1', X(E[i])); c.setAttribute('x2', X(E[i])); c.setAttribute('opacity', 1); });
      tip.innerHTML = `<b>${NAMES[current]} · epoch ${E[i]}</b><br>Baseline ${d.baseline[i][0].toFixed(2)}%<br>+ SADL <b>${d.sadl[i][0].toFixed(2)}%</b><br>Gain ${d.gain[i][0] >= 0 ? '+' : ''}${d.gain[i][0].toFixed(2)} pts`;
      const cr = card.getBoundingClientRect();
      tip.style.left = Math.min(cr.width - 190, ev.clientX - cr.left + 14) + 'px'; tip.style.top = (ev.clientY - cr.top - 70) + 'px';
      tip.classList.add('on');
    };
    const leave = () => { tip.classList.remove('on'); cross.forEach(c => c.setAttribute('opacity', 0)); };
    for (const svg of [accSvg, gainSvg]) { svg.onpointermove = move; svg.onpointerleave = leave; }
  }
  document.querySelectorAll('[data-ds]').forEach(b => b.addEventListener('click', () => {
    current = b.dataset.ds; document.querySelectorAll('[data-ds]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); draw();
  }));
  draw();
}
