// Mainstream SSL with BatchNorm projectors, aggressive recipe (lr x4, no warmup), ImageNet-100, ViT-S/16, 100 epochs, seed 0.
// Source: rpplus8:/data/junhan/ICLRExperiments/SADL-SSL-20261119 (result.json per run).
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent?.appendChild(e); return e; };

// method: { metric: [baseline, + SADL] }
export const SSL = [
  { name: 'VICReg', online: [52.02, 60.44], linear: [58.16, 64.58], knn: [51.84, 59.64] },
  { name: 'BYOL', online: [48.64, 53.52], linear: [58.18, 61.62], knn: [51.74, 54.00] },
  { name: 'MoCo v3', online: [54.20, 55.72], linear: [62.38, 64.44], knn: [56.66, 57.46] }
];
const LABEL = { online: 'Online linear accuracy', linear: 'Frozen-feature linear probe', knn: 'kNN accuracy' };

export function initSSL() {
  const svg = document.getElementById('ssl-svg'); if (!svg) return;
  const tip = document.getElementById('ssl-tip'), card = document.getElementById('ssl-card');
  let metric = 'online';
  const W = 720, P = { l: 46, r: 12, t: 20, b: 300 }, LO = 0, HI = 70;
  const Y = v => P.b - (v - LO) / (HI - LO) * (P.b - P.t);
  function draw() {
    svg.innerHTML = '';
    document.getElementById('ssl-metric').textContent = LABEL[metric] + ' (%)';
    const g = el('g', { 'font-family': 'Inter,Segoe UI,sans-serif', 'font-size': 13 }, svg);
    for (let v = LO; v <= HI; v += 10) {
      el('line', { x1: P.l, x2: W - P.r, y1: Y(v), y2: Y(v), stroke: 'rgba(255,255,255,.07)' }, g);
      el('text', { x: P.l - 10, y: Y(v) + 4, 'text-anchor': 'end', fill: '#7f8ba1', 'font-size': 12 }, g).textContent = v;
    }
    const gw = (W - P.l - P.r) / SSL.length, bw = gw * .26;
    SSL.forEach((m, i) => {
      const [b, s] = m[metric], cx = P.l + gw * (i + .5);
      for (const [k, v, col] of [[0, b, '#b9c0cc'], [1, s, '#4f8fd1']]) {
        const x = cx + (k ? 3 : -bw - 3);
        el('rect', { x, y: Y(v), width: bw, height: P.b - Y(v), rx: 4, fill: col }, g);
        el('text', { x: x + bw / 2, y: Y(v) - 8, 'text-anchor': 'middle', fill: k ? '#eef2f8' : '#b4bfd2', 'font-weight': k ? 700 : 500 }, g).textContent = v.toFixed(1);
      }
      el('text', { x: cx, y: P.b + 24, 'text-anchor': 'middle', fill: '#eef2f8', 'font-weight': 700, 'font-size': 15 }, g).textContent = m.name;
      el('text', { x: cx, y: P.b + 44, 'text-anchor': 'middle', fill: '#4f8fd1', 'font-weight': 700 }, g).textContent = `+${(s - b).toFixed(1)}`;
      const hit = el('rect', { x: cx - gw / 2, y: P.t, width: gw, height: P.b - P.t, fill: 'transparent' }, g);
      hit.addEventListener('pointermove', e => {
        tip.innerHTML = `<b>${m.name}</b><br>Online ${m.online[0].toFixed(1)} → <b>${m.online[1].toFixed(1)}</b><br>Linear probe ${m.linear[0].toFixed(1)} → <b>${m.linear[1].toFixed(1)}</b><br>kNN ${m.knn[0].toFixed(1)} → <b>${m.knn[1].toFixed(1)}</b>`;
        const c = card.getBoundingClientRect(); tip.style.left = Math.min(c.width - 220, e.clientX - c.left + 12) + 'px'; tip.style.top = (e.clientY - c.top - 90) + 'px'; tip.classList.add('on');
      });
      hit.addEventListener('pointerleave', () => tip.classList.remove('on'));
    });
    document.getElementById('ssl-table').innerHTML = '<thead><tr><th>Method</th><th>Online</th><th>Linear probe</th><th>kNN</th></tr></thead><tbody>' +
      SSL.map(m => `<tr><td>${m.name}</td>${['online', 'linear', 'knn'].map(k => `<td>${m[k][0].toFixed(2)} → ${m[k][1].toFixed(2)}</td>`).join('')}</tr>`).join('') + '</tbody>';
  }
  document.querySelectorAll('[data-ssl]').forEach(b => b.addEventListener('click', () => {
    metric = b.dataset.ssl; document.querySelectorAll('[data-ssl]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); draw();
  }));
  draw();
}
