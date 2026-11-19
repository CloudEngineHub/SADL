import { initHero } from './hero.js';
import { initGate } from './charts.js';
import { initNullSpace } from './nullspace.js';
import { initControl } from './results.js';
import { initImageCharts } from './imagecharts.js';
import { initRescue } from './rescue.js';
import { initBars } from './bars.js';

const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];

// nav background + active link
const nav = $('#nav');
const onScroll = () => nav.classList.toggle('is-solid', scrollY > 40);
addEventListener('scroll', onScroll, { passive: true }); onScroll();
const links = $$('.nav-links a');
const spy = new IntersectionObserver(entries => entries.forEach(e => {
  if (e.isIntersecting) links.forEach(a => a.setAttribute('aria-current', String(a.getAttribute('href') === '#' + e.target.id)));
}), { rootMargin: '-45% 0px -50% 0px' });
$$('main section[id]').forEach(s => spy.observe(s));

// reveal on scroll; energy bars grow when visible
const rev = new IntersectionObserver(entries => entries.forEach(e => {
  if (!e.isIntersecting) return;
  e.target.classList.add('in');
  e.target.querySelectorAll('.ebar-fill').forEach(f => { f.style.width = f.dataset.w + '%'; });
  rev.unobserve(e.target);
}), { threshold: .12 });
$$('.reveal').forEach(n => rev.observe(n));

initHero($('#hero-canvas'));
initNullSpace();
initGate();
initControl();
initImageCharts();
initRescue();
initBars();

// geometry viewer: load three.js only when the section approaches
const geoList = $('#geo-list');
const geoObs = new IntersectionObserver(async ([e]) => {
  if (!e.isIntersecting) return; geoObs.disconnect();
  const { initGeometry, REGS } = await import('./geo3d.js');
  const viewer = await initGeometry({ stage: $('#geo-stage'), canvas: $('#geo-canvas') });
  geoList.innerHTML = REGS.map((r, i) => `<button type="button" class="geo-item" data-reg="${r.id}" aria-pressed="${i === 0}"><span>${r.label}</span><span class="sr">${r.sr.toFixed(2)}%</span></button>`).join('');
  const pick = id => {
    const r = REGS.find(x => x.id === id);
    $('#geo-name').textContent = r.label; $('#geo-sr').textContent = r.sr.toFixed(2) + '%';
    $$('.geo-item').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.reg === id)));
    viewer?.select(id);
  };
  geoList.addEventListener('click', ev => { const b = ev.target.closest('[data-reg]'); if (b) pick(b.dataset.reg); });
  pick('sigreg');
}, { rootMargin: '400px' });
geoObs.observe($('#geometry'));

// attention examples
$$('[data-attn]').forEach(b => b.addEventListener('click', () => {
  const k = b.dataset.attn;
  $('#attn-in').src = `assets/figures/attention-${k}-input.webp`;
  $('#attn-base').src = `assets/figures/attention-${k}-baseline.webp`;
  $('#attn-sadl').src = `assets/figures/attention-${k}-sadl.webp`;
  $$('[data-attn]').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
}));

// copy BibTeX
$('#copy-bib').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('#bib-text').textContent); $('#copy-bib').textContent = 'Copied'; }
  catch { $('#copy-bib').textContent = 'Select and copy'; }
  setTimeout(() => { $('#copy-bib').textContent = 'Copy'; }, 1600);
});
