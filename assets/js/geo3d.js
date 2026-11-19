// 3D viewer for the real endpoint PCA (INTACT epoch 1, seed 0) of eight Demand Regularizers.
import * as THREE from '../vendor/three.module.js';
import { OrbitControls } from '../vendor/OrbitControls.js';

export const REGS = [
  { id: 'sigreg', label: 'SIGReg', sr: 95.58 },
  { id: 'visreg', label: 'VISReg', sr: 95.08 },
  { id: 'qqworld', label: 'QQWorld', sr: 95.81 },
  { id: 'energy-gaussian', label: 'Energy-Gaussian', sr: 95.33 },
  { id: 'vic-moments', label: 'VIC moments', sr: 95.03 },
  { id: 'student-t5', label: 'Student-t5', sr: 92.33 },
  { id: 'gmm-ring', label: 'GMM-Ring', sr: 95.11 },
  { id: 'sphere-shell', label: 'Sphere shell', sr: 95.44 }
];
const TASK_RGB = [[220, 73, 96], [69, 163, 138], [63, 124, 196], [213, 166, 53]];

function spriteTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d'); const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, 'rgba(255,255,255,1)'); grd.addColorStop(.62, 'rgba(255,255,255,1)'); grd.addColorStop(.8, 'rgba(255,255,255,.35)'); grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

export async function initGeometry({ stage, canvas, onSelect }) {
  let gl;
  try { gl = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true }); } catch { stage.querySelector('#geo-hint').textContent = 'WebGL unavailable'; return null; }
  const data = await (await fetch('assets/data/endpoint-pca.json')).json();
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, .1, 100);
  camera.position.set(2.5, 1.6, 2.8);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.autoRotate = !matchMedia('(prefers-reduced-motion: reduce)').matches; controls.autoRotateSpeed = .7;
  controls.minDistance = 1.6; controls.maxDistance = 8;
  controls.addEventListener('start', () => { controls.autoRotate = false; });

  const n = 4096;
  const geom = new THREE.BufferGeometry();
  const pos = new Float32Array(n * 3), target = new Float32Array(n * 3), col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { const c = TASK_RGB[Math.floor(i / 1024)]; col.set([c[0] / 255, c[1] / 255, c[2] / 255], i * 3); }
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const mat = new THREE.PointsMaterial({ size: .042, vertexColors: true, map: spriteTexture(), transparent: true, depthWrite: false, opacity: .92 });
  scene.add(new THREE.Points(geom, mat));
  const axes = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2.4, 2.4, 2.4)), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: .06 }));
  scene.add(axes);

  let loaded = false;
  function load(id) {
    const p = data.datasets[id].positions;
    // scale by RMS radius so every geometry fills the same view; shape is unchanged
    let s = 0; for (let i = 0; i < p.length; i++) s += p[i] * p[i];
    const rms = Math.sqrt(s / n) || 1;
    for (let i = 0; i < p.length; i++) target[i] = p[i] / rms * .72;
    if (!loaded) { pos.set(target); loaded = true; }
  }
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    gl.setPixelRatio(Math.min(Math.max(devicePixelRatio || 1, 1.5), 3)); gl.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(stage);
  let visible = false;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) requestAnimationFrame(tick); }).observe(stage);
  function tick() {
    if (!visible) return;
    for (let i = 0; i < pos.length; i++) pos[i] += (target[i] - pos[i]) * .08;
    geom.attributes.position.needsUpdate = true;
    controls.update(); gl.render(scene, camera);
    requestAnimationFrame(tick);
  }
  return { select(id) { load(id); onSelect?.(id); } };
}
