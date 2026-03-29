/* ══════════════════════════════════════════════════════════════
   Whiteboard Theory Explorer — Interactive Visualisations
   ══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initScrollAnimations();
  initNavScroll();
  initPyramidViz();
  initYoloViz();
  initIoUViz();
  initCRNNViz();
  initLSTMViz();
  initCTCAlignmentDemo();
  initMappingDemo();
  initDecodingDemo();
  initLatticeViz();
});

/* ══════════════════════════════════════════════════════════════
   HERO PARTICLES
   ══════════════════════════════════════════════════════════════ */
function initParticles() {
  const container = document.getElementById('hero-particles');
  if (!container) return;
  const colors = ['#6366f1', '#a855f7', '#ec4899', '#22d3ee', '#4ade80'];
  for (let i = 0; i < 30; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 4 + 2;
    p.style.width = size + 'px';
    p.style.height = size + 'px';
    p.style.left = Math.random() * 100 + '%';
    p.style.background = colors[Math.floor(Math.random() * colors.length)];
    p.style.animationDuration = (Math.random() * 15 + 10) + 's';
    p.style.animationDelay = (Math.random() * 10) + 's';
    container.appendChild(p);
  }
}

/* ══════════════════════════════════════════════════════════════
   SCROLL ANIMATIONS
   ══════════════════════════════════════════════════════════════ */
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

  document.querySelectorAll('.fade-in, .fade-in-left, .fade-in-right').forEach(el => {
    observer.observe(el);
  });
}

/* ══════════════════════════════════════════════════════════════
   NAV SCROLL PROGRESS + ACTIVE SECTION
   ══════════════════════════════════════════════════════════════ */
function initNavScroll() {
  const progressBar = document.getElementById('scroll-progress');
  const sections = document.querySelectorAll('.section');
  const navLinks = document.querySelectorAll('.nav-links a');

  window.addEventListener('scroll', () => {
    // Progress bar
    const scrolled = window.scrollY;
    const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
    const pct = (scrolled / maxScroll) * 100;
    if (progressBar) progressBar.style.width = pct + '%';

    // Active section highlight
    let current = '';
    sections.forEach(sec => {
      const top = sec.offsetTop - 100;
      if (scrolled >= top) current = sec.id;
    });
    navLinks.forEach(link => {
      link.classList.toggle('active', link.getAttribute('href') === '#' + current);
    });
  });
}

/* ══════════════════════════════════════════════════════════════
   PYRAMID VIZ (CNN Multi-Scale Features)
   ══════════════════════════════════════════════════════════════ */
function initPyramidViz() {
  const canvas = document.getElementById('feature-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const btns = document.querySelectorAll('.pyramid-btn');

  const levels = {
    1: { res: '52 × 52', channels: '128', receptive: 'Small regions', detects: 'Arrows, Small Annotations', gridSize: 26, color: '#60a5fa' },
    2: { res: '26 × 26', channels: '256', receptive: 'Medium regions', detects: 'Handwriting, Equations', gridSize: 13, color: '#4ade80' },
    3: { res: '13 × 13', channels: '512', receptive: 'Large regions', detects: 'Diagrams, Sticky Notes', gridSize: 7, color: '#fb923c' },
  };

  function drawGrid(level) {
    const data = levels[level];
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const n = data.gridSize;
    const cellW = w / n;
    const cellH = h / n;

    // Draw grid cells with random "activation"
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const activation = Math.random() * 0.6 + 0.1;
        ctx.fillStyle = hexToRGBA(data.color, activation);
        ctx.fillRect(j * cellW + 0.5, i * cellH + 0.5, cellW - 1, cellH - 1);
      }
    }

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= n; i++) {
      ctx.beginPath(); ctx.moveTo(i * cellW, 0); ctx.lineTo(i * cellW, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cellH); ctx.lineTo(w, i * cellH); ctx.stroke();
    }

    // Update info
    document.getElementById('info-resolution').textContent = data.res;
    document.getElementById('info-channels').textContent = data.channels;
    document.getElementById('info-receptive').textContent = data.receptive;
    document.getElementById('info-detects').textContent = data.detects;
  }

  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      drawGrid(parseInt(btn.dataset.level));
    });
  });

  drawGrid(1);
}

/* ══════════════════════════════════════════════════════════════
   YOLO GRID VISUALISATION
   ══════════════════════════════════════════════════════════════ */
function initYoloViz() {
  const canvas = document.getElementById('yolo-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const GRID = 7;
  const cellW = W / GRID, cellH = H / GRID;

  // Simulated detections on a whiteboard (normalised xc, yc, w, h)
  const detections = [
    { xc: 0.25, yc: 0.18, w: 0.40, h: 0.22, cls: 'Handwriting', color: '#4ade80', conf: 0.92 },
    { xc: 0.72, yc: 0.22, w: 0.38, h: 0.30, cls: 'Diagram', color: '#60a5fa', conf: 0.88 },
    { xc: 0.50, yc: 0.42, w: 0.20, h: 0.06, cls: 'Arrow', color: '#f472b6', conf: 0.85 },
    { xc: 0.30, yc: 0.62, w: 0.45, h: 0.12, cls: 'Equation', color: '#a855f7', conf: 0.91 },
    { xc: 0.78, yc: 0.75, w: 0.22, h: 0.22, cls: 'Sticky Note', color: '#fb923c', conf: 0.87 },
  ];

  let hoveredCell = null;
  let showDetections = true;

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Background page sim
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);

    // Draw simulated doc content
    drawDocContent(ctx, W, H);

    // Grid overlay
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= GRID; i++) {
      ctx.beginPath(); ctx.moveTo(i * cellW, 0); ctx.lineTo(i * cellW, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cellH); ctx.lineTo(W, i * cellH); ctx.stroke();
    }

    // Hovered cell
    if (hoveredCell) {
      ctx.fillStyle = 'rgba(99, 102, 241, 0.15)';
      ctx.fillRect(hoveredCell.col * cellW, hoveredCell.row * cellH, cellW, cellH);
      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(hoveredCell.col * cellW, hoveredCell.row * cellH, cellW, cellH);
    }

    // Detection boxes
    if (showDetections) {
      detections.forEach(det => {
        const x1 = (det.xc - det.w / 2) * W;
        const y1 = (det.yc - det.h / 2) * H;
        const bw = det.w * W;
        const bh = det.h * H;

        ctx.strokeStyle = det.color;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.strokeRect(x1, y1, bw, bh);
        ctx.setLineDash([]);

        // Label background
        const label = `${det.cls} ${det.conf.toFixed(2)}`;
        ctx.font = '600 11px Inter, sans-serif';
        const tm = ctx.measureText(label);
        ctx.fillStyle = det.color;
        ctx.fillRect(x1, y1 - 16, tm.width + 10, 16);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x1 + 5, y1 - 4);

        // Center dot
        ctx.fillStyle = det.color;
        ctx.beginPath();
        ctx.arc(det.xc * W, det.yc * H, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }

  function drawDocContent(ctx, W, H) {
    // Whiteboard surface — subtle off-white tint
    ctx.fillStyle = 'rgba(255,255,250,0.04)';
    ctx.fillRect(0, 0, W, H);

    // Handwriting region (scribble lines)
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < 4; i++) {
      const sx = W * 0.06;
      const sy = H * 0.10 + i * H * 0.055;
      const ex = sx + (0.25 + Math.random() * 0.10) * W;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.quadraticCurveTo(sx + (ex - sx) * 0.5, sy + (Math.random() - 0.5) * 8, ex, sy);
      ctx.stroke();
    }

    // Diagram region (rectangle + circle)
    ctx.strokeStyle = 'rgba(96,165,250,0.12)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(W * 0.55, H * 0.08, W * 0.16, H * 0.14);
    ctx.beginPath();
    ctx.arc(W * 0.78, H * 0.22, W * 0.07, 0, Math.PI * 2);
    ctx.stroke();
    // Connector line
    ctx.beginPath();
    ctx.moveTo(W * 0.71, H * 0.15);
    ctx.lineTo(W * 0.73, H * 0.18);
    ctx.stroke();

    // Arrow region
    ctx.strokeStyle = 'rgba(244,114,182,0.12)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(W * 0.38, H * 0.42);
    ctx.lineTo(W * 0.58, H * 0.42);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(W * 0.58, H * 0.42);
    ctx.lineTo(W * 0.55, H * 0.39);
    ctx.moveTo(W * 0.58, H * 0.42);
    ctx.lineTo(W * 0.55, H * 0.45);
    ctx.stroke();

    // Equation region
    ctx.fillStyle = 'rgba(168,85,247,0.08)';
    ctx.fillRect(W * 0.08, H * 0.56, W * 0.44, H * 0.10);
    ctx.fillStyle = 'rgba(168,85,247,0.12)';
    ctx.font = '14px JetBrains Mono';
    ctx.fillText('y = mx + b', W * 0.14, H * 0.63);

    // Sticky note region
    ctx.fillStyle = 'rgba(251,146,60,0.08)';
    ctx.fillRect(W * 0.67, H * 0.64, W * 0.22, H * 0.22);
    ctx.strokeStyle = 'rgba(251,146,60,0.12)';
    ctx.lineWidth = 1;
    ctx.strokeRect(W * 0.67, H * 0.64, W * 0.22, H * 0.22);
  }

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (W / rect.width);
    const y = (e.clientY - rect.top) * (H / rect.height);
    hoveredCell = { col: Math.floor(x / cellW), row: Math.floor(y / cellH) };
    draw();
  });

  canvas.addEventListener('mouseleave', () => {
    hoveredCell = null;
    draw();
  });

  canvas.addEventListener('click', () => {
    showDetections = !showDetections;
    draw();
  });

  draw();
}

/* ══════════════════════════════════════════════════════════════
   IoU INTERACTIVE VISUALISATION
   ══════════════════════════════════════════════════════════════ */
function initIoUViz() {
  const canvas = document.getElementById('iou-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  // Ground truth box (fixed)
  const gt = { x: 130, y: 80, w: 200, h: 180 };
  // Predicted box (draggable)
  let pred = { x: 180, y: 100, w: 200, h: 180 };
  let dragging = false;
  let dragOffset = { x: 0, y: 0 };

  function computeIoU(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w);
    const y2 = Math.min(a.y + a.h, b.y + b.h);
    const interW = Math.max(0, x2 - x1);
    const interH = Math.max(0, y2 - y1);
    const inter = interW * interH;
    const union = a.w * a.h + b.w * b.h - inter;
    return union > 0 ? inter / union : 0;
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Intersection
    const ix1 = Math.max(gt.x, pred.x);
    const iy1 = Math.max(gt.y, pred.y);
    const ix2 = Math.min(gt.x + gt.w, pred.x + pred.w);
    const iy2 = Math.min(gt.y + gt.h, pred.y + pred.h);
    if (ix2 > ix1 && iy2 > iy1) {
      ctx.fillStyle = 'rgba(99, 102, 241, 0.25)';
      ctx.fillRect(ix1, iy1, ix2 - ix1, iy2 - iy1);
    }

    // Ground truth
    ctx.strokeStyle = '#4ade80';
    ctx.lineWidth = 2.5;
    ctx.strokeRect(gt.x, gt.y, gt.w, gt.h);
    ctx.fillStyle = '#4ade80';
    ctx.font = '600 12px Inter';
    ctx.fillText('Ground Truth', gt.x, gt.y - 8);

    // Predicted
    ctx.strokeStyle = '#f472b6';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([8, 4]);
    ctx.strokeRect(pred.x, pred.y, pred.w, pred.h);
    ctx.setLineDash([]);
    ctx.fillStyle = '#f472b6';
    ctx.fillText('Predicted (drag me)', pred.x, pred.y - 8);

    // Center dots
    ctx.fillStyle = '#4ade80';
    ctx.beginPath(); ctx.arc(gt.x + gt.w/2, gt.y + gt.h/2, 4, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#f472b6';
    ctx.beginPath(); ctx.arc(pred.x + pred.w/2, pred.y + pred.h/2, 4, 0, Math.PI*2); ctx.fill();

    // Distance line
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(gt.x + gt.w/2, gt.y + gt.h/2);
    ctx.lineTo(pred.x + pred.w/2, pred.y + pred.h/2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Update IoU readout
    const iou = computeIoU(gt, pred);
    document.getElementById('iou-value').textContent = `IoU = ${iou.toFixed(3)}`;
    document.getElementById('iou-bar').style.width = (iou * 100) + '%';
  }

  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const my = (e.clientY - rect.top) * (H / rect.height);
    if (mx >= pred.x && mx <= pred.x + pred.w && my >= pred.y && my <= pred.y + pred.h) {
      dragging = true;
      dragOffset = { x: mx - pred.x, y: my - pred.y };
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const my = (e.clientY - rect.top) * (H / rect.height);
    pred.x = Math.max(0, Math.min(W - pred.w, mx - dragOffset.x));
    pred.y = Math.max(0, Math.min(H - pred.h, my - dragOffset.y));
    draw();
  });

  canvas.addEventListener('mouseup', () => { dragging = false; });
  canvas.addEventListener('mouseleave', () => { dragging = false; });

  draw();
}

/* ══════════════════════════════════════════════════════════════
   CRNN PIPELINE VISUALISATION
   ══════════════════════════════════════════════════════════════ */
function initCRNNViz() {
  // Feature columns
  const colsContainer = document.getElementById('feature-cols');
  if (colsContainer) {
    const T = 12;
    for (let i = 0; i < T; i++) {
      const col = document.createElement('div');
      col.className = 'feature-col';
      col.style.height = (20 + Math.random() * 25) + 'px';
      col.style.animationDelay = (i * 0.05) + 's';
      colsContainer.appendChild(col);
    }
  }

  // BiLSTM cells
  const bilstmViz = document.getElementById('bilstm-viz');
  if (bilstmViz) {
    const T = 8;
    for (let i = 0; i < T; i++) {
      if (i > 0) {
        const fwdArrow = document.createElement('div');
        fwdArrow.className = 'bilstm-arrow';
        fwdArrow.textContent = '⇄';
        bilstmViz.appendChild(fwdArrow);
      }
      const cell = document.createElement('div');
      cell.className = 'bilstm-cell';
      cell.innerHTML = `h<sub>${i+1}</sub>`;
      bilstmViz.appendChild(cell);
    }
  }

  // CTC output preview
  const ctcOutput = document.getElementById('ctc-output-preview');
  if (ctcOutput) {
    const sequence = ['H', 'ε', 'e', 'l', 'l', 'ε', 'l', 'o', 'ε'];
    const result = 'Hello';
    sequence.forEach(ch => {
      const cell = document.createElement('div');
      cell.className = 'ctc-out-cell ' + (ch === 'ε' ? 'blank' : 'char');
      cell.textContent = ch;
      ctcOutput.appendChild(cell);
    });
    // Arrow
    const arrow = document.createElement('div');
    arrow.style.cssText = 'color: var(--text-muted); font-size: 1.2rem; margin: 0 8px;';
    arrow.textContent = '→';
    ctcOutput.appendChild(arrow);
    // Result
    result.split('').forEach(ch => {
      const cell = document.createElement('div');
      cell.className = 'ctc-out-cell result';
      cell.textContent = ch;
      ctcOutput.appendChild(cell);
    });
  }
}

/* ══════════════════════════════════════════════════════════════
   LSTM GATE VISUALISATION (Canvas)
   ══════════════════════════════════════════════════════════════ */
function initLSTMViz() {
  const canvas = document.getElementById('lstm-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Match canvas resolution to CSS display size (1:1, no DPR scaling)
  // so mouse coordinates map directly to canvas drawing coordinates.
  let initialized = false;

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return { w: 680, h: 450 };
    canvas.width = rect.width;
    canvas.height = rect.height;
    return { w: rect.width, h: rect.height };
  }

  let displaySize = { w: 680, h: 450 }; // fallback until visible

  const gateColors = {
    forget: '#ef4444',
    input: '#60a5fa',
    cell: '#4ade80',
    output: '#fb923c'
  };

  // Gate regions are calculated dynamically based on canvas display size
  function getGateData(w, h) {
    // Layout: gates are proportionally placed
    const gw = w * 0.18;  // gate width
    const gh = h * 0.15;  // gate height
    const padL = w * 0.10; // left padding
    const midY = h * 0.40; // vertical midpoint for gates

    return {
      forget: {
        title: 'Forget Gate (fτ)',
        eq: 'fτ = σ(Wf [hτ-1; xτ] + bf)',
        desc: 'Controls what information to DISCARD from the previous cell state. Acts as a filter: values near 0 forget, values near 1 remember. This is critical for handling long sequences—it prevents irrelevant context from accumulating.',
        region: { x: padL, y: midY - gh - h * 0.04, w: gw, h: gh }
      },
      input: {
        title: 'Input Gate (iτ)',
        eq: 'iτ = σ(Wi [hτ-1; xτ] + bi)',
        desc: 'Controls what NEW information to STORE. Works with the cell candidate to determine the magnitude of new values added to the cell state. High values mean "this input is important."',
        region: { x: padL, y: midY + h * 0.04, w: gw, h: gh }
      },
      cell: {
        title: 'Cell Candidate (c̃τ)',
        eq: 'c̃τ = tanh(Wc [hτ-1; xτ] + bc)',
        desc: 'Creates candidate values that COULD BE added to the cell state. Uses tanh to output values in [-1, 1]. The input gate decides how much of this candidate actually gets stored.',
        region: { x: w * 0.38, y: midY - gh * 0.5, w: gw, h: gh }
      },
      output: {
        title: 'Output Gate (oτ)',
        eq: 'oτ = σ(Wo [hτ-1; xτ] + bo)',
        desc: 'Controls what to OUTPUT as the hidden state. Filters the cell state through tanh and gates it, producing the hidden state hτ that is passed to the next time-step and to the output layer.',
        region: { x: w * 0.66, y: midY - gh * 0.5, w: gw, h: gh }
      }
    };
  }

  let gateData = getGateData(displaySize.w, displaySize.h);
  let activeGate = null;

  function draw() {
    const w = displaySize.w;
    const h = displaySize.h;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#0d0d14';
    ctx.fillRect(0, 0, w, h);

    // Main cell outline
    const cellX = w * 0.08, cellY = h * 0.10;
    const cellW = w * 0.84, cellH = h * 0.78;
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.strokeRect(cellX, cellY, cellW, cellH);
    ctx.fillStyle = 'rgba(255,255,255,0.015)';
    ctx.fillRect(cellX, cellY, cellW, cellH);

    // Cell state highway (top)
    const csY = h * 0.20;
    drawArrow(ctx, w * 0.05, csY, w * 0.95, csY, 'rgba(74, 222, 128, 0.4)', 3);
    ctx.fillStyle = '#4ade80';
    ctx.font = '600 11px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Cell State  cτ-1 →  cτ', w * 0.5, csY - 10);
    ctx.textAlign = 'left';

    // Hidden state path (bottom)
    const hsY = h * 0.82;
    drawArrow(ctx, w * 0.05, hsY, w * 0.95, hsY, 'rgba(251, 146, 60, 0.4)', 3);
    ctx.fillStyle = '#fb923c';
    ctx.font = '600 11px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Hidden State  hτ-1 →  hτ', w * 0.5, hsY + 18);
    ctx.textAlign = 'left';

    // Input arrow (xτ)
    drawArrow(ctx, w * 0.5, h * 0.96, w * 0.5, hsY + 4, 'rgba(255,255,255,0.25)', 2);
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '600 12px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('xτ', w * 0.5, h * 0.99);
    ctx.textAlign = 'left';

    // hτ-1 arrow
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '600 11px Inter';
    ctx.fillText('hτ-1', w * 0.01, hsY - 6);

    // Draw gates (larger, more visible)
    Object.entries(gateData).forEach(([key, data]) => {
      const r = data.region;
      const isActive = activeGate === key;
      const color = gateColors[key];

      // Gate box — significantly more visible
      ctx.fillStyle = isActive ? hexToRGBA(color, 0.35) : hexToRGBA(color, 0.2);
      roundRect(ctx, r.x, r.y, r.w, r.h, 10);
      ctx.fill();

      ctx.strokeStyle = isActive ? color : hexToRGBA(color, 0.7);
      ctx.lineWidth = isActive ? 3 : 2;
      roundRect(ctx, r.x, r.y, r.w, r.h, 10);
      ctx.stroke();

      // Gate label
      ctx.fillStyle = isActive ? '#fff' : 'rgba(255,255,255,0.85)';
      ctx.font = `${isActive ? '700' : '600'} 13px Inter`;
      ctx.textAlign = 'center';
      ctx.fillText(data.title.split('(')[0].trim(), r.x + r.w / 2, r.y + r.h / 2 - 4);

      // Gate activation function symbol
      ctx.font = '500 11px JetBrains Mono';
      ctx.fillStyle = isActive ? color : hexToRGBA(color, 0.9);
      const symbol = key === 'cell' ? 'tanh' : 'σ';
      ctx.fillText(symbol, r.x + r.w / 2, r.y + r.h / 2 + 14);
      ctx.textAlign = 'left';

      // Subtle glow on active
      if (isActive) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 15;
        roundRect(ctx, r.x, r.y, r.w, r.h, 10);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    });

    // Pointwise operations
    const forgetR = gateData.forget.region;
    const inputR = gateData.input.region;
    const cellR = gateData.cell.region;
    const outputR = gateData.output.region;

    // ⊙ multiply (forget × c_prev) on cell state line
    const mulForgetX = forgetR.x + forgetR.w + (cellR.x - forgetR.x - forgetR.w) * 0.5;
    drawCircleOp(ctx, mulForgetX, csY, '×', '#ef4444');

    // ⊙ multiply (input × candidate)
    const mulInputX = cellR.x + cellR.w * 0.5;
    const mulInputY = csY + (cellR.y - csY) * 0.5;
    drawCircleOp(ctx, mulInputX + cellR.w * 0.6, mulInputY, '×', '#60a5fa');

    // + (add to cell state)
    const addX = cellR.x + cellR.w + (outputR.x - cellR.x - cellR.w) * 0.3;
    drawCircleOp(ctx, addX, csY, '+', '#4ade80');

    // ⊙ multiply (output × tanh(c))
    const mulOutX = outputR.x + outputR.w * 0.5;
    const mulOutY = outputR.y + outputR.h + (hsY - outputR.y - outputR.h) * 0.5;
    drawCircleOp(ctx, mulOutX, mulOutY, '×', '#fb923c');

    // tanh box after cell state
    const tanhX = outputR.x - w * 0.02;
    const tanhY = csY + (outputR.y - csY) * 0.3;
    ctx.fillStyle = 'rgba(168, 85, 247, 0.2)';
    roundRect(ctx, tanhX - 25, tanhY - 12, 50, 24, 6);
    ctx.fill();
    ctx.strokeStyle = 'rgba(168, 85, 247, 0.6)';
    ctx.lineWidth = 1.5;
    roundRect(ctx, tanhX - 25, tanhY - 12, 50, 24, 6);
    ctx.stroke();
    ctx.fillStyle = '#a855f7';
    ctx.font = '500 11px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText('tanh', tanhX, tanhY + 4);
    ctx.textAlign = 'left';

    // Connection lines (dashed)
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);

    // Forget gate → multiply on cell state
    drawLine(ctx, forgetR.x + forgetR.w / 2, forgetR.y, forgetR.x + forgetR.w / 2, csY + 16);

    // Input gate → multiply with candidate
    drawLine(ctx, inputR.x + inputR.w, inputR.y + inputR.h / 2, mulInputX + cellR.w * 0.6 - 16, mulInputY);

    // Cell candidate → multiply
    drawLine(ctx, cellR.x + cellR.w, cellR.y + cellR.h / 2, mulInputX + cellR.w * 0.6 - 16, mulInputY);

    // Multiply result → add on cell state
    drawLine(ctx, mulInputX + cellR.w * 0.6, mulInputY - 16, addX, csY + 16);

    // Output gate → multiply below
    drawLine(ctx, outputR.x + outputR.w / 2, outputR.y + outputR.h, mulOutX, mulOutY - 16);

    // tanh → output multiply
    drawLine(ctx, tanhX, tanhY + 12, mulOutX - 16, mulOutY);

    // Output multiply → hidden state
    drawLine(ctx, mulOutX, mulOutY + 16, mulOutX, hsY - 4);

    ctx.setLineDash([]);
  }

  function drawCircleOp(ctx, x, y, symbol, color) {
    ctx.fillStyle = hexToRGBA(color, 0.2);
    ctx.beginPath(); ctx.arc(x, y, 16, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = hexToRGBA(color, 0.6);
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, 16, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = '700 14px Inter';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(symbol, x, y);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let found = null;
    Object.entries(gateData).forEach(([key, data]) => {
      const r = data.region;
      if (mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h) {
        found = key;
      }
    });

    if (found !== activeGate) {
      activeGate = found;
      draw();
      updateGateInfo(found);
    }
    canvas.style.cursor = found ? 'pointer' : 'default';
  });

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    Object.entries(gateData).forEach(([key, data]) => {
      const r = data.region;
      if (mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h) {
        activeGate = key;
        draw();
        updateGateInfo(key);
      }
    });
  });

  // Handle canvas resize
  window.addEventListener('resize', () => {
    displaySize = resizeCanvas();
    gateData = getGateData(displaySize.w, displaySize.h);
    draw();
  });

  function updateGateInfo(gate) {
    const titleEl = document.getElementById('gate-title');
    const eqEl = document.getElementById('gate-equation');
    const descEl = document.getElementById('gate-description');
    if (!gate) {
      titleEl.textContent = 'Select a Gate';
      eqEl.textContent = 'Click or hover over any gate in the LSTM cell diagram.';
      descEl.textContent = 'The LSTM cell uses four gates to control information flow: Forget (what to discard), Input (what to store), Cell Candidate (what could be stored), and Output (what to emit as hidden state).';
      return;
    }
    const data = gateData[gate];
    titleEl.textContent = data.title;
    eqEl.textContent = data.eq;
    descEl.textContent = data.desc;
  }

  // Initialize canvas properly when it becomes visible (may be below fold)
  const initObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !initialized) {
        initialized = true;
        displaySize = resizeCanvas();
        gateData = getGateData(displaySize.w, displaySize.h);
        draw();
      }
    });
  }, { threshold: 0.1 });
  initObserver.observe(canvas);

  // Also try immediate initialization (works if canvas is already visible)
  displaySize = resizeCanvas();
  gateData = getGateData(displaySize.w, displaySize.h);
  draw();
}

/* ══════════════════════════════════════════════════════════════
   CTC ALIGNMENT DEMO
   ══════════════════════════════════════════════════════════════ */
function initCTCAlignmentDemo() {
  const container = document.getElementById('align-outputs');
  if (!container) return;

  const outputs = ['c', 'ε', 'ε', 'a', 'a', 'ε', 't', 'ε', 'ε', 'ε'];
  outputs.forEach((ch, i) => {
    const cell = document.createElement('div');
    cell.className = 'align-cell ' + (ch === 'ε' ? 'blank-cell' : 'char-cell');
    cell.textContent = ch;
    cell.style.animationDelay = (i * 0.1) + 's';
    container.appendChild(cell);
  });
}

/* ══════════════════════════════════════════════════════════════
   CTC MAPPING DEMO (Multiple paths)
   ══════════════════════════════════════════════════════════════ */
function initMappingDemo() {
  const paths = [
    { input: ['c', 'ε', 'ε', 'a', 'a', 'ε', 't'], dedup: ['c', 'ε', 'a', 'ε', 't'], result: ['c', 'a', 't'] },
    { input: ['ε', 'c', 'a', 'a', 't', 't', 'ε'], dedup: ['ε', 'c', 'a', 't', 'ε'], result: ['c', 'a', 't'] },
    { input: ['c', 'c', 'a', 'ε', 'ε', 't', 't'], dedup: ['c', 'a', 'ε', 't'], result: ['c', 'a', 't'] },
    { input: ['ε', 'ε', 'c', 'a', 't', 'ε', 'ε'], dedup: ['ε', 'c', 'a', 't', 'ε'], result: ['c', 'a', 't'] },
  ];
  let currentPath = 0;

  function renderPath(idx) {
    const path = paths[idx];
    ['mapping-input', 'mapping-dedup', 'mapping-result'].forEach(id => {
      document.getElementById(id).innerHTML = '';
    });

    path.input.forEach(ch => {
      const cell = document.createElement('div');
      cell.className = 'align-cell ' + (ch === 'ε' ? 'blank-cell' : 'char-cell');
      cell.textContent = ch;
      document.getElementById('mapping-input').appendChild(cell);
    });

    path.dedup.forEach(ch => {
      const cell = document.createElement('div');
      cell.className = 'align-cell ' + (ch === 'ε' ? 'blank-cell' : 'char-cell');
      cell.textContent = ch;
      document.getElementById('mapping-dedup').appendChild(cell);
    });

    path.result.forEach(ch => {
      const cell = document.createElement('div');
      cell.className = 'align-cell char-cell';
      cell.style.background = 'rgba(74,222,128,0.2)';
      cell.style.color = '#4ade80';
      cell.style.borderColor = 'rgba(74,222,128,0.4)';
      cell.textContent = ch;
      document.getElementById('mapping-result').appendChild(cell);
    });

    document.getElementById('mapping-counter').textContent = `Path ${idx + 1} / ${paths.length}`;
  }

  document.getElementById('mapping-next')?.addEventListener('click', () => {
    currentPath = (currentPath + 1) % paths.length;
    renderPath(currentPath);
  });

  document.getElementById('mapping-prev')?.addEventListener('click', () => {
    currentPath = (currentPath - 1 + paths.length) % paths.length;
    renderPath(currentPath);
  });

  renderPath(0);
}

/* ══════════════════════════════════════════════════════════════
   GREEDY DECODING DEMO
   ══════════════════════════════════════════════════════════════ */
function initDecodingDemo() {
  const barContainer = document.getElementById('decode-bars');
  const resultContainer = document.getElementById('decode-result');
  if (!barContainer || !resultContainer) return;

  // Simulated softmax outputs for "cat"
  const steps = [
    { probs: { c: 0.82, a: 0.05, t: 0.03, ε: 0.10 }, best: 'c' },
    { probs: { c: 0.15, a: 0.08, t: 0.02, ε: 0.75 }, best: 'ε' },
    { probs: { c: 0.04, a: 0.05, t: 0.01, ε: 0.90 }, best: 'ε' },
    { probs: { c: 0.03, a: 0.85, t: 0.02, ε: 0.10 }, best: 'a' },
    { probs: { c: 0.02, a: 0.78, t: 0.08, ε: 0.12 }, best: 'a' },
    { probs: { c: 0.01, a: 0.10, t: 0.05, ε: 0.84 }, best: 'ε' },
    { probs: { c: 0.02, a: 0.03, t: 0.88, ε: 0.07 }, best: 't' },
    { probs: { c: 0.01, a: 0.02, t: 0.12, ε: 0.85 }, best: 'ε' },
  ];

  const accentColors = { c: '#6366f1', a: '#a855f7', t: '#ec4899', ε: '#3b3b5c' };

  steps.forEach((step, i) => {
    const stepDiv = document.createElement('div');
    stepDiv.className = 'decode-step';

    // Time label
    const timeLabel = document.createElement('div');
    timeLabel.className = 'decode-label';
    timeLabel.textContent = `τ=${i+1}`;
    stepDiv.appendChild(timeLabel);

    // Bar
    const barWrap = document.createElement('div');
    barWrap.className = 'decode-bar-container';
    const bar = document.createElement('div');
    bar.className = 'decode-bar';
    bar.style.height = (step.probs[step.best] * 100) + '%';
    bar.style.background = accentColors[step.best];
    bar.style.transitionDelay = (i * 0.08) + 's';
    barWrap.appendChild(bar);
    stepDiv.appendChild(barWrap);

    // Best char
    const charDiv = document.createElement('div');
    charDiv.className = 'decode-char ' + (step.best === 'ε' ? 'is-blank' : 'is-char');
    charDiv.textContent = step.best;
    stepDiv.appendChild(charDiv);

    barContainer.appendChild(stepDiv);
  });

  // Result after B mapping
  const resultLabel = document.createElement('div');
  resultLabel.style.cssText = 'font-family: var(--font-mono); font-size: 0.82rem; color: var(--text-muted); margin-right: 8px;';
  resultLabel.textContent = 'B(π) =';
  resultContainer.appendChild(resultLabel);

  'cat'.split('').forEach(ch => {
    const cell = document.createElement('div');
    cell.className = 'ctc-out-cell result';
    cell.textContent = ch;
    resultContainer.appendChild(cell);
  });

  // Trigger bar animation on scroll
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        barContainer.querySelectorAll('.decode-bar').forEach(bar => {
          bar.style.height = bar.style.height; // re-trigger
        });
      }
    });
  }, { threshold: 0.3 });
  observer.observe(barContainer);
}

/* ══════════════════════════════════════════════════════════════
   CTC LATTICE VISUALISATION
   ══════════════════════════════════════════════════════════════ */
function initLatticeViz() {
  const canvas = document.getElementById('lattice-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  // Modified label for "cat": l' = [ε, c, ε, a, ε, t, ε]
  const modLabel = ['ε', 'c', 'ε', 'a', 'ε', 't', 'ε'];
  const T = 8; // time steps
  const S = modLabel.length;

  const marginLeft = 80, marginTop = 50, marginRight = 40, marginBottom = 50;
  const gridW = W - marginLeft - marginRight;
  const gridH = H - marginTop - marginBottom;
  const cellW = gridW / T;
  const cellH = gridH / S;

  // Simulated alpha values (forward variable)
  const alpha = [];
  for (let t = 0; t < T; t++) {
    alpha[t] = [];
    for (let s = 0; s < S; s++) {
      // Simple simulation: probability decays away from the "diagonal"
      const diagS = (s / S) * T;
      const dist = Math.abs(t - diagS);
      alpha[t][s] = Math.max(0, Math.exp(-dist * 0.8) * (0.3 + Math.random() * 0.7));
    }
  }

  let hoveredCell = null;

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '600 12px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Time-steps τ →', marginLeft + gridW / 2, H - 10);
    ctx.save();
    ctx.translate(18, marginTop + gridH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Modified label l' ↑", 0, 0);
    ctx.restore();
    ctx.textAlign = 'left';

    // Time step labels
    ctx.font = '500 11px JetBrains Mono';
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.textAlign = 'center';
    for (let t = 0; t < T; t++) {
      ctx.fillText(`τ=${t+1}`, marginLeft + t * cellW + cellW / 2, marginTop - 10);
    }

    // Label labels
    ctx.textAlign = 'right';
    for (let s = 0; s < S; s++) {
      const y = marginTop + s * cellH + cellH / 2 + 4;
      ctx.fillStyle = modLabel[s] === 'ε' ? 'rgba(255,255,255,0.25)' : 'rgba(99,102,241,0.7)';
      ctx.fillText(modLabel[s], marginLeft - 12, y);
    }
    ctx.textAlign = 'left';

    // Draw cells
    for (let t = 0; t < T; t++) {
      for (let s = 0; s < S; s++) {
        const x = marginLeft + t * cellW;
        const y = marginTop + s * cellH;
        const val = alpha[t][s];

        // Cell background
        if (val > 0.01) {
          ctx.fillStyle = hexToRGBA('#6366f1', val * 0.6);
          roundRect(ctx, x + 2, y + 2, cellW - 4, cellH - 4, 4);
          ctx.fill();
        }

        // Cell border
        const isHovered = hoveredCell && hoveredCell.t === t && hoveredCell.s === s;
        ctx.strokeStyle = isHovered ? '#6366f1' : 'rgba(255,255,255,0.05)';
        ctx.lineWidth = isHovered ? 2 : 0.5;
        roundRect(ctx, x + 2, y + 2, cellW - 4, cellH - 4, 4);
        ctx.stroke();

        // Value text for significant cells
        if (val > 0.15) {
          ctx.fillStyle = val > 0.5 ? 'white' : 'rgba(255,255,255,0.6)';
          ctx.font = '500 9px JetBrains Mono';
          ctx.textAlign = 'center';
          ctx.fillText(val.toFixed(2), x + cellW / 2, y + cellH / 2 + 3);
          ctx.textAlign = 'left';
        }
      }
    }

    // Draw transition arrows for hovered cell
    if (hoveredCell) {
      const cx = marginLeft + hoveredCell.t * cellW + cellW / 2;
      const cy = marginTop + hoveredCell.s * cellH + cellH / 2;

      // Possible predecessors
      if (hoveredCell.t > 0) {
        // Same position
        drawSmallArrow(ctx, cx - cellW, cy, cx - cellW/2 + 4, cy, '#6366f1');
        // Position above
        if (hoveredCell.s > 0) {
          drawSmallArrow(ctx, cx - cellW, cy - cellH, cx - cellW/2 + 4, cy - cellH/2 + 4, '#a855f7');
        }
        // Two positions above (skip blank for non-blank)
        if (hoveredCell.s > 1 && modLabel[hoveredCell.s] !== 'ε' && modLabel[hoveredCell.s] !== modLabel[hoveredCell.s - 2]) {
          drawSmallArrow(ctx, cx - cellW, cy - 2 * cellH, cx - cellW/2 + 4, cy - cellH + 4, '#ec4899');
        }
      }
    }
  }

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const my = (e.clientY - rect.top) * (H / rect.height);
    const t = Math.floor((mx - marginLeft) / cellW);
    const s = Math.floor((my - marginTop) / cellH);
    if (t >= 0 && t < T && s >= 0 && s < S) {
      hoveredCell = { t, s };
      const descEl = document.getElementById('lattice-desc');
      if (descEl) {
        descEl.textContent = `α(τ=${t+1}, s=${s+1}) = ${alpha[t][s].toFixed(4)} — corresponds to label symbol "${modLabel[s]}" at time-step ${t+1}`;
      }
    } else {
      hoveredCell = null;
    }
    draw();
  });

  canvas.addEventListener('mouseleave', () => {
    hoveredCell = null;
    draw();
  });

  draw();
}

/* ══════════════════════════════════════════════════════════════
   UTILITY FUNCTIONS
   ══════════════════════════════════════════════════════════════ */
function hexToRGBA(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawArrow(ctx, x1, y1, x2, y2, color, width) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  // Arrowhead
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 10;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6), y2 - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6), y2 - headLen * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawSmallArrow(ctx, x1, y1, x2, y2, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 7;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6), y2 - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6), y2 - headLen * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawLine(ctx, x1, y1, x2, y2) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}
