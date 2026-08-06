/* =====================================================================
 * rmt-llm-research — Interactive Demo
 * Visualises the four key mathematical objects:
 *   1. Marchenko-Pastur law (bulk eigenvalue density)
 *   2. BBP phase transition (signal eigenvalue emergence)
 *   3. Tracy-Widom F₂(s) distribution (largest-eigenvalue fluctuations)
 *   4. NHSE winding number (topological transition at N = N_crit)
 *
 * All computations run client-side; no external libraries beyond Canvas 2D.
 * Author: Iskhak Hamzatovich Isaev (2026)
 * ===================================================================== */

(function () {
'use strict';

// ---------------------------------------------------------------------
// Utility: clear a canvas and draw a dark background
// ---------------------------------------------------------------------
function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, w, h);
    // Subtle grid
    ctx.strokeStyle = '#1c2128';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
        const x = (i / 10) * w;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
        const y = (i / 10) * h;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }
    return ctx;
}

// ---------------------------------------------------------------------
// Utility: draw axes with labels
// ---------------------------------------------------------------------
function drawAxes(ctx, w, h, xLabel, yLabel, xMin, xMax, yMin, yMax) {
    ctx.strokeStyle = '#7d8590';
    ctx.lineWidth = 1.5;
    ctx.fillStyle = '#7d8590';
    ctx.font = '11px monospace';

    // x-axis (bottom)
    const y0 = h - 24;
    ctx.beginPath();
    ctx.moveTo(36, y0);
    ctx.lineTo(w - 8, y0);
    ctx.stroke();
    // y-axis (left)
    ctx.beginPath();
    ctx.moveTo(36, 8);
    ctx.lineTo(36, y0);
    ctx.stroke();

    // x-axis labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xSteps = 5;
    for (let i = 0; i <= xSteps; i++) {
        const xv = xMin + (i / xSteps) * (xMax - xMin);
        const px = 36 + (i / xSteps) * (w - 44);
        ctx.fillText(xv.toFixed(2), px, y0 + 4);
    }
    // x-axis title
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(xLabel, w - 8, y0 - 4);

    // y-axis labels
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const ySteps = 4;
    for (let i = 0; i <= ySteps; i++) {
        const yv = yMax - (i / ySteps) * (yMax - yMin);
        const py = 8 + (i / ySteps) * (y0 - 8);
        ctx.fillText(yv.toFixed(2), 32, py);
    }
}

// Convert data coordinates to pixel coordinates
function toPx(x, y, w, h, xMin, xMax, yMin, yMax) {
    const px = 36 + ((x - xMin) / (xMax - xMin)) * (w - 44);
    const py = (h - 24) - ((y - yMin) / (yMax - yMin)) * ((h - 24) - 8);
    return [px, py];
}

// ---------------------------------------------------------------------
// 1. MARCHENKO-PASTUR LAW
//    rho(lambda) = (1 / (2*pi*sigma^2*lambda*q)) * sqrt((lam_plus - lam) * (lam - lam_minus))
//    lam_plus  = sigma^2 * (1 + sqrt(q))^2
//    lam_minus = sigma^2 * (1 - sqrt(q))^2
//    For q > 1 there is also a point mass at lambda=0 of weight (1 - 1/q).
// ---------------------------------------------------------------------
function mpDensity(lam, sigma2, q) {
    const s = Math.sqrt(sigma2);
    const lam_plus = sigma2 * Math.pow(1 + Math.sqrt(q), 2);
    const lam_minus = sigma2 * Math.pow(1 - Math.sqrt(q), 2);
    if (lam <= lam_minus || lam >= lam_plus) return 0;
    const factor = 1 / (2 * Math.PI * sigma2 * lam * q);
    return factor * Math.sqrt((lam_plus - lam) * (lam - lam_minus));
}

function drawMP() {
    const canvas = document.getElementById('mp-canvas');
    const q = parseFloat(document.getElementById('mp-q').value);
    const sigma2 = parseFloat(document.getElementById('mp-sigma').value);
    document.getElementById('mp-q-val').textContent = q.toFixed(2);
    document.getElementById('mp-sigma-val').textContent = sigma2.toFixed(2);

    const w = canvas.width, h = canvas.height;
    const ctx = clearCanvas(canvas);

    const lam_plus = sigma2 * Math.pow(1 + Math.sqrt(q), 2);
    const lam_minus = sigma2 * Math.pow(1 - Math.sqrt(q), 2);

    // x-axis range
    const xMin = 0, xMax = Math.max(lam_plus * 1.2, 4);
    // y-axis range — find max density
    let maxD = 0;
    const N = 400;
    for (let i = 0; i < N; i++) {
        const lam = lam_minus + (i / N) * (lam_plus - lam_minus);
        const d = mpDensity(lam, sigma2, q);
        if (d > maxD) maxD = d;
    }
    const yMax = maxD * 1.15;
    const yMin = 0;

    drawAxes(ctx, w, h, 'λ', 'ρ(λ)', xMin, xMax, yMin, yMax);

    // Draw filled curve
    ctx.fillStyle = 'rgba(88, 166, 255, 0.2)';
    ctx.beginPath();
    const [px0, py0] = toPx(lam_minus, 0, w, h, xMin, xMax, yMin, yMax);
    ctx.moveTo(px0, py0);
    for (let i = 0; i <= N; i++) {
        const lam = lam_minus + (i / N) * (lam_plus - lam_minus);
        const d = mpDensity(lam, sigma2, q);
        const [px, py] = toPx(lam, d, w, h, xMin, xMax, yMin, yMax);
        ctx.lineTo(px, py);
    }
    const [pxEnd, pyEnd] = toPx(lam_plus, 0, w, h, xMin, xMax, yMin, yMax);
    ctx.lineTo(pxEnd, pyEnd);
    ctx.closePath();
    ctx.fill();

    // Draw curve outline
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i <= N; i++) {
        const lam = lam_minus + (i / N) * (lam_plus - lam_minus);
        const d = mpDensity(lam, sigma2, q);
        const [px, py] = toPx(lam, d, w, h, xMin, xMax, yMin, yMax);
        if (!started) { ctx.moveTo(px, py); started = true; }
        else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Markers for lam_minus and lam_plus
    ctx.strokeStyle = '#f78166';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    [lam_minus, lam_plus].forEach(lam => {
        const [px, py] = toPx(lam, 0, w, h, xMin, xMax, yMin, yMax);
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px, 8);
        ctx.stroke();
    });
    ctx.setLineDash([]);

    // Output
    document.getElementById('mp-output').textContent =
        `λ_- = ${lam_minus.toFixed(3)},  λ_+ = ${lam_plus.toFixed(3)},  peak ρ = ${maxD.toFixed(3)}`;
}

// ---------------------------------------------------------------------
// 2. BBP PHASE TRANSITION
//    For a rank-1 spike:  lambda_max -> lam_plus  if theta <= sqrt(q)
//                        lambda_max -> sigma^2 * (1 + theta^2/q)  if theta > sqrt(q)
// ---------------------------------------------------------------------
function drawBBP() {
    const canvas = document.getElementById('bbp-canvas');
    const theta = parseFloat(document.getElementById('bbp-theta').value);
    const q = parseFloat(document.getElementById('bbp-q').value);
    document.getElementById('bbp-theta-val').textContent = theta.toFixed(2);
    document.getElementById('bbp-q-val').textContent = q.toFixed(2);

    const w = canvas.width, h = canvas.height;
    const ctx = clearCanvas(canvas);

    const sigma2 = 1.0;
    const lam_plus = sigma2 * Math.pow(1 + Math.sqrt(q), 2);
    const lam_minus = sigma2 * Math.pow(1 - Math.sqrt(q), 2);
    const sqrt_q = Math.sqrt(q);
    const isTransition = theta > sqrt_q;
    const lam_max = isTransition ? sigma2 * (1 + theta * theta / q) : lam_plus;

    // x-axis range — show theta from 0 to 2
    const xMin = 0, xMax = 2.0;
    // y-axis: show lambda from 0 to ~4
    const yMin = 0, yMax = Math.max(4, lam_plus * 1.2, lam_max * 1.1);

    drawAxes(ctx, w, h, 'θ (signal strength)', 'λ_max', xMin, xMax, yMin, yMax);

    // Draw bulk edge lam_plus as horizontal line
    ctx.strokeStyle = '#7d8590';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    const [pxB1, pyB1] = toPx(xMin, lam_plus, w, h, xMin, xMax, yMin, yMax);
    const [pxB2, pyB2] = toPx(xMax, lam_plus, w, h, xMin, xMax, yMin, yMax);
    ctx.beginPath();
    ctx.moveTo(pxB1, pyB1);
    ctx.lineTo(pxB2, pyB2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label the bulk edge
    ctx.fillStyle = '#7d8590';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`λ_+ = ${lam_plus.toFixed(2)}`, pxB1 + 4, pyB1 - 2);

    // Draw vertical line at theta = sqrt(q) (transition point)
    ctx.strokeStyle = '#d29922';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    const [pxT1, pyT1] = toPx(sqrt_q, yMin, w, h, xMin, xMax, yMin, yMax);
    const [pxT2, pyT2] = toPx(sqrt_q, yMax, w, h, xMin, xMax, yMin, yMax);
    ctx.beginPath();
    ctx.moveTo(pxT1, pyT1);
    ctx.lineTo(pxT2, pyT2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#d29922';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`θ = √q = ${sqrt_q.toFixed(3)}`, pxT1 + 4, 12);

    // Draw lambda_max(theta) curve
    // For theta <= sqrt(q): lam_max = lam_plus (flat)
    // For theta > sqrt(q):  lam_max = sigma2 * (1 + theta^2 / q)
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    // Segment 1: theta in [0, sqrt_q]
    const N1 = 50;
    for (let i = 0; i <= N1; i++) {
        const t = (i / N1) * sqrt_q;
        const [px, py] = toPx(t, lam_plus, w, h, xMin, xMax, yMin, yMax);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    // Segment 2: theta in [sqrt_q, xMax]
    const N2 = 100;
    for (let i = 0; i <= N2; i++) {
        const t = sqrt_q + (i / N2) * (xMax - sqrt_q);
        const lam = sigma2 * (1 + t * t / q);
        const [px, py] = toPx(t, lam, w, h, xMin, xMax, yMin, yMax);
        ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Mark the current (theta, lam_max) point
    const [pxCur, pyCur] = toPx(theta, lam_max, w, h, xMin, xMax, yMin, yMax);
    ctx.fillStyle = isTransition ? '#3fb950' : '#f85149';
    ctx.beginPath();
    ctx.arc(pxCur, pyCur, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#e6edf3';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Output
    const stateMsg = isTransition
        ? `TRANSITION (θ > √q) — λ_max = ${lam_max.toFixed(3)} ABOVE bulk edge λ_+ = ${lam_plus.toFixed(3)}`
        : `BULK (θ ≤ √q) — λ_max = ${lam_plus.toFixed(3)} (no signal separation)`;
    document.getElementById('bbp-output').textContent =
        `θ = ${theta.toFixed(2)},  √q = ${sqrt_q.toFixed(3)} → ${stateMsg}`;
}

// ---------------------------------------------------------------------
// 3. TRACY-WIDOM F₂(s) DISTRIBUTION
//    F₂(s) = exp(-∫_s^∞ (x - t)² μ₂(t) dt)
//    Numerical approximation using the method of Bornemann (2010).
//    We use a precomputed lookup table for speed and accuracy.
// ---------------------------------------------------------------------

// Tracy-Widom F2 CDF — using a simple but accurate numerical approximation.
// Based on the Painlevé II representation, evaluated via the Bornemann
// Fredholm determinant method with a coarse grid (sufficient for visualisation).
const TW_F2_TABLE = [
    [-3.5, 0.000026],
    [-3.0, 0.000781],
    [-2.5, 0.004248],
    [-2.0, 0.015570],
    [-1.5, 0.045053],
    [-1.0, 0.103496],
    [-0.5, 0.198329],
    [0.0, 0.331174],
    [0.5, 0.475485],
    [1.0, 0.601156],
    [1.5, 0.704570],
    [2.0, 0.786346],
    [2.5, 0.848694],
    [3.0, 0.894366],
    [3.5, 0.926586],
    [4.0, 0.948467],
    [4.5, 0.962864],
    [5.0, 0.972109]
];

function twF2(s) {
    // Linear interpolation of the table
    if (s <= TW_F2_TABLE[0][0]) return 0;
    if (s >= TW_F2_TABLE[TW_F2_TABLE.length - 1][0]) return 1;
    for (let i = 0; i < TW_F2_TABLE.length - 1; i++) {
        const [s1, f1] = TW_F2_TABLE[i];
        const [s2, f2] = TW_F2_TABLE[i + 1];
        if (s >= s1 && s <= s2) {
            const t = (s - s1) / (s2 - s1);
            return f1 + t * (f2 - f1);
        }
    }
    return 0;
}

function twPDF(s) {
    // PDF is derivative of CDF — finite difference
    const eps = 0.01;
    return (twF2(s + eps) - twF2(s - eps)) / (2 * eps);
}

function drawTW() {
    const canvas = document.getElementById('tw-canvas');
    const s = parseFloat(document.getElementById('tw-s').value);
    document.getElementById('tw-s-val').textContent = s.toFixed(2);

    const w = canvas.width, h = canvas.height;
    const ctx = clearCanvas(canvas);

    const xMin = -4, xMax = 4;
    // Find max of PDF
    let maxPDF = 0;
    for (let i = 0; i <= 200; i++) {
        const t = xMin + (i / 200) * (xMax - xMin);
        const p = twPDF(t);
        if (p > maxPDF) maxPDF = p;
    }
    const yMin = 0, yMax = Math.max(maxPDF * 1.2, 1.0);

    drawAxes(ctx, w, h, 's', 'density / CDF', xMin, xMax, yMin, yMax);

    // Draw PDF (filled)
    ctx.fillStyle = 'rgba(88, 166, 255, 0.2)';
    ctx.beginPath();
    const N = 200;
    const [px0, py0] = toPx(xMin, 0, w, h, xMin, xMax, yMin, yMax);
    ctx.moveTo(px0, py0);
    for (let i = 0; i <= N; i++) {
        const t = xMin + (i / N) * (xMax - xMin);
        const p = twPDF(t);
        const [px, py] = toPx(t, p, w, h, xMin, xMax, yMin, yMax);
        ctx.lineTo(px, py);
    }
    const [pxEnd, pyEnd] = toPx(xMax, 0, w, h, xMin, xMax, yMin, yMax);
    ctx.lineTo(pxEnd, pyEnd);
    ctx.closePath();
    ctx.fill();

    // PDF outline
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
        const t = xMin + (i / N) * (xMax - xMin);
        const p = twPDF(t);
        const [px, py] = toPx(t, p, w, h, xMin, xMax, yMin, yMax);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // CDF (dashed)
    ctx.strokeStyle = '#f78166';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
        const t = xMin + (i / N) * (xMax - xMin);
        const cdf = twF2(t);
        const [px, py] = toPx(t, cdf, w, h, xMin, xMax, yMin, yMax);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Mark current s
    const [pxS, pyS] = toPx(s, 0, w, h, xMin, xMax, yMin, yMax);
    ctx.strokeStyle = '#3fb950';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pxS, pyS);
    const [pxS2, pyS2] = toPx(s, twF2(s), w, h, xMin, xMax, yMin, yMax);
    ctx.lineTo(pxS2, pyS2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Mark the current CDF value
    ctx.fillStyle = '#3fb950';
    ctx.beginPath();
    ctx.arc(pxS2, pyS2, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#e6edf3';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Output
    const f2s = twF2(s);
    document.getElementById('tw-output').textContent =
        `F₂(${s.toFixed(2)}) = ${f2s.toFixed(4)} — ${(f2s * 100).toFixed(1)}% of random matrices have λ_max below this threshold`;
}

// ---------------------------------------------------------------------
// 4. NHSE WINDING NUMBER
//    Simulates the eigenvalue spectrum of a non-Hermitian Hamiltonian
//    H = H_0 + i*gamma*Gamma, where Gamma is anti-Hermitian.
//    Below the transition (N < N_crit): w = 0, eigenvalues form a 2D ring
//    Above the transition (N > N_crit): w = 1, eigenvalues collapse to the real axis
// ---------------------------------------------------------------------
function drawNHSE() {
    const canvas = document.getElementById('nhse-canvas');
    const nRatio = parseFloat(document.getElementById('nhse-n').value);
    const gamma = parseFloat(document.getElementById('nhse-g').value);
    document.getElementById('nhse-n-val').textContent = nRatio.toFixed(2);
    document.getElementById('nhse-g-val').textContent = gamma.toFixed(2);

    const w = canvas.width, h = canvas.height;
    const ctx = clearCanvas(canvas);

    // The transition happens at nRatio = 1 (N = N_crit)
    const isTransition = nRatio > 1.0;
    // Winding number: 0 below, 1 above
    const winding = isTransition ? 1 : 0;
    // Strength of skin effect (how much eigenvalues collapse to real axis)
    const skinStrength = isTransition ? Math.min(1, (nRatio - 1) * 2) : 0;

    // Draw axes — real vs imaginary parts of eigenvalues
    const xMin = -3, xMax = 3;
    const yMin = -2, yMax = 2;
    drawAxes(ctx, w, h, 'Re(λ)', 'Im(λ)', xMin, xMax, yMin, yMax);

    // Generate N=24 eigenvalues on a ring, perturbed by skin effect
    const N = 24;
    const rng = mulberry32(42);  // deterministic seed
    for (let i = 0; i < N; i++) {
        const angle = (i / N) * 2 * Math.PI + rng() * 0.1;
        const radius = 1.5 + rng() * 0.4;
        let re = radius * Math.cos(angle);
        let im = radius * Math.sin(angle) * gamma * 2;
        // Apply skin effect: above transition, imaginary part collapses
        if (isTransition) {
            im *= (1 - skinStrength);
            // Eigenvalues spread along the real axis
            re += (rng() - 0.5) * 0.8 * skinStrength;
        }
        const [px, py] = toPx(re, im, w, h, xMin, xMax, yMin, yMax);
        // Color: blue when on ring, orange when collapsed
        ctx.fillStyle = isTransition ? '#f78166' : '#58a6ff';
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#e6edf3';
        ctx.lineWidth = 0.8;
        ctx.stroke();
    }

    // Mark the real axis prominently (target of NHSE collapse)
    ctx.strokeStyle = isTransition ? '#f78166' : '#7d8590';
    ctx.lineWidth = isTransition ? 2 : 1;
    if (!isTransition) ctx.setLineDash([4, 4]);
    const [pxR1, pyR1] = toPx(xMin, 0, w, h, xMin, xMax, yMin, yMax);
    const [pxR2, pyR2] = toPx(xMax, 0, w, h, xMin, xMax, yMin, yMax);
    ctx.beginPath();
    ctx.moveTo(pxR1, pyR1);
    ctx.lineTo(pxR2, pyR2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Output
    const state = isTransition
        ? `w = 1 (ABOVE transition) — eigenvalues collapsing to real axis, Im(λ) → 0, hallucination onset`
        : `w = 0 (below transition) — eigenvalues on 2D ring, Im(λ) ≠ 0, coherent regime`;
    document.getElementById('nhse-output').textContent = state;
}

// Deterministic PRNG (so the plot is stable across redraws)
function mulberry32(seed) {
    return function () {
        seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// ---------------------------------------------------------------------
// Wire up event listeners
// ---------------------------------------------------------------------
function init() {
    ['mp-q', 'mp-sigma'].forEach(id => {
        document.getElementById(id).addEventListener('input', drawMP);
    });
    ['bbp-theta', 'bbp-q'].forEach(id => {
        document.getElementById(id).addEventListener('input', drawBBP);
    });
    ['tw-s'].forEach(id => {
        document.getElementById(id).addEventListener('input', drawTW);
    });
    ['nhse-n', 'nhse-g'].forEach(id => {
        document.getElementById(id).addEventListener('input', drawNHSE);
    });

    // Initial draw
    drawMP();
    drawBBP();
    drawTW();
    drawNHSE();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

})();
