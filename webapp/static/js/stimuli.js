/**
 * Stimulus generators — the psychophysics rendering layer.
 * Each function draws one clinically-specified stimulus onto the stage.
 */

export const DIRS = ["up", "right", "down", "left"];

/** logMAR optotype: an "E" subtending 5x10^L arcmin at the viewing distance. */
export function letterHeightPx(logmar, distanceCm, pxPerCm) {
  const arcmin = 5.0 * Math.pow(10, logmar);
  const heightCm = 2 * distanceCm * Math.tan((arcmin / 60 / 2) * Math.PI / 180);
  return heightCm * pxPerCm;
}

/** Tumbling-E drawn on canvas: 5x5 grid, bar thickness = 1/5 of height. */
export function drawTumblingE(ctx, cx, cy, sizePx, dir, fg = "#000", bg = null) {
  const u = sizePx / 5;
  ctx.save();
  if (bg) { ctx.fillStyle = bg; ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height); }
  ctx.translate(cx, cy);
  const rot = { right: 0, down: 90, left: 180, up: 270 }[dir];
  ctx.rotate((rot * Math.PI) / 180);
  ctx.fillStyle = fg;
  ctx.fillRect(-2.5 * u, -2.5 * u, u, 5 * u);       // spine
  ctx.fillRect(-2.5 * u, -2.5 * u, 5 * u, u);       // top bar
  ctx.fillRect(-2.5 * u, -0.5 * u, 5 * u, u);       // middle bar
  ctx.fillRect(-2.5 * u, 1.5 * u, 5 * u, u);        // bottom bar
  ctx.restore();
}

/** Sloan-style letter set used for contrast triplets. */
export const SLOAN = ["C", "D", "H", "K", "N", "O", "R", "S", "V", "Z"];

/** sRGB code value for a Weber contrast against white, gamma-corrected. */
export function contrastToGray(logCS, background = 255, gamma = 2.2) {
  const weber = Math.pow(10, -logCS);
  const bgLin = Math.pow(background / 255, gamma);
  const fgLin = Math.max(0, bgLin * (1 - weber));
  return Math.round(255 * Math.pow(fgLin, 1 / gamma));
}

/** Astigmatic sunburst dial: n spokes over 180 degrees, plus a fixation ring. */
export function drawAstigmaticDial(ctx, cx, cy, radius, nSpokes = 12) {
  ctx.save();
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.strokeStyle = "#000";
  ctx.lineWidth = Math.max(2, radius * 0.012);
  for (let i = 0; i < nSpokes; i++) {
    const a = (i * Math.PI) / nSpokes;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * radius * 0.22, cy + Math.sin(a) * radius * 0.22);
    ctx.lineTo(cx + Math.cos(a) * radius, cy + Math.sin(a) * radius);
    ctx.moveTo(cx - Math.cos(a) * radius * 0.22, cy - Math.sin(a) * radius * 0.22);
    ctx.lineTo(cx - Math.cos(a) * radius, cy - Math.sin(a) * radius);
    ctx.stroke();
  }
  // clock numerals give the patient a vocabulary for reporting
  ctx.fillStyle = "#000";
  ctx.font = `${Math.round(radius * 0.1)}px system-ui`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (let h = 1; h <= 12; h++) {
    const a = ((h * 30) - 90) * Math.PI / 180;
    ctx.fillText(String(h), cx + Math.cos(a) * radius * 1.1, cy + Math.sin(a) * radius * 1.1);
  }
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 0.16, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

/** Amsler grid: N x N squares with central fixation dot. */
export function drawAmsler(ctx, cx, cy, sidePx, squares = 20) {
  const half = sidePx / 2, step = sidePx / squares;
  ctx.save();
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= squares; i++) {
    const o = -half + i * step;
    ctx.moveTo(cx + o, cy - half); ctx.lineTo(cx + o, cy + half);
    ctx.moveTo(cx - half, cy + o); ctx.lineTo(cx + half, cy + o);
  }
  ctx.stroke();
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(cx, cy, Math.max(3, sidePx * 0.008), 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/**
 * Pseudoisochromatic plate: numeral rendered in dots whose color differs from
 * the background only along a protan/deutan confusion direction, with dot
 * lightness randomized so the figure is not visible as a luminance edge.
 */
export function drawColorPlate(ctx, cx, cy, radius, digit, type, rng) {
  const PALETTES = {
    protan:  { fig: [[142, 118, 60], [156, 128, 72], [130, 108, 52]],
               bg:  [[126, 128, 62], [140, 140, 74], [116, 118, 54]] },
    deutan:  { fig: [[168, 108, 78], [180, 120, 88], [156, 98, 70]],
               bg:  [[150, 122, 76], [162, 134, 86], [138, 112, 68]] },
    general: { fig: [[196, 96, 84], [208, 108, 94], [184, 86, 76]],
               bg:  [[122, 152, 96], [134, 164, 106], [110, 140, 88]] },
    demo:    { fig: [[200, 90, 80], [212, 102, 90], [188, 80, 72]],
               bg:  [[140, 140, 140], [152, 152, 152], [128, 128, 128]] },
  };
  const pal = PALETTES[type] || PALETTES.general;

  // A pseudoisochromatic plate only works if the figure is invisible to the
  // LUMINANCE channel — otherwise a color-deficient viewer reads it as a
  // brightness edge and the plate tests nothing. Scale each figure colour to
  // match the mean background luma (Rec.601), then randomize lightness per dot
  // so no residual edge survives.
  const luma = (c) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
  const bgLuma = pal.bg.reduce((s, c) => s + luma(c), 0) / pal.bg.length;
  const matched = pal.fig.map((c) => {
    const k = bgLuma / Math.max(luma(c), 1);
    return c.map((v) => Math.max(0, Math.min(255, Math.round(v * k))));
  });
  const palette = { fig: matched, bg: pal.bg };

  // rasterize the digit into an offscreen mask
  const off = document.createElement("canvas");
  off.width = off.height = radius * 2;
  const og = off.getContext("2d");
  og.fillStyle = "#000"; og.fillRect(0, 0, off.width, off.height);
  og.fillStyle = "#fff";
  og.font = `bold ${Math.round(radius * 1.15)}px system-ui`;
  og.textAlign = "center"; og.textBaseline = "middle";
  og.fillText(String(digit), radius, radius);
  const mask = og.getImageData(0, 0, off.width, off.height).data;

  ctx.save();
  ctx.fillStyle = "#f2efe6";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const dots = Math.round(radius * radius * 0.055);
  for (let i = 0; i < dots; i++) {
    const a = rng() * Math.PI * 2;
    const rr = radius * Math.sqrt(rng());
    const x = Math.cos(a) * rr, y = Math.sin(a) * rr;
    const mx = Math.round(x + radius), my = Math.round(y + radius);
    const inFigure = mask[(my * off.width + mx) * 4] > 128;
    const set = inFigure ? palette.fig : palette.bg;
    const base = set[Math.floor(rng() * set.length)];
    // per-dot lightness jitter, applied identically to figure and background
    const jitter = 0.86 + rng() * 0.28;
    const c = base.map((v) => Math.max(0, Math.min(255, Math.round(v * jitter))));
    const dr = radius * (0.018 + rng() * 0.028);
    ctx.beginPath();
    ctx.arc(cx + x, cy + y, dr, 0, Math.PI * 2);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fill();
  }
  ctx.restore();
}

/** Deterministic PRNG so plates are reproducible per session. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Dynamic random-dot stereogram, red/cyan anaglyph.
 *
 * Dots are re-randomized every frame so no monocular pattern persists, and the
 * disparate region is a shape whose position is the 4AFC answer. A catch trial
 * (disparityPx = 0) contains no depth information at all: a subject who
 * "sees" the shape anyway is reading an artifact, and the scorer voids the run.
 *
 * Anaglyph note: the red channel carries ~21% of white luminance and cyan ~79%,
 * so the two eyes get very different light. Both half-images are therefore
 * drawn at matched luminance rather than full-intensity red/cyan.
 */
export function drawRDS(ctx, opts) {
  const { size, disparityPx, quadrant, dotPx = 3, density = 0.35, rng } = opts;
  const g = ctx;
  g.fillStyle = "#000";
  g.fillRect(0, 0, size, size);
  g.globalCompositeOperation = "lighter";

  const half = size / 2;
  const region = { 0: [0, 0], 1: [half, 0], 2: [0, half], 3: [half, half] }[quadrant];
  const inRegion = (x, y) =>
    x >= region[0] && x < region[0] + half && y >= region[1] && y < region[1] + half;

  const n = Math.round((size * size * density) / (dotPx * dotPx));
  const shift = disparityPx / 2;
  for (let i = 0; i < n; i++) {
    const x = rng() * size, y = rng() * size;
    const d = inRegion(x, y) ? shift : 0;
    // luminance-matched half-images: dim cyan against brighter red
    g.fillStyle = "rgb(210,0,0)";
    g.fillRect(Math.round(x - d), Math.round(y), dotPx, dotPx);
    g.fillStyle = "rgb(0,57,57)";
    g.fillRect(Math.round(x + d), Math.round(y), dotPx, dotPx);
  }
  g.globalCompositeOperation = "source-over";
}

/** Screen disparity in arcsec: 206265 * d / D. Matches the server module. */
export function disparityArcsec(pixelDisparityMm, distanceMm) {
  return distanceMm > 0 ? (206265 * pixelDisparityMm) / distanceMm : 0;
}

/**
 * Worth four-dot target: one red (top), two green (sides), one white (bottom).
 * Through red/cyan glasses the red dot reaches only the red-filtered eye, the
 * green dots only the other, and the white dot both — so the reported count
 * and colours separate fusion, suppression, and diplopia.
 */
export function drawWorthDots(ctx, cx, cy, radius) {
  const g = ctx;
  g.fillStyle = "#000";
  g.fillRect(0, 0, g.canvas.width, g.canvas.height);
  const r = Math.max(6, radius * 0.16);
  const dot = (x, y, color) => {
    g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2);
    g.fillStyle = color; g.fill();
  };
  dot(cx, cy - radius, "#d40000");            // red, top
  dot(cx - radius, cy, "#00c853");            // green, left
  dot(cx + radius, cy, "#00c853");            // green, right
  dot(cx, cy + radius, "#ffffff");            // white, bottom (both eyes)
}

/** Finest logMAR this display can draw (1-px stroke on a 5x5 optotype). */
export function renderableFloorLogmar(distanceCm, pxPerCm, minStrokePx = 1.0) {
  if (pxPerCm <= 0 || distanceCm <= 0) return Infinity;
  const targetHeightCm = (5 * minStrokePx) / pxPerCm;
  const arcmin = 2 * Math.atan(targetHeightCm / (2 * distanceCm)) * (180 / Math.PI) * 60;
  return Math.log10(arcmin / 5);
}

/** Faintest logCS an 8-bit sRGB display can present against white. */
export function displayCeilingLogCS(background = 255, bits = 8, gamma = 2.2) {
  const maxCode = (1 << bits) - 1;
  const bgLin = Math.pow(background / maxCode, gamma);
  const fgLin = Math.pow((background - 1) / maxCode, gamma);
  return -Math.log10((bgLin - fgLin) / bgLin);
}
