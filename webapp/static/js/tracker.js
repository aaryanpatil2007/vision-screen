/**
 * Live eye tracking for the capture UI.
 *
 * Runs MediaPipe FaceLandmarker in the browser purely for REAL-TIME FEEDBACK:
 * the iris/lid overlay, quality chips, and gaze trace that tell the user their
 * capture is working. The authoritative analysis still happens server-side on
 * the uploaded video — this never feeds the report.
 */
import { FilesetResolver, FaceLandmarker } from "../vendor/vision_bundle.mjs";

export const EYE = {
  left:  { corners: [33, 133],  iris: 468, ring: [469, 470, 471, 472],
           lid: [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7] },
  right: { corners: [362, 263], iris: 473, ring: [474, 475, 476, 477],
           lid: [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382] },
};

export class EyeTracker {
  constructor(video, canvas) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.landmarker = null;
    this.running = false;
    this.last = null;
    this.gazeTrace = [];
    this.onFrame = null;
    this.stats = { fps: 0, faceOk: false, eyePx: 0, brightness: 0 };
    this._frames = 0;
    this._t0 = performance.now();
    this._lumaCanvas = document.createElement("canvas");
  }

  async init() {
    const fileset = await FilesetResolver.forVisionTasks("/static/vendor/wasm");
    this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "/static/models/face_landmarker.task", delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
    });
  }

  start() {
    if (this.running) return;
    this.running = true;
    const loop = () => {
      if (!this.running) return;
      this._tick();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  stop() { this.running = false; }

  _brightness() {
    const c = this._lumaCanvas;
    c.width = 48; c.height = 36;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(this.video, 0, 0, 48, 36);
    const d = g.getImageData(0, 0, 48, 36).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += (d[i] + d[i + 1] + d[i + 2]) / 3;
    return sum / (d.length / 4);
  }

  _tick() {
    const v = this.video;
    if (!this.landmarker || v.readyState < 2) return;
    const w = v.videoWidth, h = v.videoHeight;
    if (!w || !h) return;
    if (this.canvas.width !== w) { this.canvas.width = w; this.canvas.height = h; }

    const res = this.landmarker.detectForVideo(v, performance.now());
    const lm = res.faceLandmarks && res.faceLandmarks[0];

    this._frames++;
    const dt = performance.now() - this._t0;
    if (dt > 500) { this.stats.fps = Math.round(this._frames * 1000 / dt); this._frames = 0; this._t0 = performance.now(); }
    this.stats.brightness = this._brightness();
    this.stats.faceOk = !!lm;

    this.ctx.clearRect(0, 0, w, h);
    if (!lm) { this.stats.eyePx = 0; if (this.onFrame) this.onFrame(null, this.stats); return; }

    const px = (i) => [lm[i].x * w, lm[i].y * h];
    const a = px(33), b = px(263);
    this.stats.eyePx = Math.hypot(b[0] - a[0], b[1] - a[1]);

    const data = { left: null, right: null };
    for (const side of ["left", "right"]) {
      const spec = EYE[side];
      const c = px(spec.iris);
      const ring = spec.ring.map(px);
      const r = ring.reduce((s, p) => s + Math.hypot(p[0] - c[0], p[1] - c[1]), 0) / ring.length;
      const [c0, c1] = spec.corners.map(px);
      const lo = Math.min(c0[0], c1[0]), hi = Math.max(c0[0], c1[0]);
      data[side] = { center: c, radius: r, gaze: hi > lo ? (c[0] - lo) / (hi - lo) : 0.5,
                     lid: spec.lid.map(px) };
      this._drawEye(data[side]);
    }
    this._drawGazeTrace(data, w, h);
    this.last = data;
    if (this.onFrame) this.onFrame(data, this.stats);
  }

  _drawEye(eye) {
    const g = this.ctx;
    g.save();
    // lid contour
    g.beginPath();
    eye.lid.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
    g.closePath();
    g.strokeStyle = "rgba(78,163,255,0.55)";
    g.lineWidth = 1.4;
    g.stroke();
    // iris circle
    g.beginPath();
    g.arc(eye.center[0], eye.center[1], eye.radius, 0, Math.PI * 2);
    g.strokeStyle = "rgba(56,211,159,0.95)";
    g.lineWidth = 2;
    g.stroke();
    // pupil-center crosshair
    g.beginPath();
    g.moveTo(eye.center[0] - 6, eye.center[1]); g.lineTo(eye.center[0] + 6, eye.center[1]);
    g.moveTo(eye.center[0], eye.center[1] - 6); g.lineTo(eye.center[0], eye.center[1] + 6);
    g.strokeStyle = "rgba(255,255,255,0.9)";
    g.lineWidth = 1.2;
    g.stroke();
    g.restore();
  }

  _drawGazeTrace(data, w, h) {
    const mean = (data.left.gaze + data.right.gaze) / 2;
    this.gazeTrace.push(mean);
    if (this.gazeTrace.length > 120) this.gazeTrace.shift();
    const g = this.ctx;
    const bh = 26, y0 = h - bh - 6;
    g.save();
    g.strokeStyle = "rgba(78,163,255,0.85)";
    g.lineWidth = 1.5;
    g.beginPath();
    this.gazeTrace.forEach((v, i) => {
      const x = 6 + (i / 120) * (w - 12);
      const y = y0 + bh - Math.max(0, Math.min(1, v)) * bh;
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.stroke();
    g.restore();
  }
}
