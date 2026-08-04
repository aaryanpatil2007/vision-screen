/**
 * VisionScreen capture app.
 *
 * Runs the guided battery, records the session video, logs every stimulus
 * event with timestamps, and posts both to the server for analysis. The
 * client renders stimuli and collects responses; it never scores anything —
 * scoring lives server-side so the report has a single source of truth.
 */
import { EyeTracker } from "./tracker.js";
import {
  DIRS, SLOAN, letterHeightPx, drawTumblingE, contrastToGray,
  drawAstigmaticDial, drawAmsler, drawColorPlate, mulberry32,
} from "./stimuli.js";

const PLATES = [
  { id: "p0", digit: 12, type: "demo" },
  { id: "p1", digit: 8, type: "general" },
  { id: "p2", digit: 6, type: "protan" },
  { id: "p3", digit: 5, type: "deutan" },
  { id: "p4", digit: 29, type: "general" },
  { id: "p5", digit: 3, type: "protan" },
  { id: "p6", digit: 15, type: "deutan" },
  { id: "p7", digit: 74, type: "general" },
  { id: "p8", digit: 2, type: "protan" },
  { id: "p9", digit: 7, type: "deutan" },
];

const STEPS = [
  { id: "setup", label: "Setup" },
  { id: "acuity_both", label: "Acuity" },
  { id: "acuity_right", label: "Right eye" },
  { id: "acuity_left", label: "Left eye" },
  { id: "contrast", label: "Contrast" },
  { id: "astigmatism", label: "Astigmatism" },
  { id: "color_vision", label: "Color" },
  { id: "amsler", label: "Amsler" },
  { id: "motility", label: "Tracking" },
  { id: "pupil", label: "Pupils" },
  { id: "photoref", label: "Refraction" },
];

const $ = (s) => document.querySelector(s);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class Session {
  constructor() {
    this.segments = [];
    this.t0 = 0;
    this.distanceCm = 50;
    this.pxPerCm = 96 / 2.54;   // CSS px; refined by the card calibration step
  }
  now() { return (performance.now() - this.t0) / 1000; }
  open(testId) {
    const seg = { test_id: testId, start_ts: this.now(), end_ts: null, events: [] };
    this.segments.push(seg);
    return seg;
  }
  log(seg, kind, payload) {
    seg.events.push({ ts: this.now(), kind, payload });
  }
  close(seg) { seg.end_ts = this.now(); }
  toJSON(fps) {
    return JSON.stringify({
      session_id: crypto.randomUUID(),
      px_per_cm: this.pxPerCm,
      distance_cm: this.distanceCm,
      fps,
      segments: this.segments.map((s) => ({ ...s, end_ts: s.end_ts ?? this.now() })),
    });
  }
}

class App {
  constructor() {
    this.session = new Session();
    this.tracker = null;
    this.stream = null;
    this.recorder = null;
    this.chunks = [];
    this.stepIndex = 0;
    this.pxPerCm = 96 / 2.54;
    this.rng = mulberry32(20260804);
  }

  // ---------- infrastructure ----------

  async initCamera() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      audio: false,
    });
    const video = $("#cam");
    video.srcObject = this.stream;
    await video.play();
    this.tracker = new EyeTracker(video, $("#overlay"));
    await this.tracker.init();
    this.tracker.onFrame = (eyes, stats) => this.renderStats(eyes, stats);
    this.tracker.start();
  }

  renderStats(eyes, stats) {
    const set = (sel, txt, cls) => {
      const el = $(sel);
      if (!el) return;
      el.textContent = txt;
      el.className = "chip" + (cls ? " " + cls : "");
    };
    set("#chipFace", stats.faceOk ? "face ok" : "no face", stats.faceOk ? "ok" : "bad");
    set("#chipEye", `${Math.round(stats.eyePx)}px`, stats.eyePx >= 90 ? "ok" : "bad");
    set("#chipLight", `lum ${Math.round(stats.brightness)}`,
        stats.brightness > 45 && stats.brightness < 215 ? "ok" : "bad");
    set("#chipFps", `${stats.fps}fps`, stats.fps >= 12 ? "ok" : "");
    if (eyes && $("#mGaze")) {
      $("#mGaze").textContent = ((eyes.left.gaze + eyes.right.gaze) / 2).toFixed(2);
      $("#mIris").textContent = `${eyes.left.radius.toFixed(1)}px`;
      $("#mSym").textContent = Math.abs(eyes.left.gaze - eyes.right.gaze).toFixed(3);
    }
  }

  startRecording() {
    const opts = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? { mimeType: "video/webm;codecs=vp9", videoBitsPerSecond: 2_500_000 }
      : {};
    this.recorder = new MediaRecorder(this.stream, opts);
    this.recorder.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.recorder.start(1000);
    this.session.t0 = performance.now();
  }

  setStep(id) {
    this.stepIndex = STEPS.findIndex((s) => s.id === id);
    const rail = $("#rail"), labels = $("#railLabels");
    rail.innerHTML = ""; labels.innerHTML = "";
    STEPS.forEach((s, i) => {
      const d = document.createElement("div");
      d.className = "step" + (i < this.stepIndex ? " done" : i === this.stepIndex ? " active" : "");
      rail.appendChild(d);
      const sp = document.createElement("span");
      sp.textContent = s.label;
      if (i === this.stepIndex) sp.className = "active";
      labels.appendChild(sp);
    });
  }

  stage(html, { white = false, blackout = false } = {}) {
    const st = $("#stage");
    st.className = "stage on" + (white ? " white" : "");
    document.body.classList.toggle("blackout", blackout);
    st.innerHTML = html;
    return st;
  }
  hideStage() {
    $("#stage").className = "stage";
    document.body.classList.remove("blackout");
  }

  /** Wait for one of `keys`; returns the key. Also accepts on-screen buttons. */
  waitKey(keys) {
    return new Promise((resolve) => {
      const onKey = (e) => {
        const k = e.key.replace("Arrow", "").toLowerCase();
        if (keys.includes(k)) { cleanup(); resolve(k); }
      };
      const onClick = (e) => {
        const v = e.target.closest("[data-key]")?.dataset.key;
        if (v && keys.includes(v)) { cleanup(); resolve(v); }
      };
      const cleanup = () => {
        document.removeEventListener("keydown", onKey);
        document.removeEventListener("click", onClick);
      };
      document.addEventListener("keydown", onKey);
      document.addEventListener("click", onClick);
    });
  }

  // ---------- tests ----------

  async runAcuity(eyeLabel, testId) {
    this.setStep(testId);
    const cover = eyeLabel === "both" ? "Keep both eyes open."
      : `Cover your ${eyeLabel === "right" ? "LEFT" : "RIGHT"} eye with your palm.`;
    await this.prompt(
      `Visual acuity — ${eyeLabel === "both" ? "both eyes" : eyeLabel + " eye"}`,
      `${cover} Press the arrow key pointing the same way as the E's open side.
       The letters shrink as you get them right. Guess if unsure.`,
    );

    const seg = this.session.open(testId);
    this.session.log(seg, "cover", { eye: eyeLabel });
    let logmar = 1.0, trials = 0, reversals = 0, lastCorrect = null;
    const MAX = 26, MIN_TRIALS = 12, MAX_REV = 6;

    while (trials < MAX && !(trials >= MIN_TRIALS && reversals >= MAX_REV)) {
      const dir = DIRS[Math.floor(this.rng() * 4)];
      const px = letterHeightPx(logmar, this.session.distanceCm, this.session.pxPerCm);
      const st = this.stage(`
        <div class="instruction">${cover} Which way does the E point?</div>
        <canvas id="opto" width="520" height="520" style="max-width:80vw"></canvas>
        <div class="keypad">
          <div class="spacer"></div><button data-key="up">↑</button><div class="spacer"></div>
          <button data-key="left">←</button><button data-key="down">↓</button><button data-key="right">→</button>
        </div>`, { white: true });
      const c = st.querySelector("#opto");
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, c.width, c.height);
      drawTumblingE(ctx, c.width / 2, c.height / 2, Math.max(px, 4), dir);

      const answer = await this.waitKey(DIRS);
      const correct = answer === dir;
      this.session.log(seg, "trial", { logmar: +logmar.toFixed(2), shown: dir, answered: answer });
      if (lastCorrect !== null && correct !== lastCorrect) reversals++;
      lastCorrect = correct;
      logmar = Math.min(1.3, Math.max(-0.3, logmar + (correct ? -0.1 : 0.2)));
      trials++;
    }
    this.session.close(seg);
    this.hideStage();
  }

  async runContrast() {
    this.setStep("contrast");
    await this.prompt("Contrast sensitivity",
      `Letters will fade toward the background. Type the letter you see, or press
       SPACE if you cannot read it. Sit at your normal distance and keep screen
       brightness at maximum.`);
    // Pelli-Robson protocol: triplets of three letters at each contrast level,
    // 0.15 log-unit steps, stop when a whole triplet is failed.
    const seg = this.session.open("contrast");
    const keys = [...SLOAN.map((s) => s.toLowerCase()), " "];
    for (let i = 0; i < 16; i++) {
      const logCS = +(0.15 * i).toFixed(2);
      const gray = contrastToGray(logCS);
      let correctInTriplet = 0;
      for (let k = 0; k < 3; k++) {
        const letter = SLOAN[Math.floor(this.rng() * SLOAN.length)];
        this.stage(`
          <div class="instruction">Type the letter you see — or press SPACE if you can't.</div>
          <div style="font-size:19vh;font-weight:700;color:rgb(${gray},${gray},${gray});
                      font-family:'Helvetica Neue',Arial,sans-serif">${letter}</div>
          <div class="sub">triplet ${i + 1}, letter ${k + 1} of 3</div>`, { white: true });
        const ans = await this.waitKey(keys);
        const correct = ans.toUpperCase() === letter;
        if (correct) correctInTriplet++;
        this.session.log(seg, "trial", {
          log_cs: logCS, shown: letter, answered: ans.trim() || null, correct,
        });
      }
      if (correctInTriplet < 2) break;   // whole triplet failed — chart endpoint
    }
    this.session.close(seg);
    this.hideStage();
  }

  async runAstigmatism() {
    this.setStep("astigmatism");
    await this.prompt("Astigmatism dial",
      `You'll see a sunburst of lines. If some lines look DARKER or SHARPER than the
       others, click on them. If all lines look equally clear, click "All equal".
       Repeat for each eye.`);
    const seg = this.session.open("astigmatism");
    for (const eye of ["right", "left"]) {
      const st = this.stage(`
        <div class="instruction" style="color:#333">
          Cover your ${eye === "right" ? "LEFT" : "RIGHT"} eye. Click the darkest/sharpest line —
          or press E if all look equal.</div>
        <canvas id="dial" width="620" height="620" style="max-width:78vmin;max-height:78vmin"></canvas>
        <div class="sub" style="color:#444">Press E for "all equal"</div>`, { white: true });
      const c = st.querySelector("#dial");
      drawAstigmaticDial(c.getContext("2d"), c.width / 2, c.height / 2, c.width * 0.42, 12);
      const angle = await new Promise((resolve) => {
        const onKey = (e) => { if (e.key.toLowerCase() === "e") { cleanup(); resolve(null); } };
        const onClick = (ev) => {
          const r = c.getBoundingClientRect();
          const x = ev.clientX - r.left - r.width / 2;
          const y = ev.clientY - r.top - r.height / 2;
          if (Math.hypot(x, y) < r.width * 0.1) return;
          let deg = (Math.atan2(y, x) * 180) / Math.PI;
          cleanup(); resolve(((deg % 180) + 180) % 180);
        };
        const cleanup = () => {
          document.removeEventListener("keydown", onKey);
          c.removeEventListener("click", onClick);
        };
        document.addEventListener("keydown", onKey);
        c.addEventListener("click", onClick);
      });
      this.session.log(seg, "dial", { eye, dark_meridian_deg: angle, no_preference: angle === null });
    }
    this.session.close(seg);
    this.hideStage();
  }

  async runColor() {
    this.setStep("color_vision");
    await this.prompt("Color plates",
      `Type the number you see in each circle of dots, then press ENTER.
       Press ENTER alone if you see no number. Screen color is not calibrated —
       this is a rough screen only.`);
    const seg = this.session.open("color_vision");
    for (const plate of PLATES) {
      const st = this.stage(`
        <div class="instruction" style="color:#333">What number do you see? Type it and press Enter.</div>
        <canvas id="plate" width="520" height="520" style="max-width:70vmin"></canvas>
        <input id="ans" inputmode="numeric" autocomplete="off"
               style="margin-top:26px;font-size:26px;width:140px;text-align:center;padding:10px;
                      border-radius:10px;border:1px solid #bbb">
        <div class="sub" style="color:#444">Press Enter with the box empty if you see nothing</div>`,
        { white: true });
      const c = st.querySelector("#plate");
      drawColorPlate(c.getContext("2d"), c.width / 2, c.height / 2, c.width * 0.44,
                     plate.digit, plate.type, this.rng);
      const input = st.querySelector("#ans");
      input.focus();
      const val = await new Promise((resolve) => {
        input.addEventListener("keydown", (e) => {
          if (e.key === "Enter") resolve(input.value.trim());
        });
      });
      this.session.log(seg, "plate", {
        id: plate.id, type: plate.type, shown: plate.digit,
        answered: val === "" ? null : parseInt(val, 10),
      });
    }
    this.session.close(seg);
    this.hideStage();
  }

  async runAmsler() {
    this.setStep("amsler");
    await this.prompt("Amsler grid",
      `Stare at the centre dot. Without moving your eyes, click any area where lines
       look WAVY, BLURRED or MISSING. Click "Looks normal" if the grid is clean.
       One eye at a time.`);
    const seg = this.session.open("amsler");
    for (const eye of ["right", "left"]) {
      const st = this.stage(`
        <div class="instruction">Cover your ${eye === "right" ? "LEFT" : "RIGHT"} eye.
          Stare at the centre dot and click any distorted or missing areas.</div>
        <canvas id="grid" width="620" height="620" style="max-width:76vmin;max-height:76vmin"></canvas>
        <div class="sub">
          <button class="primary" data-key="done" style="margin-right:10px">Looks normal</button>
          <button class="ghost" data-key="next">Done marking</button></div>`);
      const c = st.querySelector("#grid");
      const g = c.getContext("2d");
      const marks = [];
      const redraw = () => {
        drawAmsler(g, c.width / 2, c.height / 2, c.width * 0.86, 20);
        g.fillStyle = "rgba(255,107,107,.85)";
        marks.forEach((m) => {
          g.beginPath(); g.arc(m.px, m.py, 9, 0, Math.PI * 2); g.fill();
        });
      };
      redraw();
      c.addEventListener("click", (ev) => {
        const r = c.getBoundingClientRect();
        const px = (ev.clientX - r.left) * (c.width / r.width);
        const py = (ev.clientY - r.top) * (c.height / r.height);
        marks.push({ px, py, x: px / c.width, y: py / c.height });
        redraw();
      });
      const action = await this.waitKey(["done", "next"]);
      this.session.log(seg, "amsler", {
        eye, marks: marks.map((m) => ({ x: +m.x.toFixed(3), y: +m.y.toFixed(3) })),
        reported_normal: action === "done" && marks.length === 0,
      });
    }
    this.session.close(seg);
    this.hideStage();
  }

  async runMotility() {
    this.setStep("motility");
    await this.prompt("Eye tracking",
      `Follow the moving dot with your EYES ONLY — keep your head still.
       This measures how smoothly and symmetrically your eyes move. ~18 seconds.`);
    const seg = this.session.open("motility");
    // Dark stage + bright dot: the target itself is the Hirschberg light source,
    // so the corneal reflex needed for alignment is actually present.
    const st = this.stage(`
      <div class="instruction">Follow the bright dot with your eyes. Keep your head still.</div>
      <div id="pursuitDot"></div>
      <div class="sub" id="ptime"></div>`);
    st.classList.add("glint");
    const dot = st.querySelector("#pursuitDot");
    const DUR = 18000, t0 = performance.now();
    let lastLog = 0;
    await new Promise((resolve) => {
      const step = (now) => {
        const t = now - t0;
        if (t >= DUR) return resolve();
        const w = window.innerWidth - 40, h = window.innerHeight - 40;
        let x, y, phase;
        if (t < 9000) {                       // smooth horizontal pursuit
          phase = "pursuit_h";
          x = 0.5 + 0.42 * Math.sin((2 * Math.PI * t) / 4500);
          y = 0.45;
        } else if (t < 13000) {               // vertical pursuit
          phase = "pursuit_v";
          x = 0.5;
          y = 0.45 + 0.3 * Math.sin((2 * Math.PI * (t - 9000)) / 3500);
        } else {                              // H-pattern step saccades
          phase = "saccade";
          const idx = Math.floor((t - 13000) / 700) % 6;
          const pts = [[0.1, 0.45], [0.9, 0.45], [0.1, 0.2], [0.9, 0.2], [0.1, 0.7], [0.9, 0.7]];
          [x, y] = pts[idx];
        }
        dot.style.left = `${x * w}px`;
        dot.style.top = `${y * h}px`;
        if (now - lastLog > 33) {
          lastLog = now;
          this.session.log(seg, "dot", { x: +x.toFixed(4), y: +y.toFixed(4), phase });
        }
        st.querySelector("#ptime").textContent = `${Math.ceil((DUR - t) / 1000)}s`;
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
    this.session.close(seg);
    this.hideStage();
  }

  async runPupil() {
    this.setStep("pupil");
    await this.prompt("Pupil light response",
      `Turn the room lights DOWN. Look at the camera and hold still.
       The screen will go dark, then flash bright several times. Don't blink during the flashes.`);
    const seg = this.session.open("pupil");
    const st = this.stage(`
      <div class="instruction">Look at the camera. Hold still.</div>
      <div id="flashband" style="display:none"></div>
      <div class="sub" id="pstat">adapting to the dark…</div>`, { blackout: true });
    const band = st.querySelector("#flashband");
    const stat = st.querySelector("#pstat");

    await sleep(4000);   // dark adaptation
    for (let i = 0; i < 3; i++) {
      stat.textContent = `flash ${i + 1} of 3`;
      this.session.log(seg, "flash_on", { index: i });
      band.style.display = "block";
      document.body.style.background = "#fff";
      await sleep(900);
      band.style.display = "none";
      document.body.style.background = "";
      this.session.log(seg, "flash_off", { index: i });
      await sleep(3200);  // redilation
    }
    this.session.close(seg);
    this.hideStage();
  }

  async runPhotoref() {
    this.setStep("photoref");
    await this.prompt("Refraction estimate",
      `Keep the room dark. Look just BELOW the camera and hold very still while
       the screen lights up. This images the red reflex from your retina.`);
    const seg = this.session.open("photoref");
    this.session.log(seg, "photoref_config", {
      e_m: 0.005, d_m: this.session.distanceCm / 100,
    });
    const st = this.stage(`
      <div class="instruction">Look just below the camera. Hold still.</div>
      <div id="flashband" style="display:block"></div>
      <div class="sub" id="prstat"></div>`, { blackout: true });
    for (let s = 8; s > 0; s--) {
      st.querySelector("#prstat").textContent = `${s}s`;
      await sleep(1000);
    }
    this.session.close(seg);
    this.hideStage();
  }

  // ---------- flow ----------

  prompt(title, body) {
    return new Promise((resolve) => {
      const st = this.stage(`
        <div style="max-width:620px;text-align:center;padding:0 24px">
          <h1>${title}</h1>
          <p class="lead">${body}</p>
          <button class="primary" data-key="go">Start this test</button>
        </div>`);
      st.querySelector("[data-key=go]").addEventListener("click", () => resolve(), { once: true });
    });
  }

  async runAll() {
    $("#intro").hidden = true;
    $("#railwrap").hidden = false;
    $("#camPanel").classList.add("floating");
    document.body.classList.add("testing");
    this.startRecording();

    await this.runAcuity("both", "acuity_both");
    await this.runAcuity("right", "acuity_right");
    await this.runAcuity("left", "acuity_left");
    await this.runContrast();
    await this.runAstigmatism();
    await this.runColor();
    await this.runAmsler();
    await this.runMotility();
    await this.runPupil();
    await this.runPhotoref();

    await this.finish();
  }

  async finish() {
    this.stage(`<div style="text-align:center">
      <div class="spinner"></div>
      <h1 style="margin-top:18px">Analyzing your session…</h1>
      <p class="lead">Processing the recording frame by frame. This can take a minute.</p></div>`);
    this.tracker.stop();
    this.recorder.stop();
    await new Promise((r) => (this.recorder.onstop = r));

    const fps = 30;
    const form = new FormData();
    form.append("video", new Blob(this.chunks, { type: "video/webm" }), "session.webm");
    form.append("meta", this.session.toJSON(fps));
    const resp = await fetch("/analyze", { method: "POST", body: form });
    const html = await resp.text();
    document.open(); document.write(html); document.close();
  }
}

// ---------- bootstrap ----------

window.addEventListener("DOMContentLoaded", async () => {
  const app = new App();
  window.__app = app;

  $("#btnCamera").addEventListener("click", async () => {
    $("#btnCamera").disabled = true;
    $("#btnCamera").textContent = "Starting camera…";
    try {
      await app.initCamera();
      $("#camStatus").textContent = "Camera ready — check that both eyes are outlined below.";
      $("#btnStart").disabled = false;
      $("#btnCamera").hidden = true;
    } catch (e) {
      $("#camStatus").textContent = "Camera failed: " + e.message;
      $("#btnCamera").disabled = false;
      $("#btnCamera").textContent = "Retry camera";
    }
  });

  // card-width calibration: a credit card is 8.56 cm wide by ISO/IEC 7810
  const slider = $("#cardSlider");
  const card = $("#cardBox");
  const updateCard = () => {
    const px = +slider.value;
    card.style.width = px + "px";
    app.session.pxPerCm = px / 8.56;
    $("#pxPerCm").textContent = app.session.pxPerCm.toFixed(1);
  };
  slider.addEventListener("input", updateCard);
  updateCard();

  $("#distance").addEventListener("input", (e) => {
    app.session.distanceCm = +e.target.value;
    $("#distanceVal").textContent = e.target.value;
  });

  $("#btnStart").addEventListener("click", () => app.runAll());
});
