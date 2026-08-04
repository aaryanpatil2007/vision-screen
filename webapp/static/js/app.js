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
  drawRDS, disparityArcsec, drawWorthDots,
  renderableFloorLogmar, displayCeilingLogCS, drawSloanLetter,
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
  { id: "stereo", label: "Depth" },
  { id: "suppression", label: "Fusion" },
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
    this.canDarken = true;
    this.speak = false;
  }

  /**
   * Read text aloud. Someone who cannot comfortably read the screen still has
   * to be able to follow the protocol — and this is a vision test, so assuming
   * the instructions are legible is exactly the wrong assumption. Uses the
   * browser's own speech synthesis; no network, no dependency.
   */
  say(text) {
    if (!this.speak || !window.speechSynthesis) return;
    try {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text.replace(/\s+/g, " ").trim());
      u.rate = 0.98; u.pitch = 1.0;
      window.speechSynthesis.speak(u);
    } catch (e) { /* speech is an aid, never a dependency */ }
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

  async runAcuity(eyeLabel, testId, optotype = "sloan") {
    this.setStep(testId);
    const cover = eyeLabel === "both" ? "Keep both eyes open."
      : `Cover your ${eyeLabel === "right" ? "LEFT" : "RIGHT"} eye with your palm.`;
    await this.prompt(
      `Visual acuity — ${eyeLabel === "both" ? "both eyes" : eyeLabel + " eye"}`,
      optotype === "sloan"
        ? `${cover} Type the letter you see, then it will shrink. Letters come from
           C D H K N O R S V Z. Guess if unsure — guessing is expected.`
        : `${cover} Press the arrow key pointing the same way as the E's open side.
           The letters shrink as you get them right. Guess if unsure.`,
    );

    const seg = this.session.open(testId);
    this.session.log(seg, "cover", { eye: eyeLabel });
    this.session.log(seg, "optotype", { name: optotype });
    // Sloan is a 10-alternative task, so the staircase ratio differs to keep
    // converging on the guessing-corrected 50% point (11/9 vs 5/3).
    const STEP_UP_FOR = { sloan: 0.1 * 11 / 9, tumbling_e: 0.1 * 5 / 3 };
    let logmar = 1.0, trials = 0, reversals = 0, lastCorrect = null;
    // budget matched to the server's measured repeatability sweep:
    // 60 trials puts test-retest CoR at 0.113 logMAR, the level of the
    // printed ETDRS chart itself (0.11).
    const MAX = 60, MIN_TRIALS = 20, MAX_REV = 14;
    // Never present an optotype finer than one pixel of stroke — below that
    // the E is no longer the shape it claims to be.
    const floorLogmar = renderableFloorLogmar(
      this.session.distanceCm, this.session.pxPerCm);
    this.session.log(seg, "display_floor", { logmar: floorLogmar });
    // Kaernbach weighted up-down targeting the guessing-corrected 50% point
    // for a 4-alternative task: S_up / S_down = 5/3.
    const STEP_DOWN = 0.1, STEP_UP = STEP_UP_FOR[optotype], HALVE_AFTER = 2;

    while (trials < MAX && !(trials >= MIN_TRIALS && reversals >= MAX_REV)) {
      const px = letterHeightPx(logmar, this.session.distanceCm, this.session.pxPerCm);
      const isSloan = optotype === "sloan";
      const shown = isSloan
        ? SLOAN[Math.floor(this.rng() * SLOAN.length)]
        : DIRS[Math.floor(this.rng() * 4)];
      const keypad = isSloan
        ? `<div class="sub" style="color:#444">Type the letter, or press SPACE if you cannot read it</div>`
        : `<div class="keypad">
             <div class="spacer"></div><button data-key="up">↑</button><div class="spacer"></div>
             <button data-key="left">←</button><button data-key="down">↓</button><button data-key="right">→</button>
           </div>`;
      const st = this.stage(`
        <div class="instruction">${cover} ${isSloan ? "Which letter is this?" : "Which way does the E point?"}</div>
        <canvas id="opto" width="520" height="520" style="max-width:80vw"></canvas>
        ${keypad}`, { white: true });
      const c = st.querySelector("#opto");
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, c.width, c.height);
      if (isSloan) {
        drawSloanLetter(ctx, c.width / 2, c.height / 2, Math.max(px, 4), shown);
      } else {
        drawTumblingE(ctx, c.width / 2, c.height / 2, Math.max(px, 4), shown);
      }

      const answer = await this.waitKey(
        isSloan ? [...SLOAN.map((l) => l.toLowerCase()), " "] : DIRS);
      const correct = isSloan
        ? answer.toUpperCase() === shown
        : answer === shown;
      this.session.log(seg, "trial",
                       { logmar: +logmar.toFixed(2), shown, answered: answer });
      if (lastCorrect !== null && correct !== lastCorrect) reversals++;
      lastCorrect = correct;
      // step halves once the run has bracketed threshold; the up/down RATIO
      // is preserved so the convergence criterion does not move
      const scale = reversals >= HALVE_AFTER ? 0.5 : 1.0;
      logmar = Math.min(1.3, Math.max(Math.max(-0.3, floorLogmar),
                        logmar + (correct ? -STEP_DOWN : STEP_UP) * scale));
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
    const csCeiling = displayCeilingLogCS();
    this.session.log(seg, "display_ceiling", { log_cs: csCeiling });
    let consecutiveFails = 0;
    for (let i = 0; i < 16; i++) {
      const logCS = +(0.15 * i).toFixed(2);
      if (logCS > csCeiling) break;   // beyond this every rung is the same image
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
      // One failed triplet can be a lapse; require two in a row before
      // stopping, as reading down a printed chart does.
      consecutiveFails = correctInTriplet >= 2 ? 0 : consecutiveFails + 1;
      if (consecutiveFails >= 2) break;
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

  async runStereo() {
    this.setStep("stereo");
    const pitchMm = 10 / this.session.pxPerCm;
    const floorArcsec = disparityArcsec(pitchMm, this.session.distanceCm * 10);
    await this.prompt("Depth perception",
      `Put on the red-cyan glasses (red over your LEFT eye). A square of dots will
       appear; one quarter of it floats in front of the rest. Click the quarter
       that stands out — guess if you're not sure. Some patterns have no depth
       at all, which is intentional.`);

    const seg = this.session.open("stereo");
    this.session.log(seg, "stereo_config", {
      px_per_cm: this.session.pxPerCm,
      display_floor_arcsec: floorArcsec,
    });

    // ladder in pixel disparity; a catch trial (0 px) every fourth presentation
    const ladder = [12, 8, 6, 4, 3, 2, 1.5, 1];
    const order = [];
    ladder.forEach((d, i) => {
      order.push({ disparity: d, catch: false });
      order.push({ disparity: d, catch: false });
      if (i % 2 === 1) order.push({ disparity: 0, catch: true });
    });

    for (const item of order) {
      const quadrant = Math.floor(this.rng() * 4);
      const st = this.stage(`
        <div class="instruction">Click the quarter that floats in front.</div>
        <canvas id="rds" width="520" height="520" style="max-width:76vmin;max-height:76vmin"></canvas>
        <div class="sub">Guess if unsure — some patterns genuinely have no depth.</div>`);
      const c = st.querySelector("#rds");
      const g = c.getContext("2d");
      // re-randomize dots every frame: kills any static monocular pattern
      let raf, frames = 0;
      const animate = () => {
        drawRDS(g, { size: c.width, disparityPx: item.disparity, quadrant,
                     dotPx: 3, density: 0.35, rng: mulberry32((frames++ * 2654435761) >>> 0) });
        raf = requestAnimationFrame(animate);
      };
      animate();

      const answer = await new Promise((resolve) => {
        c.addEventListener("click", (ev) => {
          const r = c.getBoundingClientRect();
          const qx = ev.clientX - r.left < r.width / 2 ? 0 : 1;
          const qy = ev.clientY - r.top < r.height / 2 ? 0 : 1;
          resolve(qy * 2 + qx);
        }, { once: true });
      });
      cancelAnimationFrame(raf);

      const arcsec = disparityArcsec(item.disparity * pitchMm,
                                     this.session.distanceCm * 10);
      this.session.log(seg, item.catch ? "catch" : "trial", {
        arcsec, disparity_px: item.disparity,
        shown: quadrant, answered: answer, correct: answer === quadrant,
      });
    }
    this.session.close(seg);
    this.hideStage();
  }


  async runSuppression() {
    this.setStep("suppression");
    await this.prompt("Binocular fusion",
      `Keep the red-cyan glasses on, red lens over your RIGHT eye.
       You'll see a diamond of coloured dots. Count how many you see and say
       which colours — this shows whether both eyes are working together.`);
    const seg = this.session.open("suppression");

    for (const [dist, label] of [["near", "at your normal distance"],
                                 ["far", "from as far back as you can sit"]]) {
      const st = this.stage(`
        <div class="instruction">Look at the dots ${label}. How many do you see?</div>
        <canvas id="worth" width="360" height="360" style="max-width:60vmin"></canvas>
        <div class="keypad" style="grid-template-columns:repeat(2,190px)">
          <button data-key="four">4 dots</button>
          <button data-key="two_red">2 red only</button>
          <button data-key="three_green">3 green only</button>
          <button data-key="five_uncrossed">5 — reds on the right</button>
          <button data-key="five_crossed">5 — reds on the left</button>
        </div>`, { blackout: true });
      const c = st.querySelector("#worth");
      drawWorthDots(c.getContext("2d"), c.width / 2, c.height / 2, c.width * 0.3);
      const resp = await this.waitKey(["four", "two_red", "three_green",
                                       "five_uncrossed", "five_crossed"]);
      this.session.log(seg, "worth", { distance: dist, response: resp });
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

  /** Hold until the room is actually dark enough, with live feedback. */
  async waitForDarkness(maxWaitMs = 45000) {
    const st = this.stage(`
      <div style="max-width:560px;text-align:center;padding:0 24px">
        <h1>Dim the room</h1>
        <p class="lede" style="margin:0 auto">This test reads light reflected off your
        own eye, so the screen needs to be the brightest thing here. Turn off the
        lights or draw the curtains — the reading below updates live.</p>
        <p style="font-family:var(--mono);font-size:2.6rem;margin:1.8rem 0 0.4rem"
           id="lvl">—</p>
        <p class="hint" id="lvlmsg" style="margin:0 auto">waiting…</p>
        <p style="margin-top:2rem">
          <button class="ghost" data-key="skip">Skip this test instead</button></p>
      </div>`, { blackout: true });
    const t0 = performance.now();
    return new Promise((resolve) => {
      const tick = setInterval(() => {
        const b = this.tracker ? this.tracker.stats.brightness : 999;
        const el = st.querySelector("#lvl"), msg = st.querySelector("#lvlmsg");
        if (el) {
          el.textContent = Math.round(b);
          el.style.color = b <= 90 ? "var(--good)" : "var(--warn)";
        }
        if (msg) msg.textContent = b <= 90
          ? "dark enough — starting" : "still too bright, keep dimming";
        if (b <= 90) { clearInterval(tick); setTimeout(() => resolve(true), 900); }
        else if (performance.now() - t0 > maxWaitMs) { clearInterval(tick); resolve(false); }
      }, 400);
      st.querySelector("[data-key=skip]").addEventListener("click", () => {
        clearInterval(tick); resolve(false);
      }, { once: true });
    });
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
    this.say(`${title}. ${body}`);
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
    await this.runStereo();
    await this.runSuppression();
    await this.runMotility();
    if (this.canDarken) {
      const darkEnough = await this.waitForDarkness();
      if (darkEnough) {
        await this.runPupil();
        await this.runPhotoref();
      } else {
        const seg = this.session.open("skipped");
        this.session.log(seg, "skipped", { tests: ["pupil", "photoref"],
                                           reason: "room never became dark enough" });
        this.session.close(seg);
        this.hideStage();
      }
    } else {
      // Recorded explicitly so the report can say "not attempted" rather than
      // "inconclusive" — the user made a choice, the test did not fail.
      const seg = this.session.open("skipped");
      this.session.log(seg, "skipped", { tests: ["pupil", "photoref"],
                                         reason: "no dark room available" });
      this.session.close(seg);
    }

    await this.finish();
  }

  async finish() {
    const aborted = this.abortedReason
      ? `<h1 style="margin-top:18px">Something went wrong</h1>
         <p class="lead">The test stopped early
         (<code>${this.abortedReason}</code>). Analyzing the tests you did
         complete — the report will cover those.</p>`
      : `<h1 style="margin-top:18px">Analyzing your session…</h1>
         <p class="lead">Processing the recording frame by frame. This can take a minute.</p>`;
    this.stage(`<div style="text-align:center;max-width:600px;padding:0 24px">
      <div class="spinner"></div>${aborted}</div>`);
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

/**
 * A thrown stage must never leave the user staring at a frozen screen. Any
 * uncaught error aborts the battery, uploads whatever was collected, and says
 * what happened — a partial report beats an infinite wait.
 */
function installCrashHandler(app) {
  let handled = false;
  const onCrash = async (detail) => {
    if (handled) return;
    handled = true;
    console.error("battery aborted:", detail);
    try {
      // Record the reason so the analyzing screen keeps showing it — otherwise
      // finish() overwrites the explanation and the user never learns why the
      // battery stopped.
      app.abortedReason = String(detail).slice(0, 160);
      if (app.recorder && app.recorder.state !== "inactive") {
        await app.finish();
      } else {
        app.stage(`<div style="max-width:560px;text-align:center;padding:0 24px">
          <h1>Something went wrong</h1>
          <p class="lead">The test hit an error and stopped
          (<code>${app.abortedReason}</code>). Reload to start again.</p></div>`);
      }
    } catch (e) {
      console.error("could not recover:", e);
    }
  };
  window.addEventListener("error", (e) => onCrash(e.message || e.error));
  window.addEventListener("unhandledrejection", (e) => onCrash(e.reason));
}

window.addEventListener("DOMContentLoaded", async () => {
  const app = new App();
  window.__app = app;
  installCrashHandler(app);

  $("#btnCamera").addEventListener("click", async () => {
    $("#btnCamera").disabled = true;
    $("#btnCamera").textContent = "Starting camera…";
    try {
      await app.initCamera();
      $("#camStatus").textContent = "Camera ready — check that both eyes are outlined below.";
      $("#btnCamera").hidden = true;
      if (app.onCameraReady) app.onCameraReady();
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

  // Grey ramp: a subjective check that the dark end of the display is not
  // crushed. Standards want 80-320 cd/m2 and the browser cannot measure it.
  const ramp = $("#grayRamp");
  for (let i = 0; i < 16; i++) {
    const v = Math.round((i / 15) * 90);
    const seg = document.createElement("div");
    seg.style.cssText = `flex:1;background:rgb(${v},${v},${v})`;
    ramp.appendChild(seg);
  }

  const brightness = $("#brightnessOk");
  const startBtn = $("#btnStart");
  const refreshStart = () => {
    startBtn.disabled = !(app.tracker && brightness.checked);
    startBtn.textContent = brightness.checked
      ? "Begin screening"
      : "Confirm screen brightness above";
  };
  brightness.addEventListener("change", refreshStart);
  app.onCameraReady = refreshStart;

  // dark-room capability: the user's answer, not an assumption
  const darkNo = $("#darkNo");
  const syncDark = () => { app.canDarken = !(darkNo && darkNo.checked); };
  document.querySelectorAll('input[name="darkroom"]').forEach(
    (r) => r.addEventListener("change", syncDark));
  syncDark();

  // live room-light readout, so "dim the room" is verifiable rather than hopeful
  const lightNow = $("#lightNow");
  if (lightNow) {
    setInterval(() => {
      if (!app.tracker) return;
      const b = app.tracker.stats.brightness;
      const dark = b <= 90;
      lightNow.innerHTML =
        `Room light, measured from your camera: <strong style="color:${
          dark ? "var(--good)" : "var(--warn)"}">${Math.round(b)}</strong>` +
        `<span style="color:var(--dim)"> — ${dark
          ? "dark enough for the light-reflex tests"
          : "too bright for those three; dim below 90 when prompted"}</span>`;
    }, 700);
  }

  // spoken guidance — a vision test cannot assume its instructions are legible
  const speakOn = $("#speakOn");
  if (speakOn) {
    speakOn.addEventListener("change", () => {
      app.speak = speakOn.checked;
      if (app.speak) app.say("Spoken instructions are on.");
    });
  }

  startBtn.addEventListener("click", () => app.runAll());
});
