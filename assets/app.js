/* Typed Decision Models as Judges: interactive figures. Data: assets/site-data.js (built by tools/build_data.py). */
(() => {
  "use strict";
  const D = window.SITE_DATA;
  const $ = (s) => document.querySelector(s);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const still = /[?&]still\b/.test(location.search);           // ?still: no motion (for screenshots)
  const reduceMotion = still || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (still) document.documentElement.classList.add("still");

  const KIND = {
    "TDM": { label: "TDMs", v: "--tdm", sym: d3.symbolCircle },
    "LLM 2024": { label: "LLM judges of 2024", v: "--llm24", sym: d3.symbolSquare },
    "LLM 2026": { label: "LLM judges of 2026", v: "--llm26", sym: d3.symbolTriangle },
  };
  const KINDS = ["TDM", "LLM 2024", "LLM 2026"];
  const col = (kind) => css(KIND[kind].v);
  const judges = D.judges;
  const jById = Object.fromEntries(judges.map((j) => [j.id, j]));
  const jByName = Object.fromEntries(judges.map((j) => [j.name, j]));
  const tests = D.tests;
  const tByKey = Object.fromEntries(tests.map((t) => [t.key, t]));
  const cellMap = new Map(D.cells.map((c) => [c.judge + "|" + c.test, c]));
  const cell = (j, t) => cellMap.get(j + "|" + t);
  const tdms = judges.filter((j) => j.kind === "TDM");
  const med = (a) => d3.median(a.filter((x) => x != null));
  const per100 = (x) => (x * 100).toFixed(0);
  const signed = (x, d = 2) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
  const testName = (k) => (tByKey[k] ? tByKey[k].name : k);

  /* ---------- tooltip ---------- */
  const tip = $("#tooltip");
  function showTip(html, ev) {
    tip.innerHTML = html; tip.style.opacity = 1;
    const r = tip.getBoundingClientRect();
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - 14;
    if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  const hideTip = () => { tip.style.opacity = 0; };

  /* ---------- small helpers ---------- */
  function chips(container, items, selected, onSelect) {
    const el = typeof container === "string" ? $(container) : container;
    el.innerHTML = "";
    const buttons = items.map((it) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "chip"; b.innerHTML = esc(it.label) + (it.ours ? '<span class="ours" aria-label="our addition">†</span>' : "");
      b.setAttribute("aria-pressed", String(it.key === selected));
      b.addEventListener("click", () => { buttons.forEach((x) => x.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true"); onSelect(it.key, true); });
      el.appendChild(b); return b;
    });
    return { select(key) { buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(items[i].key === key))); } };
  }
  function legend(container, entries) {
    $(container).innerHTML = entries.map((e) => `<span>${e.html}${esc(e.label)}</span>`).join("");
  }
  const swatchHTML = (kind) => `<i class="swatch sw-${kind === "TDM" ? "tdm" : kind === "LLM 2024" ? "llm24" : "llm26"}"></i>`;
  const kindLegend = (keys = KINDS) => keys.map((k) => ({ html: swatchHTML(k), label: KIND[k].label }));
  function svgIn(sel, w, h) {
    const root = d3.select(sel); root.selectAll("*").remove();
    return root.append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("preserveAspectRatio", "xMidYMid meet");
  }
  const width = (sel, min = 300) => Math.max(min, Math.round($(sel).getBoundingClientRect().width || 600));

  /* ---------- static lists ---------- */
  function renderTests() {
    $("#tests").innerHTML = tests.map((t) => `
      <div class="test"><div class="test-top"><h4>${esc(t.name)}</h4>
      <span class="tag ${t.ours ? "ours" : ""}">${t.ours ? "ours · " : ""}${t.dataset === "fact" ? "fact-related" : "alignment"}</span></div>
      <p>${esc(t.what)}</p></div>`).join("");
    $("#judges").innerHTML = KINDS.map((k) => `
      <div class="judge-group"><h4>${swatchHTML(k)}${esc(KIND[k].label)}</h4><ul>
      ${judges.filter((j) => j.kind === k).map((j) => `<li>${esc(j.name)}</li>`).join("")}</ul></div>`).join("");
  }

  /* ---------- hero: the scale ---------- */
  const CUES = [
    { key: "sentiment", label: "Sentiment", token: "angry tone", on: "the rewritten answer" },
    { key: "authority, better response", label: "Authority (on better)", token: "fake quote", on: "the better answer", ours: true },
    { key: "authority", label: "Authority", token: "fake quote", on: "the worse answer" },
    { key: "bandwagon", label: "Bandwagon", token: "“90% prefer it”", on: "the worse answer" },
    { key: "position", label: "Position", token: "shown second", on: "the answer shown second" },
    { key: "compassion", label: "Compassion fade", token: "“GPT-5.6”", on: "the worse answer" },
    { key: "verbosity, better response", label: "Verbosity (on better)", token: "padded", on: "the better answer", ours: true },
    { key: "distraction", label: "Distraction", token: "loves pasta", on: "the answer with the note" },
    { key: "diversity", label: "Diversity", token: "identity note", on: "the answer with the note" },
  ];
  let scaleState = { key: "sentiment", auto: !reduceMotion, timer: null };
  function scaleValues(key) {
    const cs = tdms.map((j) => cell(j.id, key));
    return { shift: med(cs.map((c) => c.shift)), excess: med(cs.map((c) => c.e)), biased: cs.filter((c) => c.c === "bias").length };
  }
  function renderScale() {
    const svg = d3.select("#scale"); svg.selectAll("*").remove();
    const ink = css("--ink"), ink2 = css("--ink-2"), rule = css("--rule"), accent = css("--accent"), card = css("--card"), mark = css("--mark");
    const cx = 260, cy = 74, L = 170;
    svg.append("path").attr("d", `M${cx - 62},232 L${cx + 62},232`).attr("stroke", ink).attr("stroke-width", 1.4);
    svg.append("path").attr("d", `M${cx - 30},232 L${cx},212 L${cx + 30},232`).attr("fill", "none").attr("stroke", ink).attr("stroke-width", 1.2);
    svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", cy).attr("y2", 214).attr("stroke", ink).attr("stroke-width", 1.4);
    // angle ticks behind the beam
    const arcG = svg.append("g").attr("opacity", 0.9);
    [-20, -10, 0, 10, 20].forEach((a) => {
      const r1 = 40, r2 = a === 0 ? 50 : 46, rad = (a - 90) * Math.PI / 180;
      arcG.append("line").attr("x1", cx + r1 * Math.cos(rad)).attr("y1", cy + r1 * Math.sin(rad))
        .attr("x2", cx + r2 * Math.cos(rad)).attr("y2", cy + r2 * Math.sin(rad)).attr("stroke", rule).attr("stroke-width", 1);
    });
    const beam = svg.append("g").attr("class", "beam");
    beam.append("line").attr("x1", cx - L).attr("x2", cx + L).attr("y1", cy).attr("y2", cy).attr("stroke", ink).attr("stroke-width", 2.2).attr("stroke-linecap", "round");
    beam.append("line").attr("x1", cx).attr("x2", cx).attr("y1", cy).attr("y2", cy - 34).attr("stroke", accent).attr("stroke-width", 1.4);
    beam.append("circle").attr("cx", cx).attr("cy", cy - 36).attr("r", 2.6).attr("fill", accent);
    svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 4.5).attr("fill", card).attr("stroke", ink).attr("stroke-width", 1.4);
    function pan(side) {
      const g = svg.append("g").attr("class", "pan-" + side);
      g.append("path").attr("d", "M0,0 L-38,62 M0,0 L38,62").attr("stroke", ink2).attr("stroke-width", 0.9);
      g.append("path").attr("d", "M-50,62 Q0,84 50,62 Z").attr("fill", card).attr("stroke", ink).attr("stroke-width", 1.3);
      g.append("circle").attr("r", 2.4).attr("fill", ink);
      g.append("text").attr("y", 106).attr("text-anchor", "middle").attr("class", "lbl").style("font-size", "15px");
      return g;
    }
    const left = pan("left"), right = pan("right");
    right.select("text").text("the other answer");
    const token = left.append("g").attr("class", "token");
    token.append("rect").attr("height", 24).attr("y", 35).attr("fill", mark).attr("stroke", ink).attr("stroke-width", 0.8);
    token.append("text").attr("y", 52).attr("text-anchor", "middle").attr("class", "lbl-strong").style("font-size", "14px");
    svg.append("text").attr("x", cx).attr("y", 254).attr("text-anchor", "middle").attr("class", "lbl-small").style("font-size", "12px").text("tilt = mean shift of the probability, median TDM");
    scaleState.parts = { beam, left, right, token, cx, cy, L };
    setScale(scaleState.key, false);
  }
  function setScale(key, animate = true) {
    scaleState.key = key;
    const cue = CUES.find((c) => c.key === key), v = scaleValues(key), P = scaleState.parts;
    if (!P) return;
    const angle = Math.max(-22, Math.min(22, -v.shift * 70));
    const place = (deg) => {
      const r = deg * Math.PI / 180;
      P.beam.attr("transform", `rotate(${deg},${P.cx},${P.cy})`);
      P.left.attr("transform", `translate(${P.cx - P.L * Math.cos(r)},${P.cy - P.L * Math.sin(r)})`);
      P.right.attr("transform", `translate(${P.cx + P.L * Math.cos(r)},${P.cy + P.L * Math.sin(r)})`);
    };
    const from = scaleState.angle ?? 0;
    scaleState.angle = angle;
    P.left.select(".lbl").text(cue.on);
    P.token.select("text").text(cue.token);
    const tw = Math.max(64, cue.token.length * 8.2 + 16);
    P.token.select("rect").attr("x", -tw / 2).attr("width", tw);
    if (animate && !reduceMotion) {
      d3.select(P.beam.node()).interrupt();
      d3.transition().duration(1100).ease(d3.easeBackOut.overshoot(1.6)).tween("tilt", () => { const i = d3.interpolateNumber(from, angle); return (t) => place(i(t)); });
    } else place(angle);
    const word = v.shift < -0.02 ? "punished" : v.shift > 0.02 ? "rewarded" : "barely moved";
    $("#scale-readout").innerHTML = `
      <span><b>${signed(v.shift)}</b>probability of ${esc(cue.on)} (${word})</span>
      <span><b>${per100(v.excess)} of 100</b>verdicts changed beyond chance</span>
      <span><b>${v.biased} of 8</b>TDMs biased</span>`;
  }
  function initScale() {
    const ctl = chips("#scale-chips", CUES, scaleState.key, (k, user) => { if (user) stopAuto(); setScale(k); });
    scaleState.ctl = ctl;
    renderScale();
    const stopAuto = () => { scaleState.auto = false; clearInterval(scaleState.timer); };
    if (scaleState.auto) {
      let i = 0;
      scaleState.timer = setInterval(() => {
        if (!scaleState.auto || document.hidden) return;
        const rect = $("#scale").getBoundingClientRect();
        if (rect.bottom < 0 || rect.top > window.innerHeight) return;
        i = (i + 1) % CUES.length; setScale(CUES[i].key); ctl.select(CUES[i].key);
      }, 3600);
      ["pointerdown", "keydown"].forEach((e) => $("#scale-chips").addEventListener(e, stopAuto, { once: true }));
    }
  }

  /* ---------- Fig. 1: fingerprints ---------- */
  let selectedJudge = null;
  const SHORT = { "Sentiment": "Sentiment", "Position": "Position", "Authority (on better)": "Auth. (better)", "Authority": "Authority",
    "Bandwagon": "Bandwagon", "Compassion fade": "Compassion", "Verbosity (on better)": "Verb. (better)", "Verbosity": "Verbosity",
    "Distraction": "Distraction", "Diversity": "Diversity", "Fallacy oversight": "Fallacy", "Neutral note": "Neutral note" };
  const n = tests.length, step = (2 * Math.PI) / n;
  function petals(g, R, r0, values, opts = {}) {
    const rScale = d3.scaleSqrt().domain([0, 0.4]).range([r0, R]).clamp(true);
    const arc = d3.arc().cornerRadius(1.5).padAngle(0.035);
    g.append("circle").attr("r", rScale(0.05)).attr("fill", "none").attr("stroke", css("--accent")).attr("stroke-width", 0.7).attr("opacity", 0.55);
    values.forEach((v, i) => {
      const a = i * step;
      const outer = Math.max(r0 + 1.5, rScale(Math.max(0, v.e ?? 0)));
      const d = arc({ innerRadius: r0, outerRadius: outer, startAngle: a - step / 2, endAngle: a + step / 2 });
      const p = g.append("path").attr("d", d);
      if (opts.key) p.attr("fill", css("--rule")).attr("stroke", css("--muted")).attr("stroke-width", 0.6);
      else if (v.c === "bias") p.attr("fill", opts.color);
      else if (v.c === "unclear") p.attr("fill", opts.color).attr("fill-opacity", 0.2).attr("stroke", opts.color).attr("stroke-width", 0.9);
      else p.attr("fill", css("--rule")).attr("fill-opacity", 0.8);
      if (opts.onHover) p.style("cursor", "pointer").on("mousemove", (ev) => opts.onHover(ev, v, i)).on("mouseleave", hideTip);
    });
  }
  function renderPrints() {
    // the key
    const ks = svgIn("#print-key", 260, 240), kg = ks.append("g").attr("transform", "translate(130,120)");
    petals(kg, 70, 10, tests.map(() => ({ e: 0.4 })), { key: true });
    tests.forEach((t, i) => {
      const a = i * step - Math.PI / 2, x = Math.cos(a) * 82, y = Math.sin(a) * 82;
      kg.append("text").attr("x", x).attr("y", y + 4).attr("text-anchor", Math.abs(x) < 8 ? "middle" : x > 0 ? "start" : "end")
        .attr("class", "lbl-small").style("fill", t.ours ? css("--accent") : null).text(SHORT[t.name] + (t.ours ? "†" : ""));
    });
    kg.append("text").attr("y", 4).attr("text-anchor", "middle").attr("class", "lbl-small").text("key");
    // the judges
    const box = $("#prints"); box.innerHTML = "";
    judges.forEach((j) => {
      const b = document.createElement("button"); b.type = "button"; b.className = "print" + (selectedJudge === j.id ? " active" : "");
      b.setAttribute("aria-label", `${j.name}: ${j.nBiased} of 12 tests with a bias`);
      box.appendChild(b);
      const s = d3.select(b).append("svg").attr("viewBox", "0 0 120 120");
      const g = s.append("g").attr("transform", "translate(60,60)");
      const vals = tests.map((t) => cell(j.id, t.key));
      petals(g, 54, 6, vals, { color: col(j.kind), onHover: (ev, v, i) => showTip(
        `<b>${esc(j.name)}</b> · ${esc(tests[i].name)}<br>excess <span class="tt-num">${per100(v.e)}</span> per 100 <span class="tt-num">[${per100(v.lo)}, ${per100(v.hi)}]</span><br>${v.c === "bias" ? "bias" : v.c === "unclear" ? "unclear" : "no bias"}`, ev) });
      const nm = document.createElement("div"); nm.className = "print-name"; nm.textContent = j.name; b.appendChild(nm);
      const ct = document.createElement("div"); ct.className = "print-count"; ct.textContent = `${j.nBiased} of 12 biased`; b.appendChild(ct);
      b.addEventListener("click", () => {
        selectedJudge = selectedJudge === j.id ? null : j.id;
        document.querySelectorAll(".print").forEach((x) => x.classList.remove("active"));
        if (selectedJudge) b.classList.add("active");
        renderForest();
      });
    });
  }

  /* ---------- Fig. 2: forest plot per test ---------- */
  let forestTest = "position";
  function renderForest() {
    const W = width("#forest", 320), narrow = W < 560;
    const rowH = 22, gap = 18, left = narrow ? 112 : 150, right = 20, top = 26;
    const rows = []; let y = top;
    KINDS.forEach((k) => { y += gap * (rows.length ? 1 : 0); judges.filter((j) => j.kind === k).forEach((j, i) => { rows.push({ j, y, first: i === 0 }); y += rowH; }); });
    const H = y + 34;
    const svg = svgIn("#forest", W, H);
    const all = D.cells; const lo = d3.min(all, (c) => c.lo), hi = d3.max(all, (c) => c.hi);
    const x = d3.scaleLinear().domain([Math.min(-0.1, lo), Math.max(0.5, hi)]).range([left, W - right]).nice();
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - 26})`)
      .call(d3.axisBottom(x).ticks(narrow ? 5 : 8).tickFormat((d) => (d * 100).toFixed(0)).tickSize(0)).call((g) => g.select(".domain").remove());
    svg.append("text").attr("x", W - right).attr("y", H - 2).attr("text-anchor", "end").attr("class", "lbl-small").text("excess per 100 judgments");
    x.ticks(narrow ? 5 : 8).forEach((t) => svg.append("line").attr("class", "grid-line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", top - 12).attr("y2", H - 34));
    svg.append("line").attr("class", "ref-line").attr("x1", x(0.05)).attr("x2", x(0.05)).attr("y1", top - 14).attr("y2", H - 34);
    svg.append("text").attr("class", "ref-text").attr("x", x(0.05) + 4).attr("y", top - 6).text("margin 5");
    rows.forEach((r) => {
      const c = cell(r.j.id, forestTest), color = col(r.j.kind), sel = selectedJudge === r.j.id;
      const g = svg.append("g").attr("transform", `translate(0,${r.y})`);
      if (sel) g.append("rect").attr("x", 0).attr("y", -rowH / 2).attr("width", W).attr("height", rowH).attr("fill", css("--mark")).attr("opacity", 0.55);
      if (r.first) g.append("text").attr("class", "group-label").attr("x", 0).attr("y", -rowH / 2 - 4).text(KIND[r.j.kind].label);
      g.append("text").attr("x", left - 10).attr("y", 4).attr("text-anchor", "end").attr("class", sel ? "lbl-strong" : "lbl").text(r.j.name);
      const ci = g.append("line").attr("y1", 0).attr("y2", 0).attr("stroke", color).attr("stroke-width", 1.6).attr("stroke-linecap", "round")
        .attr("opacity", c.c === "none" ? 0.35 : 0.9);
      const size = c.c === "none" ? 26 : 70;
      const dot = g.append("path").attr("d", d3.symbol(KIND[r.j.kind].sym, size)())
        .attr("fill", c.c === "bias" ? color : c.c === "unclear" ? css("--card") : css("--rule"))
        .attr("stroke", c.c === "none" ? "none" : color).attr("stroke-width", 1.5);
      const hit = g.append("rect").attr("x", left).attr("y", -rowH / 2).attr("width", W - left - right).attr("height", rowH).attr("fill", "transparent")
        .on("mousemove", (ev) => showTip(`<b>${esc(r.j.name)}</b> · ${esc(testName(forestTest))}<br>excess <span class="tt-num">${per100(c.e)}</span> <span class="tt-num">[${per100(c.lo)}, ${per100(c.hi)}]</span><br>robustness rate <span class="tt-num">${c.rr.toFixed(2)}</span>, consistency rate <span class="tt-num">${c.cr.toFixed(2)}</span>`, ev))
        .on("mouseleave", hideTip);
      const prev = forestPrev[r.j.id];
      const set = (cc) => { ci.attr("x1", x(cc.lo)).attr("x2", x(cc.hi)); dot.attr("transform", `translate(${x(cc.e)},0)`); };
      if (prev && !reduceMotion) {
        set(prev);
        ci.transition().duration(600).ease(d3.easeCubicOut).attr("x1", x(c.lo)).attr("x2", x(c.hi));
        dot.transition().duration(600).ease(d3.easeCubicOut).attr("transform", `translate(${x(c.e)},0)`);
      } else set(c);
      forestPrev[r.j.id] = c;
      void hit;
    });
  }
  const forestPrev = {};

  /* ---------- Fig. 3: medians per test ---------- */
  function renderDumbbell() {
    const W = width("#dumbbell", 320), narrow = W < 560, left = narrow ? 130 : 170, right = 24, top = 10, rowH = 26;
    const rows = tests.map((t) => {
      const v = (k) => judges.filter((j) => j.kind === k).map((j) => cell(j.id, t.key).e);
      return { t, tdm: med(v("TDM")), l26: med(v("LLM 2026")), l24: d3.mean(v("LLM 2024")) };
    }).sort((a, b) => (b.tdm - b.l26) - (a.tdm - a.l26));
    const H = top + rows.length * rowH + 34;
    const svg = svgIn("#dumbbell", W, H);
    const x = d3.scaleLinear().domain([-0.02, d3.max(rows, (r) => Math.max(r.tdm, r.l26, r.l24)) + 0.03]).range([left, W - right]).nice();
    x.ticks(6).forEach((t) => svg.append("line").attr("class", "grid-line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", top).attr("y2", H - 30));
    svg.append("line").attr("class", "ref-line").attr("x1", x(0.05)).attr("x2", x(0.05)).attr("y1", top).attr("y2", H - 30);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - 26})`).call(d3.axisBottom(x).ticks(6).tickFormat((d) => (d * 100).toFixed(0)).tickSize(0)).call((g) => g.select(".domain").remove());
    svg.append("text").attr("x", W - right).attr("y", H - 2).attr("text-anchor", "end").attr("class", "lbl-small").text("median excess per 100 judgments");
    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2, g = svg.append("g");
      g.append("text").attr("x", left - 10).attr("y", y + 4).attr("text-anchor", "end").attr("class", "lbl").text(r.t.name + (r.t.ours ? " †" : ""));
      g.append("line").attr("x1", x(r.tdm)).attr("x2", x(r.l26)).attr("y1", y).attr("y2", y).attr("stroke", css("--muted")).attr("stroke-width", 1.2);
      [["LLM 2024", r.l24], ["LLM 2026", r.l26], ["TDM", r.tdm]].forEach(([k, v]) => {
        g.append("path").attr("d", d3.symbol(KIND[k].sym, 64)()).attr("transform", `translate(${x(v)},${y})`).attr("fill", col(k)).attr("stroke", css("--card")).attr("stroke-width", 1.5);
      });
      g.append("rect").attr("x", 0).attr("y", y - rowH / 2).attr("width", W).attr("height", rowH).attr("fill", "transparent")
        .on("mousemove", (ev) => showTip(`<b>${esc(r.t.name)}</b><br>TDMs <span class="tt-num">${per100(r.tdm)}</span> · LLM 2026 <span class="tt-num">${per100(r.l26)}</span> · LLM 2024 <span class="tt-num">${per100(r.l24)}</span>`, ev))
        .on("mouseleave", hideTip);
    });
  }

  /* ---------- Fig. 4: overlap ---------- */
  function renderOverlap() {
    const W = width("#overlap", 320), narrow = W < 560, left = narrow ? 140 : 210, right = 50;
    const names = { "typed - typed": "TDM and TDM", "typed - LLM 2026": "TDM and 2026 LLM judge", "LLM 2026 - LLM 2026": "2026 LLM and 2026 LLM judge" };
    const order = ["typed - typed", "typed - LLM 2026", "LLM 2026 - LLM 2026"];
    const rows = order.map((k) => D.overlap.find((o) => o.pair === k));
    const bh = 10, groupH = 40, H = rows.length * groupH + 30;
    const svg = svgIn("#overlap", W, H);
    const x = d3.scaleLinear().domain([0, 0.35]).range([left, W - right]);
    x.ticks(7).forEach((t) => svg.append("line").attr("class", "grid-line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", 0).attr("y2", H - 26));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - 22})`).call(d3.axisBottom(x).ticks(7).tickFormat(d3.format(".0%")).tickSize(0)).call((g) => g.select(".domain").remove());
    rows.forEach((r, i) => {
      const y = i * groupH + 6;
      svg.append("text").attr("x", left - 10).attr("y", y + bh + 4).attr("text-anchor", "end").attr("class", "lbl").text(names[r.pair]);
      [["flips", css("--ink-2"), r.flips], ["stays", css("--rule"), r.stays]].forEach(([k, c, v], m) => {
        const yy = y + m * (bh + 3);
        svg.append("rect").attr("x", left).attr("y", yy).attr("height", bh).attr("fill", c).attr("width", 0)
          .transition().duration(reduceMotion ? 0 : 700).delay(i * 80).attr("width", x(v) - left);
        svg.append("text").attr("x", x(v) + 6).attr("y", yy + bh - 3).attr("class", "tick-label").text(Math.round(v * 100) + "%");
      });
    });
    legend("#overlap-legend", [
      { html: `<i class="swatch" style="background:${css("--ink-2")}"></i>`, label: "second judge flips, where the first flips" },
      { html: `<i class="swatch" style="background:${css("--rule")}"></i>`, label: "second judge flips, where the first stays" }]);
  }

  /* ---------- Fig. 5: accuracy before and after ---------- */
  const SLOPES = [
    { key: "citation", label: "Fake citation", worse: "authority", better: "authority, better response" },
    { key: "padding", label: "Padding", worse: "verbosity", better: "verbosity, better response" },
    { key: "tone", label: "Emotional tone", worse: "sentiment, worse", better: "sentiment, better" },
  ];
  let slopeKey = "citation";
  function slopePanel(sel, testKey, title) {
    const W = width(sel, 280), H = 300, left = 40, right = 120, top = 30, bottom = 30;
    const svg = svgIn(sel, W, H);
    const rows = D.accuracy.filter((a) => a.test === testKey).map((a) => ({ ...a, j: jByName[a.judge] })).filter((a) => a.j);
    const y = d3.scaleLinear().domain([0.2, 1]).range([H - bottom, top]);
    const x0 = left, x1 = W - right;
    svg.append("text").attr("x", 0).attr("y", 14).attr("class", "lbl-strong").text(title);
    y.ticks(5).forEach((t) => {
      svg.append("line").attr("class", "grid-line").attr("x1", x0).attr("x2", x1).attr("y1", y(t)).attr("y2", y(t));
      svg.append("text").attr("class", "tick-label").attr("x", x0 - 8).attr("y", y(t) + 3).attr("text-anchor", "end").text(t.toFixed(1));
    });
    [["original", x0], ["with the cue", x1]].forEach(([t, xx]) => svg.append("text").attr("class", "lbl-small").attr("x", xx).attr("y", H - 8).attr("text-anchor", xx === x0 ? "start" : "end").text(t));
    rows.forEach((r) => {
      svg.append("line").attr("x1", x0).attr("x2", x1).attr("y1", y(r.before)).attr("y2", y(r.after)).attr("stroke", col(r.j.kind)).attr("stroke-width", 1).attr("opacity", 0.35)
        .on("mousemove", (ev) => showTip(`<b>${esc(r.judge)}</b><br>accuracy <span class="tt-num">${r.before.toFixed(2)} → ${r.after.toFixed(2)}</span>`, ev)).on("mouseleave", hideTip)
        .style("cursor", "pointer");
    });
    const meds = ["TDM", "LLM 2026"].map((k) => {
      const rs = rows.filter((r) => r.j.kind === k);
      return { k, b: med(rs.map((r) => r.before)), a: med(rs.map((r) => r.after)) };
    });
    const labels = [];
    meds.forEach((m) => {
      const line = svg.append("line").attr("x1", x0).attr("x2", x0).attr("y1", y(m.b)).attr("y2", y(m.b)).attr("stroke", col(m.k)).attr("stroke-width", 3).attr("stroke-linecap", "round");
      line.transition().duration(reduceMotion ? 0 : 800).attr("x2", x1).attr("y2", y(m.a));
      svg.append("circle").attr("cx", x0).attr("cy", y(m.b)).attr("r", 3.5).attr("fill", col(m.k));
      svg.append("circle").attr("cx", x1).attr("cy", y(m.a)).attr("r", 3.5).attr("fill", col(m.k));
      labels.push({ y: y(m.a), text: `${m.k === "TDM" ? "median TDM" : "median 2026 LLM"} ${m.b.toFixed(2)} → ${m.a.toFixed(2)}`, c: col(m.k) });
    });
    labels.sort((a, b) => a.y - b.y);
    if (labels.length === 2 && labels[1].y - labels[0].y < 14) { const mid = (labels[0].y + labels[1].y) / 2; labels[0].y = mid - 8; labels[1].y = mid + 8; }
    labels.forEach((l) => svg.append("text").attr("x", x1 + 8).attr("y", l.y + 4).attr("class", "lbl-small").style("fill", l.c).style("font-weight", 600).text(l.text));
  }
  function renderSlope() {
    const s = SLOPES.find((x) => x.key === slopeKey);
    slopePanel("#slope-worse", s.worse, "Cue on the worse answer");
    slopePanel("#slope-better", s.better, "Cue on the better answer");
  }

  /* ---------- Fig. 6: iceberg ---------- */
  function renderIceberg() {
    const W = width("#iceberg", 320), narrow = W < 640;
    const keys = Object.keys(D.hidden).sort((a, b) => (D.hidden[b].flip + D.hidden[b].hidden) - (D.hidden[a].flip + D.hidden[a].hidden));
    const top = 26, mid = 150, bottom = narrow ? 120 : 100, H = mid + 120 + bottom, left = 40, right = 10;
    const svg = svgIn("#iceberg", W, H);
    const x = d3.scaleBand().domain(keys).range([left, W - right]).padding(0.5);
    const y = d3.scaleLinear().domain([0, 0.32]).range([0, 120]);
    const c = col("TDM");
    [0.1, 0.2, 0.3].forEach((t) => {
      [mid - y(t), mid + y(t)].forEach((yy) => svg.append("line").attr("class", "grid-line").attr("x1", left).attr("x2", W - right).attr("y1", yy).attr("y2", yy));
      svg.append("text").attr("class", "tick-label").attr("x", left - 6).attr("y", mid - y(t) + 3).attr("text-anchor", "end").text(Math.round(t * 100) + "%");
      svg.append("text").attr("class", "tick-label").attr("x", left - 6).attr("y", mid + y(t) + 3).attr("text-anchor", "end").text(Math.round(t * 100) + "%");
    });
    svg.append("text").attr("class", "lbl-small").attr("x", left).attr("y", top - 10).text("↑ flips");
    keys.forEach((k, i) => {
      const v = D.hidden[k], bw = Math.min(30, x.bandwidth()), xx = x(k) + (x.bandwidth() - bw) / 2;
      svg.append("rect").attr("x", xx).attr("width", bw).attr("y", mid).attr("height", 0).attr("fill", c)
        .transition().duration(reduceMotion ? 0 : 700).delay(i * 40).attr("y", mid - y(v.flip)).attr("height", y(v.flip));
      svg.append("rect").attr("x", xx).attr("width", bw).attr("y", mid + 2).attr("height", 0).attr("fill", c).attr("fill-opacity", 0.3)
        .attr("stroke", c).attr("stroke-width", 0.8)
        .transition().duration(reduceMotion ? 0 : 700).delay(i * 40 + 120).attr("height", y(v.hidden));
      svg.append("rect").attr("x", xx - 3).attr("width", bw + 6).attr("y", mid - 120).attr("height", 240).attr("fill", "transparent")
        .on("mousemove", (ev) => showTip(`<b>${esc(testName(k))}</b><br>flips <span class="tt-num">${Math.round(v.flip * 100)}%</span><br>hidden shifts <span class="tt-num">${Math.round(v.hidden * 100)}%</span>`, ev))
        .on("mouseleave", hideTip);
      const tx = xx + bw / 2, ty = mid + 120 + 18;
      svg.append("text").attr("class", "lbl-small").attr("transform", `translate(${tx},${ty}) rotate(${narrow ? -55 : -32})`).attr("text-anchor", "end").text(testName(k));
    });
    svg.append("line").attr("x1", left).attr("x2", W - right).attr("y1", mid + 1).attr("y2", mid + 1).attr("stroke", css("--ink")).attr("stroke-width", 1.4);
    legend("#iceberg-legend", [
      { html: `<i class="swatch" style="background:${c}"></i>`, label: "above the line: the verdict flips" },
      { html: `<i class="swatch" style="background:${c};opacity:.3;outline:1px solid ${c}"></i>`, label: "below: the verdict stays, but the probability moves by more than 0.10" }]);
  }

  /* ---------- Fig. 7: confidence curve ---------- */
  let curveKey = "others";
  function renderCurve() {
    const W = width("#curve", 320), H = 300, left = 46, right = 20, top = 16, bottom = 42;
    const svg = svgIn("#curve", W, H);
    const x = d3.scaleLinear().domain([0.5, 1]).range([left, W - right]);
    const y = d3.scaleLinear().domain([0, 0.6]).range([H - bottom, top]);
    y.ticks(6).forEach((t) => {
      svg.append("line").attr("class", "grid-line").attr("x1", left).attr("x2", W - right).attr("y1", y(t)).attr("y2", y(t));
      svg.append("text").attr("class", "tick-label").attr("x", left - 8).attr("y", y(t) + 3).attr("text-anchor", "end").text(Math.round(t * 100) + "%");
    });
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - bottom + 6})`).call(d3.axisBottom(x).ticks(5).tickSize(0)).call((g) => g.select(".domain").remove());
    svg.append("text").attr("x", W - right).attr("y", H - 4).attr("text-anchor", "end").attr("class", "lbl-small").text("probability the TDM gave its choice on the original prompt");
    const pts = (k) => (D.curves[k] || []).map((b) => ({ x: (b.lo + b.hi) / 2, y: b.flip, n: b.n, lo: b.lo, hi: b.hi }));
    const line = d3.line().x((d) => x(d.x)).y((d) => y(d.y)).curve(d3.curveMonotoneX);
    if (curveKey !== "others") svg.append("path").attr("d", line(pts("others"))).attr("fill", "none").attr("stroke", css("--muted")).attr("stroke-width", 1.4).attr("opacity", 0.6);
    const p = pts(curveKey), c = col("TDM");
    const path = svg.append("path").attr("d", line(p)).attr("fill", "none").attr("stroke", c).attr("stroke-width", 2.4);
    if (!reduceMotion) { const L = path.node().getTotalLength(); path.attr("stroke-dasharray", `${L} ${L}`).attr("stroke-dashoffset", L).transition().duration(900).attr("stroke-dashoffset", 0); }
    p.forEach((d) => {
      svg.append("circle").attr("cx", x(d.x)).attr("cy", y(d.y)).attr("r", 4).attr("fill", c).attr("stroke", css("--card")).attr("stroke-width", 2)
        .on("mousemove", (ev) => showTip(`confidence <span class="tt-num">${d.lo.toFixed(2)} to ${d.hi.toFixed(2)}</span><br>flipped <span class="tt-num">${Math.round(d.y * 100)}%</span> of <span class="tt-num">${d.n.toLocaleString("en")}</span> verdicts`, ev))
        .on("mouseleave", hideTip);
    });
    if (curveKey !== "others") { const o = pts("others"), last = o[o.length - 1]; svg.append("text").attr("x", x(last.x)).attr("y", y(last.y) - 8).attr("text-anchor", "end").attr("class", "lbl-small").text("other tests"); }
  }

  /* ---------- Fig. 8: cost against bias ---------- */
  function renderCost() {
    const W = width("#cost", 320), H = 330, left = 46, right = 20, top = 18, bottom = 44;
    const svg = svgIn("#cost", W, H);
    const x = d3.scaleLog().domain([0.01, 12]).range([left, W - right]);
    const y = d3.scaleLinear().domain([0, 9]).range([H - bottom, top]);
    [0.01, 0.1, 1, 10].forEach((t) => {
      svg.append("line").attr("class", "grid-line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", top).attr("y2", H - bottom);
      svg.append("text").attr("class", "tick-label").attr("x", x(t)).attr("y", H - bottom + 16).attr("text-anchor", "middle").text("$" + t);
    });
    d3.range(0, 10, 2).forEach((t) => {
      svg.append("line").attr("class", "grid-line").attr("x1", left).attr("x2", W - right).attr("y1", y(t)).attr("y2", y(t));
      svg.append("text").attr("class", "tick-label").attr("x", left - 8).attr("y", y(t) + 3).attr("text-anchor", "end").text(t);
    });
    svg.append("text").attr("x", W - right).attr("y", H - 6).attr("text-anchor", "end").attr("class", "lbl-small").text("US$ per 1,000 judgments (log scale)");
    svg.append("text").attr("x", left).attr("y", top - 6).attr("class", "lbl-small").text("tests with a bias (of 12)");
    const pts = judges.map((j) => ({ j, px: x(j.usd1000), py: y(j.nBiased) }));
    pts.sort((a, b) => a.px - b.px);
    pts.forEach((p, i) => { for (let k = 0; k < i; k++) { const q = pts[k]; if (Math.abs(q.px - p.px) < 9 && Math.abs(q.py - p.py) < 9) p.py -= 9; } });
    const placed = [];
    pts.forEach((p) => {
      const j = p.j;
      svg.append("path").attr("d", d3.symbol(KIND[j.kind].sym, 80)()).attr("transform", `translate(${p.px},${p.py})`).attr("fill", col(j.kind)).attr("stroke", css("--card")).attr("stroke-width", 1.5)
        .style("cursor", "pointer")
        .on("mousemove", (ev) => showTip(`<b>${esc(j.name)}</b><br>US$<span class="tt-num">${j.usd1000.toFixed(3)}</span> per 1,000 judgments<br>median latency <span class="tt-num">${j.latency.toFixed(2)}</span> s<br>repeats <span class="tt-num">${(j.crAl * 100).toFixed(1)}%</span> of its verdicts<br><span class="tt-num">${j.nBiased}</span> of 12 tests with a bias`, ev))
        .on("mouseleave", hideTip);
      const lw = j.name.length * 6.1;
      let ly = p.py + 4, lx = p.px + 8;
      if (lx + lw > W - right) lx = p.px - 8 - lw;
      for (let tries = 0; tries < 6 && placed.some((q) => Math.abs(q.y - ly) < 12 && lx < q.x + q.w && lx + lw > q.x); tries++) ly += 12;
      placed.push({ x: lx, y: ly, w: lw });
      svg.append("text").attr("x", lx).attr("y", ly).attr("class", "lbl-small").text(j.name);
    });
  }

  /* ---------- appendix minis ---------- */
  function renderDose() {
    const W = width("#dose", 280), H = 220, left = 34, right = 16, top = 10, bottom = 30;
    const svg = svgIn("#dose", W, H);
    const x = d3.scalePoint().domain([60, 70, 80, 90]).range([left, W - right]);
    const y = d3.scaleLinear().domain([-0.05, 0.4]).range([H - bottom, top]);
    y.ticks(5).forEach((t) => { svg.append("line").attr("class", "grid-line").attr("x1", left).attr("x2", W - right).attr("y1", y(t)).attr("y2", y(t)); svg.append("text").attr("class", "tick-label").attr("x", left - 6).attr("y", y(t) + 3).attr("text-anchor", "end").text(Math.round(t * 100)); });
    [60, 70, 80, 90].forEach((p) => svg.append("text").attr("class", "tick-label").attr("x", x(p)).attr("y", H - bottom + 16).attr("text-anchor", "middle").text(p + "%"));
    svg.append("text").attr("class", "lbl-small").attr("x", W - right).attr("y", H - 2).attr("text-anchor", "end").text("claimed majority for the worse answer");
    const line = d3.line().x((d) => x(d.pct)).y((d) => y(d.e));
    const byJ = d3.group(D.bandwagon, (d) => d.judge);
    const paths = [];
    byJ.forEach((rows, id) => {
      const j = jById[id]; rows.sort((a, b) => a.pct - b.pct);
      const p = svg.append("path").attr("d", line(rows)).attr("fill", "none").attr("stroke", col(j.kind)).attr("stroke-width", 1.4).attr("opacity", 0.75);
      paths.push({ p, j, rows });
    });
    svg.append("rect").attr("x", left).attr("y", top).attr("width", W - left - right).attr("height", H - top - bottom).attr("fill", "transparent")
      .on("mousemove", (ev) => {
        const [mx, my] = d3.pointer(ev);
        let best = null, bd = 1e9;
        paths.forEach((o) => o.rows.forEach((r) => { const dd = Math.hypot(x(r.pct) - mx, y(r.e) - my); if (dd < bd) { bd = dd; best = o; } }));
        paths.forEach((o) => o.p.attr("opacity", o === best ? 1 : 0.2).attr("stroke-width", o === best ? 2.6 : 1.2));
        showTip(`<b>${esc(best.j.name)}</b><br>${best.rows.map((r) => `${r.pct}%: <span class="tt-num">${per100(r.e)}</span>`).join(" · ")}`, ev);
      })
      .on("mouseleave", () => { paths.forEach((o) => o.p.attr("opacity", 0.75).attr("stroke-width", 1.4)); hideTip(); });
    legend("#dose-legend", kindLegend());
  }
  function renderEce() {
    const W = width("#ece", 280), H = 230, left = 40, right = 20, top = 14, bottom = 34;
    const svg = svgIn("#ece", W, H);
    const pts = tdms.map((j) => ({ j, x: j.eceAl, y: j.meanExcess }));
    const x = d3.scaleLinear().domain([0.04, 0.28]).range([left, W - right]);
    const y = d3.scaleLinear().domain([0.06, 0.15]).range([H - bottom, top]);
    x.ticks(5).forEach((t) => { svg.append("line").attr("class", "grid-line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", top).attr("y2", H - bottom); svg.append("text").attr("class", "tick-label").attr("x", x(t)).attr("y", H - bottom + 14).attr("text-anchor", "middle").text(t.toFixed(2)); });
    y.ticks(4).forEach((t) => { svg.append("line").attr("class", "grid-line").attr("x1", left).attr("x2", W - right).attr("y1", y(t)).attr("y2", y(t)); svg.append("text").attr("class", "tick-label").attr("x", left - 6).attr("y", y(t) + 3).attr("text-anchor", "end").text(Math.round(t * 100)); });
    svg.append("text").attr("class", "lbl-small").attr("x", W - right).attr("y", H - 2).attr("text-anchor", "end").text("calibration error (ECE), alignment set");
    svg.append("text").attr("class", "lbl-small").attr("x", left).attr("y", top - 4).text("mean excess");
    const labs = pts.map((p) => ({ p, x: x(p.x), y: y(p.y) + 4, right: x(p.x) > W - 100 })).sort((a, b) => a.y - b.y);
    for (let k = 1; k < labs.length; k++) for (let m = 0; m < k; m++) {
      const a = labs[m], b = labs[k];
      if (Math.abs(a.x - b.x) < 90 && b.y - a.y < 12) b.y = a.y + 12;
    }
    labs.forEach((l) => {
      const p = l.p, cx = x(p.x), cy = y(p.y);
      svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 4.5).attr("fill", col("TDM")).attr("stroke", css("--card")).attr("stroke-width", 1.5)
        .on("mousemove", (ev) => showTip(`<b>${esc(p.j.name)}</b><br>ECE <span class="tt-num">${p.x.toFixed(3)}</span>, mean excess <span class="tt-num">${per100(p.y)}</span>`, ev)).on("mouseleave", hideTip);
      const lx = cx + (l.right ? -8 : 8);
      if (Math.abs(l.y - 4 - cy) > 3) svg.append("line").attr("x1", cx + (l.right ? -4 : 4)).attr("y1", cy).attr("x2", lx).attr("y2", l.y - 4).attr("stroke", css("--rule")).attr("stroke-width", 0.8);
      svg.append("text").attr("x", lx + (l.right ? -2 : 2)).attr("y", l.y).attr("text-anchor", l.right ? "end" : "start").attr("class", "lbl-small").text(p.j.name);
    });
  }
  function renderIdentity() {
    const groups = [...new Set(D.identity.map((d) => d.group))];
    const neutral = { TDM: med(tdms.map((j) => cell(j.id, "diversity, neutral note").e)),
      "LLM 2026": med(judges.filter((j) => j.kind === "LLM 2026").map((j) => cell(j.id, "diversity, neutral note").e)) };
    const rows = [...groups.map((g) => ({ g, v: Object.fromEntries(D.identity.filter((d) => d.group === g).map((d) => [d.kind, d.e])) })),
      { g: "Neutral note", v: neutral, neutral: true }];
    const W = width("#identity", 280), rowH = 20, left = 120, right = 20, H = rows.length * rowH + 30;
    const svg = svgIn("#identity", W, H);
    const x = d3.scaleLinear().domain([0, 0.06]).range([left, W - right]);
    x.ticks(6).forEach((t) => svg.append("line").attr("class", "grid-line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", 0).attr("y2", H - 26));
    rows.forEach((r, i) => {
      const y = i * rowH + 10;
      if (r.neutral) svg.append("line").attr("x1", 0).attr("x2", W).attr("y1", y - rowH / 2).attr("y2", y - rowH / 2).attr("stroke", css("--rule"));
      svg.append("text").attr("x", left - 8).attr("y", y + 4).attr("text-anchor", "end").attr("class", r.neutral ? "lbl-strong" : "lbl").text(r.g);
      ["LLM 2026", "TDM"].forEach((k) => svg.append("path").attr("d", d3.symbol(KIND[k].sym, 54)()).attr("transform", `translate(${x(r.v[k])},${y})`).attr("fill", col(k)).attr("stroke", css("--card")).attr("stroke-width", 1.2)
        .on("mousemove", (ev) => showTip(`<b>${esc(r.g)}</b> · median ${esc(KIND[k].label)}<br>excess <span class="tt-num">${(r.v[k] * 100).toFixed(1)}</span> per 100`, ev)).on("mouseleave", hideTip));
    });
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - 22})`).call(d3.axisBottom(x).ticks(6).tickFormat((d) => (d * 100).toFixed(0)).tickSize(0)).call((g) => g.select(".domain").remove());
    svg.append("text").attr("class", "lbl-small").attr("x", W - right).attr("y", H - 2).attr("text-anchor", "end").text("median excess per 100, both answers pooled");
    legend("#identity-legend", kindLegend(["TDM", "LLM 2026"]));
  }
  function renderFormats() {
    const rows = tdms.map((j) => ({ j, v: j.layNative })).sort((a, b) => b.v - a.v);
    const W = width("#formats", 280), rowH = 20, left = 120, right = 44, H = rows.length * rowH + 26;
    const svg = svgIn("#formats", W, H);
    const x = d3.scaleLinear().domain([0, 0.12]).range([left, W - right]);
    rows.forEach((r, i) => {
      const y = i * rowH;
      svg.append("text").attr("x", left - 8).attr("y", y + 13).attr("text-anchor", "end").attr("class", "lbl").text(r.j.name);
      svg.append("rect").attr("x", left).attr("y", y + 4).attr("height", 12).attr("fill", col("TDM")).attr("width", 0)
        .transition().duration(reduceMotion ? 0 : 700).delay(i * 50).attr("width", x(r.v) - left);
      svg.append("text").attr("x", x(r.v) + 6).attr("y", y + 14).attr("class", "tick-label").text((r.v * 100).toFixed(1) + "%");
    });
    svg.append("text").attr("class", "lbl-small").attr("x", W - right).attr("y", H - 2).attr("text-anchor", "end").text("verdicts that change when the text moves into fields");
  }
  function renderOrders() {
    const rows = D.bothOrders.map((r) => ({ ...r, j: jById[r.judge] })).sort((a, b) => b.change - a.change);
    const W = width("#orders", 280), rowH = 20, left = 120, right = 20, H = rows.length * rowH + 34;
    const svg = svgIn("#orders", W, H);
    const x = d3.scaleLinear().domain([-0.04, 0.07]).range([left, W - right]);
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 0).attr("y2", H - 30).attr("stroke", css("--ink-2")).attr("stroke-width", 1);
    rows.forEach((r, i) => {
      const y = i * rowH + 10;
      svg.append("text").attr("x", left - 8).attr("y", y + 4).attr("text-anchor", "end").attr("class", "lbl").text(r.j.name);
      svg.append("line").attr("x1", x(r.lo)).attr("x2", x(r.hi)).attr("y1", y).attr("y2", y).attr("stroke", col("TDM")).attr("stroke-width", 1.5).attr("opacity", 0.7);
      svg.append("circle").attr("cx", x(r.change)).attr("cy", y).attr("r", 4).attr("fill", r.lo > 0 ? col("TDM") : css("--card")).attr("stroke", col("TDM")).attr("stroke-width", 1.5)
        .on("mousemove", (ev) => showTip(`<b>${esc(r.j.name)}</b><br>accuracy <span class="tt-num">${r.one.toFixed(3)} → ${r.both.toFixed(3)}</span><br>change <span class="tt-num">${signed(r.change * 100, 1)}</span> points <span class="tt-num">[${signed(r.lo * 100, 1)}, ${signed(r.hi * 100, 1)}]</span>`, ev)).on("mouseleave", hideTip);
    });
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - 26})`).call(d3.axisBottom(x).ticks(5).tickFormat((d) => signed(d * 100, 0)).tickSize(0)).call((g) => g.select(".domain").remove());
    svg.append("text").attr("class", "lbl-small").attr("x", W - right).attr("y", H - 2).attr("text-anchor", "end").text("change in accuracy, points");
  }

  /* ---------- § 4: the example browser ---------- */
  let exKey = "sentiment";
  function segHTML(seg) { return seg.map(([op, t]) => op === "ins" ? `<ins>${esc(t)}</ins>` : op === "del" ? `<del>${esc(t)}</del>` : esc(t)).join(" "); }
  function renderExample() {
    const ex = D.examples.find((e) => e.test === exKey), t = tByKey[exKey], fallacy = exKey === "fallacy";
    $("#ex-what").innerHTML = `<b>${esc(t.name)}${t.ours ? " (our addition)" : ""}.</b> ${esc(t.what)} Source of this item: ${esc(ex.source.replace(/^.*\//, ""))}.`;
    const tagFor = (isR1, after) => isR1 ? (fallacy && after ? '<span class="better">reasoning broken</span>' : '<span class="better">better</span>') : (fallacy && after ? '<span class="better">now better</span>' : "");
    const col1 = ["A", "B"].map((s) => `
      <div class="ex-ans"><div class="ex-ans-head"><span>${esc(ex.orig[s].label)}</span>${tagFor(ex.orig[s].r1, false)}</div>
      <div class="ex-text">${esc(ex.orig[s].text)}</div></div>`).join("");
    const col2 = ["A", "B"].map((s) => `
      <div class="ex-ans"><div class="ex-ans-head"><span>${esc(ex.pert[s].label)}</span>${tagFor(ex.pert[s].r1, true)}</div>
      <div class="ex-text">${segHTML(ex.pert[s].seg)}</div></div>`).join("");
    $("#ex-grid").innerHTML = `
      <div class="ex-col"><h4>Original prompt</h4><div class="ex-q">${esc(ex.question)}</div>${col1}${ex.orig.note ? `<div class="ex-note">${esc(ex.orig.note)}</div>` : ""}</div>
      <div class="ex-col"><h4>With the cue</h4><div class="ex-q">${esc(ex.question)}</div>${col2}${ex.pert.note ? `<div class="ex-note">${esc(ex.pert.note)}</div>` : ""}</div>`;
    // TDM arrows
    const W = width("#ex-tdm", 280), rowH = 24, left = 120, right = 16, H = ex.tdm.length * rowH + 30;
    const svg = svgIn("#ex-tdm", W, H);
    const x = d3.scaleLinear().domain([0, 1]).range([left, W - right]);
    svg.append("line").attr("x1", x(0.5)).attr("x2", x(0.5)).attr("y1", 0).attr("y2", H - 26).attr("stroke", css("--accent")).attr("stroke-width", 1);
    svg.append("text").attr("class", "ref-text").attr("x", x(0.5) + 4).attr("y", 9).text("0.5");
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => svg.append("text").attr("class", "tick-label").attr("x", x(t)).attr("y", H - 10).attr("text-anchor", "middle").text(t));
    const defs = svg.append("defs");
    ex.tdm.forEach((r, i) => {
      const y = i * rowH + 14, j = jById[r.judge], flip = (r.before > 0.5) !== (r.after > 0.5), c = flip ? css("--accent") : css("--ink-2");
      const id = "ah" + i;
      defs.append("marker").attr("id", id).attr("viewBox", "0 0 10 10").attr("refX", 8).attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto-start-reverse")
        .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", c);
      svg.append("text").attr("x", left - 8).attr("y", y + 4).attr("text-anchor", "end").attr("class", flip ? "lbl-strong" : "lbl").text(j.name);
      svg.append("line").attr("class", "grid-line").attr("x1", left).attr("x2", W - right).attr("y1", y).attr("y2", y);
      svg.append("circle").attr("cx", x(r.before)).attr("cy", y).attr("r", 3.5).attr("fill", css("--card")).attr("stroke", c).attr("stroke-width", 1.4);
      const ln = svg.append("line").attr("x1", x(r.before)).attr("x2", x(r.before)).attr("y1", y).attr("y2", y).attr("stroke", c).attr("stroke-width", 1.8);
      if (Math.abs(x(r.after) - x(r.before)) > 6) ln.attr("marker-end", `url(#${id})`);
      ln.transition().duration(reduceMotion ? 0 : 700).delay(i * 60).attr("x2", x(r.after));
      svg.append("rect").attr("x", left).attr("y", y - rowH / 2).attr("width", W - left - right).attr("height", rowH).attr("fill", "transparent")
        .on("mousemove", (ev) => showTip(`<b>${esc(j.name)}</b><br>P(${fallacy ? "R1" : "better"}) <span class="tt-num">${r.before.toFixed(2)} → ${r.after.toFixed(2)}</span>${flip ? "<br>verdict flips" : ""}`, ev)).on("mouseleave", hideTip);
    });
    const mark = (v) => v === true ? '<span class="ok">✓</span>' : v === false ? '<span class="bad">✗</span>' : '<span class="nov">none</span>';
    $("#ex-llm").innerHTML = ex.llm.map((r) => `<tr><td>${esc(jById[r.judge].name)}</td><td>${mark(r.before)} → ${mark(r.after)}</td></tr>`).join("");
  }

  /* ---------- slider ---------- */
  const SLIDES = [
    { img: "assets/fig/excess_heatmap.png", title: "Every judge, every test", text: "The excess of each judge in each test: verdicts changed per 100 beyond those a plain repetition changes. Framed cells are biases, whose whole 95% interval lies above 5. On the right, the number of tests with a bias per judge. Figure 1 of the paper." },
    { img: "assets/fig/probabilities.png", title: "What the probabilities show", text: "(a) How far each cue moves the probability of the response it targets. (b) Prompts on which the median TDM's verdict flips, or stays while its probability moves by more than 0.10. (c) Flipped verdicts by the confidence on the original prompt. Figure 2 of the paper." },
    { img: "assets/fig/versions.png", title: "Results per version", text: "Robustness rate per version of a test: bandwagon by the claimed majority, authority by the kind of citation and the response it is on, sentiment by tone and response, and diversity by identity group. From the appendix." },
    { img: "assets/fig/flips_by_margin.png", title: "Flips and confidence", text: "Share of flipped verdicts by the probability that the judge gave its choice on the original prompt, all TDMs pooled. The band shows the range of the eight models for the tests other than sentiment and authority (on better). From the appendix." },
    { img: "assets/fig/reliability.png", title: "Reliability diagrams", text: "Each point is one bin of the confidence with its share of correct verdicts, on the original alignment prompts. Points below the diagonal are overconfident, and the marker area grows with the number of prompts in the bin. From the appendix." },
    { img: "assets/fig/calibration_summary.png", title: "Two measures of calibration", text: "Expected calibration error and Brier score of the eight TDMs on the original prompts, alignment set (filled) and fact-related set (open). Lower is better. From the appendix." },
  ];
  function initSlider() {
    const track = $("#track"), dots = $("#dots");
    track.innerHTML = SLIDES.map((s, i) => `
      <div class="slide" role="group" aria-roledescription="slide" aria-label="${i + 1} of ${SLIDES.length}">
        <div class="slide-img"><img src="${s.img}" alt="${esc(s.title)}" loading="lazy"></div>
        <div class="slide-cap"><h4>${esc(s.title)}</h4><p>${esc(s.text)}</p></div>
      </div>`).join("");
    dots.innerHTML = SLIDES.map((s, i) => `<button class="dot" type="button" aria-label="Figure ${i + 1}: ${esc(s.title)}"></button>`).join("");
    let i = 0;
    const go = (k) => {
      i = (k + SLIDES.length) % SLIDES.length;
      track.style.transform = `translateX(${-100 * i}%)`;
      dots.querySelectorAll(".dot").forEach((d, m) => d.setAttribute("aria-current", String(m === i)));
      $("#count").textContent = `${i + 1} / ${SLIDES.length}`;
    };
    $("#prev").addEventListener("click", () => go(i - 1));
    $("#next").addEventListener("click", () => go(i + 1));
    dots.querySelectorAll(".dot").forEach((d, m) => d.addEventListener("click", () => go(m)));
    $("#slider").addEventListener("keydown", (e) => { if (e.key === "ArrowLeft") go(i - 1); if (e.key === "ArrowRight") go(i + 1); });
    let sx = null;
    track.addEventListener("pointerdown", (e) => { sx = e.clientX; });
    track.addEventListener("pointerup", (e) => { if (sx != null && Math.abs(e.clientX - sx) > 40) go(i + (e.clientX < sx ? 1 : -1)); sx = null; });
    go(0);
  }

  /* ---------- theme, nav, reveal, copy ---------- */
  function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem("theme"); } catch (e) { saved = null; }
    const asked = (location.search.match(/[?&]theme=(light|dark)\b/) || [])[1];      // ?theme=dark for a link or a screenshot
    if (asked) saved = asked;
    if (saved === "light" || saved === "dark") document.documentElement.setAttribute("data-theme", saved);
    $("#theme-toggle").addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      const next = cur === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* storage unavailable */ }
      renderAll();
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => renderAll());
  }
  function initNav() {
    const nav = $("#nav");
    const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 10);
    window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
    const links = [...document.querySelectorAll(".nav-links a")];
    const io = new IntersectionObserver((entries) => entries.forEach((en) => {
      if (en.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + en.target.id));
    }), { rootMargin: "-40% 0px -55% 0px" });
    document.querySelectorAll("main section[id]").forEach((s) => io.observe(s));
  }
  function initReveal() {
    const els = document.querySelectorAll(".reveal");
    if (reduceMotion || !("IntersectionObserver" in window)) { els.forEach((e) => e.classList.add("in")); return; }
    const io = new IntersectionObserver((entries) => entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } }), { rootMargin: "0px 0px -8% 0px" });
    els.forEach((e) => io.observe(e));
  }
  function initCopy() {
    $("#copy").addEventListener("click", async () => {
      const text = $("#bibtex").innerText;
      try { await navigator.clipboard.writeText(text); }
      catch (e) { const r = document.createRange(); r.selectNodeContents($("#bibtex")); const s = window.getSelection(); s.removeAllRanges(); s.addRange(r); }
      $("#copy").textContent = "Copied"; setTimeout(() => { $("#copy").textContent = "Copy"; }, 1600);
    });
  }

  /* ---------- wiring ---------- */
  function renderAll() {
    renderScale(); renderPrints(); renderForest(); renderDumbbell(); renderOverlap(); renderSlope();
    renderIceberg(); renderCurve(); renderCost(); renderDose(); renderEce(); renderIdentity(); renderFormats(); renderOrders(); renderExample();
    legend("#forest-legend", [...kindLegend(), { html: '<i class="lg-dot"></i>', label: "bias" }, { html: '<i class="lg-ring"></i>', label: "unclear" }, { html: '<i class="lg-faint"></i>', label: "no bias" }]);
    legend("#dumbbell-legend", kindLegend());
    legend("#slope-legend", kindLegend());
    legend("#cost-legend", kindLegend());
  }
  function init() {
    renderTests();
    initTheme(); initNav(); initReveal(); initCopy(); initSlider();
    const testItems = tests.map((t) => ({ key: t.key, label: t.name, ours: t.ours }));
    chips("#forest-chips", testItems, forestTest, (k) => { forestTest = k; renderForest(); });
    chips("#slope-chips", SLOPES, slopeKey, (k) => { slopeKey = k; renderSlope(); });
    chips("#curve-chips", [{ key: "others", label: "All but sentiment and authority (on better)" }, ...testItems.filter((t) => t.key !== "fallacy")], curveKey, (k) => { curveKey = k; renderCurve(); });
    chips("#ex-chips", testItems, exKey, (k) => { exKey = k; renderExample(); });
    initScale();
    renderAll();
    let rt = null;
    window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(renderAll, 180); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
