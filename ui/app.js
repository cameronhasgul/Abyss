"use strict";

// Replays executed mission logs. It draws what the logs record and never computes
// scientific results of its own. Route interpolation between recorded cells is visual only.
// Both panels share one clock in energy units, so they are always compared at equal spend.
// Reveal files are not fetched until the viewer asks for the reveal.

const PANE_COLORS = ["#f5b544", "#6ee7c7"];
const ISOBATH_STEP_M = 250;
const SCENARIO_NAMES = {
  random: "Independent random",
  clustered: "Spatial clusters",
  ridge: "Ridge-associated",
  reversed: "Flat-associated (reversed)",
  nonlinear: "Nonlinear mixed",
};
const TERRAIN_NAMES = {
  gebco_2025: "Real bathymetry: GEBCO 2025, Monterey Canyon area, 30 x 30 planning grid.",
  synthetic: "Synthetic procedural terrain.",
};
const PERCEPTION_NAMES = {
  ideal: "Ideal perception: labels come directly from the simulator.",
  stub: "Stub perception: a fake model with made-up error rates.",
};
const GEBCO_CREDIT =" GEBCO Compilation Group (2025) GEBCO 2025 Grid (doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29).";
const LAYER_LABELS = {
  predicted_richness: "Predicted richness",
  uncertainty: "Model uncertainty",
  hidden: "Hidden expected richness (simulated)",
};
const UTILITY_LABELS = {
  expected_richness: "Predicted richness",
  information_gain: "Information gain",
  coverage: "Coverage incentive",
  energy_cost: "Energy cost",
  utility: "Utility",
};

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const fmt = (v, digits = 1) => Number(v).toFixed(digits);

const app = {
  entries: [],
  worldId: null,
  panes: [],
  energy: 0,
  maxEnergy: 150,
  budget: 150,
  playing: false,
  revealed: false,
  layer: "terrain",
  lastFrame: 0,
  loadToken: 0,
  tourIndex: -1,
  benchmark: null,
};

async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

function showNotice(message, isError = false) {
  const notice = $("notice");
  notice.textContent = message;
  notice.hidden = !message;
  notice.classList.toggle("error", isError);
}

// Timeline -------------------------------------------------------------------

function buildTimeline(log) {
  let energy = 0;
  return log.steps.map((step) => {
    const start = energy;
    const arrive = start + step.energy_travel;
    const end = step.cumulative.energy_used;
    energy = end;
    return { step, start, arrive, end };
  });
}

function missionAt(pane, energy) {
  const completed = [];
  let current = null;
  for (const segment of pane.timeline) {
    if (segment.end <= energy + 1e-6) completed.push(segment);
    else { current = segment; break; }
  }
  const home = pane.log.terrain.home;
  const last = completed[completed.length - 1];
  let position = last ? last.step.position : home;
  const trail = completed.flatMap((s) => s.step.route);
  let phase = last ? "idle" : "start";

  if (current && energy > current.start) {
    const route = current.step.route;
    const travel = current.arrive - current.start;
    const progress = travel > 0 ? clamp((energy - current.start) / travel, 0, 1) : 1;
    const along = progress * (route.length - 1);
    const index = Math.floor(along);
    const a = route[index];
    const b = route[Math.min(index + 1, route.length - 1)];
    const f = along - index;
    position = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
    trail.push(...route.slice(0, index + 1), position);
    phase = progress < 1 ? "travel" : "survey";
  }
  if (!current) phase = "done";
  return { completed, current, position, trail, phase };
}

// Colour ---------------------------------------------------------------------

function ramp(stops) {
  return (t) => {
    const x = clamp(t, 0, 1) * (stops.length - 1);
    const i = Math.min(Math.floor(x), stops.length - 2);
    const f = x - i;
    return stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  };
}

const DEPTH_RAMP = ramp([[126, 182, 186], [58, 128, 150], [28, 84, 118], [14, 48, 84], [6, 24, 50]]);
const BELIEF_RAMP = ramp([[12, 30, 60], [34, 88, 130], [60, 160, 150], [190, 220, 120], [250, 245, 190]]);
const HIDDEN_RAMP = ramp([[24, 14, 44], [96, 30, 90], [190, 56, 90], [245, 130, 80], [252, 225, 160]]);

function gridExtent(grid) {
  const values = grid.flat();
  return [Math.min(...values), Math.max(...values)];
}

function fieldImage(grid, color, shade) {
  const rows = grid.length;
  const cols = grid[0].length;
  const canvas = document.createElement("canvas");
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(cols, rows);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const rgb = color(grid[r][c]);
      const light = shade ? shade(r, c) : 1;
      const i = (r * cols + c) * 4;
      image.data[i] = clamp(rgb[0] * light, 0, 255);
      image.data[i + 1] = clamp(rgb[1] * light, 0, 255);
      image.data[i + 2] = clamp(rgb[2] * light, 0, 255);
      image.data[i + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

function terrainImage(terrain) {
  const depth = terrain.depth_m;
  const [lo, hi] = gridExtent(depth);
  const rows = depth.length;
  const cols = depth[0].length;
  const at = (r, c) => depth[clamp(r, 0, rows - 1)][clamp(c, 0, cols - 1)];
  // Hillshade lit from the north-west. Elevation is the negative of depth.
  const shade = (r, c) => {
    const dx = -(at(r, c + 1) - at(r, c - 1));
    const dy = -(at(r + 1, c) - at(r - 1, c));
    return clamp(1 + (dy - dx) / (0.35 * (hi - lo || 1)), 0.6, 1.5);
  };
  return fieldImage(depth, (d) => DEPTH_RAMP((d - lo) / (hi - lo || 1)), shade);
}

function isobaths(depth) {
  const [lo, hi] = gridExtent(depth);
  const segments = [];
  const first = Math.ceil(lo / ISOBATH_STEP_M) * ISOBATH_STEP_M;
  for (let level = first; level <= hi; level += ISOBATH_STEP_M) {
    for (let r = 0; r < depth.length - 1; r += 1) {
      for (let c = 0; c < depth[0].length - 1; c += 1) {
        const corners = [[r, c], [r, c + 1], [r + 1, c + 1], [r + 1, c]];
        const points = [];
        corners.forEach((a, k) => {
          const b = corners[(k + 1) % 4];
          const va = depth[a[0]][a[1]];
          const vb = depth[b[0]][b[1]];
          if ((va < level) !== (vb < level)) {
            const f = (level - va) / (vb - va);
            points.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]);
          }
        });
        if (points.length >= 2) segments.push([points[0], points[1]]);
        if (points.length === 4) segments.push([points[2], points[3]]);
      }
    }
  }
  return segments;
}

// Map ------------------------------------------------------------------------

function sizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function latestDiagnostics(mission) {
  for (let i = mission.completed.length - 1; i >= 0; i -= 1) {
    const diagnostics = mission.completed[i].step.planner_diagnostics;
    if (diagnostics && Object.keys(diagnostics).length) return diagnostics;
  }
  return {};
}

function overlayGrid(pane, mission) {
  if (app.layer === "hidden") return app.revealed ? pane.reveal.expected_richness_map : null;
  if (app.layer === "terrain") return null;
  return latestDiagnostics(mission)[app.layer] || null;
}

function drawMap(pane, mission) {
  const { ctx, width } = sizeCanvas(pane.el.map);
  const [rows, cols] = pane.log.terrain.shape;
  const cell = width / cols;
  const toXY = ([r, c]) => [(c + 0.5) * cell, (r + 0.5) * cell];

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(pane.terrainImage, 0, 0, width, rows * cell);

  const grid = overlayGrid(pane, mission);
  let legend = "";
  if (grid) {
    const [lo, hi] = gridExtent(grid);
    const color = app.layer === "hidden" ? HIDDEN_RAMP : BELIEF_RAMP;
    const span = hi - lo || 1;
    ctx.drawImage(fieldImage(grid, (v) => color((v - lo) / span)), 0, 0, width, rows * cell);
    legend = `${LAYER_LABELS[app.layer]}: ${fmt(lo, 2)} (dark) to ${fmt(hi, 2)} (bright)`;
  } else if (app.layer === "terrain") {
    const [lo, hi] = gridExtent(pane.log.terrain.depth_m);
    legend = `Depth ${Math.round(lo)} to ${Math.round(hi)} m. Lines every ${ISOBATH_STEP_M} m. North is up.`;
  } else if (pane.timeline.some((s) => s.step.planner_diagnostics?.[app.layer])) {
    legend = `${LAYER_LABELS[app.layer]}: no data yet. The first survey hasn't finished.`;
  } else {
    legend = `${LAYER_LABELS[app.layer] || "Layer"}: this planner has no model, so it logs no map.`;
  }
  pane.el.legend.textContent = legend;

  ctx.lineWidth = 0.8;
  ctx.strokeStyle = grid ? "rgba(0,0,0,.28)" : "rgba(255,255,255,.2)";
  ctx.beginPath();
  for (const [a, b] of pane.isobaths) {
    const [x1, y1] = toXY(a);
    const [x2, y2] = toXY(b);
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
  }
  ctx.stroke();

  const color = pane.color;
  const line = (points, stroke, lineWidth) => {
    if (points.length < 2) return;
    ctx.beginPath();
    points.forEach((p, i) => {
      const [x, y] = toXY(p);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = "round";
    ctx.stroke();
  };
  line(mission.trail, "rgba(4,10,18,.75)", 3.5);
  line(mission.trail, color, 1.6);

  if (mission.current && mission.current.step.observation) {
    const target = mission.current.step.route[mission.current.step.route.length - 1];
    const [tx, ty] = toXY(target);
    const [rx, ry] = toXY(mission.position);
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(rx, ry);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(tx, ty, Math.max(5, cell * 0.45), 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  for (const { step } of mission.completed) {
    if (!step.observation) continue;
    const [x, y] = toXY(step.observation.waypoint);
    const richness = step.observation.richness;
    const radius = richness ? 2.5 + Math.sqrt(richness) * 1.8 : 2.2;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(4,10,18,.9)";
    ctx.fillStyle = richness ? color : "rgba(4,10,18,.75)";
    ctx.fill();
    ctx.stroke();
    if (step.new_discoveries.length) {
      ctx.beginPath();
      ctx.arc(x, y, radius + 2.5, 0, Math.PI * 2);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  }

  const [hx, hy] = toXY(pane.log.terrain.home);
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#04101c";
  ctx.lineWidth = 1.5;
  ctx.fillRect(hx - 5, hy - 5, 10, 10);
  ctx.strokeRect(hx - 5, hy - 5, 10, 10);

  const [px, py] = toXY(mission.position);
  ctx.beginPath();
  ctx.arc(px, py, 6.5, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.stroke();
}

// Panel text -----------------------------------------------------------------

const STATUS_TEXT = {
  start: "At home station",
  idle: "Choosing target",
  travel: "Travelling",
  survey: "Surveying",
  done: "Home safely",
};

function updatePaneText(pane, mission) {
  const { el, log, reveal } = pane;
  const last = mission.completed[mission.completed.length - 1];
  let found = last ? last.step.cumulative.distinct_categories : 0;
  const isReal = log.provenance.perception !== "ideal";
  if (app.revealed && isReal) {
    // After the reveal, score against the true labels rather than the model's predictions.
    const lastStep = last ? last.step.step : 0;
    const seen = reveal.true_observations.filter((o) => o.step <= lastStep).flatMap((o) => o.categories);
    found = new Set(seen).size;
  }
  el.foundLabel.textContent = !isReal ? "Categories found" : app.revealed ? "True categories found" : "Categories perceived";
  el.unknownStat.hidden = !isReal;
  el.unknown.textContent = last ? last.step.cumulative.unknown_encounters : 0;
  const surveys = last ? last.step.cumulative.surveys : 0;
  const used = last ? last.step.cumulative.energy_used : 0;
  const budget = log.energy.budget;
  const shown = Math.min(app.energy, log.outcome.energy_used);

  el.found.textContent = found;
  if (app.revealed) {
    const small = document.createElement("small");
    small.textContent = ` of ${reveal.categories_available_count}`;
    el.found.append(small);
  }
  el.surveys.textContent = surveys;
  el.left.textContent = fmt(budget - Math.max(used, shown));
  el.energyBar.style.width = `${(100 * (budget - shown)) / budget}%`;
  el.reserve.style.width = `${(100 * log.energy.reserve) / budget}%`;

  let status = STATUS_TEXT[mission.phase];
  if (mission.phase === "done") {
    status = log.outcome.safe_return ? STATUS_TEXT.done : "Return not verified";
    if (log.outcome.forced_return) status += " (target rejected)";
  }
  el.status.textContent = status;

  const utility = mission.current ? mission.current.step.planner_diagnostics?.target_utility : null;
  el.why.hidden = !utility;
  if (utility) {
    el.why.replaceChildren();
    const title = document.createElement("h3");
    title.textContent = "Why this target";
    el.why.append(title);
    for (const [key, value] of Object.entries(utility)) {
      const row = document.createElement("div");
      const name = document.createElement("span");
      name.textContent = UTILITY_LABELS[key] || key;
      const number = document.createElement("span");
      number.textContent = fmt(value, 3);
      row.append(name, number);
      el.why.append(row);
    }
  }
}

function readout(pane, cell) {
  if (!cell) {
    pane.el.readout.textContent = "Hover over the map to inspect a cell.";
    return;
  }
  const [r, c] = cell;
  const terrain = pane.log.terrain;
  const parts = [
    `(${r}, ${c})`,
    `${Math.round(terrain.depth_m[r][c])} m`,
    `slope ${fmt(terrain.slope_deg[r][c])}°`,
    `BPI ${Math.round(terrain.bpi_m[r][c])} m`,
  ];
  if (app.revealed) {
    parts.push(`hidden: ${pane.reveal.encounter_richness_map[r][c]} categories here`);
  }
  pane.el.readout.textContent = parts.join("  |  ");
}

function categoryLabel(name) {
  return name.replace("category_", "Cat ");
}

function renderFeed(pane, mission) {
  const surveys = mission.completed.filter((s) => s.step.observation);
  if (pane.feedCount === surveys.length) return;
  pane.feedCount = surveys.length;
  const items = surveys.slice().reverse().map(({ step, end }) => {
    const item = document.createElement("li");
    const obs = step.observation;
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = `Energy ${fmt(end)} at (${obs.waypoint[0]}, ${obs.waypoint[1]})`;
    item.append(meta);
    if (!obs.categories.length && !obs.image_refs.length) {
      const none = document.createElement("span");
      none.className = "none";
      none.textContent = "No organisms in this survey";
      item.append(none);
    }
    for (const category of obs.categories) {
      const chip = document.createElement("span");
      const isNew = step.new_discoveries.includes(category);
      chip.className = isNew ? "chip new" : "chip";
      chip.textContent = isNew ? `New: ${categoryLabel(category)}` : categoryLabel(category);
      item.append(chip);
    }
    obs.unknown_flags.forEach((unknown, index) => {
      if (!unknown) return;
      const chip = document.createElement("span");
      chip.className = "chip unknown";
      chip.textContent = `Unfamiliar, novelty ${fmt(obs.novelty_scores[index], 2)}`;
      item.append(chip);
    });
    for (const ref of obs.image_refs) {
      if (ref.includes("://")) continue; // opaque IDs, not image URLs
      const image = document.createElement("img");
      image.addEventListener("error", () => image.remove());
      image.src = ref;
      image.alt = "Marine image used as a simulated camera encounter";
      item.append(image);
    }
    return item;
  });
  pane.el.feedList.replaceChildren(...items);
}

// Chart ----------------------------------------------------------------------

function drawChart() {
  const canvas = $("chart");
  const { ctx, width, height } = sizeCanvas(canvas);
  const margin = { left: 34, right: 16, top: 10, bottom: 34 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  ctx.clearRect(0, 0, width, height);
  if (!app.panes.length) return;

  const available = app.revealed ? Math.max(...app.panes.map((p) => p.reveal.categories_available_count)) : 0;
  const finals = app.panes.map((p) => p.log.outcome.distinct_categories);
  const yMax = Math.max(4, available, ...finals);
  const x = (e) => margin.left + (e / app.budget) * plotW;
  const y = (v) => margin.top + plotH - (v / yMax) * plotH;

  ctx.font = "11px ui-monospace, Consolas, monospace";
  ctx.fillStyle = "#93a7b8";
  ctx.strokeStyle = "#24384b";
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const yStep = yMax > 10 ? 4 : 2;
  for (let v = 0; v <= yMax; v += yStep) {
    ctx.beginPath();
    ctx.moveTo(margin.left, y(v));
    ctx.lineTo(width - margin.right, y(v));
    ctx.stroke();
    ctx.fillText(String(v), margin.left - 6, y(v));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let e = 0; e <= app.budget; e += 30) ctx.fillText(String(e), x(e), height - margin.bottom + 5);
  ctx.fillText("Energy used", margin.left + plotW / 2, height - 14);

  if (app.revealed) {
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = "#ff7a8a";
    ctx.beginPath();
    ctx.moveTo(margin.left, y(available));
    ctx.lineTo(width - margin.right, y(available));
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "#ffc2c9";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("In this world", margin.left + 6, y(available) + 3);
  }

  const labels = [];
  app.panes.forEach((pane) => {
    const mission = pane.mission;
    let value = 0;
    ctx.beginPath();
    ctx.moveTo(x(0), y(0));
    for (const segment of mission.completed) {
      ctx.lineTo(x(segment.end), y(value));
      value = segment.step.cumulative.distinct_categories;
      ctx.lineTo(x(segment.end), y(value));
    }
    const end = Math.min(app.energy, pane.log.outcome.energy_used);
    ctx.lineTo(x(end), y(value));
    ctx.strokeStyle = pane.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();
    labels.push({ px: x(end), py: y(value), text: pane.entry.label, color: pane.color });
  });

  labels.sort((a, b) => a.py - b.py);
  labels.forEach((label, i) => {
    if (i && label.py - labels[i - 1].py < 13) label.py = labels[i - 1].py + 13;
    ctx.fillStyle = label.color;
    const flip = label.px > width - 110;
    ctx.textAlign = flip ? "right" : "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(label.text, label.px + (flip ? -4 : 4), label.py - 2);
  });
}

// Render loop ----------------------------------------------------------------

function render() {
  for (const pane of app.panes) {
    pane.mission = missionAt(pane, app.energy);
    drawMap(pane, pane.mission);
    updatePaneText(pane, pane.mission);
    renderFeed(pane, pane.mission);
  }
  drawChart();
  $("scrub").value = app.energy;
  $("clock").textContent = `${fmt(app.energy)} / ${fmt(app.maxEnergy)} energy`;
}

function tick(now) {
  if (!app.playing) return;
  const elapsed = (now - app.lastFrame) / 1000;
  app.lastFrame = now;
  app.energy = Math.min(app.maxEnergy, app.energy + elapsed * Number($("speed").value));
  if (app.energy >= app.maxEnergy) setPlaying(false);
  render();
  if (app.playing) requestAnimationFrame(tick);
}

function setPlaying(playing) {
  if (playing && app.energy >= app.maxEnergy) app.energy = 0;
  app.playing = playing;
  $("play").textContent = playing ? "Pause" : "Play";
  if (playing) {
    app.lastFrame = performance.now();
    requestAnimationFrame(tick);
  }
}

function jumpEvent(direction) {
  setPlaying(false);
  const events = app.panes.flatMap((p) => p.timeline.map((s) => s.end)).sort((a, b) => a - b);
  const target = direction > 0
    ? events.find((e) => e > app.energy + 1e-6)
    : events.reverse().find((e) => e < app.energy - 1e-6);
  app.energy = target === undefined ? (direction > 0 ? app.maxEnergy : 0) : target;
  render();
}

// Loading --------------------------------------------------------------------

function worlds() {
  const seen = new Map();
  for (const entry of app.entries) if (!seen.has(entry.world_id)) seen.set(entry.world_id, entry);
  return [...seen.values()];
}

function createPane(index) {
  const node = $("pane-template").content.firstElementChild.cloneNode(true);
  node.style.setProperty("--pane-color", PANE_COLORS[index]);
  const feed = document.createElement("section");
  feed.className = "feed";
  feed.style.setProperty("--pane-color", PANE_COLORS[index]);
  const title = document.createElement("h2");
  const list = document.createElement("ol");
  feed.append(title, list);
  const el = {
    root: node,
    map: node.querySelector(".map"),
    planner: node.querySelector(".planner"),
    status: node.querySelector(".status"),
    readout: node.querySelector(".readout"),
    legend: node.querySelector(".legend"),
    found: node.querySelector(".found"),
    foundLabel: node.querySelector(".found-label"),
    unknown: node.querySelector(".unknown-count"),
    unknownStat: node.querySelector(".unknown-stat"),
    surveys: node.querySelector(".surveys"),
    left: node.querySelector(".left"),
    energyBar: node.querySelector(".energy i"),
    reserve: node.querySelector(".energy b"),
    why: node.querySelector(".why"),
    feed,
    feedTitle: title,
    feedList: list,
  };
  $("panes").append(node);
  $("feeds").append(feed);
  return el;
}

async function loadPane(pane, entry) {
  const log = await getJson(entry.log);
  pane.entry = entry;
  pane.log = log;
  pane.reveal = null;
  pane.timeline = buildTimeline(log);
  pane.terrainImage = terrainImage(log.terrain);
  pane.isobaths = isobaths(log.terrain.depth_m);
  pane.feedCount = -1;
  pane.el.feedTitle.textContent = `${entry.label}: survey feed`;
}

async function fetchReveals() {
  await Promise.all(app.panes.map(async (pane) => {
    if (!pane.reveal) pane.reveal = await getJson(pane.entry.reveal);
  }));
}

function updateLayerControls() {
  const hasBelief = (key) => app.panes.some((pane) =>
    pane.timeline.some((s) => s.step.planner_diagnostics && s.step.planner_diagnostics[key]));
  for (const input of document.querySelectorAll('input[name="layer"]')) {
    if (input.value === "terrain") continue;
    input.disabled = input.value === "hidden" ? !app.revealed : !hasBelief(input.value);
    input.parentElement.title = input.disabled
      ? (input.value === "hidden" ? "Reveal the hidden world first" : "This mission's planner does not log this map")
      : "";
    if (input.disabled && input.checked) {
      document.querySelector('input[name="layer"][value="terrain"]').checked = true;
      app.layer = "terrain";
    }
  }
}

function updateClockRange() {
  app.budget = Math.max(...app.panes.map((p) => p.log.energy.budget));
  app.maxEnergy = Math.max(...app.panes.map((p) => p.log.outcome.energy_used));
  app.energy = Math.min(app.energy, app.maxEnergy);
  $("scrub").max = app.maxEnergy;
}

function updateProvenance() {
  const log = app.panes[0].log;
  const source = log.provenance.terrain;
  const credit = source === "gebco_2025" ? GEBCO_CREDIT : "";
  const model = log.provenance.perception;
  $("provenance").textContent =
    `${TERRAIN_NAMES[source] || source}${credit} Biology is simulated, not observed. ` +
    `${PERCEPTION_NAMES[model] || `Perception model: ${model}.`} ` +
    "Energy is in abstract units. Every route and count comes from an executed mission log.";
  const isReal = app.panes.some((p) => p.log.provenance.perception !== "ideal");
  $("chart-title").textContent = `Distinct categories ${isReal ? "perceived" : "found"} against energy used`;
  const stub = app.panes.some((p) => p.log.provenance.perception === "stub");
  if (stub) showNotice("Stub perception: a fake model for testing, with no real images. Not a vision result.");
}

function setRevealButton() {
  const button = $("reveal");
  button.setAttribute("aria-pressed", String(app.revealed));
  button.textContent = app.revealed ? "Hide hidden world" : "Reveal hidden world";
}

async function selectWorld(worldId) {
  const token = (app.loadToken += 1);
  setPlaying(false);
  exitTour();
  app.worldId = worldId;
  const options = app.entries.filter((e) => e.world_id === worldId);
  $("panes").replaceChildren();
  $("feeds").replaceChildren();
  app.panes = [];
  app.revealed = false;
  setRevealButton();

  const count = Math.min(options.length, 2);
  $("panes").style.gridTemplateColumns = count === 1 ? "minmax(0, 720px)" : "";
  const panes = [];
  for (let i = 0; i < count; i += 1) {
    const el = createPane(i);
    const pane = { el, color: PANE_COLORS[i] };
    for (const entry of options) {
      const option = document.createElement("option");
      option.value = entry.log;
      option.textContent = entry.label;
      el.planner.append(option);
    }
    el.planner.value = options[i].log;
    el.planner.addEventListener("change", () => changePlanner(pane, options));
    el.map.addEventListener("mousemove", (event) => {
      const rect = el.map.getBoundingClientRect();
      const [rows, cols] = pane.log.terrain.shape;
      const r = clamp(Math.floor(((event.clientY - rect.top) / rect.height) * rows), 0, rows - 1);
      const c = clamp(Math.floor(((event.clientX - rect.left) / rect.width) * cols), 0, cols - 1);
      readout(pane, [r, c]);
    });
    el.map.addEventListener("mouseleave", () => readout(pane, null));
    panes.push(pane);
  }
  await Promise.all(panes.map((pane, i) => loadPane(pane, options[i])));
  if (token !== app.loadToken) return;

  app.panes = panes;
  showNotice(count === 1 ? "Only one mission has been logged for this world." : "");
  app.energy = 0;
  updateClockRange();
  updateLayerControls();
  updateProvenance();
  render();
}

async function changePlanner(pane, options) {
  const entry = options.find((e) => e.log === pane.el.planner.value);
  setPlaying(false);
  await loadPane(pane, entry);
  if (app.revealed) await fetchReveals();
  updateClockRange();
  updateLayerControls();
  render();
}

function setLayer(layer) {
  const input = document.querySelector(`input[name="layer"][value="${layer}"]`);
  if (input.disabled) return;
  input.checked = true;
  app.layer = layer;
}

async function setRevealed(revealed) {
  if (revealed) await fetchReveals();
  app.revealed = revealed;
  setRevealButton();
  updateLayerControls();
  if (revealed) setLayer("hidden");
}

async function toggleReveal() {
  if (!app.panes.length) return;
  await setRevealed(!app.revealed);
  render();
}

// Benchmark ------------------------------------------------------------------

const PLANNER_NAMES = {
  abyss: "ABYSS",
  frontier: "Frontier",
  heuristic: "Heuristic",
  lawnmower: "Lawnmower",
  random: "Random",
  abyss_no_info: "No uncertainty",
  abyss_no_learning: "No learning",
};
const TERRAIN_LABELS = { gebco: "GEBCO Monterey crop", synthetic: "Synthetic (one per seed)" };

function worldLabel(world) {
  const name = SCENARIO_NAMES[world.scenario] || world.scenario;
  return world.scenario === "random" ? name : `${name}, signal ${world.rho}`;
}

function renderBenchmark() {
  const data = app.benchmark;
  const terrain = $("bench-terrain").value;
  const other = $("bench-other").value;
  const worlds = data.worlds.filter((w) => w.terrain === terrain);
  const span = Math.max(1, ...worlds.flatMap((w) => {
    const c = w.abyss_vs[other];
    return [Math.abs(c.ci95_low), Math.abs(c.ci95_high)];
  }));
  const position = (v) => `${50 + (50 * v) / span}%`;

  const rows = worlds.map((world) => {
    const c = world.abyss_vs[other];
    const verdict = c.ci95_low > 0 ? "higher" : c.ci95_high < 0 ? "lower" : "unclear";
    const row = document.createElement("div");
    row.className = `forest-row ci-${verdict}`;
    const label = document.createElement("span");
    label.textContent = worldLabel(world);
    const track = document.createElement("div");
    track.className = "forest-track";
    const zero = document.createElement("i");
    zero.className = "forest-zero";
    zero.style.left = "50%";
    const ci = document.createElement("i");
    ci.className = "forest-ci";
    ci.style.left = position(c.ci95_low);
    ci.style.width = `${(50 * (c.ci95_high - c.ci95_low)) / span}%`;
    const dot = document.createElement("i");
    dot.className = "forest-dot";
    dot.style.left = position(c.mean_difference);
    track.append(zero, ci, dot);
    const value = document.createElement("span");
    value.className = "forest-value";
    const word = document.createElement("b");
    word.textContent = { higher: "ABYSS higher", lower: "ABYSS lower", unclear: "No clear difference" }[verdict];
    const sign = c.mean_difference >= 0 ? "+" : "";
    value.append(word, ` ${sign}${fmt(c.mean_difference, 2)} [${fmt(c.ci95_low, 2)}, ${fmt(c.ci95_high, 2)}]`);
    row.append(label, track, value);
    return row;
  });
  $("bench-forest").replaceChildren(...rows);
  $("bench-note").textContent =
    `Difference in distinct categories found, ABYSS minus ${PLANNER_NAMES[other]}, on the same ${data.seeds.length} worlds. ` +
    "Bars show 95% intervals. Zero is the centre line.";

  const planners = Object.keys(PLANNER_NAMES);
  const head = document.createElement("tr");
  for (const title of ["Mean categories found", ...planners.map((p) => PLANNER_NAMES[p])]) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = title;
    head.append(th);
  }
  const body = worlds.map((world) => {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = worldLabel(world);
    tr.append(name);
    const best = Math.max(...planners.map((p) => world.planners[p].mean));
    for (const planner of planners) {
      const td = document.createElement("td");
      const mean = world.planners[planner].mean;
      td.textContent = fmt(mean, 2);
      if (planner === "abyss") td.classList.add("abyss-col");
      if (mean === best) td.classList.add("best");
      tr.append(td);
    }
    return tr;
  });
  $("bench-table").replaceChildren(head, ...body);

  const safety = data.safety;
  $("bench-meta").textContent =
    `${safety.missions} missions, ${safety.safe_returns} safe returns, ${safety.forced_returns} forced returns. ` +
    `Scored on true labels with ideal perception. Biology is simulated and energy is in abstract units. ` +
    `ABYSS settings were frozen on separate development seeds. The maximum is ` +
    `${data.frozen_config.simulation.n_categories} categories, so results near it can't show larger differences. ` +
    "Intervals are not corrected for multiple comparisons.";
}

async function loadBenchmark() {
  const response = await fetch("logs/benchmark.json");
  if (!response.ok) return; // only appears once a real benchmark has been run
  app.benchmark = await response.json();
  const terrains = [...new Set(app.benchmark.worlds.map((w) => w.terrain))];
  $("bench-terrain").replaceChildren(...terrains.map((t) => new Option(TERRAIN_LABELS[t] || t, t)));
  const others = Object.keys(app.benchmark.worlds[0].abyss_vs);
  $("bench-other").replaceChildren(...others.map((p) => new Option(PLANNER_NAMES[p] || p, p)));
  $("bench-terrain").addEventListener("change", renderBenchmark);
  $("bench-other").addEventListener("change", renderBenchmark);
  $("bench").hidden = false;
  renderBenchmark();
}

// Guided demo ----------------------------------------------------------------

function abyssPane() {
  return app.panes.find((p) => p.entry.planner === "abyss") || app.panes[0];
}

function otherPane() {
  return app.panes.find((p) => p !== abyssPane()) || app.panes[0];
}

const TOUR_STAGES = [
  {
    title: "Stage 1: Deployment",
    apply: async () => { await setRevealed(false); setLayer("terrain"); app.energy = 0; },
    body: () => "Two robots start at the same station on the same terrain, with the same energy. " +
      "Neither knows where the organisms are. This terrain is real bathymetry; the biology is simulated.",
  },
  {
    title: "Stage 2: First observations",
    apply: async () => {
      await setRevealed(false);
      setLayer("terrain");
      const first = abyssPane().timeline.find((s) => s.step.new_discoveries.length);
      app.energy = first ? first.end : 0;
    },
    body: () => {
      const first = abyssPane().timeline.find((s) => s.step.new_discoveries.length);
      return first
        ? `ABYSS makes its first discovery at energy ${fmt(first.end)}. Each survey now updates its habitat model. See the survey feed below the map.`
        : "ABYSS found no organisms in this mission.";
    },
  },
  {
    title: "Stage 3: Behaviour emerges",
    apply: async () => {
      await setRevealed(false);
      setLayer("predicted_richness");
      const line = abyssPane().timeline;
      app.energy = line[Math.min(5, line.length - 2)].start + 0.01;
    },
    body: () => {
      const line = abyssPane().timeline;
      const terms = line[Math.min(5, line.length - 2)].step.planner_diagnostics?.target_utility;
      const numbers = terms
        ? ` For this target: predicted richness ${fmt(terms.expected_richness, 2)}, information gain ${fmt(terms.information_gain, 2)}, ` +
          `coverage ${fmt(terms.coverage, 2)}, energy ${fmt(terms.energy_cost, 2)}.`
        : "";
      return "This map is what ABYSS currently believes. Its next target comes from a single formula: " +
        `predicted richness, uncertainty and coverage, per unit of energy.${numbers} No discovery triggers a scripted route.`;
    },
  },
  {
    title: "Stage 4: Mission complete",
    apply: async () => { await setRevealed(false); setLayer("terrain"); app.energy = app.maxEnergy; },
    body: () => {
      const safe = app.panes.every((p) => p.log.outcome.safe_return);
      const counts = app.panes.map((p) => `${p.entry.label} found ${p.log.outcome.distinct_categories}`).join(", ");
      return `${counts}. ${safe ? "Both returned home within budget." : "Check the return status."} One world can't settle anything, whichever planner did better.`;
    },
  },
  {
    title: "Stage 5: The hidden world",
    apply: async () => {
      await setRevealed(true);
      app.energy = app.maxEnergy;
      $("bench").scrollIntoView({ behavior: "smooth", block: "start" });
    },
    body: () => (app.benchmark
      ? "The map now shows the simulator's hidden richness. The benchmark compares the planners across many worlds, including those where ABYSS has no advantage."
      : "The map now shows the simulator's hidden richness, which the robots never saw."),
  },
];

async function showTourStage(index) {
  app.tourIndex = index;
  setPlaying(false);
  const stage = TOUR_STAGES[index];
  await stage.apply();
  $("tour").hidden = false;
  $("tour-title").textContent = stage.title;
  $("tour-body").textContent = stage.body();
  $("tour-back").disabled = index === 0;
  $("tour-next").disabled = index === TOUR_STAGES.length - 1;
  if (index < TOUR_STAGES.length - 1) window.scrollTo({ top: 0, behavior: "smooth" });
  render();
}

function exitTour() {
  app.tourIndex = -1;
  $("tour").hidden = true;
}

function bindControls() {
  $("world").addEventListener("change", (event) => selectWorld(event.target.value));
  $("play").addEventListener("click", () => setPlaying(!app.playing));
  $("prev").addEventListener("click", () => jumpEvent(-1));
  $("next").addEventListener("click", () => jumpEvent(1));
  $("scrub").addEventListener("input", (event) => {
    setPlaying(false);
    app.energy = Number(event.target.value);
    render();
  });
  $("reveal").addEventListener("click", () => toggleReveal().catch((e) => showNotice(e.message, true)));
  const tourStep = (index) => showTourStage(index).catch((e) => showNotice(e.message, true));
  $("tour-start").addEventListener("click", () => { if (app.panes.length) tourStep(0); });
  $("tour-back").addEventListener("click", () => tourStep(app.tourIndex - 1));
  $("tour-next").addEventListener("click", () => tourStep(app.tourIndex + 1));
  $("tour-exit").addEventListener("click", exitTour);
  $("layers").addEventListener("change", (event) => {
    app.layer = event.target.value;
    render();
  });
  document.addEventListener("keydown", (event) => {
    // Focused controls keep their own keyboard behaviour.
    if (event.target.closest("select, input, button")) return;
    if (event.code === "Space") { event.preventDefault(); setPlaying(!app.playing); }
    if (event.key === "ArrowRight") jumpEvent(1);
    if (event.key === "ArrowLeft") jumpEvent(-1);
  });
  new ResizeObserver(() => { if (app.panes.length) render(); }).observe($("panes"));
}

async function init() {
  bindControls();
  try {
    app.entries = (await getJson("logs/index.json")).entries;
  } catch (error) {
    const hint = location.protocol === "file:"
      ? "Browsers block loading logs from file:// pages. Run: python -m http.server 8000 --directory ui"
      : "Build the logs with: python -m scripts.build_ui_logs";
    showNotice(`Could not load mission logs. ${hint}`, true);
    return;
  }
  const select = $("world");
  for (const world of worlds()) {
    const option = document.createElement("option");
    option.value = world.world_id;
    option.textContent = `${SCENARIO_NAMES[world.scenario] || world.scenario}, seed ${world.seed}`;
    select.append(option);
  }
  await selectWorld(select.value);
  await loadBenchmark();
}

init().catch((error) => showNotice(error.message, true));
