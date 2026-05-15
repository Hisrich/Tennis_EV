/* api.js — shared API client and UI utilities */

const API = (() => {
  // Flask serves everything — API and frontend are on the same origin
  const BASE = "";

  async function req(method, path, body) {
    const opts = {
      method,
      headers: { "Content-Type": "application/json" },
    };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(BASE + path, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || res.statusText);
    }
    return res.json();
  }

  return {
    get:   (p)      => req("GET",   p),
    post:  (p, b)   => req("POST",  p, b),
    patch: (p, b)   => req("PATCH", p, b),

    getOverview:          ()     => API.get("/api/stats/overview"),
    getOpportunities:     (edge=5) => API.get(`/api/opportunities?min_edge=${edge}`),
    getPerformanceSummary:(days=30)=> API.get(`/api/performance/summary?days=${days}`),
    getBankrollHistory:   (days=90)=> API.get(`/api/performance/bankroll-history?days=${days}`),
    getMatches:           (up=false)=>API.get(`/api/matches?upcoming_only=${up}`),
    getBets:              (status) => API.get(`/api/bets${status ? "?status="+status : ""}`),
    getModels:            ()     => API.get("/api/models"),
    getPlayers:           (q="") => API.get(`/api/players${q ? "?search="+encodeURIComponent(q) : ""}`),
    getEloHistory:        (id)   => API.get(`/api/players/${id}/elo-history`),

    recordBet:   (data)         => API.post("/api/bets", data),
    settleBet:   (id, status)   => API.patch(`/api/bets/${id}/settle`, { status }),
    runBacktest: (data)         => API.post("/api/backtest", data),
    triggerTask: (task)         => API.post(`/api/admin/${task}`),
    triggerTrain:(algo="xgboost")=> API.post(`/api/models/train?algorithm=${algo}`),
  };
})();

/* ── Toast notifications ─────────────────────────────────── */
const Toast = (() => {
  let wrap = null;
  function init() {
    if (!wrap) {
      wrap = document.createElement("div");
      wrap.className = "toast-wrap";
      document.body.appendChild(wrap);
    }
  }
  function show(msg, type = "success") {
    init();
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = msg;
    wrap.appendChild(el);
    setTimeout(() => el.remove(), 3500);
  }
  return { success: m => show(m, "success"), error: m => show(m, "error") };
})();

/* ── Mini chart using Canvas ─────────────────────────────── */
function drawLineChart(canvas, data, { xKey, yKey, color = "#22c55e", fill = true } = {}) {
  const ctx = canvas.getContext("2d");
  const W = canvas.offsetWidth || canvas.width;
  const H = canvas.offsetHeight || canvas.height;
  canvas.width  = W;
  canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  if (!data || data.length < 2) {
    ctx.fillStyle = "#6b7280";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No data yet", W / 2, H / 2);
    return;
  }

  const vals = data.map(d => d[yKey]);
  const min  = Math.min(...vals);
  const max  = Math.max(...vals);
  const pad  = { top: 16, right: 10, bottom: 32, left: 54 };
  const iW   = W - pad.left - pad.right;
  const iH   = H - pad.top  - pad.bottom;
  const range = max - min || 1;

  const px = (i) => pad.left + (i / (data.length - 1)) * iW;
  const py = (v) => pad.top  + iH - ((v - min) / range) * iH;

  // Grid lines
  ctx.strokeStyle = "#1f2937";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (iH / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
  }

  // Y-axis labels
  ctx.fillStyle = "#6b7280";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const v = max - ((max - min) / 4) * i;
    const y = pad.top + (iH / 4) * i;
    ctx.fillText(v.toFixed(0), pad.left - 6, y + 3);
  }

  // X-axis labels (first, middle, last)
  ctx.textAlign = "center";
  [0, Math.floor(data.length / 2), data.length - 1].forEach(i => {
    const raw = data[i][xKey];
    const label = raw ? new Date(raw).toLocaleDateString("en-GB", { month: "short", day: "numeric" }) : "";
    ctx.fillText(label, px(i), H - pad.bottom + 16);
  });

  // Fill gradient
  if (fill) {
    const grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
    grad.addColorStop(0, color + "33");
    grad.addColorStop(1, color + "00");
    ctx.beginPath();
    ctx.moveTo(px(0), H - pad.bottom);
    data.forEach((d, i) => ctx.lineTo(px(i), py(d[yKey])));
    ctx.lineTo(px(data.length - 1), H - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  }

  // Line
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  data.forEach((d, i) => i === 0 ? ctx.moveTo(px(i), py(d[yKey])) : ctx.lineTo(px(i), py(d[yKey])));
  ctx.stroke();

  // Dots at ends
  [0, data.length - 1].forEach(i => {
    ctx.beginPath();
    ctx.arc(px(i), py(vals[i]), 3, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  });
}

/* ── Sidebar active state ────────────────────────────────── */
function initNav() {
  const page = window.location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".nav-item").forEach(a => {
    const href = a.getAttribute("href") || "";
    const active =
      (page === "index.html" && (href === "index.html" || href === "./")) ||
      href === page;
    if (active) a.classList.add("active");
  });
}

document.addEventListener("DOMContentLoaded", initNav);
