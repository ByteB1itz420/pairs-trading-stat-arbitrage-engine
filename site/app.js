"use strict";

const COLORS = ["#087f65", "#6c87ba"];
const state = { rows: [], symbols: [], name: "Synthetic demo" };
const $ = (selector) => document.querySelector(selector);

function setStatus(message, mode = "") {
  const status = $("#data-status");
  status.className = `data-status ${mode ? `is-${mode}` : ""}`;
  status.replaceChildren();
  const icon = document.createElement("span");
  icon.className = "status-icon";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = mode === "error" ? "!" : mode === "success" ? "✓" : "i";
  const text = document.createElement("span");
  text.textContent = message;
  status.append(icon, text);
}

function csvRows(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted && character === '"' && text[index + 1] === '"') {
      cell += '"';
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(cell);
      if (row.some((value) => value.trim() !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += character;
    }
  }
  if (quoted) throw new Error("CSV contains a quote that was never closed.");
  row.push(cell);
  if (row.some((value) => value.trim() !== "")) rows.push(row);
  return rows;
}

function parsePrices(text) {
  const table = csvRows(text);
  if (table.length < 2) throw new Error("CSV has a header but no price rows.");
  const headers = table[0].map((header) => header.trim().toLowerCase());
  const dateIndex = headers.indexOf("timestamp");
  const symbolIndex = headers.indexOf("symbol");
  const closeIndex = headers.indexOf("adjusted_close");
  if ([dateIndex, symbolIndex, closeIndex].some((index) => index < 0)) {
    throw new Error("Required CSV headers are timestamp, symbol, and adjusted_close.");
  }
  const openIndex = headers.indexOf("open");
  const values = new Map();
  const symbols = new Set();
  const parsed = [];
  for (let line = 1; line < table.length; line += 1) {
    const row = table[line];
    const timestamp = (row[dateIndex] || "").trim();
    const symbol = (row[symbolIndex] || "").trim().toUpperCase();
    const close = Number(row[closeIndex]);
    const openValue = openIndex < 0 ? "" : (row[openIndex] || "").trim();
    const open = openValue === "" ? null : Number(openValue);
    if (!timestamp || !Number.isFinite(Date.parse(timestamp))) {
      throw new Error(`Invalid timestamp on CSV row ${line + 1}.`);
    }
    if (!symbol) throw new Error(`Missing symbol on CSV row ${line + 1}.`);
    if (!Number.isFinite(close) || close <= 0) {
      throw new Error(`Adjusted close must be a positive number on CSV row ${line + 1}.`);
    }
    if (open !== null && (!Number.isFinite(open) || open <= 0)) {
      throw new Error(`Open must be a positive number on CSV row ${line + 1}.`);
    }
    const key = `${timestamp}\u0000${symbol}`;
    if (values.has(key)) throw new Error(`Duplicate ${symbol} observation on ${timestamp}.`);
    values.set(key, close);
    symbols.add(symbol);
    parsed.push({ timestamp, symbol, close, open });
  }
  const orderedSymbols = [...symbols].sort();
  if (orderedSymbols.length < 2) throw new Error("Choose a CSV with at least two symbols.");
  parsed.sort((left, right) => left.timestamp.localeCompare(right.timestamp));
  return { rows: parsed, symbols: orderedSymbols };
}

function deterministicRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function demoPrices() {
  const random = deterministicRandom(731);
  const count = 180;
  let logBeta = 4.5;
  let residual = 0;
  const rows = [];
  const businessDates = [];
  for (
    let milliseconds = Date.UTC(2024, 0, 2);
    businessDates.length < count;
    milliseconds += 24 * 60 * 60 * 1000
  ) {
    const date = new Date(milliseconds);
    if (date.getUTCDay() !== 0 && date.getUTCDay() !== 6) businessDates.push(date);
  }
  for (let index = 0; index < count; index += 1) {
    logBeta += (random() - 0.5) * 0.026;
    residual = 0.72 * residual + (random() - 0.5) * 0.02;
    if (index >= 125 && index < 130) residual += 0.04;
    const date = businessDates[index];
    const timestamp = date.toISOString().slice(0, 10);
    const closeA = Math.exp(0.35 + 0.8 * logBeta + residual);
    const closeB = Math.exp(logBeta);
    rows.push({
      timestamp,
      symbol: "ALPHA",
      close: closeA,
      open: closeA * (1 + 0.001 * Math.sin(index * 0.73)),
    });
    rows.push({
      timestamp,
      symbol: "BETA",
      close: closeB,
      open: closeB * (1 + 0.001 * Math.cos(index * 0.61)),
    });
  }
  return { rows, symbols: ["ALPHA", "BETA"] };
}

function dateLabel(value) {
  const parsed = new Date(/^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00Z` : value);
  return Number.isNaN(parsed.valueOf())
    ? value
    : parsed.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
}

function populateSelect(select, symbols, selected) {
  select.replaceChildren();
  for (const symbol of symbols) {
    const option = document.createElement("option");
    option.value = symbol;
    option.textContent = symbol;
    select.append(option);
  }
  select.value = selected;
}

function currentSeries() {
  const symbolA = $("#symbol-a").value;
  const symbolB = $("#symbol-b").value;
  if (symbolA === symbolB) throw new Error("Select two different assets.");
  const byDate = new Map();
  for (const row of state.rows) {
    if (row.symbol !== symbolA && row.symbol !== symbolB) continue;
    if (!byDate.has(row.timestamp)) byDate.set(row.timestamp, {});
    const day = byDate.get(row.timestamp);
    day[row.symbol] = row.close;
    if (row.symbol === symbolA) day.openA = row.open;
    if (row.symbol === symbolB) day.openB = row.open;
  }
  const dates = [...byDate.keys()].sort();
  const aligned = dates
    .filter((timestamp) => Number.isFinite(byDate.get(timestamp)[symbolA]) && Number.isFinite(byDate.get(timestamp)[symbolB]))
    .map((timestamp) => ({
      timestamp,
      a: byDate.get(timestamp)[symbolA],
      b: byDate.get(timestamp)[symbolB],
      openA: byDate.get(timestamp).openA,
      openB: byDate.get(timestamp).openB,
    }));
  if (aligned.length < 20) {
    throw new Error(`Only ${aligned.length} common price dates. At least 20 are needed to chart a pair.`);
  }
  return { symbolA, symbolB, aligned };
}

function setBacktestStatus(message, mode = "") {
  const status = $("#backtest-status");
  status.className = `backtest-status ${mode ? `is-${mode}` : ""}`;
  status.textContent = message;
  status.hidden = false;
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "—";
}

function showBacktestMetrics(metrics) {
  const items = [
    ["Net return", formatPercent(metrics.cumulative_net_return)],
    ["Maximum drawdown", formatPercent(metrics.maximum_drawdown)],
    ["Sharpe ratio", Number.isFinite(metrics.sharpe_ratio) ? metrics.sharpe_ratio.toFixed(2) : "—"],
    ["Completed trades", String(metrics.trade_count)],
  ];
  const container = $("#backtest-metrics");
  container.replaceChildren();
  for (const [label, value] of items) {
    const card = document.createElement("div");
    const title = document.createElement("span");
    title.className = "summary-label";
    title.textContent = label;
    const result = document.createElement("strong");
    result.textContent = value;
    card.append(title, result);
    container.append(card);
  }
}

async function runPythonBacktest() {
  const button = $("#run-backtest");
  button.disabled = true;
  $("#backtest-results").hidden = true;
  try {
    const rawUrl = $("#api-url").value.trim();
    if (!rawUrl) throw new Error("Enter the public Railway API base URL first.");
    const apiUrl = new URL(rawUrl);
    if (apiUrl.protocol !== "https:" && !["localhost", "127.0.0.1"].includes(apiUrl.hostname)) {
      throw new Error("Use an HTTPS Railway URL (HTTP is allowed only for localhost).");
    }
    const { symbolA, symbolB, aligned } = currentSeries();
    const trainingCount = Math.floor(aligned.length * 0.7);
    if (trainingCount < 100 || aligned.length - trainingCount < 2) {
      throw new Error("The selected pair needs at least 100 training dates and 2 test dates.");
    }
    const rows = [];
    for (const row of aligned) {
      if (!Number.isFinite(row.openA) || !Number.isFinite(row.openB)) {
        throw new Error("Backtesting requires a positive open price for both assets on every selected date.");
      }
      rows.push(
        { timestamp: row.timestamp, symbol: symbolA, adjusted_close: row.a, open: row.openA },
        { timestamp: row.timestamp, symbol: symbolB, adjusted_close: row.b, open: row.openB },
      );
    }
    const trainEnd = aligned[trainingCount - 1].timestamp;
    const testStart = aligned[trainingCount].timestamp;
    const payload = {
      rows,
      symbol_a: symbolA,
      symbol_b: symbolB,
      train_end: trainEnd,
      test_start: testStart,
      formation_window_days: Math.min(252, trainingCount),
      min_observations: 100,
      zscore_window_days: Math.min(Number($("#zscore-window").value), trainingCount),
    };
    if (rows.length > 40_000) {
      throw new Error("The research API accepts at most 40,000 selected-pair rows.");
    }
    const requestBody = JSON.stringify(payload);
    if (new Blob([requestBody]).size > 8 * 1024 * 1024) {
      throw new Error("The research API accepts request bodies up to 8 MiB.");
    }
    sessionStorage.setItem("pairwise-api-url", apiUrl.origin);
    setBacktestStatus(`Sending ${rows.length.toLocaleString()} selected-pair price rows to the research API…`);
    const headers = { "Content-Type": "application/json" };
    const apiKey = $("#api-key").value.trim();
    if (apiKey) headers["X-API-Key"] = apiKey;
    const response = await fetch(`${apiUrl.origin}/api/v1/backtest`, {
      method: "POST",
      headers,
      body: requestBody,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : "The research API rejected this backtest.");
    }
    if (data.status === "not_cointegrated") {
      setBacktestStatus(
        `No backtest run: ${symbolA}/${symbolB} did not pass training-period Engle–Granger screening (p=${Number(data.pair.engle_granger_pvalue).toPrecision(3)}).`,
        "error",
      );
      return;
    }
    showBacktestMetrics(data.metrics);
    renderChart($("#backtest-equity-chart"), [
      { values: data.equity_curve.map((point) => point.net_equity), color: COLORS[0] },
    ], {
      firstLabel: dateLabel(data.out_of_sample.start),
      lastLabel: dateLabel(data.out_of_sample.end),
    });
    $("#backtest-summary").textContent =
      `${data.pair.symbol_a}/${data.pair.symbol_b} · hedge ratio ${Number(data.pair.hedge_ratio).toFixed(4)} · ` +
      `Engle–Granger p=${Number(data.pair.engle_granger_pvalue).toPrecision(3)} · ` +
      `${data.out_of_sample.observations} out-of-sample sessions · ${data.trades.length} completed trades`;
    $("#backtest-results").hidden = false;
    setBacktestStatus("Backtest complete. Results are computed from your uploaded prices and were not stored by the API.", "success");
  } catch (error) {
    setBacktestStatus(error instanceof Error ? error.message : "Could not complete the backtest.", "error");
  } finally {
    button.disabled = false;
  }
}

function fitOls(rows) {
  const x = rows.map((row) => Math.log(row.b));
  const y = rows.map((row) => Math.log(row.a));
  const xMean = x.reduce((sum, value) => sum + value, 0) / x.length;
  const yMean = y.reduce((sum, value) => sum + value, 0) / y.length;
  const covariance = x.reduce((sum, value, index) => sum + (value - xMean) * (y[index] - yMean), 0);
  const variance = x.reduce((sum, value) => sum + (value - xMean) ** 2, 0);
  if (!(variance > 0)) throw new Error("The second asset's training prices have zero variance.");
  const beta = covariance / variance;
  const alpha = yMean - beta * xMean;
  const spread = rows.map((row) => Math.log(row.a) - alpha - beta * Math.log(row.b));
  return { alpha, beta, spread };
}

function zScores(values, window) {
  return values.map((value, index) => {
    if (index < window - 1) return null;
    const slice = values.slice(index - window + 1, index + 1);
    const mean = slice.reduce((sum, item) => sum + item, 0) / window;
    const variance = slice.reduce((sum, item) => sum + (item - mean) ** 2, 0) / (window - 1);
    const deviation = Math.sqrt(variance);
    return deviation > 1e-12 ? (value - mean) / deviation : null;
  });
}

function pathFor(values, width, height, padding, min, max) {
  const span = max - min || 1;
  const innerWidth = width - 2 * padding.left;
  const innerHeight = height - padding.top - padding.bottom;
  return values
    .map((value, index) => {
      if (!Number.isFinite(value)) return "";
      const x = padding.left + (values.length <= 1 ? 0 : (index / (values.length - 1)) * innerWidth);
      const y = padding.top + (1 - (value - min) / span) * innerHeight;
      return `${index === 0 || !Number.isFinite(values[index - 1]) ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .filter(Boolean)
    .join(" ");
}

function renderChart(svg, series, options = {}) {
  const width = Math.max(320, Math.floor(svg.parentElement.clientWidth || 600));
  const height = Math.max(190, Math.floor(svg.parentElement.clientHeight || 220));
  const padding = { left: 48, right: 12, top: 14, bottom: 28 };
  const finiteValues = series.flatMap((line) => line.values.filter(Number.isFinite));
  if (!finiteValues.length && !options.emptyMessage) {
    throw new Error("There are not enough finite values to draw this chart.");
  }
  let min = finiteValues.length ? Math.min(...finiteValues) : Math.min(...(options.thresholds || [0]));
  let max = finiteValues.length ? Math.max(...finiteValues) : Math.max(...(options.thresholds || [0]));
  if (options.includeZero) {
    min = Math.min(0, min);
    max = Math.max(0, max);
  }
  if (options.thresholds) {
    min = Math.min(min, ...options.thresholds);
    max = Math.max(max, ...options.thresholds);
  }
  const extra = (max - min || Math.abs(max) * 0.1 || 1) * 0.12;
  min -= extra;
  max += extra;
  const span = max - min;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.replaceChildren();
  const ns = "http://www.w3.org/2000/svg";
  const make = (name, attributes) => {
    const element = document.createElementNS(ns, name);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    return element;
  };
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = max - (span * tick) / 4;
    const y = padding.top + ((height - padding.top - padding.bottom) * tick) / 4;
    svg.append(make("line", { x1: padding.left, x2: width - padding.right, y1: y, y2: y, class: "grid-line" }));
    const label = make("text", { x: padding.left - 8, y: y + 4, "text-anchor": "end", class: "axis-text" });
    label.textContent = options.percent ? `${value.toFixed(0)}%` : value.toLocaleString(undefined, { maximumFractionDigits: 2 });
    svg.append(label);
  }
  for (const threshold of options.thresholds || []) {
    const y = padding.top + (1 - (threshold - min) / span) * (height - padding.top - padding.bottom);
    svg.append(make("line", { x1: padding.left, x2: width - padding.right, y1: y, y2: y, class: threshold === 0 ? "zero-line" : "reference-line" }));
  }
  for (const line of series) {
    const d = pathFor(line.values, width, height, padding, min, max);
    if (!d) continue;
    svg.append(make("path", { d, class: "series-line", stroke: line.color }));
  }
  const first = make("text", { x: padding.left, y: height - 5, class: "axis-text" });
  first.textContent = options.firstLabel || "";
  const last = make("text", { x: width - padding.right, y: height - 5, "text-anchor": "end", class: "axis-text" });
  last.textContent = options.lastLabel || "";
  svg.append(first, last);
  if (!finiteValues.length) {
    const message = make("text", {
      x: width / 2,
      y: height / 2,
      "text-anchor": "middle",
      class: "axis-text",
    });
    message.textContent = options.emptyMessage;
    svg.append(message);
  }
}

function renderLegend(target, labels) {
  target.replaceChildren();
  for (const item of labels) {
    const wrapper = document.createElement("span");
    const line = document.createElement("i");
    line.className = "legend-line";
    line.style.backgroundColor = item.color;
    wrapper.append(line, document.createTextNode(item.label));
    target.append(wrapper);
  }
}

function renderWorkspace() {
  try {
    const { symbolA, symbolB, aligned } = currentSeries();
    const fit = fitOls(aligned);
    const window = Number($("#zscore-window").value);
    const zscore = zScores(fit.spread, window);
    const a = aligned.map((row) => row.a);
    const b = aligned.map((row) => row.b);
    const rebasedA = a.map((value) => (value / a[0]) * 100);
    const rebasedB = b.map((value) => (value / b[0]) * 100);
    const firstDate = dateLabel(aligned[0].timestamp);
    const lastDate = dateLabel(aligned[aligned.length - 1].timestamp);
    renderChart($("#price-chart"), [
      { values: a, color: COLORS[0] },
      { values: b, color: COLORS[1] },
    ], { firstLabel: firstDate, lastLabel: lastDate });
    renderLegend($("#price-legend"), [
      { label: symbolA, color: COLORS[0] },
      { label: symbolB, color: COLORS[1] },
    ]);
    renderChart($("#rebased-chart"), [
      { values: rebasedA, color: COLORS[0] },
      { values: rebasedB, color: COLORS[1] },
    ], { firstLabel: firstDate, lastLabel: lastDate });
    renderLegend($("#rebased-legend"), [
      { label: symbolA, color: COLORS[0] },
      { label: symbolB, color: COLORS[1] },
    ]);
    renderChart($("#spread-chart"), [
      { values: fit.spread, color: COLORS[0] },
    ], { firstLabel: firstDate, lastLabel: lastDate, thresholds: [0], includeZero: true });
    $("#spread-summary").textContent = `Full-sample OLS hedge ratio: ${fit.beta.toFixed(4)} · ${aligned.length} aligned observations`;
    renderChart($("#zscore-chart"), [
      { values: zscore, color: COLORS[0] },
    ], {
      firstLabel: firstDate,
      lastLabel: lastDate,
      thresholds: [-2, 0, 2],
      includeZero: true,
      emptyMessage: "Spread variation is too small for a z-score.",
    });
    $("#charts").hidden = false;
    $("#empty-state").hidden = true;
    $("#chart-footnote").hidden = false;
    setStatus(`Ready: ${aligned.length} common price dates for ${symbolA} / ${symbolB}. File data is kept in this browser session.`, "success");
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "Could not draw this pair.", "error");
  }
}

function installDataset(dataset, name) {
  state.rows = dataset.rows;
  state.symbols = dataset.symbols;
  state.name = name;
  populateSelect($("#symbol-a"), state.symbols, state.symbols[0]);
  populateSelect($("#symbol-b"), state.symbols, state.symbols[1]);
  const timestamps = state.rows.map((row) => row.timestamp).sort();
  $("#summary-name").textContent = name;
  $("#summary-rows").textContent = state.rows.length.toLocaleString();
  $("#summary-symbols").textContent = state.symbols.join(", ");
  $("#summary-range").textContent = `${dateLabel(timestamps[0])} – ${dateLabel(timestamps[timestamps.length - 1])}`;
  $("#dataset-summary").hidden = false;
  $("#pair-controls").hidden = false;
  $("#empty-state").hidden = true;
  $("#charts").hidden = false;
  renderWorkspace();
}

function activateSource(button) {
  for (const item of document.querySelectorAll(".source-button")) item.classList.toggle("is-active", item === button);
}

function initializeTheme() {
  const toggle = $("#theme-toggle");
  const system = window.matchMedia("(prefers-color-scheme: dark)");
  const label = () => {
    const dark = document.documentElement.dataset.theme === "dark";
    toggle.textContent = dark ? "Light mode" : "Dark mode";
    toggle.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    toggle.setAttribute("aria-pressed", String(dark));
    document.querySelector('meta[name="theme-color"]').content = dark ? "#101d1b" : "#f5f7f4";
  };
  toggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("pairwise-theme", next); } catch (_) { /* storage disabled */ }
    label();
  });
  system.addEventListener("change", (event) => {
    try { if (localStorage.getItem("pairwise-theme")) return; } catch (_) { /* storage disabled */ }
    document.documentElement.dataset.theme = event.matches ? "dark" : "light";
    label();
  });
  label();
}

function initialize() {
  initializeTheme();
  $("#demo-button").addEventListener("click", () => {
    activateSource($("#demo-button"));
    $("#csv-file").value = "";
    installDataset(demoPrices(), "Synthetic demo");
  });
  $("#empty-demo-button").addEventListener("click", () => $("#demo-button").click());
  $("#upload-button").addEventListener("click", () => {
    activateSource($("#upload-button"));
    $("#csv-file").click();
  });
  $("#csv-file").addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      setStatus("No file selected. Choose a CSV to load it in this browser.", "");
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      setStatus("This visual explorer accepts CSV files up to 25 MB.", "error");
      event.target.value = "";
      return;
    }
    setStatus(`Reading ${file.name} locally…`, "");
    try {
      const dataset = parsePrices(await file.text());
      installDataset(dataset, file.name);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not parse this CSV.", "error");
      $("#dataset-summary").hidden = true;
      $("#pair-controls").hidden = true;
      $("#charts").hidden = true;
      $("#empty-state").hidden = false;
      event.target.value = "";
    }
  });
  $("#symbol-a").addEventListener("change", () => {
    const symbolB = $("#symbol-b");
    if ($("#symbol-a").value === symbolB.value) {
      symbolB.value = state.symbols.find((symbol) => symbol !== $("#symbol-a").value) || "";
    }
    renderWorkspace();
  });
  $("#symbol-b").addEventListener("change", renderWorkspace);
  $("#zscore-window").addEventListener("change", renderWorkspace);
  $("#api-url").value = sessionStorage.getItem("pairwise-api-url")
    || "https://pairs-trading-api-production.up.railway.app";
  $("#run-backtest").addEventListener("click", runPythonBacktest);
  window.addEventListener("resize", () => {
    if (!$("#charts").hidden) renderWorkspace();
  });
  $("#empty-state").hidden = true;
  installDataset(demoPrices(), "Synthetic demo");
}

document.addEventListener("DOMContentLoaded", initialize);
