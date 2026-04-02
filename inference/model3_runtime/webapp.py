from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .runtime import Model3InferenceRuntime

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response
except ModuleNotFoundError:
    FastAPI = Any
    HTTPException = RuntimeError
    HTMLResponse = Any
    Response = Any


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Model3 Runtime</title>
  <style>
    body { font-family: monospace; background: #11161a; color: #eef3f5; margin: 20px; }
    .tabs { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
    .tabs button.active { background: #1d2a31; border-color: #7cd1f0; }
    .view { display: none; }
    .view.active { display: block; }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; align-items: center; }
    button, input { background: #0c1013; color: #eef3f5; border: 1px solid #334049; padding: 8px 10px; }
    pre { background: #0c1013; border: 1px solid #334049; padding: 12px; overflow: auto; }
    .row { display: grid; grid-template-columns: minmax(480px, 1.4fr) minmax(360px, 1fr); gap: 16px; }
    img { width: 100%; border: 1px solid #334049; display: block; background: #0c1013; }
    .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }
    .chart-card { background: #0c1013; border: 1px solid #334049; padding: 12px; }
    .settings-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; width: 100%; }
    .settings-grid label { display: grid; gap: 4px; font-size: 12px; color: #c8d8df; }
    .chart-title { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 8px; }
    .chart-title h3 { margin: 0; font-size: 15px; }
    .chart-meta { color: #97aeb9; font-size: 12px; }
    .chart-summary { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
    .chip { border: 1px solid #2d434e; background: #111a1f; color: #bfefff; padding: 2px 6px; font-size: 12px; }
    .svg-wrap { border: 1px solid #263840; background: linear-gradient(180deg, #0d1519 0%, #0a1013 100%); }
    .svg-wrap svg { display: block; width: 100%; height: auto; }
    .top-list { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; margin-top: 10px; font-size: 12px; color: #c8d8df; }
    .top-list .prob { color: #7cd1f0; text-align: right; }
    .full-list-wrap { margin-top: 10px; border-top: 1px solid #22343d; padding-top: 10px; }
    .full-list-title { color: #97aeb9; font-size: 12px; margin-bottom: 6px; }
    .full-list { max-height: 220px; overflow: auto; display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; font-size: 12px; color: #c8d8df; }
    .full-list .prob { color: #7cd1f0; text-align: right; }
    .bar-list { display: grid; gap: 6px; margin-top: 6px; }
    .bar-row { display: grid; grid-template-columns: minmax(110px, 160px) 1fr auto; gap: 8px; align-items: center; font-size: 12px; }
    .bar-label { color: #c8d8df; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bar-track { height: 10px; background: #121c21; border: 1px solid #263840; position: relative; }
    .bar-fill { position: absolute; inset: 0 auto 0 0; background: linear-gradient(90deg, #3dd9c5 0%, #7cd1f0 100%); }
    .bar-value { color: #7cd1f0; min-width: 52px; text-align: right; }
    .empty-state { background: #0c1013; border: 1px dashed #334049; padding: 20px; color: #97aeb9; }
    .decision-box { background: #0c1013; border: 1px solid #334049; padding: 12px; margin-bottom: 14px; }
    .decision-title { font-size: 14px; margin: 0 0 8px 0; }
    .decision-line { margin: 4px 0; color: #c8d8df; font-size: 12px; }
    .decision-line strong { color: #eef3f5; }
  </style>
</head>
<body>
  <h1>Model3 Runtime</h1>
  <div class="tabs">
    <button id="tab-view" class="active" onclick="switchView('view')">view</button>
    <button id="tab-logits" onclick="switchView('logits')">logits</button>
  </div>

  <div id="view-view" class="view active">
    <div class="toolbar">
      <label>harness url
        <input id="harness-url" type="text" size="42">
      </label>
      <button onclick="applyHarnessUrl()">set url</button>
      <button onclick="post('/api/connect')">connect</button>
      <button onclick="post('/api/disconnect')">disconnect</button>
      <button onclick="post('/api/start')">start</button>
      <button onclick="post('/api/pause')">pause</button>
      <button onclick="post('/api/reset')">reset</button>
      <label>cache window
        <input id="cache-window" type="number" min="1" step="1" placeholder="no drop">
      </label>
      <button onclick="setCacheWindow()">apply</button>
      <label>refresh fps
        <input id="refresh-fps" type="number" min="0.1" max="30" step="0.1" value="2">
      </label>
      <button onclick="refreshComposite()">refresh now</button>
      <button onclick="post('/api/record/start')">record start</button>
      <button onclick="post('/api/record/stop')">record stop</button>
    </div>
    <div class="row">
      <div>
        <img id="composite" src="" alt="runtime composite">
      </div>
      <div>
        <h2>State</h2>
        <pre id="state">loading...</pre>
      </div>
    </div>
  </div>

  <div id="view-logits" class="view">
    <div class="toolbar">
      <label>player
        <input id="player-index" type="number" min="0" max="4" step="1" value="0">
      </label>
      <button onclick="refreshLogits()">load logits</button>
    </div>
    <div class="toolbar">
      <div class="settings-grid">
        <label>keyboard threshold
          <input id="keyboard-threshold" type="number" min="0" max="1" step="0.01" value="0.12">
        </label>
        <label>mouse zero bias
          <input id="mouse-zero-bias" type="number" step="0.1" value="0">
        </label>
        <label>mouse scale
          <input id="mouse-scale" type="number" min="0" step="0.05" value="1.0">
        </label>
        <label>mouse temperature
          <input id="mouse-temperature" type="number" min="0.01" step="0.05" value="1.0">
        </label>
        <label>mouse top-k
          <input id="mouse-top-k" type="number" min="1" step="1" value="1">
        </label>
        <label>mouse deadzone
          <input id="mouse-deadzone" type="number" min="0" step="0.05" value="0.35">
        </label>
      </div>
      <button onclick="applyActionDecode()">apply decode</button>
    </div>
    <div class="row">
      <div>
      <div id="decision-path" class="decision-box">
        <h2 class="decision-title">Selected Path</h2>
        <div class="decision-line">no decoded actions yet</div>
      </div>
      <h2>Distributions</h2>
      <div id="logits-graphs" class="charts-grid">
        <div class="empty-state">paused inspection only</div>
      </div>
      </div>
      <div>
        <h2>Last State</h2>
        <pre id="state-secondary">loading...</pre>
      </div>
    </div>
  </div>
  <script>
    let refreshTimer = null;
    const stickyInputIds = [
      'harness-url',
      'cache-window',
      'keyboard-threshold',
      'mouse-zero-bias',
      'mouse-scale',
      'mouse-temperature',
      'mouse-top-k',
      'mouse-deadzone',
    ];

    function markDirty(event) {
      event.target.dataset.dirty = 'true';
    }

    function clearDirty(ids) {
      for (const id of ids) {
        const el = document.getElementById(id);
        if (el) {
          delete el.dataset.dirty;
        }
      }
    }

    function syncInputValue(id, value) {
      const el = document.getElementById(id);
      if (!el) return;
      if (el.dataset.dirty === 'true') return;
      if (document.activeElement === el) return;
      el.value = value ?? '';
    }

    function switchView(name) {
      for (const id of ['view', 'logits']) {
        document.getElementById(`tab-${id}`).classList.toggle('active', id === name);
        document.getElementById(`view-${id}`).classList.toggle('active', id === name);
      }
    }

    async function post(path, payload) {
      const res = await fetch(path, {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: payload ? JSON.stringify(payload) : '{}',
      });
      const data = await res.json();
      await refreshState();
      await refreshComposite();
      return data;
    }

    async function refreshState() {
      const res = await fetch('/api/state');
      const data = await res.json();
      document.getElementById('state').textContent = JSON.stringify(data, null, 2);
      document.getElementById('state-secondary').textContent = JSON.stringify(data, null, 2);
      syncInputValue('harness-url', data.harness_url || '');
      const decode = data?.runtime_options?.action_decode || {};
      if (decode.keyboard_threshold !== undefined) syncInputValue('keyboard-threshold', decode.keyboard_threshold);
      if (decode.mouse_zero_bias !== undefined) syncInputValue('mouse-zero-bias', decode.mouse_zero_bias);
      if (decode.mouse_scale !== undefined) syncInputValue('mouse-scale', decode.mouse_scale);
      if (decode.mouse_temperature !== undefined) syncInputValue('mouse-temperature', decode.mouse_temperature);
      if (decode.mouse_top_k !== undefined) syncInputValue('mouse-top-k', decode.mouse_top_k);
      if (decode.mouse_deadzone !== undefined) syncInputValue('mouse-deadzone', decode.mouse_deadzone);
      syncInputValue('cache-window', data.cache_window_frames ?? '');
    }

    function fmtProb(value) {
      return `${(100 * Number(value || 0)).toFixed(1)}%`;
    }

    function sampleTickIndices(count, maxTicks = 6) {
      if (count <= maxTicks) {
        return Array.from({length: count}, (_, idx) => idx);
      }
      const indices = [];
      for (let i = 0; i < maxTicks; i++) {
        indices.push(Math.round(i * (count - 1) / (maxTicks - 1)));
      }
      return [...new Set(indices)];
    }

    function createSvgElement(tag, attrs = {}) {
      const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [key, value] of Object.entries(attrs)) {
        el.setAttribute(key, String(value));
      }
      return el;
    }

    function renderCategoricalChart(panel) {
      const width = 640;
      const height = 230;
      const left = 56;
      const right = 18;
      const top = 16;
      const bottom = 40;
      const plotWidth = width - left - right;
      const plotHeight = height - top - bottom;
      const points = panel.probabilities || [];
      const labels = panel.x_labels || [];
      const svg = createSvgElement('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img'});

      svg.appendChild(createSvgElement('rect', {
        x: left,
        y: top,
        width: plotWidth,
        height: plotHeight,
        fill: '#0b1317',
        stroke: '#263840',
      }));

      for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
        const y = top + (1 - tick) * plotHeight;
        svg.appendChild(createSvgElement('line', {
          x1: left,
          y1: y,
          x2: left + plotWidth,
          y2: y,
          stroke: '#22343d',
          'stroke-width': 1,
        }));
        const label = createSvgElement('text', {
          x: left - 8,
          y: y + 4,
          fill: '#8da5b0',
          'font-size': 11,
          'text-anchor': 'end',
        });
        label.textContent = tick.toFixed(2);
        svg.appendChild(label);
      }

      if (points.length > 0) {
        const step = points.length > 1 ? plotWidth / (points.length - 1) : 0;
        const coords = points.map((value, idx) => {
          const x = left + (idx * step);
          const y = top + (1 - Number(value)) * plotHeight;
          return [x, y];
        });
        const linePoints = coords.map(([x, y]) => `${x},${y}`).join(' ');
        const areaPoints = `${left},${top + plotHeight} ${linePoints} ${left + plotWidth},${top + plotHeight}`;
        svg.appendChild(createSvgElement('polygon', {
          points: areaPoints,
          fill: 'rgba(61, 217, 197, 0.18)',
          stroke: 'none',
        }));
        svg.appendChild(createSvgElement('polyline', {
          points: linePoints,
          fill: 'none',
          stroke: '#7cd1f0',
          'stroke-width': 2,
        }));

        const predIdx = Number(panel.predicted_index || 0);
        if (predIdx >= 0 && predIdx < coords.length) {
          const [px, py] = coords[predIdx];
          svg.appendChild(createSvgElement('circle', {
            cx: px,
            cy: py,
            r: 4,
            fill: '#ffce6e',
            stroke: '#f6f1d1',
            'stroke-width': 1.2,
          }));
        }

        for (const idx of sampleTickIndices(labels.length)) {
          const x = left + (idx * step);
          svg.appendChild(createSvgElement('line', {
            x1: x,
            y1: top + plotHeight,
            x2: x,
            y2: top + plotHeight + 5,
            stroke: '#59717c',
          }));
          const text = createSvgElement('text', {
            x,
            y: top + plotHeight + 18,
            fill: '#8da5b0',
            'font-size': 10,
            'text-anchor': 'middle',
          });
          text.textContent = labels[idx];
          svg.appendChild(text);
        }
      }

      return svg;
    }

    function renderBarList(panel) {
      const wrap = document.createElement('div');
      wrap.className = 'bar-list';
      const labels = panel.x_labels || [];
      const probs = panel.probabilities || [];
      for (let i = 0; i < labels.length; i++) {
        const row = document.createElement('div');
        row.className = 'bar-row';

        const label = document.createElement('div');
        label.className = 'bar-label';
        label.textContent = labels[i];

        const track = document.createElement('div');
        track.className = 'bar-track';
        const fill = document.createElement('div');
        fill.className = 'bar-fill';
        fill.style.width = `${Math.max(0, Math.min(100, 100 * Number(probs[i] || 0)))}%`;
        track.appendChild(fill);

        const value = document.createElement('div');
        value.className = 'bar-value';
        value.textContent = fmtProb(probs[i]);

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(value);
        wrap.appendChild(row);
      }
      return wrap;
    }

    function createChip(text) {
      const chip = document.createElement('div');
      chip.className = 'chip';
      chip.textContent = text;
      return chip;
    }

    function renderPanel(panel) {
      const card = document.createElement('article');
      card.className = 'chart-card';

      const title = document.createElement('div');
      title.className = 'chart-title';
      const h3 = document.createElement('h3');
      h3.textContent = panel.head_name;
      const meta = document.createElement('div');
      meta.className = 'chart-meta';
      meta.textContent = panel.representation || '';
      title.appendChild(h3);
      title.appendChild(meta);
      card.appendChild(title);

      const summary = document.createElement('div');
      summary.className = 'chart-summary';
      if (panel.predicted_label !== undefined) {
        summary.appendChild(createChip(`top: ${panel.predicted_label}`));
      }
      if (panel.predicted_probability !== undefined) {
        summary.appendChild(createChip(`p=${fmtProb(panel.predicted_probability)}`));
      }
      if (panel.active_actions && panel.active_actions.length) {
        summary.appendChild(createChip(`active: ${panel.active_actions.join(', ')}`));
      }
      card.appendChild(summary);

      const plot = document.createElement('div');
      plot.className = 'svg-wrap';
      if (panel.kind === 'categorical') {
        plot.appendChild(renderCategoricalChart(panel));
      } else {
        plot.appendChild(renderBarList(panel));
      }
      card.appendChild(plot);

      const topList = document.createElement('div');
      topList.className = 'top-list';
      const topCategories = panel.top_categories || [];
      for (const item of topCategories) {
        const label = document.createElement('div');
        label.textContent = item.label;
        const prob = document.createElement('div');
        prob.className = 'prob';
        prob.textContent = fmtProb(item.probability);
        topList.appendChild(label);
        topList.appendChild(prob);
      }
      card.appendChild(topList);

      const fullWrap = document.createElement('div');
      fullWrap.className = 'full-list-wrap';
      const fullTitle = document.createElement('div');
      fullTitle.className = 'full-list-title';
      fullTitle.textContent = 'all bins';
      const fullList = document.createElement('div');
      fullList.className = 'full-list';
      const fullCategories = panel.full_categories || [];
      for (const item of fullCategories) {
        const label = document.createElement('div');
        label.textContent = item.label;
        const prob = document.createElement('div');
        prob.className = 'prob';
        prob.textContent = fmtProb(item.probability);
        fullList.appendChild(label);
        fullList.appendChild(prob);
      }
      fullWrap.appendChild(fullTitle);
      fullWrap.appendChild(fullList);
      card.appendChild(fullWrap);
      return card;
    }

    function renderLogits(data) {
      const container = document.getElementById('logits-graphs');
      container.innerHTML = '';
      const panels = Array.isArray(data?.panels) ? data.panels : [];
      if (!panels.length) {
        const empty = document.createElement('div');
        empty.className = 'empty-state';
        empty.textContent = 'no logits available yet';
        container.appendChild(empty);
        return;
      }
      for (const panel of panels) {
        container.appendChild(renderPanel(panel));
      }
    }

    function renderDecisionPath(data) {
      const wrap = document.getElementById('decision-path');
      wrap.innerHTML = '';
      const title = document.createElement('h2');
      title.className = 'decision-title';
      title.textContent = 'Selected Path';
      wrap.appendChild(title);

      const summary = data?.action_summary || {};
      const trace = summary?.decision_trace || {};
      if (!Object.keys(trace).length) {
        const line = document.createElement('div');
        line.className = 'decision-line';
        line.textContent = 'no decoded actions yet';
        wrap.appendChild(line);
        return;
      }

      const selectedKeys = (trace.keyboard?.selected_actions || []).join(', ') || 'none';
      const keyLine = document.createElement('div');
      keyLine.className = 'decision-line';
      keyLine.innerHTML = `<strong>keyboard:</strong> threshold ${Number(trace.keyboard?.threshold || 0).toFixed(2)} -> ${selectedKeys}`;
      wrap.appendChild(keyLine);

      const dropped = trace.keyboard?.contrary_resolutions || [];
      if (dropped.length) {
        const dropLine = document.createElement('div');
        dropLine.className = 'decision-line';
        dropLine.innerHTML = `<strong>contrary filter:</strong> ${dropped.map(item => `${item.dropped} dropped for ${item.winner}`).join('; ')}`;
        wrap.appendChild(dropLine);
      }

      for (const axis of ['mouse_x', 'mouse_y']) {
        const axisTrace = trace[axis] || {};
        const line = document.createElement('div');
        line.className = 'decision-line';
        const selected = axisTrace.selected_bin;
        const delta = Number(axisTrace.selected_delta || 0).toFixed(2);
        const zeroProb = fmtProb(axisTrace.zero_bin_probability || 0);
        line.innerHTML = `<strong>${axis}:</strong> zero-bias ${Number(axisTrace.zero_bias || 0).toFixed(2)}, temp ${Number(axisTrace.temperature || 1).toFixed(2)}, top-k ${axisTrace.top_k || 1} -> bin ${selected}, delta ${delta}, zero-bin ${zeroProb}`;
        wrap.appendChild(line);

        const candidates = axisTrace.top_k_candidates || [];
        if (candidates.length) {
          const candidateLine = document.createElement('div');
          candidateLine.className = 'decision-line';
          candidateLine.textContent = `candidates: ${candidates.map(item => `#${item.bin_index} ${fmtProb(item.probability)}`).join(', ')}`;
          wrap.appendChild(candidateLine);
        }
      }

      const eventLine = document.createElement('div');
      eventLine.className = 'decision-line';
      eventLine.innerHTML = `<strong>emitted:</strong> ${(summary.emitted_events || []).length ? JSON.stringify(summary.emitted_events) : 'no events'}`;
      wrap.appendChild(eventLine);
    }

    async function refreshLogits() {
      const playerIndex = Number(document.getElementById('player-index').value || '0');
      const res = await fetch(`/api/logits/${playerIndex}`);
      const data = await res.json();
      renderDecisionPath(data);
      renderLogits(data);
    }

    async function refreshComposite() {
      const img = document.getElementById('composite');
      img.src = `/api/composite.jpg?ts=${Date.now()}`;
    }

    async function applyHarnessUrl() {
      const harnessUrl = document.getElementById('harness-url').value.trim();
      clearDirty(['harness-url']);
      await post('/api/harness', {harness_url: harnessUrl});
    }

    async function setCacheWindow() {
      const raw = document.getElementById('cache-window').value.trim();
      const payload = raw ? {cache_window_frames: Number(raw)} : {cache_window_frames: null};
      clearDirty(['cache-window']);
      await post('/api/cache_window', payload);
    }

    async function applyActionDecode() {
      clearDirty([
        'keyboard-threshold',
        'mouse-zero-bias',
        'mouse-scale',
        'mouse-temperature',
        'mouse-top-k',
        'mouse-deadzone',
      ]);
      await post('/api/action_decode', {
        keyboard_threshold: Number(document.getElementById('keyboard-threshold').value || '0.12'),
        mouse_zero_bias: Number(document.getElementById('mouse-zero-bias').value || '0'),
        mouse_scale: Number(document.getElementById('mouse-scale').value || '1'),
        mouse_temperature: Number(document.getElementById('mouse-temperature').value || '1'),
        mouse_top_k: Number(document.getElementById('mouse-top-k').value || '1'),
        mouse_deadzone: Number(document.getElementById('mouse-deadzone').value || '0.35'),
      });
      await refreshLogits();
    }

    function applyRefreshLoop() {
      if (refreshTimer) {
        clearInterval(refreshTimer);
      }
      const fps = Math.max(0.1, Number(document.getElementById('refresh-fps').value || '2'));
      refreshTimer = setInterval(refreshComposite, Math.max(50, Math.floor(1000 / fps)));
    }

    for (const id of stickyInputIds) {
      document.getElementById(id).addEventListener('input', markDirty);
    }
    document.getElementById('refresh-fps').addEventListener('change', applyRefreshLoop);
    refreshState();
    refreshComposite();
    refreshLogits();
    applyRefreshLoop();
    setInterval(refreshState, 1000);
  </script>
</body>
</html>
"""


def create_app(runtime: Model3InferenceRuntime) -> FastAPI:
    if FastAPI is Any:
        raise RuntimeError("fastapi is not installed. Install inference requirements first.")

    app = FastAPI(title="model3-runtime")
    app.state.runtime = runtime

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return INDEX_HTML

    @app.get("/api/state")
    async def state():
        return asdict(await runtime.snapshot())

    @app.get("/api/composite.jpg")
    async def composite():
        payload = await runtime.latest_composite_jpeg()
        if payload is None:
            raise HTTPException(status_code=404, detail="no composite available yet")
        return Response(content=payload, media_type="image/jpeg")

    @app.get("/api/logits/{player_index}")
    async def logits(player_index: int):
        try:
            return await runtime.latest_logits(player_index)
        except IndexError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/connect")
    async def connect():
        try:
            await runtime.connect()
            return {"ok": True}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/disconnect")
    async def disconnect():
        await runtime.disconnect()
        return {"ok": True}

    @app.post("/api/start")
    async def start():
        try:
            await runtime.start()
            return {"ok": True}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/pause")
    async def pause():
        await runtime.pause()
        return {"ok": True}

    @app.post("/api/reset")
    async def reset():
        await runtime.reset()
        return {"ok": True}

    @app.post("/api/cache_window")
    async def cache_window(payload: dict[str, Any]):
        cache_window_frames = payload.get("cache_window_frames")
        if cache_window_frames is not None:
            cache_window_frames = int(cache_window_frames)
        await runtime.set_cache_window_frames(cache_window_frames)
        return {"ok": True}

    @app.post("/api/harness")
    async def set_harness(payload: dict[str, Any]):
        harness_url = str(payload.get("harness_url", "")).strip()
        if not harness_url:
            raise HTTPException(status_code=400, detail="harness_url is required")
        await runtime.set_harness_url(harness_url)
        return {"ok": True}

    @app.post("/api/action_decode")
    async def set_action_decode(payload: dict[str, Any]):
        try:
            updated = await runtime.set_action_decode_config(payload)
            return {"ok": True, "action_decode": updated}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/record/start")
    async def record_start():
        try:
            path = await runtime.start_recording()
            return {"ok": True, "path": path}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/record/stop")
    async def record_stop():
        path = await runtime.stop_recording()
        return {"ok": True, "path": path}

    return app
