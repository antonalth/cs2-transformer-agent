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
    <div class="row">
      <div>
      <h2>Logits</h2>
      <pre id="logits">paused inspection only</pre>
      </div>
      <div>
        <h2>Last State</h2>
        <pre id="state-secondary">loading...</pre>
      </div>
    </div>
  </div>
  <script>
    let refreshTimer = null;

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
      document.getElementById('harness-url').value = data.harness_url || '';
    }

    async function refreshLogits() {
      const playerIndex = Number(document.getElementById('player-index').value || '0');
      const res = await fetch(`/api/logits/${playerIndex}`);
      const data = await res.json();
      document.getElementById('logits').textContent = JSON.stringify(data, null, 2);
    }

    async function refreshComposite() {
      const img = document.getElementById('composite');
      img.src = `/api/composite.jpg?ts=${Date.now()}`;
    }

    async function applyHarnessUrl() {
      const harnessUrl = document.getElementById('harness-url').value.trim();
      await post('/api/harness', {harness_url: harnessUrl});
    }

    async function setCacheWindow() {
      const raw = document.getElementById('cache-window').value.trim();
      const payload = raw ? {cache_window_frames: Number(raw)} : {cache_window_frames: null};
      await post('/api/cache_window', payload);
    }

    function applyRefreshLoop() {
      if (refreshTimer) {
        clearInterval(refreshTimer);
      }
      const fps = Math.max(0.1, Number(document.getElementById('refresh-fps').value || '2'));
      refreshTimer = setInterval(refreshComposite, Math.max(50, Math.floor(1000 / fps)));
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
