from __future__ import annotations

from pathlib import Path

from .config import load_config
from .supervisor import HarnessSupervisor


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CS2 Sim Harness</title>
  <style>
    body { font-family: monospace; background: #101416; color: #f2f4f5; margin: 24px; }
    h1, h2 { margin: 0 0 12px 0; }
    .tabs { display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }
    .tabs button { background: #0f1518; color: #f2f4f5; border: 1px solid #2b3338; padding: 8px 12px; }
    .tabs button.active { background: #21343c; border-color: #6ab7d6; }
    .view { display: none; }
    .view.active { display: block; }
    .layout { display: grid; grid-template-columns: minmax(420px, 1.4fr) minmax(460px, 1fr); gap: 20px; align-items: start; }
    .panel { background: #161d21; border: 1px solid #2b3338; padding: 16px; }
    img { width: 100%; border: 1px solid #2b3338; display: block; background: #0b0f11; }
    .meta { color: #a8b3ba; margin-top: 10px; font-size: 12px; }
    .toolbar { display: flex; gap: 10px; align-items: center; margin: 10px 0 12px 0; flex-wrap: wrap; }
    .toolbar label { color: #a8b3ba; }
    .toolbar input { width: 80px; background: #0f1518; color: #f2f4f5; border: 1px solid #2b3338; padding: 4px 6px; }
    .toolbar select { background: #0f1518; color: #f2f4f5; border: 1px solid #2b3338; padding: 4px 6px; }
    table { border-collapse: collapse; width: 100%; }
    td, th { border: 1px solid #2b3338; padding: 8px; text-align: left; vertical-align: top; }
    th { background: #0f1518; }
    button { margin-right: 8px; margin-bottom: 6px; }
    .status-ready { color: #87f7a5; }
    .status-error { color: #ff8f8f; }
    .status-launching, .status-discovering, .status-stopping { color: #ffd37a; }
    .mono { white-space: pre-wrap; word-break: break-word; }
    .slot-tabs { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
    .slot-tabs button { background: #0f1518; color: #f2f4f5; border: 1px solid #2b3338; padding: 6px 10px; }
    .slot-tabs button.active { background: #21343c; border-color: #6ab7d6; }
    .control-layout { display: grid; grid-template-columns: minmax(540px, 1.4fr) minmax(320px, 1fr); gap: 20px; align-items: start; }
    .control-surface { position: relative; border: 1px solid #2b3338; background: #0b0f11; }
    .control-surface img { display: block; width: 100%; border: 0; }
    .control-surface.armed { outline: 2px solid #6ab7d6; }
    .control-hint { position: absolute; left: 12px; bottom: 12px; background: rgba(0, 0, 0, 0.55); padding: 6px 8px; font-size: 12px; }
    .event-log { background: #0f1518; border: 1px solid #2b3338; min-height: 120px; max-height: 300px; overflow: auto; padding: 8px; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>CS2 Sim Harness</h1>
  <div class="tabs">
    <button id="tab-overview" class="active" onclick="switchView('overview')">Overview</button>
    <button id="tab-control" onclick="switchView('control')">Control</button>
  </div>

  <div id="view-overview" class="view active">
    <div class="layout">
      <div class="panel">
        <h2>Composite</h2>
        <div class="toolbar">
          <label for="refresh-fps">Refresh FPS</label>
          <input id="refresh-fps" type="number" min="0.1" max="30" step="0.1">
          <button onclick="refreshComposite()">refresh now</button>
        </div>
        <img id="composite" src="/api/composite.jpg" alt="composite view">
        <div class="meta" id="composite-meta">waiting for refresh</div>
      </div>
      <div class="panel">
        <h2>Slots</h2>
        <table id="slots"></table>
      </div>
    </div>
  </div>

  <div id="view-control" class="view">
    <div class="panel">
      <h2>Control</h2>
      <div class="slot-tabs" id="control-slot-tabs"></div>
      <div class="control-layout">
        <div class="panel">
          <div class="toolbar">
            <label for="control-refresh-fps">View FPS</label>
            <input id="control-refresh-fps" type="number" min="0.1" max="30" step="0.1">
            <button id="arm-control" onclick="toggleControlArmed()">arm control</button>
            <button onclick="calibrateTopLeft()">calibrate top-left</button>
            <button onclick="refreshControlFrame()">refresh now</button>
          </div>
          <div id="control-surface" class="control-surface" tabindex="0">
            <img id="control-frame" src="" alt="slot control view">
            <div class="control-hint" id="control-hint">disarmed</div>
          </div>
          <div class="meta" id="control-meta">select a slot</div>
        </div>
        <div class="panel">
          <h2>Selected Slot</h2>
          <div class="mono" id="control-slot-details"></div>
          <h2 style="margin-top:16px;">Recent Events</h2>
          <div class="event-log" id="event-log">no events yet</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const browserFps = __BROWSER_FPS__;
    let refreshTimer = null;
    let controlRefreshTimer = null;
    let currentView = 'overview';
    let latestSlots = [];
    let selectedControlSlot = null;
    let controlArmed = false;
    let pressedKeys = new Set();
    let pressedButtons = new Set();
    let lastMouseSendMs = 0;
    const browserKeyMap = {
      KeyA: 'a', KeyB: 'b', KeyC: 'c', KeyD: 'd', KeyE: 'e', KeyF: 'f', KeyG: 'g', KeyH: 'h',
      KeyI: 'i', KeyJ: 'j', KeyK: 'k', KeyL: 'l', KeyM: 'm', KeyN: 'n', KeyO: 'o', KeyP: 'p',
      KeyQ: 'q', KeyR: 'r', KeyS: 's', KeyT: 't', KeyU: 'u', KeyV: 'v', KeyW: 'w', KeyX: 'x',
      KeyY: 'y', KeyZ: 'z',
      Digit0: '0', Digit1: '1', Digit2: '2', Digit3: '3', Digit4: '4',
      Digit5: '5', Digit6: '6', Digit7: '7', Digit8: '8', Digit9: '9',
      Space: 'space', Enter: 'enter', Tab: 'tab', Escape: 'esc', Backspace: 'backspace',
      ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down',
      ShiftLeft: 'lshift', ShiftRight: 'rshift', ControlLeft: 'lctrl', ControlRight: 'rctrl',
      AltLeft: 'lalt', AltRight: 'ralt',
      Minus: 'minus', Equal: 'equals', Comma: 'comma', Period: 'period', Slash: 'slash',
      Semicolon: 'semicolon', Quote: 'apostrophe', BracketLeft: 'leftbracket', BracketRight: 'rightbracket',
      Backslash: 'backslash', Backquote: 'grave',
      F1: 'f1', F2: 'f2', F3: 'f3', F4: 'f4', F5: 'f5', F6: 'f6',
      F7: 'f7', F8: 'f8', F9: 'f9', F10: 'f10', F11: 'f11', F12: 'f12',
    };
    const mouseButtonMap = { 0: 1, 1: 2, 2: 3, 3: 4, 4: 5 };

    function esc(value) {
      return String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    }

    function statusClass(status) {
      return 'status-' + status;
    }

    function refreshComposite() {
      const ts = Date.now();
      const img = document.getElementById('composite');
      img.src = '/api/composite.jpg?ts=' + ts;
      const fps = getRefreshFps();
      document.getElementById('composite-meta').textContent =
        'last refresh: ' + new Date(ts).toLocaleTimeString() + ' | refresh fps: ' + fps.toFixed(1);
    }

    function getRefreshFps() {
      const raw = Number(document.getElementById('refresh-fps').value);
      if (!Number.isFinite(raw)) {
        return browserFps;
      }
      return Math.min(30, Math.max(0.1, raw));
    }

    function scheduleCompositeRefresh() {
      if (refreshTimer !== null) {
        clearInterval(refreshTimer);
      }
      const fps = getRefreshFps();
      refreshTimer = setInterval(refreshComposite, Math.max(100, Math.floor(1000 / fps)));
    }

    function getControlRefreshFps() {
      const raw = Number(document.getElementById('control-refresh-fps').value);
      if (!Number.isFinite(raw)) {
        return browserFps;
      }
      return Math.min(30, Math.max(0.1, raw));
    }

    function switchView(view) {
      currentView = view;
      document.getElementById('view-overview').classList.toggle('active', view === 'overview');
      document.getElementById('view-control').classList.toggle('active', view === 'control');
      document.getElementById('tab-overview').classList.toggle('active', view === 'overview');
      document.getElementById('tab-control').classList.toggle('active', view === 'control');
      if (view === 'control') {
        refreshControlFrame();
      }
    }

    function updateSlotTabs() {
      const root = document.getElementById('control-slot-tabs');
      if (!selectedControlSlot && latestSlots.length > 0) {
        selectedControlSlot = latestSlots[0].name;
      }
      root.innerHTML = latestSlots.map(slot => (
        `<button class="${slot.name === selectedControlSlot ? 'active' : ''}" onclick="selectControlSlot('${slot.name}')">${slot.name}</button>`
      )).join('');
    }

    function selectControlSlot(slotName) {
      selectedControlSlot = slotName;
      updateSlotTabs();
      updateSelectedSlotPanel();
      refreshControlFrame();
    }

    function selectedSlotSnapshot() {
      return latestSlots.find(slot => slot.name === selectedControlSlot) ?? null;
    }

    function updateSelectedSlotPanel() {
      const slot = selectedSlotSnapshot();
      const details = document.getElementById('control-slot-details');
      if (!slot) {
        details.textContent = 'no slot selected';
        return;
      }
      details.textContent = [
        `name=${slot.name}`,
        `status=${slot.status}`,
        `user=${slot.user}`,
        `pid=${slot.process_id ?? ''}`,
        `node=${slot.pipewire_node_id ?? ''}`,
        `client=${slot.pipewire_client_id ?? ''}`,
        `gamescope_socket=${slot.gamescope_socket ?? ''}`,
        `eis_socket=${slot.eis_socket ?? ''}`,
        `error=${slot.error ?? ''}`,
      ].join('\\n');
    }

    function logEvent(line) {
      const root = document.getElementById('event-log');
      const stamp = new Date().toLocaleTimeString();
      const next = `[${stamp}] ${line}\\n`;
      root.textContent = (next + root.textContent).slice(0, 4000);
    }

    async function sendControlEvent(payload) {
      if (!selectedControlSlot) {
        return;
      }
      const res = await fetch(`/api/slots/${selectedControlSlot}/input`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        let detail = `http ${res.status}`;
        try {
          const body = await res.json();
          detail = body.detail ?? JSON.stringify(body);
        } catch (_) {}
        logEvent(`input error: ${detail}`);
      }
    }

    async function calibrateTopLeft() {
      if (!selectedControlSlot) {
        return;
      }
      await sendControlEvent({ t: 'mouse_abs', x: 0.0, y: 0.0, visible: true });
      logEvent(`calibrate top-left sent to ${selectedControlSlot}`);
      refreshControlFrame();
    }

    function releaseAllPressed() {
      for (const code of Array.from(pressedKeys)) {
        const mapped = browserKeyMap[code];
        if (mapped) {
          sendControlEvent({ t: 'key', key: mapped, down: false });
        }
      }
      for (const button of Array.from(pressedButtons)) {
        sendControlEvent({ t: 'mouse_btn', button, down: false });
      }
      pressedKeys.clear();
      pressedButtons.clear();
      sendControlEvent({ t: 'cursor', x: 0.5, y: 0.5, visible: false });
    }

    function updateControlArmedUi() {
      document.getElementById('arm-control').textContent = controlArmed ? 'disarm control' : 'arm control';
      document.getElementById('control-hint').textContent = controlArmed
        ? `armed for ${selectedControlSlot ?? 'no slot'}`
        : 'disarmed';
      document.getElementById('control-surface').classList.toggle('armed', controlArmed);
    }

    function toggleControlArmed() {
      controlArmed = !controlArmed;
      if (!controlArmed) {
        releaseAllPressed();
      } else {
        document.getElementById('control-surface').focus();
      }
      updateControlArmedUi();
    }

    function scheduleControlRefresh() {
      if (controlRefreshTimer !== null) {
        clearInterval(controlRefreshTimer);
      }
      const fps = getControlRefreshFps();
      controlRefreshTimer = setInterval(() => {
        if (currentView === 'control') {
          refreshControlFrame();
        }
      }, Math.max(100, Math.floor(1000 / fps)));
    }

    function refreshControlFrame() {
      const slot = selectedSlotSnapshot();
      const img = document.getElementById('control-frame');
      const meta = document.getElementById('control-meta');
      if (!slot) {
        img.removeAttribute('src');
        meta.textContent = 'select a slot';
        return;
      }
      const ts = Date.now();
      img.src = `/api/slots/${slot.name}/frame.jpg?ts=${ts}`;
      meta.textContent =
        `slot=${slot.name} | status=${slot.status} | refresh fps=${getControlRefreshFps().toFixed(1)} | last refresh=${new Date(ts).toLocaleTimeString()}`;
    }

    function controlSurfacePosition(event) {
      const root = document.getElementById('control-surface');
      const rect = root.getBoundingClientRect();
      if (rect.width <= 1 || rect.height <= 1) {
        return null;
      }
      const x = Math.min(Math.max(event.clientX - rect.left, 0), rect.width - 1);
      const y = Math.min(Math.max(event.clientY - rect.top, 0), rect.height - 1);
      return {
        x: x / Math.max(1, rect.width - 1),
        y: y / Math.max(1, rect.height - 1),
      };
    }

    async function reload() {
      const res = await fetch('/api/slots');
      latestSlots = await res.json();
      const slots = latestSlots;
      const rows = [
        '<tr><th>slot</th><th>status</th><th>details</th><th>actions</th></tr>'
      ];
      for (const slot of slots) {
        const details = [
          `user=${esc(slot.user)}`,
          `pid=${esc(slot.process_id ?? '')}`,
          `node=${esc(slot.pipewire_node_id ?? '')}`,
          `client=${esc(slot.pipewire_client_id ?? '')}`,
          `gamescope_socket=${esc(slot.gamescope_socket ?? '')}`,
          `eis_socket=${esc(slot.eis_socket ?? '')}`,
          `error=${esc(slot.error ?? '')}`,
        ].join('\\n');
        rows.push(
          `<tr>
            <td>${slot.name}</td>
            <td class="${statusClass(slot.status)}">${slot.status}</td>
            <td class="mono">${details}</td>
            <td>
              <button onclick="act('${slot.name}', 'start')">start</button>
              <button onclick="act('${slot.name}', 'restart')">restart</button>
              <button onclick="act('${slot.name}', 'stop')">stop</button>
            </td>
          </tr>`
        );
      }
      document.getElementById('slots').innerHTML = rows.join('');
      updateSlotTabs();
      updateSelectedSlotPanel();
    }
    async function act(slot, op) {
      await fetch(`/api/slots/${slot}/${op}`, { method: 'POST' });
      await reload();
      refreshComposite();
      refreshControlFrame();
    }
    document.getElementById('refresh-fps').value = browserFps.toFixed(1);
    document.getElementById('control-refresh-fps').value = browserFps.toFixed(1);
    document.getElementById('refresh-fps').addEventListener('change', () => {
      document.getElementById('refresh-fps').value = getRefreshFps().toFixed(1);
      scheduleCompositeRefresh();
      refreshComposite();
    });
    document.getElementById('control-refresh-fps').addEventListener('change', () => {
      document.getElementById('control-refresh-fps').value = getControlRefreshFps().toFixed(1);
      scheduleControlRefresh();
      refreshControlFrame();
    });
    document.getElementById('control-surface').addEventListener('contextmenu', (event) => {
      if (controlArmed) {
        event.preventDefault();
      }
    });
    document.getElementById('control-surface').addEventListener('mousemove', (event) => {
      if (!controlArmed) {
        return;
      }
      const now = performance.now();
      if (now - lastMouseSendMs < 16) {
        return;
      }
      lastMouseSendMs = now;
      const pos = controlSurfacePosition(event);
      if (!pos) {
        return;
      }
      sendControlEvent({ t: 'mouse_abs', x: pos.x, y: pos.y, visible: true });
    });
    document.getElementById('control-surface').addEventListener('mouseleave', () => {
      if (controlArmed) {
        sendControlEvent({ t: 'cursor', x: 0.5, y: 0.5, visible: false });
      }
    });
    document.getElementById('control-surface').addEventListener('mousedown', (event) => {
      if (!controlArmed) {
        return;
      }
      event.preventDefault();
      const button = mouseButtonMap[event.button];
      if (!button || pressedButtons.has(button)) {
        return;
      }
      pressedButtons.add(button);
      sendControlEvent({ t: 'mouse_btn', button, down: true });
      const pos = controlSurfacePosition(event);
      if (pos) {
        sendControlEvent({ t: 'mouse_abs', x: pos.x, y: pos.y, visible: true });
      }
    });
    window.addEventListener('mouseup', (event) => {
      if (!controlArmed) {
        return;
      }
      const button = mouseButtonMap[event.button];
      if (!button || !pressedButtons.has(button)) {
        return;
      }
      pressedButtons.delete(button);
      sendControlEvent({ t: 'mouse_btn', button, down: false });
    });
    window.addEventListener('keydown', (event) => {
      if (!controlArmed) {
        return;
      }
      const mapped = browserKeyMap[event.code];
      if (!mapped) {
        return;
      }
      event.preventDefault();
      if (pressedKeys.has(event.code)) {
        return;
      }
      pressedKeys.add(event.code);
      sendControlEvent({ t: 'key', key: mapped, down: true });
    });
    window.addEventListener('keyup', (event) => {
      if (!controlArmed) {
        return;
      }
      const mapped = browserKeyMap[event.code];
      if (!mapped) {
        return;
      }
      event.preventDefault();
      pressedKeys.delete(event.code);
      sendControlEvent({ t: 'key', key: mapped, down: false });
    });
    window.addEventListener('blur', () => {
      if (controlArmed) {
        releaseAllPressed();
      }
    });
    reload();
    refreshComposite();
    refreshControlFrame();
    updateControlArmedUi();
    setInterval(reload, 1000);
    scheduleCompositeRefresh();
    scheduleControlRefresh();
  </script>
</body>
</html>
"""


def create_app(config_path: str | Path = "sim_harness.toml"):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse, Response
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FastAPI is not installed. Install inference requirements first: "
            "pip install -r requirements-inference.txt"
        ) from exc

    config = load_config(config_path)
    supervisor = HarnessSupervisor(config)
    app = FastAPI(title="cs2-sim-harness")
    app.state.supervisor = supervisor

    @app.on_event("startup")
    async def on_startup() -> None:
        await supervisor.start()

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        await supervisor.stop()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML.replace("__BROWSER_FPS__", str(config.web.browser_fps))

    @app.get("/api/slots")
    async def list_slots():
        return JSONResponse(supervisor.list_slots())

    @app.post("/api/slots/{slot_name}/start")
    async def start_slot(slot_name: str):
        try:
            return JSONResponse(await supervisor.start_slot(slot_name))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/slots/{slot_name}/stop")
    async def stop_slot(slot_name: str):
        try:
            return JSONResponse(await supervisor.stop_slot(slot_name))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/slots/{slot_name}/restart")
    async def restart_slot(slot_name: str):
        try:
            return JSONResponse(await supervisor.restart_slot(slot_name))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/slots/{slot_name}/input")
    async def send_input(slot_name: str, payload: dict):
        try:
            await supervisor.send_input(slot_name, payload)
            return {"ok": True}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/slots/{slot_name}/frame.jpg")
    async def slot_frame(slot_name: str):
        try:
            worker = supervisor.get_worker(slot_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        payload = await worker.latest_jpeg()
        if payload is None:
            return Response(status_code=204)
        return Response(payload, media_type="image/jpeg")

    @app.get("/api/composite.jpg")
    async def composite():
        try:
            return Response(await supervisor.render_composite(), media_type="image/jpeg")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app
