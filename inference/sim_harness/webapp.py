from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any

from .config import load_config
from .remote_protocol import decode_stream_text, encode_stream_ack, encode_stream_error, encode_stream_pong
from .supervisor import HarnessSupervisor

try:
    from fastapi import WebSocket
except ModuleNotFoundError:
    WebSocket = Any


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
    .type-row { display: flex; gap: 8px; align-items: stretch; flex-wrap: wrap; margin-top: 12px; }
    .type-row input[type="password"], .type-row input[type="text"] { flex: 1 1 320px; background: #0f1518; color: #f2f4f5; border: 1px solid #2b3338; padding: 8px; }
    .type-row button { margin: 0; }
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
          <button onclick="cleanupAll()">cleanup all</button>
        </div>
        <img id="composite" src="/api/composite.jpg" alt="composite view">
        <div class="meta" id="composite-meta">waiting for refresh</div>
        <div class="meta" id="system-meta">idle</div>
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
            <button id="toggle-audio" onclick="toggleAudioPlayback()">audio off</button>
            <button onclick="refreshControlFrame()">refresh now</button>
          </div>
          <div id="control-surface" class="control-surface" tabindex="0">
            <img id="control-frame" src="" alt="slot control view">
            <div class="control-hint" id="control-hint">disarmed</div>
          </div>
          <div class="type-row">
            <input id="type-text" type="password" placeholder="type or paste text here">
            <button id="toggle-type-visibility" onclick="toggleTypedTextVisibility()">show</button>
            <button onclick="typeTextField()">type text</button>
            <button onclick="clearTypeText()">clear</button>
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
    let pasteInFlight = false;
    let audioEnabled = false;
    let audioContext = null;
    let audioWorkletNode = null;
    let audioModuleReady = false;
    let audioUseWorklet = false;
    let audioPollTimer = null;
    let audioPollInFlight = false;
    let audioLastSeq = 0;
    let audioFallbackPlayhead = 0;
    const audioFrameHz = 32;
    const audioPollMs = Math.max(8, Math.floor(1000 / (audioFrameHz * 2)));
    const audioChannels = 2;
    const audioSampleRate = 24000;
    const audioWorkletSource = `
class PcmPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.current = null;
    this.offset = 0;
    this.primed = false;
    this.targetFrames = 4096;
    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (data.t === 'reset') {
        this.queue = [];
        this.current = null;
        this.offset = 0;
        this.primed = false;
        return;
      }
      if (data.t === 'chunk' && Array.isArray(data.channels)) {
        const chunk = data.channels.map((buffer) => new Float32Array(buffer));
        if (chunk.length > 0 && chunk[0].length > 0) {
          this.queue.push(chunk);
        }
      }
    };
  }

  queuedFrames() {
    let total = 0;
    if (this.current) {
      total += this.current[0].length - this.offset;
    }
    for (const chunk of this.queue) {
      total += chunk[0].length;
    }
    return total;
  }

  process(inputs, outputs) {
    const output = outputs[0];
    const frameCount = output[0].length;
    for (let ch = 0; ch < output.length; ch += 1) {
      output[ch].fill(0);
    }
    if (!this.primed) {
      if (this.queuedFrames() < this.targetFrames) {
        return true;
      }
      this.primed = true;
    }
    let pos = 0;
    while (pos < frameCount) {
      if (!this.current || this.offset >= this.current[0].length) {
        this.current = this.queue.shift() || null;
        this.offset = 0;
        if (!this.current) {
          this.primed = false;
          break;
        }
      }
      const available = this.current[0].length - this.offset;
      const take = Math.min(frameCount - pos, available);
      for (let ch = 0; ch < output.length; ch += 1) {
        const src = this.current[Math.min(ch, this.current.length - 1)];
        output[ch].set(src.subarray(this.offset, this.offset + take), pos);
      }
      this.offset += take;
      pos += take;
    }
    return true;
  }
}

registerProcessor('pcm-player', PcmPlayerProcessor);
`;
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
    const pastedCharMap = {
      'a': ['a', false], 'b': ['b', false], 'c': ['c', false], 'd': ['d', false], 'e': ['e', false],
      'f': ['f', false], 'g': ['g', false], 'h': ['h', false], 'i': ['i', false], 'j': ['j', false],
      'k': ['k', false], 'l': ['l', false], 'm': ['m', false], 'n': ['n', false], 'o': ['o', false],
      'p': ['p', false], 'q': ['q', false], 'r': ['r', false], 's': ['s', false], 't': ['t', false],
      'u': ['u', false], 'v': ['v', false], 'w': ['w', false], 'x': ['x', false], 'y': ['y', false],
      'z': ['z', false],
      'A': ['a', true], 'B': ['b', true], 'C': ['c', true], 'D': ['d', true], 'E': ['e', true],
      'F': ['f', true], 'G': ['g', true], 'H': ['h', true], 'I': ['i', true], 'J': ['j', true],
      'K': ['k', true], 'L': ['l', true], 'M': ['m', true], 'N': ['n', true], 'O': ['o', true],
      'P': ['p', true], 'Q': ['q', true], 'R': ['r', true], 'S': ['s', true], 'T': ['t', true],
      'U': ['u', true], 'V': ['v', true], 'W': ['w', true], 'X': ['x', true], 'Y': ['y', true],
      'Z': ['z', true],
      '0': ['0', false], '1': ['1', false], '2': ['2', false], '3': ['3', false], '4': ['4', false],
      '5': ['5', false], '6': ['6', false], '7': ['7', false], '8': ['8', false], '9': ['9', false],
      ')': ['0', true], '!': ['1', true], '@': ['2', true], '#': ['3', true], '$': ['4', true],
      '%': ['5', true], '^': ['6', true], '&': ['7', true], '*': ['8', true], '(': ['9', true],
      ' ': ['space', false], '\\n': ['enter', false], '\\r': ['enter', false], '\\t': ['tab', false],
      '-': ['minus', false], '_': ['minus', true],
      '=': ['equals', false], '+': ['equals', true],
      ',': ['comma', false], '<': ['comma', true],
      '.': ['period', false], '>': ['period', true],
      '/': ['slash', false], '?': ['slash', true],
      ';': ['semicolon', false], ':': ['semicolon', true],
      "'": ['apostrophe', false], '"': ['apostrophe', true],
      '[': ['leftbracket', false], '{': ['leftbracket', true],
      ']': ['rightbracket', false], '}': ['rightbracket', true],
      '\\\\': ['backslash', false], '|': ['backslash', true],
      '`': ['grave', false], '~': ['grave', true],
    };

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
      if (audioEnabled) {
        audioLastSeq = 0;
        resetAudioWorklet();
        pollAudioChunk();
      }
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

    async function sendKeyStroke(keyName, useShift) {
      if (useShift) {
        await sendControlEvent({ t: 'key', key: 'lshift', down: true });
      }
      await sendControlEvent({ t: 'key', key: keyName, down: true });
      await sendControlEvent({ t: 'key', key: keyName, down: false });
      if (useShift) {
        await sendControlEvent({ t: 'key', key: 'lshift', down: false });
      }
    }

    async function typeClipboardText(text) {
      if (pasteInFlight) {
        return;
      }
      pasteInFlight = true;
      try {
        let sent = 0;
        let skipped = 0;
        for (const ch of text) {
          const spec = pastedCharMap[ch];
          if (!spec) {
            skipped += 1;
            continue;
          }
          await sendKeyStroke(spec[0], spec[1]);
          sent += 1;
          await new Promise(resolve => setTimeout(resolve, 12));
        }
        logEvent(`paste typed: sent=${sent} skipped=${skipped}`);
      } finally {
        pasteInFlight = false;
      }
    }

    function typedTextField() {
      return document.getElementById('type-text');
    }

    function toggleTypedTextVisibility() {
      const field = typedTextField();
      const button = document.getElementById('toggle-type-visibility');
      const showing = field.type === 'text';
      field.type = showing ? 'password' : 'text';
      button.textContent = showing ? 'show' : 'hide';
    }

    function clearTypeText() {
      typedTextField().value = '';
      logEvent('typed text field cleared');
    }

    async function typeTextField() {
      const text = typedTextField().value;
      if (!text) {
        logEvent('type text ignored: field empty');
        return;
      }
      logEvent(`type text requested: ${text.length} chars`);
      await typeClipboardText(text);
    }

    async function pasteFromClipboardApi() {
      if (!navigator.clipboard || !navigator.clipboard.readText) {
        throw new Error('Clipboard API unavailable');
      }
      const text = await navigator.clipboard.readText();
      if (!text) {
        throw new Error('Clipboard text empty');
      }
      logEvent(`clipboard api read: ${text.length} chars`);
      await typeClipboardText(text);
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

    function updateAudioUi() {
      document.getElementById('toggle-audio').textContent = audioEnabled ? 'audio on' : 'audio off';
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

    async function ensureAudioContext() {
      if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }
      if (audioContext.audioWorklet && !audioModuleReady) {
        const blob = new Blob([audioWorkletSource], { type: 'text/javascript' });
        const url = URL.createObjectURL(blob);
        try {
          await audioContext.audioWorklet.addModule(url);
        } finally {
          URL.revokeObjectURL(url);
        }
        audioModuleReady = true;
      }
      if (audioModuleReady && !audioWorkletNode) {
        audioWorkletNode = new AudioWorkletNode(audioContext, 'pcm-player', {
          numberOfInputs: 0,
          numberOfOutputs: 1,
          outputChannelCount: [audioChannels],
        });
        audioWorkletNode.connect(audioContext.destination);
      }
      audioUseWorklet = !!audioWorkletNode;
    }

    function resetAudioWorklet() {
      if (audioWorkletNode) {
        audioWorkletNode.port.postMessage({ t: 'reset' });
      }
      audioFallbackPlayhead = 0;
    }

    function pcm16ToPlanarFloat(pcmView, channels) {
      const frameCount = Math.floor(pcmView.length / channels);
      if (frameCount <= 0) {
        return [];
      }
      const planar = Array.from({ length: channels }, () => new Float32Array(frameCount));
      for (let i = 0; i < frameCount; i += 1) {
        const base = i * channels;
        for (let ch = 0; ch < channels; ch += 1) {
          planar[ch][i] = pcmView[base + ch] / 32768.0;
        }
      }
      return planar;
    }

    function resamplePlanar(planar, srcRate, dstRate) {
      if (planar.length === 0 || srcRate === dstRate) {
        return planar;
      }
      const srcFrames = planar[0].length;
      const dstFrames = Math.max(1, Math.round(srcFrames * dstRate / srcRate));
      return planar.map((channel) => {
        const out = new Float32Array(dstFrames);
        for (let i = 0; i < dstFrames; i += 1) {
          const srcPos = i * srcRate / dstRate;
          const idx0 = Math.floor(srcPos);
          const idx1 = Math.min(idx0 + 1, srcFrames - 1);
          const frac = srcPos - idx0;
          const a = channel[Math.min(idx0, srcFrames - 1)];
          const b = channel[idx1];
          out[i] = a + (b - a) * frac;
        }
        return out;
      });
    }

    async function pollAudioChunk() {
      if (!audioEnabled || !selectedControlSlot || audioPollInFlight) {
        return;
      }
      audioPollInFlight = true;
      try {
        const res = await fetch(`/api/slots/${selectedControlSlot}/audio.pcm?ts=${Date.now()}`);
        if (res.status === 204) {
          return;
        }
        if (!res.ok) {
          logEvent(`audio error: http ${res.status}`);
          return;
        }
        const seq = Number(res.headers.get('x-audio-seq') ?? '0');
        if (!Number.isFinite(seq) || seq <= audioLastSeq) {
          return;
        }
        const sampleRate = Number(res.headers.get('x-audio-sample-rate') ?? String(audioSampleRate));
        const channels = Number(res.headers.get('x-audio-channels') ?? String(audioChannels));
        audioLastSeq = seq;
        const pcm = await res.arrayBuffer();
        await ensureAudioContext();
        const pcmView = new Int16Array(pcm);
        if (!Number.isFinite(channels) || channels < 1 || pcmView.length === 0) {
          return;
        }
        let planar = pcm16ToPlanarFloat(pcmView, channels);
        if (planar.length === 0) {
          return;
        }
        planar = resamplePlanar(planar, sampleRate, audioContext.sampleRate);
        if (audioUseWorklet) {
          audioWorkletNode.port.postMessage(
            { t: 'chunk', channels: planar.map((channel) => channel.buffer) },
            planar.map((channel) => channel.buffer),
          );
        } else {
          const frameCount = planar[0]?.length ?? 0;
          if (frameCount <= 0) {
            return;
          }
          const audioBuffer = audioContext.createBuffer(planar.length, frameCount, audioContext.sampleRate);
          for (let ch = 0; ch < planar.length; ch += 1) {
            audioBuffer.copyToChannel(planar[ch], ch);
          }
          const source = audioContext.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(audioContext.destination);
          const now = audioContext.currentTime;
          if (audioFallbackPlayhead < now + 0.12) {
            audioFallbackPlayhead = now + 0.12;
          }
          source.start(audioFallbackPlayhead);
          audioFallbackPlayhead += audioBuffer.duration;
        }
      } catch (err) {
        logEvent(`audio playback failed: ${err}`);
      } finally {
        audioPollInFlight = false;
      }
    }

    function startAudioPolling() {
      if (audioPollTimer !== null) {
        clearInterval(audioPollTimer);
      }
      audioPollTimer = setInterval(() => {
        pollAudioChunk();
      }, audioPollMs);
    }

    function stopAudioPolling() {
      if (audioPollTimer !== null) {
        clearInterval(audioPollTimer);
        audioPollTimer = null;
      }
      audioLastSeq = 0;
      audioPollInFlight = false;
      resetAudioWorklet();
    }

    async function toggleAudioPlayback() {
      audioEnabled = !audioEnabled;
      updateAudioUi();
      if (!audioEnabled) {
        stopAudioPolling();
        return;
      }
      try {
        await ensureAudioContext();
        startAudioPolling();
        pollAudioChunk();
      } catch (err) {
        audioEnabled = false;
        updateAudioUi();
        logEvent(`audio enable failed: ${err}`);
      }
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

    function controlFrameRect() {
      const frame = document.getElementById('control-frame');
      const rect = frame.getBoundingClientRect();
      if (rect.width > 1 && rect.height > 1) {
        return rect;
      }
      return document.getElementById('control-surface').getBoundingClientRect();
    }

    function controlSurfacePosition(event) {
      const rect = controlFrameRect();
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

    async function cleanupAll() {
      const meta = document.getElementById('system-meta');
      meta.textContent = 'cleanup running...';
      try {
        const res = await fetch('/api/cleanup-all', { method: 'POST' });
        if (!res.ok) {
          let detail = `http ${res.status}`;
          try {
            const body = await res.json();
            detail = body.detail ?? JSON.stringify(body);
          } catch (_) {}
          meta.textContent = `cleanup failed: ${detail}`;
          return;
        }
        const body = await res.json();
        const killed = (body.cleanup ?? []).map(item =>
          `${item.slot}:${(item.killed ?? []).join(',') || 'none'}`
        ).join(' | ');
        const locks = (body.removed_global_locks ?? []).length;
        meta.textContent = `cleanup complete | global locks removed=${locks} | ${killed}`;
        await reload();
        refreshComposite();
        refreshControlFrame();
      } catch (err) {
        meta.textContent = `cleanup failed: ${err}`;
      }
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
      if ((event.ctrlKey || event.metaKey) && event.code === 'KeyV') {
        event.preventDefault();
        logEvent(`paste shortcut detected for ${selectedControlSlot ?? 'no slot'}`);
        pasteFromClipboardApi().catch((err) => {
          logEvent(`clipboard api failed: ${err}`);
          logEvent('use browser paste event as fallback');
        });
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
    window.addEventListener('paste', (event) => {
      if (!controlArmed || !selectedControlSlot) {
        return;
      }
      event.preventDefault();
      const text = event.clipboardData?.getData('text') ?? '';
      if (!text) {
        logEvent('paste ignored: clipboard text empty');
        return;
      }
      logEvent(`paste event read: ${text.length} chars`);
      typeClipboardText(text);
    });
    reload();
    refreshComposite();
    refreshControlFrame();
    updateControlArmedUi();
    updateAudioUi();
    setInterval(reload, 1000);
    scheduleCompositeRefresh();
    scheduleControlRefresh();
  </script>
</body>
</html>
"""


async def _run_model_stream(websocket, supervisor: HarnessSupervisor, stream_interval_s: float) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()
    next_tick = loop.time()
    try:
        while True:
            timeout = max(0.0, next_tick - loop.time())
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
            except asyncio.TimeoutError:
                await websocket.send_bytes(await supervisor.observation_payload())
                next_tick += stream_interval_s
                if next_tick < loop.time():
                    next_tick = loop.time()
                continue

            try:
                message = decode_stream_text(raw)
            except Exception as exc:
                await websocket.send_text(encode_stream_error(code="bad_json", detail=str(exc)))
                continue

            op = str(message.get("op", ""))
            if op == "actions":
                actions = message.get("actions")
                if not isinstance(actions, dict):
                    await websocket.send_text(
                        encode_stream_error(code="bad_actions", detail="actions must be an object")
                    )
                    continue
                server_recv_ns = time.time_ns()
                try:
                    results = await supervisor.apply_actions(actions)
                except KeyError as exc:
                    await websocket.send_text(encode_stream_error(code="unknown_slot", detail=str(exc)))
                    continue
                await websocket.send_text(
                    encode_stream_ack(
                        client_send_ns=message.get("client_send_ns"),
                        server_recv_ns=server_recv_ns,
                        server_send_ns=time.time_ns(),
                        results=results,
                    )
                )
                continue

            if op == "ping":
                await websocket.send_text(encode_stream_pong(message.get("client_send_ns")))
                continue

            await websocket.send_text(
                encode_stream_error(code="bad_op", detail=f"unsupported op: {op or '<missing>'}")
            )
    except Exception as exc:
        if exc.__class__.__name__ in {"WebSocketDisconnect"}:
            return
        if isinstance(exc, RuntimeError):
            return
        raise


async def model_stream_endpoint(websocket: WebSocket) -> None:
    supervisor = websocket.app.state.supervisor
    stream_interval_s = websocket.app.state.stream_interval_s
    await _run_model_stream(websocket, supervisor, stream_interval_s)


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
    stream_hz = min((slot.audio_frame_hz for slot in config.slots), default=32)
    stream_interval_s = 1.0 / max(1, stream_hz)
    app = FastAPI(title="cs2-sim-harness")
    app.state.supervisor = supervisor
    app.state.stream_interval_s = stream_interval_s

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

    @app.get("/api/model/observation.bin")
    async def model_observation():
        try:
            payload = await supervisor.observation_payload()
            return Response(payload, media_type="application/octet-stream")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/model/actions")
    async def model_actions(payload: dict):
        try:
            actions = payload.get("actions")
            if not isinstance(actions, dict):
                raise HTTPException(status_code=400, detail="payload.actions must be an object")
            return JSONResponse({"results": await supervisor.apply_actions(actions)})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    app.add_api_websocket_route("/api/model/ws", model_stream_endpoint)

    @app.post("/api/cleanup-all")
    async def cleanup_all():
        try:
            return JSONResponse(await supervisor.cleanup_all())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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

    @app.get("/api/slots/{slot_name}/audio.pcm")
    async def slot_audio(slot_name: str):
        try:
            worker = supervisor.get_worker(slot_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        item = await worker.latest_audio_pcm()
        if item is None:
            return Response(status_code=204)
        seq, ts_ns, payload = item
        return Response(
            payload,
            media_type="application/octet-stream",
            headers={
                "x-audio-seq": str(seq),
                "x-audio-time-ns": str(ts_ns),
                "x-audio-sample-rate": "24000",
                "x-audio-channels": "2",
                "x-audio-format": "s16le",
            },
        )

    @app.get("/api/composite.jpg")
    async def composite():
        try:
            return Response(await supervisor.render_composite(), media_type="image/jpeg")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app
