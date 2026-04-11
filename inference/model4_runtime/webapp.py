from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from .runtime import (
    DECODER_KEYBOARD_METRICS,
    DECODER_MOUSE_METRICS,
    DEFAULT_DECODER_CALIBRATION_KEYBOARD_FN_COST_OVERRIDES,
    DEFAULT_DECODER_CALIBRATION_KEYBOARD_FP_COST_OVERRIDES,
    Model4InferenceRuntime,
)
from .actions import KEYBOARD_ONLY_ACTIONS

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
  <title>Model4 Runtime</title>
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
    .keyboard-threshold-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; width: 100%; margin-top: 10px; }
    .keyboard-threshold-grid label { display: grid; gap: 4px; font-size: 12px; color: #c8d8df; }
    .helper-text { font-size: 12px; color: #97aeb9; margin-top: 6px; }
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
    .status-box { background: #0c1013; border: 1px solid #334049; padding: 12px; margin-bottom: 14px; }
    .capture-frame-box { background: #0c1013; border: 1px solid #334049; padding: 12px; }
    .capture-controls { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }
    .capture-controls input[type="range"] { flex: 1; min-width: 240px; }
    progress { width: 100%; height: 18px; margin: 8px 0; }
  </style>
</head>
<body>
  <h1>Model4 Runtime</h1>
  <div class="tabs">
    <button id="tab-view" class="active" onclick="switchView('view')">view</button>
    <button id="tab-logits" onclick="switchView('logits')">logits</button>
    <button id="tab-capture" onclick="switchView('capture')">capture</button>
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
        <input id="player-index" type="number" min="0" max="0" step="1" value="0">
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
        <label>mouse scale x
          <input id="mouse-scale-x" type="number" step="0.05" value="1.0">
        </label>
        <label>mouse scale y
          <input id="mouse-scale-y" type="number" step="0.05" value="1.0">
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
      <div style="width: 100%;">
        <div style="font-size: 12px; color: #97aeb9; margin-bottom: 6px;">keyboard thresholds per action</div>
        <div id="keyboard-threshold-grid" class="keyboard-threshold-grid"></div>
      </div>
      <button onclick="applyActionDecode()">apply decode</button>
      <button onclick="calibrateMouse()">calibrate mouse</button>
    </div>
    <div class="row">
      <div>
      <div class="status-box">
        <h2 class="decision-title">Mouse Calibration</h2>
        <pre id="calibration-status">no calibration run yet</pre>
      </div>
      <div class="status-box">
        <h2 class="decision-title">Decoder Calibration</h2>
        <div class="settings-grid">
          <label>max samples
            <input id="decoder-cal-max-samples" type="number" min="1" step="1" value="8">
          </label>
          <label>frames / sample
            <input id="decoder-cal-frames-per-sample" type="number" min="8" step="8" value="128">
          </label>
          <label>keyboard false-positive cost
            <input id="decoder-cal-keyboard-fp-cost" type="number" min="0" step="0.1" value="1.0">
          </label>
          <label>keyboard metric
            <select id="decoder-cal-keyboard-metric">
              <option value="cost_weighted_accuracy">cost-weighted accuracy</option>
              <option value="balanced_accuracy">balanced accuracy</option>
              <option value="f1">f1</option>
            </select>
          </label>
          <label>keyboard false-negative cost
            <input id="decoder-cal-keyboard-fn-cost" type="number" min="0" step="0.1" value="1.0">
          </label>
          <label>mouse move-vs-bin weight
            <input id="decoder-cal-mouse-move-weight" type="number" min="0.01" max="0.99" step="0.01" value="0.70">
          </label>
          <label>mouse move metric
            <select id="decoder-cal-mouse-metric">
              <option value="balanced_accuracy">balanced accuracy</option>
              <option value="f1">f1</option>
            </select>
          </label>
          <label style="display:flex; align-items:center; gap:8px; align-self:end;">
            <input id="decoder-cal-mouse-stochastic" type="checkbox">
            stochastic mouse calibration
          </label>
        </div>
        <div style="width: 100%; margin-top: 10px;">
          <div style="font-size: 12px; color: #97aeb9; margin-bottom: 6px;">per-key keyboard false-negative cost overrides</div>
          <div id="decoder-cal-keyboard-fn-cost-grid" class="keyboard-threshold-grid"></div>
          <div class="helper-text">raise FN cost for keys where missing a press is worse than a false press</div>
        </div>
        <div style="width: 100%; margin-top: 10px;">
          <div style="font-size: 12px; color: #97aeb9; margin-bottom: 6px;">per-key keyboard false-positive cost overrides</div>
          <div id="decoder-cal-keyboard-fp-cost-grid" class="keyboard-threshold-grid"></div>
          <div class="helper-text">raise FP cost for noisy or low-value keys like score/inspect/turn</div>
        </div>
        <div class="toolbar" style="margin-top: 10px; margin-bottom: 0;">
          <button onclick="startDecoderCalibration()">calibrate decoder</button>
          <button onclick="cancelDecoderCalibration()">cancel calibration</button>
          <button onclick="exportSettings()">export settings</button>
          <button onclick="triggerSettingsImport()">import settings</button>
          <input id="settings-import-input" type="file" accept="application/json,.json" style="display:none" onchange="importSettings(event)">
        </div>
        <progress id="decoder-calibration-progress" max="1" value="0"></progress>
        <div id="decoder-calibration-progress-text" class="decision-line">idle</div>
        <pre id="decoder-calibration-status">no decoder calibration run yet</pre>
      </div>
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

  <div id="view-capture" class="view">
    <div class="toolbar">
      <button onclick="startCapture()">start capture</button>
      <button onclick="stopCapture()">stop capture</button>
      <button onclick="clearCapture()">clear capture</button>
      <label>player
        <input id="capture-player-index" type="number" min="0" max="0" step="1" value="0">
      </label>
      <button id="capture-play-button" onclick="toggleCapturePlayback()">play</button>
    </div>
    <div class="capture-frame-box">
      <div class="capture-controls">
        <input id="capture-frame-slider" type="range" min="0" max="0" step="1" value="0" oninput="loadCaptureFrame()">
        <span id="capture-frame-label">no capture loaded</span>
      </div>
      <img id="capture-frame-image" src="" alt="capture frame">
    </div>
    <div class="row" style="margin-top: 16px;">
      <div>
        <div id="capture-decision-path" class="decision-box">
          <h2 class="decision-title">Captured Path</h2>
          <div class="decision-line">no capture frame selected</div>
        </div>
        <h2>Captured Distributions</h2>
        <div id="capture-graphs" class="charts-grid">
          <div class="empty-state">no capture frame selected</div>
        </div>
      </div>
      <div>
        <h2>Capture State</h2>
        <pre id="capture-state">loading...</pre>
        <h2>Captured Frame</h2>
        <pre id="capture-frame-state">no capture frame selected</pre>
      </div>
    </div>
  </div>
  <script>
    const keyboardActionLabels = __KEYBOARD_ACTION_LABELS_JSON__;
    const defaultDecoderCalibrationKeyboardFnCostOverrides = __DEFAULT_DECODER_CALIBRATION_KEYBOARD_FN_COST_OVERRIDES_JSON__;
    const defaultDecoderCalibrationKeyboardFpCostOverrides = __DEFAULT_DECODER_CALIBRATION_KEYBOARD_FP_COST_OVERRIDES_JSON__;
    const decoderKeyboardMetricChoices = __DECODER_KEYBOARD_METRICS_JSON__;
    const decoderMouseMetricChoices = __DECODER_MOUSE_METRICS_JSON__;
    let refreshTimer = null;
    let capturePlaybackTimer = null;
    let uiSettingsSaveTimer = null;
    const stickyInputIds = [
      'harness-url',
      'cache-window',
      'keyboard-threshold',
      'mouse-zero-bias',
      'mouse-scale',
      'mouse-scale-x',
      'mouse-scale-y',
      'mouse-temperature',
      'mouse-top-k',
      'mouse-deadzone',
    ];
    const persistentUiInputIds = [
      'refresh-fps',
      'player-index',
      'capture-player-index',
      'decoder-cal-max-samples',
      'decoder-cal-frames-per-sample',
      'decoder-cal-keyboard-fp-cost',
      'decoder-cal-keyboard-fn-cost',
      'decoder-cal-keyboard-metric',
      'decoder-cal-mouse-move-weight',
      'decoder-cal-mouse-metric',
      'decoder-cal-mouse-stochastic',
    ];

    function keyboardThresholdInputId(action) {
      return `keyboard-threshold-${action.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`;
    }

    function decoderCalibrationFnCostInputId(action) {
      return `decoder-cal-keyboard-fn-cost-${action.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`;
    }

    function decoderCalibrationFpCostInputId(action) {
      return `decoder-cal-keyboard-fp-cost-${action.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`;
    }

    function ensureKeyboardThresholdControls() {
      const grid = document.getElementById('keyboard-threshold-grid');
      const fnCostGrid = document.getElementById('decoder-cal-keyboard-fn-cost-grid');
      const fpCostGrid = document.getElementById('decoder-cal-keyboard-fp-cost-grid');
      if (!grid || !fnCostGrid || !fpCostGrid || (grid.dataset.ready === 'true' && fnCostGrid.dataset.ready === 'true' && fpCostGrid.dataset.ready === 'true')) return;
      for (const action of keyboardActionLabels) {
        if (grid.dataset.ready !== 'true') {
          const label = document.createElement('label');
          label.textContent = action;
          const input = document.createElement('input');
          input.id = keyboardThresholdInputId(action);
          input.type = 'number';
          input.min = '0';
          input.max = '1';
          input.step = '0.01';
          input.value = '0.12';
          input.addEventListener('input', markDirty);
          label.appendChild(input);
          grid.appendChild(label);
          stickyInputIds.push(input.id);
        }
        if (fnCostGrid.dataset.ready !== 'true') {
          const fnLabel = document.createElement('label');
          fnLabel.textContent = action;
          const fnInput = document.createElement('input');
          fnInput.id = decoderCalibrationFnCostInputId(action);
          fnInput.type = 'number';
          fnInput.min = '0';
          fnInput.step = '0.1';
          fnInput.placeholder = 'global';
          if (defaultDecoderCalibrationKeyboardFnCostOverrides[action] !== undefined) {
            fnInput.value = String(defaultDecoderCalibrationKeyboardFnCostOverrides[action]);
          }
          fnInput.addEventListener('input', markUiSettingDirty);
          fnLabel.appendChild(fnInput);
          fnCostGrid.appendChild(fnLabel);
        }
        if (fpCostGrid.dataset.ready !== 'true') {
          const fpLabel = document.createElement('label');
          fpLabel.textContent = action;
          const fpInput = document.createElement('input');
          fpInput.id = decoderCalibrationFpCostInputId(action);
          fpInput.type = 'number';
          fpInput.min = '0';
          fpInput.step = '0.1';
          fpInput.placeholder = 'global';
          if (defaultDecoderCalibrationKeyboardFpCostOverrides[action] !== undefined) {
            fpInput.value = String(defaultDecoderCalibrationKeyboardFpCostOverrides[action]);
          }
          fpInput.addEventListener('input', markUiSettingDirty);
          fpLabel.appendChild(fpInput);
          fpCostGrid.appendChild(fpLabel);
        }
      }
      grid.dataset.ready = 'true';
      fnCostGrid.dataset.ready = 'true';
      fpCostGrid.dataset.ready = 'true';
    }

    function markDirty(event) {
      event.target.dataset.dirty = 'true';
    }

    function markUiSettingDirty(event) {
      markDirty(event);
      scheduleUiSettingsPersist();
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

    function syncKeyboardThresholdInputs(globalThreshold, overrides) {
      for (const action of keyboardActionLabels) {
        const value = overrides && overrides[action] !== undefined ? overrides[action] : globalThreshold;
        syncInputValue(keyboardThresholdInputId(action), value);
      }
    }

    function syncDecoderCalibrationCostInputs(globalCost, overrides, inputIdFn) {
      for (const action of keyboardActionLabels) {
        const input = document.getElementById(inputIdFn(action));
        if (!input) continue;
        const hasOverride = overrides && overrides[action] !== undefined;
        const value = hasOverride ? overrides[action] : '';
        syncInputValue(input.id, value);
        input.placeholder = `${globalCost}`;
      }
    }

    function collectUiSettingsPayload() {
      const decoderCalibration = {
        max_samples: Number(document.getElementById('decoder-cal-max-samples').value || '8'),
        frames_per_sample: Number(document.getElementById('decoder-cal-frames-per-sample').value || '128'),
        keyboard_metric: document.getElementById('decoder-cal-keyboard-metric').value || 'cost_weighted_accuracy',
        keyboard_false_positive_cost: Number(document.getElementById('decoder-cal-keyboard-fp-cost').value || '1.0'),
        keyboard_false_negative_cost: Number(document.getElementById('decoder-cal-keyboard-fn-cost').value || '1.0'),
        mouse_metric: document.getElementById('decoder-cal-mouse-metric').value || 'balanced_accuracy',
        mouse_move_weight: Number(document.getElementById('decoder-cal-mouse-move-weight').value || '0.70'),
        mouse_stochastic_sampling: Boolean(document.getElementById('decoder-cal-mouse-stochastic').checked),
        keyboard_false_positive_cost_overrides: {},
        keyboard_false_negative_cost_overrides: {},
      };
      const globalKeyboardFpCost = decoderCalibration.keyboard_false_positive_cost;
      const globalKeyboardFnCost = decoderCalibration.keyboard_false_negative_cost;
      for (const action of keyboardActionLabels) {
        const fpInput = document.getElementById(decoderCalibrationFpCostInputId(action));
        if (fpInput) {
          const raw = `${fpInput.value || ''}`.trim();
          if (raw) {
            const value = Number(raw);
            if (Number.isFinite(value) && Math.abs(value - globalKeyboardFpCost) > 1e-9) {
              decoderCalibration.keyboard_false_positive_cost_overrides[action] = value;
            }
          }
        }
        const fnInput = document.getElementById(decoderCalibrationFnCostInputId(action));
        if (fnInput) {
          const raw = `${fnInput.value || ''}`.trim();
          if (raw) {
            const value = Number(raw);
            if (Number.isFinite(value) && Math.abs(value - globalKeyboardFnCost) > 1e-9) {
              decoderCalibration.keyboard_false_negative_cost_overrides[action] = value;
            }
          }
        }
      }
      return {
        browser_fps: Number(document.getElementById('refresh-fps').value || '2'),
        player_index: Number(document.getElementById('player-index').value || '0'),
        capture_player_index: Number(document.getElementById('capture-player-index').value || '0'),
        decoder_calibration: decoderCalibration,
      };
    }

    async function persistUiSettingsNow() {
      const res = await fetch('/api/ui_settings', {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify(collectUiSettingsPayload()),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload.detail || `failed to persist ui settings (${res.status})`);
      }
      clearDirty(persistentUiInputIds);
      for (const action of keyboardActionLabels) {
        clearDirty([decoderCalibrationFnCostInputId(action), decoderCalibrationFpCostInputId(action)]);
      }
    }

    function scheduleUiSettingsPersist() {
      if (uiSettingsSaveTimer) {
        clearTimeout(uiSettingsSaveTimer);
      }
      uiSettingsSaveTimer = setTimeout(() => {
        uiSettingsSaveTimer = null;
        persistUiSettingsNow().catch((error) => console.error(error));
      }, 250);
    }

    function switchView(name) {
      for (const id of ['view', 'logits', 'capture']) {
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
      syncKeyboardThresholdInputs(
        decode.keyboard_threshold !== undefined ? decode.keyboard_threshold : 0.12,
        decode.keyboard_thresholds || {}
      );
      if (decode.mouse_zero_bias !== undefined) syncInputValue('mouse-zero-bias', decode.mouse_zero_bias);
      if (decode.mouse_scale !== undefined) syncInputValue('mouse-scale', decode.mouse_scale);
      if (decode.mouse_scale_x !== undefined) syncInputValue('mouse-scale-x', decode.mouse_scale_x);
      if (decode.mouse_scale_y !== undefined) syncInputValue('mouse-scale-y', decode.mouse_scale_y);
      if (decode.mouse_temperature !== undefined) syncInputValue('mouse-temperature', decode.mouse_temperature);
      if (decode.mouse_top_k !== undefined) syncInputValue('mouse-top-k', decode.mouse_top_k);
      if (decode.mouse_deadzone !== undefined) syncInputValue('mouse-deadzone', decode.mouse_deadzone);
      syncInputValue('cache-window', data.cache_window_frames ?? '');
      const uiSettings = data.ui_settings || {};
      if (uiSettings.browser_fps !== undefined) syncInputValue('refresh-fps', uiSettings.browser_fps);
      if (uiSettings.player_index !== undefined) syncInputValue('player-index', uiSettings.player_index);
      if (uiSettings.capture_player_index !== undefined) syncInputValue('capture-player-index', uiSettings.capture_player_index);
      const decoderSettings = uiSettings.decoder_calibration || {};
      if (decoderSettings.max_samples !== undefined) syncInputValue('decoder-cal-max-samples', decoderSettings.max_samples);
      if (decoderSettings.frames_per_sample !== undefined) syncInputValue('decoder-cal-frames-per-sample', decoderSettings.frames_per_sample);
      if (decoderSettings.keyboard_false_positive_cost !== undefined) syncInputValue('decoder-cal-keyboard-fp-cost', decoderSettings.keyboard_false_positive_cost);
      if (decoderSettings.keyboard_false_negative_cost !== undefined) syncInputValue('decoder-cal-keyboard-fn-cost', decoderSettings.keyboard_false_negative_cost);
      if (decoderSettings.keyboard_metric !== undefined) syncInputValue('decoder-cal-keyboard-metric', decoderSettings.keyboard_metric);
      if (decoderSettings.mouse_move_weight !== undefined) syncInputValue('decoder-cal-mouse-move-weight', decoderSettings.mouse_move_weight);
      if (decoderSettings.mouse_metric !== undefined) syncInputValue('decoder-cal-mouse-metric', decoderSettings.mouse_metric);
      const stochasticEl = document.getElementById('decoder-cal-mouse-stochastic');
      if (stochasticEl && stochasticEl.dataset.dirty !== 'true' && document.activeElement !== stochasticEl) {
        stochasticEl.checked = Boolean(decoderSettings.mouse_stochastic_sampling);
      }
      syncDecoderCalibrationCostInputs(
        decoderSettings.keyboard_false_negative_cost !== undefined ? decoderSettings.keyboard_false_negative_cost : 1.0,
        decoderSettings.keyboard_false_negative_cost_overrides || {},
        decoderCalibrationFnCostInputId,
      );
      syncDecoderCalibrationCostInputs(
        decoderSettings.keyboard_false_positive_cost !== undefined ? decoderSettings.keyboard_false_positive_cost : 1.0,
        decoderSettings.keyboard_false_positive_cost_overrides || {},
        decoderCalibrationFpCostInputId,
      );
      renderCalibration(data.calibration || {});
      renderDecoderCalibration(data.decoder_calibration || {});
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

    function renderLogitsInto(containerId, data) {
      const container = document.getElementById(containerId);
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

    function renderLogits(data) {
      renderLogitsInto('logits-graphs', data);
    }

    function renderDecisionPathInto(containerId, data, titleText) {
      const wrap = document.getElementById(containerId);
      wrap.innerHTML = '';
      const title = document.createElement('h2');
      title.className = 'decision-title';
      title.textContent = titleText;
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

      const thresholdHits = trace.keyboard?.threshold_hits || [];
      if (thresholdHits.length) {
        const hitLine = document.createElement('div');
        hitLine.className = 'decision-line';
        hitLine.innerHTML = `<strong>threshold hits:</strong> ${thresholdHits.map(item => `${item.action} ${fmtProb(item.probability)} >= ${fmtProb(item.threshold)}`).join('; ')}`;
        wrap.appendChild(hitLine);
      }

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

    function renderDecisionPath(data) {
      renderDecisionPathInto('decision-path', data, 'Selected Path');
    }

    async function refreshLogits() {
      const playerIndex = Number(document.getElementById('player-index').value || '0');
      const res = await fetch(`/api/logits/${playerIndex}`);
      const data = await res.json();
      renderDecisionPath(data);
      renderLogits(data);
    }

    function renderCaptureSummary(data) {
      document.getElementById('capture-state').textContent = JSON.stringify(data, null, 2);
      const slider = document.getElementById('capture-frame-slider');
      const frameCount = Number(data?.frame_count || 0);
      slider.max = String(Math.max(0, frameCount - 1));
      if (frameCount === 0) {
        slider.value = '0';
        document.getElementById('capture-frame-label').textContent = data?.active ? 'capturing, no processed frames yet' : 'no capture loaded';
        document.getElementById('capture-frame-state').textContent = 'no capture frame selected';
        document.getElementById('capture-frame-image').src = '';
        renderDecisionPathInto('capture-decision-path', {}, 'Captured Path');
        renderLogitsInto('capture-graphs', {});
        return;
      }
      const current = Math.min(frameCount - 1, Number(slider.value || '0'));
      slider.value = String(current);
      document.getElementById('capture-frame-label').textContent = `frame ${current + 1} / ${frameCount}${data?.active ? ' (capturing)' : ''}`;
    }

    async function refreshCaptureSummary() {
      const res = await fetch('/api/capture');
      const data = await res.json();
      renderCaptureSummary(data);
      const frameCount = Number(data?.frame_count || 0);
      if (frameCount > 0 && !data?.active) {
        const current = Math.min(frameCount - 1, Number(document.getElementById('capture-frame-slider').value || '0'));
        document.getElementById('capture-frame-slider').value = String(current);
        await loadCaptureFrame();
      }
    }

    async function loadCaptureFrame() {
      const slider = document.getElementById('capture-frame-slider');
      const frameIndex = Number(slider.value || '0');
      const playerIndex = Number(document.getElementById('capture-player-index').value || '0');
      const res = await fetch(`/api/capture/frame/${frameIndex}/${playerIndex}`);
      if (!res.ok) {
        document.getElementById('capture-frame-state').textContent = 'no capture frame selected';
        document.getElementById('capture-frame-image').src = '';
        renderDecisionPathInto('capture-decision-path', {}, 'Captured Path');
        renderLogitsInto('capture-graphs', {});
        return;
      }
      const data = await res.json();
      document.getElementById('capture-frame-state').textContent = JSON.stringify(data, null, 2);
      const frameCount = Number(document.getElementById('capture-frame-slider').max || '0') + 1;
      document.getElementById('capture-frame-label').textContent = `frame ${frameIndex + 1} / ${frameCount} | ${data.player_name} | seq ${data?.slot?.frame_seq ?? 'n/a'}`;
      document.getElementById('capture-frame-image').src = `/api/capture/frame/${frameIndex}/${playerIndex}/image?ts=${Date.now()}`;
      renderDecisionPathInto('capture-decision-path', data, 'Captured Path');
      renderLogitsInto('capture-graphs', data);
    }

    async function startCapture() {
      const data = await post('/api/capture/start');
      renderCaptureSummary(data.capture || data);
    }

    async function stopCapture() {
      const data = await post('/api/capture/stop');
      renderCaptureSummary(data.capture || data);
      await loadCaptureFrame();
    }

    async function clearCapture() {
      pauseCapturePlayback();
      const data = await post('/api/capture/clear');
      renderCaptureSummary(data.capture || data);
    }

    function pauseCapturePlayback() {
      if (capturePlaybackTimer) {
        clearInterval(capturePlaybackTimer);
        capturePlaybackTimer = null;
      }
      document.getElementById('capture-play-button').textContent = 'play';
    }

    function toggleCapturePlayback() {
      if (capturePlaybackTimer) {
        pauseCapturePlayback();
        return;
      }
      const slider = document.getElementById('capture-frame-slider');
      capturePlaybackTimer = setInterval(async () => {
        const max = Number(slider.max || '0');
        let next = Number(slider.value || '0') + 1;
        if (next > max) {
          pauseCapturePlayback();
          return;
        }
        slider.value = String(next);
        await loadCaptureFrame();
      }, 100);
      document.getElementById('capture-play-button').textContent = 'pause';
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
      const keyboardThreshold = Number(document.getElementById('keyboard-threshold').value || '0.12');
      const keyboardThresholds = {};
      for (const action of keyboardActionLabels) {
        const id = keyboardThresholdInputId(action);
        const raw = document.getElementById(id).value;
        const value = Number(raw || String(keyboardThreshold));
        if (Math.abs(value - keyboardThreshold) > 1e-9) {
          keyboardThresholds[action] = value;
        }
      }
      clearDirty([
        'keyboard-threshold',
        'mouse-zero-bias',
        'mouse-scale',
        'mouse-scale-x',
        'mouse-scale-y',
        'mouse-temperature',
        'mouse-top-k',
        'mouse-deadzone',
        ...keyboardActionLabels.map(keyboardThresholdInputId),
      ]);
      await post('/api/action_decode', {
        keyboard_threshold: keyboardThreshold,
        keyboard_thresholds: keyboardThresholds,
        mouse_zero_bias: Number(document.getElementById('mouse-zero-bias').value || '0'),
        mouse_scale: Number(document.getElementById('mouse-scale').value || '1'),
        mouse_scale_x: Number(document.getElementById('mouse-scale-x').value || '1'),
        mouse_scale_y: Number(document.getElementById('mouse-scale-y').value || '1'),
        mouse_temperature: Number(document.getElementById('mouse-temperature').value || '1'),
        mouse_top_k: Number(document.getElementById('mouse-top-k').value || '1'),
        mouse_deadzone: Number(document.getElementById('mouse-deadzone').value || '0.35'),
      });
      await refreshLogits();
    }

    function renderCalibration(data) {
      const el = document.getElementById('calibration-status');
      if (!el) return;
      if (!data || (!data.running && !data.updated_at && !data.last_error && !data.last_result)) {
        el.textContent = 'no calibration run yet';
        return;
      }
      el.textContent = JSON.stringify(data, null, 2);
    }

    function renderDecoderCalibration(data) {
      const statusEl = document.getElementById('decoder-calibration-status');
      const progressEl = document.getElementById('decoder-calibration-progress');
      const textEl = document.getElementById('decoder-calibration-progress-text');
      if (!statusEl || !progressEl || !textEl) return;
      if (!data || (!data.running && !data.updated_at && !data.last_error && !data.last_result)) {
        progressEl.max = 1;
        progressEl.value = 0;
        textEl.textContent = 'idle';
        statusEl.textContent = 'no decoder calibration run yet';
        return;
      }
      const total = Math.max(1, Number(data.progress_total || 1));
      const current = Math.max(0, Number(data.progress_current || 0));
      progressEl.max = total;
      progressEl.value = Math.min(total, current);
      let text = data.running ? `${data.current_phase || 'running'} (${current}/${total})` : 'idle';
      if (data.eta_seconds !== null && data.eta_seconds !== undefined && Number.isFinite(Number(data.eta_seconds))) {
        text += ` | eta ${Math.max(0, Math.round(Number(data.eta_seconds)))}s`;
      }
      if (data.last_error) {
        text += ` | error: ${data.last_error}`;
      }
      textEl.textContent = text;
      statusEl.textContent = JSON.stringify(data, null, 2);
    }

    async function calibrateMouse() {
      const result = await post('/api/calibrate_mouse');
      renderCalibration(result.calibration || result);
      await refreshLogits();
    }

    async function startDecoderCalibration() {
      const uiSettings = collectUiSettingsPayload();
      await persistUiSettingsNow();
      await post('/api/calibrate_decoder', uiSettings.decoder_calibration);
    }

    async function cancelDecoderCalibration() {
      await post('/api/calibrate_decoder/cancel');
    }

    async function exportSettings() {
      const res = await fetch('/api/settings/export');
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'model4-runtime-settings.json';
      link.click();
      URL.revokeObjectURL(url);
    }

    function triggerSettingsImport() {
      const input = document.getElementById('settings-import-input');
      if (!input) return;
      input.value = '';
      input.click();
    }

    async function importSettings(event) {
      const file = event?.target?.files?.[0];
      if (!file) return;
      const text = await file.text();
      const payload = JSON.parse(text);
      await post('/api/settings/import', payload);
      clearDirty(stickyInputIds);
      clearDirty(persistentUiInputIds);
      for (const action of keyboardActionLabels) {
        clearDirty([
          keyboardThresholdInputId(action),
          decoderCalibrationFnCostInputId(action),
          decoderCalibrationFpCostInputId(action),
        ]);
      }
      await refreshLogits();
      await refreshCaptureSummary();
    }

    function applyRefreshLoop() {
      if (refreshTimer) {
        clearInterval(refreshTimer);
      }
      const fps = Math.max(0.1, Number(document.getElementById('refresh-fps').value || '2'));
      refreshTimer = setInterval(refreshComposite, Math.max(50, Math.floor(1000 / fps)));
    }

    ensureKeyboardThresholdControls();
    for (const id of stickyInputIds) {
      document.getElementById(id).addEventListener('input', markDirty);
    }
    for (const id of persistentUiInputIds) {
      const el = document.getElementById(id);
      if (!el) continue;
      el.addEventListener('input', markUiSettingDirty);
      el.addEventListener('change', markUiSettingDirty);
    }
    document.getElementById('refresh-fps').addEventListener('change', applyRefreshLoop);
    document.getElementById('capture-player-index').addEventListener('change', loadCaptureFrame);
    refreshState();
    refreshComposite();
    refreshLogits();
    refreshCaptureSummary();
    applyRefreshLoop();
    setInterval(refreshState, 1000);
    setInterval(refreshCaptureSummary, 1000);
  </script>
</body>
</html>
"""

INDEX_HTML = INDEX_HTML.replace("__KEYBOARD_ACTION_LABELS_JSON__", json.dumps(KEYBOARD_ONLY_ACTIONS))
INDEX_HTML = INDEX_HTML.replace(
    "__DEFAULT_DECODER_CALIBRATION_KEYBOARD_FN_COST_OVERRIDES_JSON__",
    json.dumps(DEFAULT_DECODER_CALIBRATION_KEYBOARD_FN_COST_OVERRIDES),
)
INDEX_HTML = INDEX_HTML.replace(
    "__DEFAULT_DECODER_CALIBRATION_KEYBOARD_FP_COST_OVERRIDES_JSON__",
    json.dumps(DEFAULT_DECODER_CALIBRATION_KEYBOARD_FP_COST_OVERRIDES),
)
INDEX_HTML = INDEX_HTML.replace("__DECODER_KEYBOARD_METRICS_JSON__", json.dumps(DECODER_KEYBOARD_METRICS))
INDEX_HTML = INDEX_HTML.replace("__DECODER_MOUSE_METRICS_JSON__", json.dumps(DECODER_MOUSE_METRICS))


def create_app(runtime: Model4InferenceRuntime) -> FastAPI:
    if FastAPI is Any:
        raise RuntimeError("fastapi is not installed. Install inference requirements first.")

    app = FastAPI(title="model4-runtime")
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

    @app.post("/api/ui_settings")
    async def set_ui_settings(payload: dict[str, Any]):
        try:
            return {"ok": True, "ui_settings": await runtime.set_ui_settings(payload)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/settings/export")
    async def export_settings():
        return await runtime.export_settings()

    @app.post("/api/settings/import")
    async def import_settings(payload: dict[str, Any]):
        try:
            return {"ok": True, "settings": await runtime.import_settings(payload)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/calibrate_decoder")
    async def calibrate_decoder(payload: dict[str, Any]):
        try:
            return await runtime.start_decoder_calibration(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/calibrate_decoder/cancel")
    async def cancel_calibrate_decoder():
        try:
            return await runtime.cancel_decoder_calibration()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/capture")
    async def capture():
        return await runtime.capture_snapshot()

    @app.post("/api/capture/start")
    async def capture_start():
        return {"ok": True, "capture": await runtime.start_capture()}

    @app.post("/api/capture/stop")
    async def capture_stop():
        return {"ok": True, "capture": await runtime.stop_capture()}

    @app.post("/api/capture/clear")
    async def capture_clear():
        return {"ok": True, "capture": await runtime.clear_capture()}

    @app.get("/api/capture/frame/{frame_index}/{player_index}")
    async def capture_frame(frame_index: int, player_index: int):
        try:
            return await runtime.capture_frame(frame_index, player_index)
        except IndexError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/capture/frame/{frame_index}/{player_index}/image")
    async def capture_frame_image(frame_index: int, player_index: int):
        try:
            payload = await runtime.capture_frame_jpeg(frame_index, player_index)
            return Response(content=payload, media_type="image/jpeg")
        except IndexError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/calibrate_mouse")
    async def calibrate_mouse():
        try:
            result = await runtime.calibrate_mouse()
            return {"ok": True, "calibration": result}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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
