#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import torch

from config import DatasetConfig, GlobalConfig, ModelConfig, TrainConfig
from dataset import DatasetRoot, GroundTruth, TrainingSample
from lightning_module import CS2PredictorModule, recursive_apply_to_floats, recursive_to_device
from model import ModelPrediction
from model_loss import mu_law_decode, mu_law_encode
from visualize_inference import (
    ITEM_NAMES,
    KEYBOARD_ONLY_ACTIONS,
    attach_gt_mouse_bins,
    frame_to_bgr,
    gt_frame_data,
    load_global_config,
    pred_frame_data,
    select_random_indices,
    teacher_forced_prev_actions,
)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, Response
except ModuleNotFoundError:
    FastAPI = Any
    HTTPException = RuntimeError
    HTMLResponse = Any
    JSONResponse = Any
    Response = Any


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Model4 Inspect</title>
  <style>
    body { font-family: monospace; background: #11161a; color: #eef3f5; margin: 20px; }
    h1, h2, h3 { margin: 0; }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; align-items: end; }
    button, input, select { background: #0c1013; color: #eef3f5; border: 1px solid #334049; padding: 8px 10px; }
    label { display: grid; gap: 4px; font-size: 12px; color: #c8d8df; }
    .row { display: grid; grid-template-columns: minmax(520px, 1.15fr) minmax(520px, 1fr); gap: 16px; align-items: start; }
    img { width: 100%; border: 1px solid #334049; display: block; background: #0c1013; }
    pre { background: #0c1013; border: 1px solid #334049; padding: 12px; overflow: auto; margin: 0; }
    .status-box { background: #0c1013; border: 1px solid #334049; padding: 12px; margin-bottom: 14px; }
    .status-line { margin: 6px 0; color: #c8d8df; font-size: 12px; }
    .status-line strong { color: #eef3f5; }
    .slider-wrap { background: #0c1013; border: 1px solid #334049; padding: 12px; margin-top: 12px; }
    .slider-row { display: flex; gap: 10px; align-items: center; }
    .slider-row input[type="range"] { flex: 1; }
    .summary-grid { display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 12px; }
    .summary-card { background: #0c1013; border: 1px solid #334049; padding: 12px; }
    .summary-title { font-size: 13px; color: #97aeb9; margin-bottom: 8px; }
    .summary-list { display: grid; gap: 4px; font-size: 12px; color: #c8d8df; }
    .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }
    .chart-card { background: #0c1013; border: 1px solid #334049; padding: 12px; }
    .chart-box { background: #0c1013; border: 1px solid #334049; padding: 12px; margin-bottom: 14px; }
    .chart-box h2 { font-size: 15px; margin-bottom: 10px; }
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
    .gt-row .bar-label { color: #b7ffc9; font-weight: 700; }
    .gt-row .bar-track { border-color: #39d17d; box-shadow: inset 0 0 0 1px rgba(57, 209, 125, 0.25); }
    .gt-row .bar-fill { background: linear-gradient(90deg, #39d17d 0%, #77f2aa 100%); }
    .gt-row .bar-value { color: #b7ffc9; font-weight: 700; }
    .empty-state { background: #0c1013; border: 1px dashed #334049; padding: 20px; color: #97aeb9; }
    details { background: #0c1013; border: 1px solid #334049; padding: 12px; }
    details summary { cursor: pointer; color: #c8d8df; }
  </style>
</head>
<body>
  <h1>Model4 Inspect</h1>
  <div class="toolbar">
    <label>split
      <select id="split">
        <option value="val">val</option>
        <option value="train">train</option>
      </select>
    </label>
    <label>sequence length
      <input id="sequence-length" type="number" min="8" step="8" value="512">
    </label>
    <label>mode
      <select id="mode" onchange="loadFrame()">
        <option value="teacher_forced">teacher forced</option>
        <option value="autoregressive">autoregressive</option>
        <option value="masked_teacher">teacher forced + SOS masking</option>
        <option value="sos_only">SOS only</option>
      </select>
    </label>
    <label>mask fraction
      <input id="mask-fraction" type="number" min="0" max="1" step="0.05" value="0.60">
    </label>
    <button onclick="loadRandomSample()">random sample</button>
    <button onclick="rerunCurrent()">rerun current</button>
  </div>

  <div class="row">
    <div>
      <img id="frame-image" src="" alt="frame">
      <div class="slider-wrap">
        <div class="slider-row">
          <input id="frame-slider" type="range" min="0" max="0" step="1" value="0" oninput="loadFrame()">
          <span id="frame-label">frame 0 / 0</span>
        </div>
      </div>
      <div class="summary-grid">
        <div class="summary-card">
          <div class="summary-title">Ground Truth</div>
          <div id="gt-summary" class="summary-list"></div>
        </div>
        <div class="summary-card">
          <div class="summary-title">Prediction</div>
          <div id="pred-summary" class="summary-list"></div>
        </div>
      </div>
    </div>
    <div>
      <div class="status-box">
        <h2 style="font-size: 15px; margin-bottom: 8px;">Current Sample</h2>
        <div id="status-lines">
          <div class="status-line">no sample loaded</div>
        </div>
      </div>
      <div class="status-box">
        <h2 style="font-size: 15px; margin-bottom: 8px;">Current Frame</h2>
        <div id="frame-status-lines">
          <div class="status-line">no frame loaded</div>
        </div>
      </div>
      <div class="chart-box">
        <h2>Distributions</h2>
        <div id="logits-graphs" class="charts-grid">
          <div class="empty-state">no logits available yet</div>
        </div>
      </div>
      <details>
        <summary>raw frame payload</summary>
        <pre id="frame-json">no frame loaded</pre>
      </details>
    </div>
  </div>

  <script>
    let currentState = null;
    let currentFrame = 0;

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
        const barWidth = plotWidth / points.length;
        const coords = points.map((value, idx) => {
          const x = left + (idx * barWidth) + barWidth / 2;
          const y = top + (1 - Number(value)) * plotHeight;
          return [x, y];
        });
        const gtIdx = panel.gt_index;
        const predIdx = Number(panel.predicted_index || 0);

        for (let idx = 0; idx < points.length; idx++) {
          const value = Number(points[idx] || 0);
          const barHeight = Math.max(1, value * plotHeight);
          const x = left + (idx * barWidth) + 1;
          const y = top + plotHeight - barHeight;
          const isGt = gtIdx !== undefined && gtIdx !== null && Number(gtIdx) === idx;
          const isPred = predIdx === idx;
          svg.appendChild(createSvgElement('rect', {
            x,
            y,
            width: Math.max(1, barWidth - 2),
            height: barHeight,
            fill: isGt ? 'rgba(57, 209, 125, 0.65)' : 'rgba(124, 209, 240, 0.78)',
            stroke: isPred ? '#ffce6e' : (isGt ? '#39d17d' : 'none'),
            'stroke-width': isPred || isGt ? 2 : 0,
          }));
        }

        if (gtIdx !== undefined && gtIdx !== null) {
          const gt = Number(gtIdx);
          if (gt >= 0 && gt < coords.length) {
            const [gx] = coords[gt];
            svg.appendChild(createSvgElement('line', {
              x1: gx,
              y1: top,
              x2: gx,
              y2: top + plotHeight,
              stroke: '#39d17d',
              'stroke-width': 2,
              'stroke-dasharray': '4 3',
            }));
          }
        }

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
          const x = left + (idx * barWidth) + barWidth / 2;
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
      const gtIndex = panel.gt_index;
      const gtIndices = new Set(Array.isArray(panel.gt_indices) ? panel.gt_indices.map(Number) : []);
      for (let i = 0; i < labels.length; i++) {
        const row = document.createElement('div');
        row.className = 'bar-row';
        if ((gtIndex !== undefined && gtIndex !== null && Number(gtIndex) === i) || gtIndices.has(i)) {
          row.classList.add('gt-row');
        }

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
      if (panel.gt_label !== undefined) {
        summary.appendChild(createChip(`gt: ${panel.gt_label}`));
      }
      if (panel.gt_actions && panel.gt_actions.length) {
        summary.appendChild(createChip(`gt active: ${panel.gt_actions.join(', ')}`));
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

    function modeValue() {
      return document.getElementById('mode').value;
    }

    function splitValue() {
      return document.getElementById('split').value;
    }

    function maskFractionValue() {
      return Number(document.getElementById('mask-fraction').value || 0.6);
    }

    function sequenceLengthValue() {
      return Number(document.getElementById('sequence-length').value || 512);
    }

    async function postJson(url, body) {
      const res = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body || {}),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `${res.status}`);
      }
      return await res.json();
    }

    function setStatusLoading(text) {
      const wrap = document.getElementById('status-lines');
      wrap.innerHTML = `<div class="status-line">${text}</div>`;
    }

    async function loadRandomSample() {
      setStatusLoading('loading random sample...');
      try {
        const state = await postJson('/api/sample/random', {
          split: splitValue(),
          sequence_length: sequenceLengthValue(),
          mask_fraction: maskFractionValue(),
        });
        applyState(state);
      } catch (err) {
        setStatusLoading(`error: ${String(err)}`);
      }
    }

    async function rerunCurrent() {
      setStatusLoading('rerunning current sample...');
      try {
        const state = await postJson('/api/sample/rerun', {
          mask_fraction: maskFractionValue(),
        });
        applyState(state);
      } catch (err) {
        setStatusLoading(`error: ${String(err)}`);
      }
    }

    function renderSummary(containerId, summary) {
      const wrap = document.getElementById(containerId);
      wrap.innerHTML = '';
      const lines = [
        `alive=${summary.alive}`,
        `hp=${summary.hp ?? '-'} armor=${summary.armor ?? '-'} money=${summary.money ?? '-'}`,
        `weapon=${summary.weapon ?? '-'}`,
        `mouse=(${Number(summary.mouse_x || 0).toFixed(2)}, ${Number(summary.mouse_y || 0).toFixed(2)}) bins=[${summary.mouse_x_bin}, ${summary.mouse_y_bin}]`,
        `keys=${summary.keys ?? '-'}`,
        `eco_buy=${summary.eco_buy ?? '-'} buy_now=${summary.eco_purchase}`,
      ];
      if (summary.eco_purchase_prob !== undefined) {
        lines.push(`eco_purchase_prob=${Number(summary.eco_purchase_prob).toFixed(3)}`);
      }
      if (summary.prev_action_source) {
        lines.push(`prev_action_source=${summary.prev_action_source}`);
      }
      for (const line of lines) {
        const div = document.createElement('div');
        div.textContent = line;
        wrap.appendChild(div);
      }
    }

    function renderFrameStatus(data) {
      const wrap = document.getElementById('frame-status-lines');
      wrap.innerHTML = '';
      const pred = data?.prediction || {};
      const gt = data?.ground_truth || {};
      const lines = [
        `frame=${data?.frame_index ?? 0}`,
        `prev_action_source=${pred.prev_action_source || '-'}`,
        `gt_weapon=${gt.weapon || '-'}  pred_weapon=${pred.eco_buy || '-'}`,
        `gt_keys=${gt.keys || '-'}  pred_keys=${pred.keys || '-'}`,
        `gt_mouse_bins=[${gt.mouse_x_bin}, ${gt.mouse_y_bin}]  pred_mouse_bins=[${pred.mouse_x_bin}, ${pred.mouse_y_bin}]`,
      ];
      for (const text of lines) {
        const div = document.createElement('div');
        div.className = 'status-line';
        div.textContent = text;
        wrap.appendChild(div);
      }
    }

    function applyState(state) {
      currentState = state;
      currentFrame = 0;
      const slider = document.getElementById('frame-slider');
      slider.min = 0;
      slider.max = Math.max(0, Number(state.num_frames || 1) - 1);
      slider.value = 0;

      const wrap = document.getElementById('status-lines');
      wrap.innerHTML = '';
      const lines = [
        `split=${state.split} sequence_length=${state.sequence_length} realized_frames=${state.num_frames}`,
        `demo=${state.demo_name} round=${state.round_num} team=${state.team} player=${state.player_name} player_idx=${state.player_idx}`,
        `sample_index=${state.sample_index} sample_start_frame=${state.start_frame}`,
        `mask_fraction_setting=${Number(state.mask_fraction_setting || 0).toFixed(2)}`,
      ];
      if (state.masked_frames) {
        lines.push(`masked_teacher realized=${state.masked_frames.masked_teacher ?? 0} frames (${(100 * Number(state.mask_fraction_realized?.masked_teacher || 0)).toFixed(1)}%)`);
      }
      for (const text of lines) {
        const div = document.createElement('div');
        div.className = 'status-line';
        div.innerHTML = `<strong>${text.split('=')[0]}:</strong> ${text.slice(text.indexOf('=') + 1)}`;
        wrap.appendChild(div);
      }
      loadFrame();
    }

    async function loadFrame() {
      if (!currentState) {
        return;
      }
      currentFrame = Number(document.getElementById('frame-slider').value || 0);
      document.getElementById('frame-label').textContent = `frame ${currentFrame} / ${Math.max(0, Number(currentState.num_frames) - 1)}`;
      document.getElementById('frame-image').src = `/api/image/${currentFrame}?token=${encodeURIComponent(currentState.sample_token)}`;
      const res = await fetch(`/api/frame/${currentFrame}?mode=${encodeURIComponent(modeValue())}`);
      const data = await res.json();
      renderSummary('gt-summary', data.ground_truth || {});
      renderSummary('pred-summary', data.prediction || {});
      renderFrameStatus(data);
      renderLogitsInto('logits-graphs', data);
      document.getElementById('frame-json').textContent = JSON.stringify(data, null, 2);
    }

    window.addEventListener('load', async () => {
      try {
        const res = await fetch('/api/state');
        const state = await res.json();
        if (state && state.loaded) {
          document.getElementById('split').value = state.split;
          document.getElementById('sequence-length').value = state.sequence_length;
          document.getElementById('mask-fraction').value = Number(state.mask_fraction_setting || 0.6).toFixed(2);
          applyState(state);
        } else {
          await loadRandomSample();
        }
      } catch (_) {
        await loadRandomSample();
      }
    });
  </script>
</body>
</html>
"""


@dataclass
class LoadedInspection:
    sample_token: str
    split: str
    sequence_length: int
    sample_index: int
    demo_name: str
    round_num: int
    team: str
    player_idx: int
    player_name: str
    start_frame: int
    num_frames: int
    mask_fraction_setting: float
    masked_frames: dict[str, int]
    mask_fraction_realized: dict[str, float]
    images_cpu: torch.Tensor
    frames_by_mode: dict[str, list[dict[str, Any]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./dataset0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--fast_preprocess", action="store_true")
    parser.add_argument("--default_split", choices=["train", "val"], default="val")
    parser.add_argument("--default_sequence_length", type=int, default=512)
    parser.add_argument("--default_mode", choices=["teacher_forced", "autoregressive", "masked_teacher", "sos_only"], default="teacher_forced")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _top_categories(labels: list[str], probabilities: list[float], limit: int = 8) -> list[dict[str, Any]]:
    pairs = sorted(enumerate(probabilities), key=lambda item: float(item[1]), reverse=True)[:limit]
    return [
        {"label": str(labels[idx]), "probability": float(prob), "index": int(idx)}
        for idx, prob in pairs
    ]


def _mouse_labels(cfg: ModelConfig) -> list[str]:
    labels: list[str] = []
    center = cfg.mouse_bins_count // 2
    for idx in range(cfg.mouse_bins_count):
        decoded = mu_law_decode(torch.tensor(idx), cfg.mouse_mu, cfg.mouse_max, cfg.mouse_bins_count).item()
        if idx == center:
            label = "0"
        else:
            magnitude = abs(decoded)
            if magnitude < 1.0:
                label = f"{decoded:.3f}"
            elif magnitude < 10.0:
                label = f"{decoded:.2f}"
            else:
                label = f"{decoded:.1f}"
            if label in {"-0.000", "0.000", "-0.00", "0.00", "-0.0", "0.0"}:
                label = "0"
            else:
                label = label.rstrip("0").rstrip(".")
        labels.append(label)
    return labels


def _eco_label(idx: int) -> str:
    if idx == len(ITEM_NAMES):
        return "NO_BUY"
    if 0 <= idx < len(ITEM_NAMES):
        return ITEM_NAMES[idx]
    return f"item_{idx}"


def _binary_panel(head_name: str, logit_value: float, labels: list[str]) -> dict[str, Any]:
    yes_probability = float(torch.sigmoid(torch.tensor(logit_value)).item())
    probabilities = [1.0 - yes_probability, yes_probability]
    predicted_index = int(yes_probability >= 0.5)
    return {
        "head_name": head_name,
        "kind": "binary",
        "representation": "sigmoid",
        "x_labels": labels,
        "probabilities": probabilities,
        "raw_logits": [0.0, float(logit_value)],
        "predicted_index": predicted_index,
        "predicted_label": labels[predicted_index],
        "predicted_probability": float(probabilities[predicted_index]),
        "top_categories": _top_categories(labels, probabilities, limit=2),
        "full_categories": [
            {"label": str(label), "probability": float(probability), "index": int(idx)}
            for idx, (label, probability) in enumerate(zip(labels, probabilities))
        ],
    }


def _categorical_panel(head_name: str, values: torch.Tensor, labels: list[str], *, gt_index: int | None = None) -> dict[str, Any]:
    probabilities = torch.softmax(values.float(), dim=0)
    predicted_index = int(torch.argmax(probabilities).item())
    probs_list = [float(value) for value in probabilities.tolist()]
    panel = {
        "head_name": head_name,
        "kind": "categorical",
        "representation": "softmax",
        "x_labels": labels,
        "probabilities": probs_list,
        "raw_logits": [float(value) for value in values.tolist()],
        "predicted_index": predicted_index,
        "predicted_label": labels[predicted_index],
        "predicted_probability": float(probs_list[predicted_index]),
        "top_categories": _top_categories(labels, probs_list),
        "full_categories": [
            {"label": str(label), "probability": float(probability), "index": int(idx)}
            for idx, (label, probability) in enumerate(zip(labels, probs_list))
        ],
    }
    if gt_index is not None and 0 <= gt_index < len(labels):
        panel["gt_index"] = int(gt_index)
        panel["gt_label"] = str(labels[gt_index])
    return panel


def _keyboard_panel(values: torch.Tensor, *, gt_mask: int | None = None) -> dict[str, Any]:
    probabilities = torch.sigmoid(values.float())
    probs_list = [float(value) for value in probabilities.tolist()]
    thresholds = [0.5 for _ in KEYBOARD_ONLY_ACTIONS]
    active_actions = [
        label for label, probability, threshold in zip(KEYBOARD_ONLY_ACTIONS, probs_list, thresholds) if probability >= threshold
    ]
    return {
        "head_name": "keyboard_logits",
        "kind": "bernoulli",
        "representation": "sigmoid",
        "x_labels": KEYBOARD_ONLY_ACTIONS,
        "probabilities": probs_list,
        "raw_logits": [float(value) for value in values.tolist()],
        "active_actions": active_actions,
        "threshold": 0.5,
        "thresholds": {label: 0.5 for label in KEYBOARD_ONLY_ACTIONS},
        "top_categories": _top_categories(KEYBOARD_ONLY_ACTIONS, probs_list),
        "full_categories": [
            {
                "label": str(label),
                "probability": float(probability),
                "index": int(idx),
                "threshold": 0.5,
            }
            for idx, (label, probability) in enumerate(zip(KEYBOARD_ONLY_ACTIONS, probs_list))
        ],
    }
    if gt_mask is not None:
        gt_indices = [idx for idx in range(len(KEYBOARD_ONLY_ACTIONS)) if (int(gt_mask) >> idx) & 1]
        panel["gt_indices"] = gt_indices
        panel["gt_actions"] = [KEYBOARD_ONLY_ACTIONS[idx] for idx in gt_indices]
    return panel


def sample_prev_action_sos_mask(
    truth: GroundTruth,
    train_cfg: TrainConfig,
    *,
    seed: int,
    mask_fraction: float | None = None,
) -> torch.Tensor:
    B, T = truth.keyboard_mask.shape
    device = truth.keyboard_mask.device
    p = float(train_cfg.teacher_forcing_sos_dropout if mask_fraction is None else mask_fraction)
    num_windows = int(getattr(train_cfg, "teacher_forcing_sos_windows", 0))
    window_frac = float(getattr(train_cfg, "teacher_forcing_sos_window_frac", 0.0))

    generator = torch.Generator(device=device.type if device.type != "cpu" else "cpu")
    generator.manual_seed(int(seed))

    if p <= 0.0:
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    elif p >= 1.0:
        mask = torch.ones(B, T, dtype=torch.bool, device=device)
    else:
        mask = torch.rand(B, T, device=device, generator=generator) < p

    if num_windows > 0 and window_frac > 0.0 and T > 0:
        window_len = max(1, int(round(T * window_frac)))
        window_len = min(window_len, T)
        max_start = T - window_len
        for b in range(B):
            for _ in range(num_windows):
                start = 0
                if max_start > 0:
                    start = int(torch.randint(0, max_start + 1, (1,), device=device, generator=generator).item())
                mask[b, start : start + window_len] = True

    mask[:, 0] = True
    return mask


class Model4Inspector:
    def __init__(
        self,
        checkpoint: str,
        *,
        data_root: str,
        device: str = "cuda",
        fast_preprocess: bool = False,
        seed: int = 0,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.global_cfg = load_global_config(self.checkpoint, data_root)
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.dataset_cache: dict[tuple[str, int], Any] = {}
        self.loaded: LoadedInspection | None = None

        self.model = CS2PredictorModule.load_from_checkpoint(str(self.checkpoint), global_cfg=self.global_cfg, strict=False)
        self.model.to(self.device)
        self.model.eval()
        if fast_preprocess and self.global_cfg.dataset.epoch_video_decoding_device == "cpu":
            self.model.model.video.set_fast_preprocess(True)

        self.target_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "no": torch.float32,
        }.get(self.global_cfg.train.mixed_precision, torch.float32)

    def _dataset_for(self, split: str, sequence_length: int):
        key = (split, sequence_length)
        cached = self.dataset_cache.get(key)
        if cached is not None:
            return cached
        dataset_cfg = DatasetConfig(**asdict(self.global_cfg.dataset))
        dataset_cfg.epoch_round_sample_length = int(sequence_length)
        ds_root = DatasetRoot(dataset_cfg)
        ds = ds_root.build_dataset(split)
        if len(ds) == 0 and split == "val":
            ds = ds_root.build_dataset("train")
        if len(ds) == 0:
            raise RuntimeError("dataset is empty")
        self.dataset_cache[key] = ds
        return ds

    def _prepare_sample(self, sample: TrainingSample) -> TrainingSample:
        sample = copy.deepcopy(sample)
        sample = recursive_to_device(sample, self.device)
        sample.truth = recursive_apply_to_floats(sample.truth, lambda t: t.to(dtype=self.target_dtype))
        if sample.images.ndim == 4:
            sample.images = sample.images.unsqueeze(0)
        if sample.audio.ndim == 2:
            sample.audio = sample.audio.unsqueeze(0)
        if sample.truth.keyboard_mask.ndim == 1:
            for field_name, value in sample.truth.__dict__.items():
                if torch.is_tensor(value):
                    setattr(sample.truth, field_name, value.unsqueeze(0))
        return sample

    def _predict_teacher_forced(self, sample: TrainingSample) -> tuple[ModelPrediction, torch.Tensor | None]:
        prev_actions = teacher_forced_prev_actions(sample.truth, self.global_cfg.model)
        pred = ModelPrediction(**self.model(sample.images, sample.audio, **prev_actions))
        return pred, None

    def _predict_masked_teacher(self, sample: TrainingSample, seed: int, mask_fraction: float | None) -> tuple[ModelPrediction, torch.Tensor]:
        prev_actions = teacher_forced_prev_actions(sample.truth, self.global_cfg.model)
        sos_mask = sample_prev_action_sos_mask(sample.truth, self.global_cfg.train, seed=seed, mask_fraction=mask_fraction)
        prev_actions["prev_action_sos_mask"] = sos_mask
        pred = ModelPrediction(**self.model(sample.images, sample.audio, **prev_actions))
        return pred, sos_mask

    def _predict_sos_only(self, sample: TrainingSample) -> tuple[ModelPrediction, torch.Tensor]:
        B, T = sample.truth.keyboard_mask.shape
        prev_actions = teacher_forced_prev_actions(sample.truth, self.global_cfg.model)
        sos_mask = torch.ones(B, T, dtype=torch.bool, device=sample.truth.keyboard_mask.device)
        prev_actions["prev_action_sos_mask"] = sos_mask
        pred = ModelPrediction(**self.model(sample.images, sample.audio, **prev_actions))
        return pred, sos_mask

    def _predict_autoregressive(self, sample: TrainingSample) -> tuple[ModelPrediction, torch.Tensor | None]:
        B, T, _, _, _ = sample.images.shape
        samples_per_frame = sample.audio.shape[-1] // T
        ar_state = self.model.model.init_autoregressive_state()
        mouse_x: list[torch.Tensor] = []
        mouse_y: list[torch.Tensor] = []
        keyboard: list[torch.Tensor] = []
        eco_buy: list[torch.Tensor] = []
        eco_purchase: list[torch.Tensor] = []

        for t in range(T):
            audio_start = t * samples_per_frame
            audio_end = (t + 1) * samples_per_frame if t < T - 1 else sample.audio.shape[-1]
            step_pred_dict, ar_state = self.model.model.forward_step(
                sample.images[:, t : t + 1],
                sample.audio[:, :, audio_start:audio_end],
                state=ar_state,
            )
            mouse_x.append(step_pred_dict["mouse_x"])
            mouse_y.append(step_pred_dict["mouse_y"])
            keyboard.append(step_pred_dict["keyboard_logits"])
            eco_buy.append(step_pred_dict["eco_buy_logits"])
            eco_purchase.append(step_pred_dict["eco_purchase_logits"])

        pred = ModelPrediction(
            mouse_x=torch.cat(mouse_x, dim=1),
            mouse_y=torch.cat(mouse_y, dim=1),
            keyboard_logits=torch.cat(keyboard, dim=1),
            eco_buy_logits=torch.cat(eco_buy, dim=1),
            eco_purchase_logits=torch.cat(eco_purchase, dim=1),
        )
        return pred, None

    def _panels_for_frame(self, pred: ModelPrediction, truth: GroundTruth, t: int) -> list[dict[str, Any]]:
        cfg = self.global_cfg.model
        gt_mouse_x = int(mu_law_encode(truth.mouse_delta[0, t, 0].float().cpu(), cfg.mouse_mu, cfg.mouse_max, cfg.mouse_bins_count).item())
        gt_mouse_y = int(mu_law_encode(truth.mouse_delta[0, t, 1].float().cpu(), cfg.mouse_mu, cfg.mouse_max, cfg.mouse_bins_count).item())
        gt_eco_buy = int(truth.eco_buy_idx[0, t].item())
        gt_eco_purchase = int(truth.eco_mask[0, t, 0].item() != 0)
        gt_keyboard_mask = int(truth.keyboard_mask[0, t].item())
        return [
            _categorical_panel("mouse_x", pred.mouse_x[0, t], _mouse_labels(cfg), gt_index=gt_mouse_x),
            _categorical_panel("mouse_y", pred.mouse_y[0, t], _mouse_labels(cfg), gt_index=gt_mouse_y),
            _keyboard_panel(pred.keyboard_logits[0, t], gt_mask=gt_keyboard_mask),
            _categorical_panel(
                "eco_buy_logits",
                pred.eco_buy_logits[0, t],
                [_eco_label(idx) for idx in range(cfg.eco_dim)],
                gt_index=gt_eco_buy if 0 <= gt_eco_buy < cfg.eco_dim else None,
            ),
            {
                **_binary_panel("eco_purchase_logits", float(pred.eco_purchase_logits[0, t, 0].item()), ["no", "yes"]),
                "gt_index": gt_eco_purchase,
                "gt_label": ["no", "yes"][gt_eco_purchase],
            },
        ]

    def _frame_payloads(
        self,
        sample: TrainingSample,
        pred: ModelPrediction,
        *,
        mode: str,
        sos_mask: torch.Tensor | None,
    ) -> list[dict[str, Any]]:
        T = sample.images.shape[1]
        payloads: list[dict[str, Any]] = []
        for t in range(T):
            gt = attach_gt_mouse_bins(gt_frame_data(sample.truth, t), self.global_cfg.model)
            pred_summary = pred_frame_data(pred, t, self.global_cfg.model)
            if mode == "teacher_forced":
                pred_summary["prev_action_source"] = "sos" if t == 0 else "teacher"
            elif mode == "autoregressive":
                pred_summary["prev_action_source"] = "sos" if t == 0 else "predicted"
            elif mode == "sos_only":
                pred_summary["prev_action_source"] = "sos"
            else:
                masked = bool(sos_mask[0, t].item()) if sos_mask is not None else (t == 0)
                pred_summary["prev_action_source"] = "sos" if masked else "teacher"

            payloads.append(
                {
                    "frame_index": t,
                    "ground_truth": gt,
                    "prediction": pred_summary,
                    "panels": self._panels_for_frame(pred, sample.truth, t),
                }
            )
        return payloads

    def _select_random_sample_index(self, ds) -> int:
        seed = self.rng.randint(0, 2**31 - 1)
        return select_random_indices(ds, 1, seed)[0]

    def _state_dict(self, loaded: LoadedInspection | None) -> dict[str, Any]:
        if loaded is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "sample_token": loaded.sample_token,
            "split": loaded.split,
            "sequence_length": loaded.sequence_length,
            "available_modes": ["teacher_forced", "autoregressive", "masked_teacher", "sos_only"],
            "sample_index": loaded.sample_index,
            "demo_name": loaded.demo_name,
            "round_num": loaded.round_num,
            "team": loaded.team,
            "player_idx": loaded.player_idx,
            "player_name": loaded.player_name,
            "start_frame": loaded.start_frame,
            "num_frames": loaded.num_frames,
            "mask_fraction_setting": loaded.mask_fraction_setting,
            "masked_frames": loaded.masked_frames,
            "mask_fraction_realized": loaded.mask_fraction_realized,
        }

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self._state_dict(self.loaded)

    def image_bytes(self, frame_index: int) -> bytes:
        with self.lock:
            loaded = self.loaded
            if loaded is None:
                raise HTTPException(status_code=404, detail="no sample loaded")
            if not (0 <= frame_index < loaded.num_frames):
                raise HTTPException(status_code=404, detail="frame out of range")
            frame_bgr = frame_to_bgr(loaded.images_cpu[frame_index])
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise HTTPException(status_code=500, detail="failed to encode image")
        return encoded.tobytes()

    def frame_payload(self, frame_index: int, *, mode: str) -> dict[str, Any]:
        with self.lock:
            loaded = self.loaded
            if loaded is None:
                raise HTTPException(status_code=404, detail="no sample loaded")
            if not (0 <= frame_index < loaded.num_frames):
                raise HTTPException(status_code=404, detail="frame out of range")
            frames = loaded.frames_by_mode.get(mode)
            if frames is None:
                raise HTTPException(status_code=404, detail=f"mode not available: {mode}")
            payload = dict(frames[frame_index])
            payload["sample_token"] = loaded.sample_token
            payload["mode"] = mode
            return payload

    def _run_sample(self, *, split: str, sequence_length: int, sample_index: int | None = None, mask_fraction: float | None = None) -> dict[str, Any]:
        ds = self._dataset_for(split, sequence_length)
        if sample_index is None:
            sample_index = self._select_random_sample_index(ds)
        if not (0 <= sample_index < len(ds)):
            raise IndexError(f"sample_index out of range: {sample_index}")

        raw_sample = ds[sample_index]
        prepared = self._prepare_sample(raw_sample)
        mask_seed = self.rng.randint(0, 2**31 - 1)

        with torch.no_grad():
            pred_tf, sos_tf = self._predict_teacher_forced(prepared)
            pred_ar, sos_ar = self._predict_autoregressive(prepared)
            pred_masked, sos_masked = self._predict_masked_teacher(prepared, mask_seed, mask_fraction)
            pred_sos, sos_all = self._predict_sos_only(prepared)

        roundsample = raw_sample._roundsample
        if roundsample is None:
            raise RuntimeError("dataset sample missing _roundsample metadata")

        num_frames = int(raw_sample.images.shape[0])
        masked_frames = {
            "teacher_forced": int(sos_tf.sum().item()) if sos_tf is not None else 0,
            "autoregressive": int(sos_ar.sum().item()) if sos_ar is not None else 0,
            "masked_teacher": int(sos_masked.sum().item()) if sos_masked is not None else 0,
            "sos_only": int(sos_all.sum().item()) if sos_all is not None else 0,
        }
        mask_fraction_realized = {
            key: float(value / max(1, num_frames))
            for key, value in masked_frames.items()
        }
        loaded = LoadedInspection(
            sample_token=f"{time.time_ns()}-{sample_index}",
            split=split,
            sequence_length=sequence_length,
            sample_index=sample_index,
            demo_name=roundsample.round.game.demo_name,
            round_num=roundsample.round.round_num,
            team=roundsample.round.team,
            player_idx=roundsample.player_idx,
            player_name=roundsample.player_name,
            start_frame=roundsample.start_frame,
            num_frames=num_frames,
            mask_fraction_setting=float(self.global_cfg.train.teacher_forcing_sos_dropout if mask_fraction is None else mask_fraction),
            masked_frames=masked_frames,
            mask_fraction_realized=mask_fraction_realized,
            images_cpu=raw_sample.images.detach().cpu(),
            frames_by_mode={
                "teacher_forced": self._frame_payloads(prepared, pred_tf, mode="teacher_forced", sos_mask=sos_tf),
                "autoregressive": self._frame_payloads(prepared, pred_ar, mode="autoregressive", sos_mask=sos_ar),
                "masked_teacher": self._frame_payloads(prepared, pred_masked, mode="masked_teacher", sos_mask=sos_masked),
                "sos_only": self._frame_payloads(prepared, pred_sos, mode="sos_only", sos_mask=sos_all),
            },
        )

        self.loaded = loaded
        return self._state_dict(loaded)

    def random_sample(self, *, split: str, sequence_length: int, mask_fraction: float | None = None) -> dict[str, Any]:
        with self.lock:
            return self._run_sample(split=split, sequence_length=sequence_length, sample_index=None, mask_fraction=mask_fraction)

    def rerun_current(self, *, mask_fraction: float | None = None) -> dict[str, Any]:
        with self.lock:
            if self.loaded is None:
                raise HTTPException(status_code=404, detail="no current sample")
            return self._run_sample(
                split=self.loaded.split,
                sequence_length=self.loaded.sequence_length,
                sample_index=self.loaded.sample_index,
                mask_fraction=mask_fraction,
            )


def create_app(inspector: Model4Inspector, *, default_split: str, default_sequence_length: int, default_mode: str) -> FastAPI:
    app = FastAPI(title="Model4 Inspect")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse(inspector.state())

    @app.post("/api/sample/random")
    async def api_sample_random(payload: dict[str, Any]) -> JSONResponse:
        split = str(payload.get("split", default_split))
        sequence_length = int(payload.get("sequence_length", default_sequence_length))
        mask_fraction = payload.get("mask_fraction")
        mask_fraction = None if mask_fraction is None else float(mask_fraction)
        return JSONResponse(inspector.random_sample(split=split, sequence_length=sequence_length, mask_fraction=mask_fraction))

    @app.post("/api/sample/rerun")
    async def api_sample_rerun(payload: dict[str, Any]) -> JSONResponse:
        mask_fraction = payload.get("mask_fraction")
        mask_fraction = None if mask_fraction is None else float(mask_fraction)
        return JSONResponse(inspector.rerun_current(mask_fraction=mask_fraction))

    @app.get("/api/frame/{frame_index}")
    async def api_frame(frame_index: int, mode: str = default_mode) -> JSONResponse:
        return JSONResponse(inspector.frame_payload(frame_index, mode=mode))

    @app.get("/api/image/{frame_index}")
    async def api_image(frame_index: int) -> Response:
        return Response(
            content=inspector.image_bytes(frame_index),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    return app


def main() -> None:
    args = parse_args()
    inspector = Model4Inspector(
        args.checkpoint,
        data_root=args.data_root,
        device=args.device,
        fast_preprocess=args.fast_preprocess,
        seed=args.seed,
    )
    app = create_app(
        inspector,
        default_split=args.default_split,
        default_sequence_length=args.default_sequence_length,
        default_mode=args.default_mode,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
