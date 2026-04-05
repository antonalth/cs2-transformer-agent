#!/usr/bin/env python3
"""
Copyright 2025 Anton Althoff

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import lmdb
import msgpack
import msgpack_numpy as mpnp
import numpy as np
import torch
import torch.nn.functional as F
from torchcodec.decoders import AudioDecoder, VideoDecoder

from config import DatasetConfig

TICK_RATE = 64
FRAME_RATE = 32
TICKS_PER_FRAME = TICK_RATE // FRAME_RATE


def ticks_to_framecount(start_tick: int, end_tick: int) -> int:
    if end_tick < start_tick:
        return 0
    return ((end_tick - start_tick) // TICKS_PER_FRAME) + 1


@dataclass
class GroundTruth:
    alive_mask: torch.BoolTensor      # [T] selected player alive
    team_alive_mask: torch.BoolTensor # [T, 5] full team alive mask
    stats: torch.FloatTensor          # [T, 3]
    mouse_delta: torch.FloatTensor    # [T, 2]
    position: torch.FloatTensor       # [T, 3]
    keyboard_mask: torch.IntTensor    # [T]
    eco_mask: torch.LongTensor        # [T, 4]
    eco_buy_idx: torch.LongTensor     # [T]
    inventory_mask: torch.LongTensor  # [T, 2]
    active_weapon_idx: torch.IntTensor# [T]
    round_number: torch.IntTensor     # [T]
    round_state_mask: torch.ByteTensor# [T]
    enemy_positions: torch.FloatTensor# [T, 5, 3]
    enemy_alive_mask: torch.BoolTensor# [T, 5]


@dataclass
class TrainingSample:
    images: torch.Tensor # [T, C, H, W]
    audio: torch.Tensor # [2, samples]
    truth: GroundTruth
    _roundsample: Optional["RoundSample"] = None


def cs2_collate_fn(batch: List[TrainingSample]) -> TrainingSample:
    imgs = torch.stack([s.images for s in batch])
    audio_raw = torch.stack([s.audio for s in batch])

    first_gt = batch[0].truth
    gt_fields = {}
    for f in fields(first_gt):
        gt_fields[f.name] = torch.stack([getattr(s.truth, f.name) for s in batch])
    batched_truth = GroundTruth(**gt_fields)

    return TrainingSample(
        images=imgs,
        audio=audio_raw,
        truth=batched_truth,
        _roundsample=None,
    )


@dataclass
class Game:
    demo_name: str
    lmdb_path: str
    rounds: List["Round"] = field(default_factory=list)


@dataclass
class Round:
    game: Game
    round_num: int
    team: str
    start_tick: int
    end_tick: int
    pov_video: List[str]
    pov_audio: List[str]
    pov_start_ticks: List[int]
    pov_end_ticks: List[int]
    pov_player_names: List[str]

    @property
    def frame_count(self) -> int:
        return ticks_to_framecount(self.start_tick, self.end_tick)


@dataclass
class RoundSample:
    round: Round
    player_idx: int
    start_tick: int
    start_frame: int
    length_frames: int
    player_start_tick: int
    player_end_tick: int
    player_name: str

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.length_frames

    @property
    def start_time(self) -> float:
        return self.start_frame / FRAME_RATE

    @property
    def end_time(self) -> float:
        return self.end_frame / FRAME_RATE

    @property
    def player_frame_count(self) -> int:
        return ticks_to_framecount(self.player_start_tick, self.player_end_tick)


class Epoch(torch.utils.data.Dataset):
    samples: List[RoundSample]
    config: DatasetConfig
    lmdb_envs: dict[str, lmdb.Environment]

    def __init__(self, config: DatasetConfig, samples: List[RoundSample]):
        self.config = config
        self.samples = samples
        self.lmdb_envs: dict[str, lmdb.Environment] = {}

    def _get_lmdb_env(self, lmdb_path: str) -> lmdb.Environment:
        env = self.lmdb_envs.get(lmdb_path)
        if env is None:
            env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=2048)
            self.lmdb_envs[lmdb_path] = env
        return env

    @staticmethod
    def _lmdb_key(demoname: str, round_num: int, team: str, tick: int) -> bytes:
        return f"{demoname}_round_{round_num:03d}_team_{team}_tick_{tick:08d}".encode("utf-8")

    @staticmethod
    def _bitmask_to_weapon_index(mask: np.ndarray) -> int:
        if mask.sum() == 0:
            return -1
        for i in range(128):
            if (mask[i // 64] >> np.uint64(i % 64)) & np.uint64(1):
                return i
        return -1

    @staticmethod
    def _bitmask_to_item_index(mask: np.ndarray) -> int:
        if mask.sum() == 0:
            return -1
        for i in range(4):
            if mask[i] == 0:
                continue
            for bit in range(64):
                if (mask[i] >> np.uint64(bit)) & np.uint64(1):
                    return i * 64 + bit
        return -1

    @staticmethod
    def pad_or_truncate_to(x: torch.Tensor, length: int, dim: int = 0) -> torch.Tensor:
        current = x.size(dim)
        if current >= length:
            index = [slice(None)] * x.ndim
            index[dim] = slice(0, length)
            return x[tuple(index)]

        pad_shape = list(x.shape)
        pad_shape[dim] = length - current
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=dim)

    def _decode_video(self, sample: RoundSample) -> torch.Tensor:
        pov = sample.round.pov_video[sample.player_idx]
        decoder = VideoDecoder(pov, device=self.config.epoch_video_decoding_device, dimension_order="NCHW")
        frames = decoder.get_frames_in_range(sample.start_frame, sample.start_frame + sample.length_frames).data
        del decoder
        return self.pad_or_truncate_to(frames, self.config.epoch_round_sample_length, dim=0)

    def _decode_audio(self, sample: RoundSample) -> torch.Tensor:
        window_secs = sample.length_frames / FRAME_RATE
        target_samples = int(round(window_secs * self.config.audio_sample_rate))

        decoder = AudioDecoder(sample.round.pov_audio[sample.player_idx], sample_rate=int(self.config.audio_sample_rate))
        try:
            waveform = decoder.get_samples_played_in_range(
                start_seconds=sample.start_time,
                stop_seconds=sample.end_time,
            ).data
        except RuntimeError:
            waveform = torch.empty(0)
        del decoder

        if waveform.numel() == 0:
            waveform = torch.zeros(2, target_samples)
        elif waveform.shape[-1] < target_samples:
            waveform = F.pad(waveform, (0, target_samples - waveform.shape[-1]))
        else:
            waveform = waveform[..., :target_samples]

        return waveform

    def _get_truth(self, sample: RoundSample) -> GroundTruth:
        r = sample.round
        g = r.game
        env = self._get_lmdb_env(g.lmdb_path)
        T = self.config.epoch_round_sample_length

        self_alive = np.zeros((T,), dtype=np.bool_)
        team_alive = np.zeros((T, 5), dtype=np.bool_)
        stats = np.zeros((T, 3), dtype=np.float32)
        mouse = np.zeros((T, 2), dtype=np.float32)
        pos = np.zeros((T, 3), dtype=np.float32)
        kbd = np.zeros((T,), dtype=np.uint32)
        eco = np.zeros((T, 4), dtype=np.uint64)
        eco_idx = np.full((T,), -1, dtype=np.int32)
        inv = np.zeros((T, 2), dtype=np.uint64)
        wep = np.full((T,), -1, dtype=np.int32)
        rnd_num = np.full((T,), r.round_num, dtype=np.int32)
        rnd_state = np.zeros((T,), dtype=np.uint8)
        enemy_pos = np.zeros((T, 5, 3), dtype=np.float32)
        enemy_alive = np.zeros((T, 5), dtype=np.bool_)

        ticks = sample.start_tick + (np.arange(T, dtype=np.int32) * TICKS_PER_FRAME)

        with env.begin(write=False) as txn:
            for frame_idx, tick in enumerate(ticks):
                blob = txn.get(self._lmdb_key(g.demo_name, r.round_num, r.team, int(tick)))
                if not blob:
                    continue

                payload = msgpack.unpackb(blob, raw=False, object_hook=mpnp.decode)
                gs = payload.get("game_state")
                if not gs:
                    continue
                gs = gs[0]

                alive_slots = [i for i in range(5) if (int(gs["team_alive"]) >> i) & 1]
                for slot in alive_slots:
                    team_alive[frame_idx, slot] = True
                self_alive[frame_idx] = sample.player_idx in alive_slots

                enemy_alive_slots = [i for i in range(5) if (int(gs["enemy_alive"]) >> i) & 1]
                for slot in enemy_alive_slots:
                    enemy_alive[frame_idx, slot] = True

                rnd_state[frame_idx] = gs["round_state"]
                enemy_pos[frame_idx] = gs["enemy_pos"]

                pdl = payload.get("player_data")
                if not pdl or len(alive_slots) != len(pdl):
                    continue

                for p_idx, p_data_arr in zip(alive_slots, pdl):
                    if p_idx != sample.player_idx:
                        continue
                    player = p_data_arr[0]
                    stats[frame_idx] = [player["health"], player["armor"], player["money"]]
                    mouse[frame_idx] = player["mouse"]
                    pos[frame_idx] = player["pos"]
                    kbd[frame_idx] = player["keyboard_bitmask"]
                    eco[frame_idx] = player["eco_bitmask"]
                    eco_idx[frame_idx] = self._bitmask_to_item_index(player["eco_bitmask"])
                    inv[frame_idx] = player["inventory_bitmask"]
                    wep[frame_idx] = self._bitmask_to_weapon_index(player["active_weapon_bitmask"])
                    break

        return GroundTruth(
            alive_mask=torch.from_numpy(self_alive),
            team_alive_mask=torch.from_numpy(team_alive),
            stats=torch.from_numpy(stats),
            mouse_delta=torch.from_numpy(mouse),
            position=torch.from_numpy(pos),
            keyboard_mask=torch.from_numpy(kbd.astype(np.int32)),
            eco_mask=torch.from_numpy(eco.astype(np.int64)),
            eco_buy_idx=torch.from_numpy(eco_idx.astype(np.int64)),
            inventory_mask=torch.from_numpy(inv.astype(np.int64)),
            active_weapon_idx=torch.from_numpy(wep),
            round_number=torch.from_numpy(rnd_num),
            round_state_mask=torch.from_numpy(rnd_state),
            enemy_positions=torch.from_numpy(enemy_pos),
            enemy_alive_mask=torch.from_numpy(enemy_alive),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TrainingSample:
        sample = self.samples[index]
        images = self._decode_video(sample)
        audio = self._decode_audio(sample)
        truth = self._get_truth(sample)
        return TrainingSample(
            images=images,
            audio=audio,
            truth=truth,
            _roundsample=sample,
        )


class DatasetRoot:
    dataset_path: str
    manifest: Dict[str, Any]
    train: List[Game]
    val: List[Game]
    store: "DatasetRoot.LmdbCache"
    config: DatasetConfig

    class LmdbCache:
        def __init__(self):
            self.opened: Dict[str, lmdb.Environment] = {}
            self.info_cache: Dict[str, Dict[str, Any]] = {}

        def open(self, lmdb_path: str) -> lmdb.Environment:
            if lmdb_path not in self.opened:
                self.opened[lmdb_path] = lmdb.open(
                    lmdb_path,
                    readonly=True,
                    lock=False,
                    max_readers=512,
                    readahead=True,
                )
            return self.opened[lmdb_path]

        def getinfo(self, game: Game) -> Dict[str, Any]:
            cache_key = (game.demo_name, game.lmdb_path)
            if cache_key in self.info_cache:
                return self.info_cache[cache_key]
            with self.open(game.lmdb_path).begin(write=False) as txn:
                blob = txn.get(f"{game.demo_name}_INFO".encode("utf-8"))
                if blob is None:
                    raise FileNotFoundError(f"Missing _INFO for {game.demo_name}")
                info = json.loads(blob.decode("utf-8"))
            self.info_cache[cache_key] = info
            return info

        def close_all(self) -> None:
            for env in self.opened.values():
                env.close()
            self.opened.clear()

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.store = self.LmdbCache()
        self.dataset_path = os.path.abspath(config.data_root)
        manifest_path = os.path.join(self.dataset_path, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.train = self.build_games("train")
        self.val = self.build_games("val")

    def build_games(self, split: str) -> List[Game]:
        games: List[Game] = []
        for name in self.manifest.get(split, []):
            games.append(
                Game(
                    demo_name=name,
                    lmdb_path=os.path.join(self.dataset_path, "lmdb", f"{name}.lmdb"),
                )
            )
        for game in games:
            self.build_rounds(game)
        return games

    @staticmethod
    def _parse_recording_filename(path: str) -> Dict[str, Any]:
        stem = Path(path).stem
        parts = stem.split("_")
        if len(parts) < 5:
            raise ValueError(f"Unexpected recording filename format: {path}")
        try:
            start_tick = int(parts[-2])
            stop_tick = int(parts[-1])
        except ValueError as exc:
            raise ValueError(f"Could not parse recording tick range from filename: {path}") from exc
        player_name = "_".join(parts[2:-2]) or "unknown_player"
        return {
            "player_name": player_name,
            "start_tick": start_tick,
            "stop_tick": stop_tick,
        }

    def build_rounds(self, game: Game):
        info = self.store.getinfo(game)
        for round_info in info["rounds"]:
            if len(round_info.get("pov_videos", [])) != 5 or len(round_info.get("pov_audio", [])) != 5:
                continue

            def _resolve(path: str) -> str:
                return os.path.abspath(os.path.join(self.dataset_path, "recordings", game.demo_name, path))

            videos = [_resolve(path) for path in round_info["pov_videos"]]
            audio = [_resolve(path) for path in round_info["pov_audio"]]
            if not all(os.path.exists(path) for path in videos + audio):
                if self.config.warn_skip:
                    logging.warning(
                        "Skipping %s/%s: missing media file.",
                        game.demo_name,
                        round_info["round_num"],
                    )
                continue

            pov_meta = [self._parse_recording_filename(path) for path in videos]
            game.rounds.append(
                Round(
                    game=game,
                    round_num=int(round_info["round_num"]),
                    team=str(round_info["team"]).upper(),
                    start_tick=int(round_info["start_tick"]),
                    end_tick=int(round_info["end_tick"]),
                    pov_video=videos,
                    pov_audio=audio,
                    pov_start_ticks=[meta["start_tick"] for meta in pov_meta],
                    pov_end_ticks=[meta["stop_tick"] for meta in pov_meta],
                    pov_player_names=[meta["player_name"] for meta in pov_meta],
                )
            )

    def build_dataset(self, split: str) -> Epoch:
        samples: List[RoundSample] = []
        games = self.train if split == "train" else self.val
        stride = self.config.sample_stride
        target_length = self.config.epoch_round_sample_length

        for game in games:
            for round_info in game.rounds:
                for player_idx in range(5):
                    player_start_tick = max(round_info.start_tick, int(round_info.pov_start_ticks[player_idx]))
                    player_end_tick = min(round_info.end_tick, int(round_info.pov_end_ticks[player_idx]))
                    player_frame_count = ticks_to_framecount(player_start_tick, player_end_tick)
                    if player_frame_count <= 0:
                        continue

                    sample_length = min(target_length, player_frame_count)
                    max_start_frame = max(0, player_frame_count - target_length)

                    for start_frame in range(0, max_start_frame + 1, stride):
                        samples.append(
                            RoundSample(
                                round=round_info,
                                player_idx=player_idx,
                                start_tick=player_start_tick + TICKS_PER_FRAME * start_frame,
                                start_frame=start_frame,
                                length_frames=sample_length,
                                player_start_tick=player_start_tick,
                                player_end_tick=player_end_tick,
                                player_name=round_info.pov_player_names[player_idx],
                            )
                        )

                    if (max_start_frame % stride) != 0 and player_frame_count >= target_length:
                        samples.append(
                            RoundSample(
                                round=round_info,
                                player_idx=player_idx,
                                start_tick=player_start_tick + TICKS_PER_FRAME * max_start_frame,
                                start_frame=max_start_frame,
                                length_frames=target_length,
                                player_start_tick=player_start_tick,
                                player_end_tick=player_end_tick,
                                player_name=round_info.pov_player_names[player_idx],
                            )
                        )
        # Close DatasetRoot LMDB handles before handing control to the Epoch
        # dataset, which opens its own readonly environments.
        self.store.close_all()
        return Epoch(self.config, samples)
