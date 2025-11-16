#!/usr/bin/env python3
from __future__ import annotations

import os, json, random, logging, argparse
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import lmdb, msgpack, numpy as np, msgpack_numpy as mpnp
import contextlib

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

#from nvidia.dali import fn, types, pipeline_def
#from nvidia.dali.plugin.pytorch import DALIGenericIterator
#from nvidia.dali.plugin.base_iterator import LastBatchPolicy

from model import CS2Transformer, CS2Config

TICK_RATE = 64
FRAME_RATE = 32
TICKS_PER_FRAME = TICK_RATE // FRAME_RATE

def ticks_to_framecount(start_tick: int, end_tick: int) -> int:
    if end_tick < start_tick: return 0
    return ((end_tick - start_tick) // TICKS_PER_FRAME) + 1

@dataclass
class TrainConfig:
    data_root: str #root dir of dataset
    run_dir: str #directory for temp files, logs etc

    epoch_gen_random_seed: int = 42
    epoch_windows_per_round: int = 3 #how many random windows
    epoch_round_sample_length: int = 128 #number of frames per window

@dataclass
class Game:
    demo_name: str
    lmdb_path: str
    rounds: List[Round] = field(default_factory=list)

@dataclass
class Round:
    game: Game
    round_num: str
    team: str
    start_tick: int
    end_tick: int
    pov_video: List[str]
    pov_audio: List[str]
    @property
    def frame_count(self) -> int:
        return ticks_to_framecount(self.start_tick, self.end_tick)
    
class RoundSample:
    round: Round
    start_tick: int
    start_frame: int
    length_frames: int

class Epoch(torch.utils.data.Dataset):
    epoch_idx: int
    samples: List[RoundSample]
    config: TrainConfig
    temp_dir: str
    lmdb_envs: dict[str, lmdb.Environment]

    def __init__(self, config: TrainConfig, epoch_idx: int, samples: List[RoundSample]):
        self.config = config
        self.epoch_idx = epoch_idx
        self.samples = samples
        self.lmdb_envs = {}
        #self.temp_dir = os.path.abspath(config.run_dir) / f"epoch{epoch_idx}"
        #os.makedirs(self.temp_dir, exist_ok=True)
        #self.vid_paths = [os.path.join(self.temp_dir, f"pov{k}_video.txt") for k in range(5)]
        #self.aud_paths = [os.path.join(self.temp_dir, f"pov{k}_audio.txt") for k in range(5)]
        #self.write_files() 
    
    #might not even be needed, since testing non-dali
    def write_files(self):
        with contextlib.ExitStack() as stack:
            files = [stack.enter_context(open(p, "w")) for p in self.vid_paths + self.aud_paths]
            vid_fs, aud_fs = files[:5], files[5:]

            for id, s in enumerate(self.samples):
                for k in range(5):
                    vid_fs[k].write(f"{s.round.pov_video[k]} {id} {s.start_frame} {s.length_frames}\n")
                    aud_fs[k].write(f"{s.round.pov_audio[k]} {s.start_frame}\n")
        return
    
    def _get_lmdb_env(self, lmdb_path: str):
        env = self.lmdb_envs.get(lmdb_path)
        if env is None:
            env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=2048)
            self.lmdb_envs[lmdb_path] = env
        return env

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        
class DatasetRoot:
    dataset_path: str
    manifest: Dict[str, Any]
    train: List[Game]
    val: List[Game]
    store: LmdbCache
    config: TrainConfig

    class LmdbCache:
        def __init__(self):
            self.opened: Dict[str, lmdb.Environment] = {}
            self.info_cache: Dict[str, Dict[str, Any]] = {}

        def open(self, lmdb_path: str) -> lmdb.Environment:
            if lmdb_path not in self.opened:
                self.opened[lmdb_path] = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=512, readahead=True)
            return self.opened[lmdb_path]
        
        def getinfo(self, game: Game) -> Dict[str, Any]:
            cache_key = (game.demo_name, game.lmdb_path)
            if cache_key in self.info_cache: return self.info_cache[cache_key]
            with self.open(game.lmdb_path).begin(write=False) as txn:
                blob = txn.get(f"{game.demo_name}_INFO".encode("utf-8"))
                if blob is None: raise FileNotFoundError(f"Missing _INFO for {game.demo_name}")
                info = json.loads(blob.decode("utf-8"))
            self.info_cache[cache_key] = info
            return info
    
    def __init__(self, config: TrainConfig):
        self.store = self.LmdbCache()
        self.dataset_path = os.path.abspath(config.data_root)
        manifest_path = self.dataset_path / "mainfest.json"
        with open(manifest_path, "r", encoding="utf-8") as f: self.manifest = json.load(f)
        self.train = self.build_games("train")
        self.val = self.build_games("val")
        self.config = config
        
    def build_games(self, split: str) -> List[Game]:
        games: List[Game] = []
        for name in self.manifest.get(split, []):
            games.append(Game(demo_name=name, 
                            lmdb_path=os.path.join(self.dataset_path, "lmdb", f"{name}.lmdb")))
        for g in games: self.build_rounds(g)
        return games
    
    def build_rounds(self, game: Game):
        info = self.store.getinfo(game)
        for r in info["rounds"]:
            if len(r.get("pov_videos", [])) != 5 or len(r.get("pov_audio", [])) != 5: continue
            def _resolve(p): return os.path.abspath(os.path.join(self.dataset_path, "recordings", game.demo_name, p))
            videos, audio = [_resolve(pv) for pv in r["pov_videos"]], [_resolve(pa) for pa in r["pov_audio"]]
            if not all(os.path.exists(p) for p in videos + audio):
                logging.warning(f"Skipping {game.demo_name}/{r['round_num']}: missing media file.")
                continue
            round = Round(game=game, round_num=int(r["round_num"]),
                          team=str(r["team"]).upper(), start_tick=int(r["start_tick"]),
                          end_tick=int(r["end_tick"]), pov_videos=videos, pov_audio=audio)
            game.rounds.append(round)

    def build_epoch(self, split: str, epoch_idx: int) -> Epoch:
        rnd = random.Random(self.config.epoch_gen_random_seed + epoch_idx)
        samples: List[RoundSample] = []
        games = self.train if split == "train" else self.val
        for g in games:
            for r in g.rounds:
                for _ in range(self.config.epoch_windows_per_round):
                    start_f = rnd.randint(0, max(0, r.frame_count - self.config.epoch_round_sample_length))
                    length_frames = self.config.epoch_round_sample_length if r.frame_count >= self.config.epoch_round_sample_length else r.frame_count
                    samples.append(
                        RoundSample(
                            round = r,
                            start_tick = r.start_tick + TICKS_PER_FRAME * start_f,
                            start_frame = start_f,
                            length_frames = length_frames
                        )
                    )
        return Epoch(self.config, epoch_idx, samples)