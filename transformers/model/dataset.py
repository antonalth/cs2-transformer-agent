#!/usr/bin/env python3
from __future__ import annotations

import os, json, random, logging, argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

import lmdb, msgpack, numpy as np, msgpack_numpy as mpnp

import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torchcodec.decoders import VideoDecoder, AudioDecoder  # pip install torchcodec
import torchaudio

TICK_RATE = 64
FRAME_RATE = 32
TICKS_PER_FRAME = TICK_RATE // FRAME_RATE

def ticks_to_framecount(start_tick: int, end_tick: int) -> int:
    if end_tick < start_tick: return 0
    return ((end_tick - start_tick) // TICKS_PER_FRAME) + 1

@dataclass
class DatasetConfig:
    data_root: str #root dir of dataset
    run_dir: str #directory for temp files, logs etc

    epoch_gen_random_seed: int = 42
    epoch_windows_per_round: int = 3 #how many random windows
    epoch_round_sample_length: int = 128 #number of frames per window
    epoch_video_decoding_device: str = "cpu"
    epoch_audio_decoding_device: str = "cpu"
    audio_sample_rate: float = 24000.0
    audio_n_fft: int = 1024
    audio_hop_length: int = audio_sample_rate // FRAME_RATE
    audio_mel_bins: int = 128

@dataclass
class GroundTruth:
    alive_mask: torch.BoolTensor      # [T, 5]
    stats: torch.FloatTensor          # [T, 5, 3]
    mouse_delta: torch.FloatTensor    # [T, 5, 2]
    position: torch.FloatTensor       # [T, 5, 3]
    keyboard_mask: torch.IntTensor    # [T, 5]
    eco_mask: torch.LongTensor        # [T, 5, 4]
    inventory_mask: torch.LongTensor  # [T, 5, 2]
    active_weapon_idx: torch.IntTensor# [T, 5]
    round_number: torch.IntTensor     # [T]
    round_state_mask: torch.ByteTensor# [T]
    enemy_positions: torch.FloatTensor# [T, 5, 3]

@dataclass
class TrainingSample:
    images: torch.Tensor
    audio: torch.Tensor
    truth: GroundTruth

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

@dataclass
class RoundSample:
    round: Round
    start_tick: int
    start_frame: int
    length_frames: int
    @property
    def end_frame(self) -> int:
        return self.start_frame + self.length_frames
    @property
    def start_time(self) -> float:
        return self.start_frame / FRAME_RATE
    @property
    def end_time(self) -> float:
        return self.end_frame / FRAME_RATE


class Epoch(torch.utils.data.Dataset):
    epoch_idx: int
    samples: List[RoundSample]
    config: DatasetConfig
    temp_dir: str
    lmdb_envs: dict[str, lmdb.Environment]
    video_decoders: dict[str, VideoDecoder]
    audio_decoders: dict[str, AudioDecoder]

    def __init__(self, config: DatasetConfig, epoch_idx: int, samples: List[RoundSample]):
        self.config = config
        self.epoch_idx = epoch_idx
        self.samples = samples
        self.lmdb_envs: dict[str, lmdb.Environment] = {}
        self.video_decoders: dict[str, VideoDecoder] = {}
        self.audio_decoders: dict[str, AudioDecoder] = {}
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.audio_sample_rate,
            n_fft=config.audio_n_fft,
            hop_length=config.audio_hop_length,
            n_mels=config.audio_mel_bins,
            center=False,
            pad_mode="reflect",
            power=2.0,
        )

    def _get_lmdb_env(self, lmdb_path: str) -> lmdb.Environment:
        env = self.lmdb_envs.get(lmdb_path)
        if env is None:
            env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=2048)
            self.lmdb_envs[lmdb_path] = env
        return env

    @staticmethod
    def _lmdb_key(demoname: str, round_num: int, team: int, tick: int) -> bytes:
        return f"{demoname}_round_{round_num:03d}_team_{team}_tick_{tick:08d}".encode("utf-8")
    
    @staticmethod
    def _bitmask_to_weapon_index(mask: np.ndarray) -> int:
        """Converts a [2] uint64 weapon bitmask to a single item index."""
        if mask.sum() == 0:
            return -1
        for i in range(128):
            if (mask[i // 64] >> np.uint64(i % 64)) & np.uint64(1):
                return i
        return -1

    #todo understand
    def pad_or_truncate_to(self, x: torch.Tensor, T: int, dim: int = 0) -> torch.Tensor:
        """
        Pad or truncate `x` along dimension `dim` to length `T` using zeros.

        - If size(dim) >= T: return x sliced along dim.
        - If size(dim) <  T: zero-pad at the end along dim.
        """
        t = x.size(dim)
        if t >= T:
            # build a slice that keeps everything except trim dim
            index = [slice(None)] * x.ndim
            index[dim] = slice(0, T)
            return x[tuple(index)]

        # need to pad
        pad_shape = list(x.shape)
        pad_shape[dim] = T - t
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=dim)

    def _get_video_decoder(self, path: str) -> VideoDecoder:
        dec = self.video_decoders.get(path)
        if dec is None:
            dec = VideoDecoder(path, device=self.config.epoch_video_decoding_device)
            self.video_decoders[path] = dec
        return dec
    
    def _get_audio_decoder(self, path: str) -> AudioDecoder:
        dec = self.audio_decoders.get(path)
        if dec is None:
            dec = AudioDecoder(path, device=self.config.epoch_audio_decoding_device)
            self.audio_decoders[path] = dec
        return dec
    
    def _decode_video(self, sample: RoundSample):
        pov_tensors = []

        for pov in sample.round.pov_video:
            decoder = self._get_video_decoder(pov)
            frames = decoder.get_frames_in_range(sample.start_frame, sample.start_frame + sample.length_frames).data
            pov_tensors.append(self.pad_or_truncate_to(frames, self.config.epoch_round_sample_length, dim=0))

        return torch.stack(pov_tensors, dim=0).permute(1, 0, 2, 3, 4)
    
    def _decode_audio(self, sample: RoundSample):
        pov_mels = []

        for pov in sample.round.pov_audio:
            decoder = self._get_audio_decoder(pov)
            waveform = decoder.get_samples_played_in_range(start_seconds=sample.start_time, stop_seconds=sample.end_time).data
            if waveform.numel() == 0:
                waveform = torch.zeros(2, int(self.config.audio_sample_rate / FRAME_RATE))
            mel = self.pad_or_truncate_to(self.mel_transform(waveform), self.config.epoch_round_sample_length, dim=-1)
            pov_mels.append(mel.permute(2, 0, 1).unsqueeze(-1))

        return torch.stack(pov_mels, dim=1)

    def _get_truth(self, sample: RoundSample) -> GroundTruth:
        r = sample.round
        g = r.game
        env = self._get_lmdb_env(g.lmdb_path)
        T = self.config.epoch_round_sample_length

        # Preallocate numpy arrays (cheap, NumPy is fine here)
        alive = np.zeros((T, 5), dtype=np.bool_)
        stats = np.zeros((T, 5, 3), dtype=np.float32)
        mouse = np.zeros((T, 5, 2), dtype=np.float32)
        pos   = np.zeros((T, 5, 3), dtype=np.float32)
        kbd   = np.zeros((T, 5),     dtype=np.uint32)
        eco   = np.zeros((T, 5, 4),  dtype=np.uint64)
        inv   = np.zeros((T, 5, 2),  dtype=np.uint64)
        wep   = np.full((T, 5), -1,  dtype=np.int32)
        rnd_num   = np.full((T,), r.round_num, dtype=np.int32)
        rnd_state = np.zeros((T,), dtype=np.uint8)
        enemy_pos = np.zeros((T, 5, 3), dtype=np.float32)

        ticks = sample.start_tick + (np.arange(T, dtype=np.int32) * TICKS_PER_FRAME)

        with env.begin(write=False) as txn:
            for f, tick in enumerate(ticks):
                blob = txn.get(self._lmdb_key(g.demo_name, r.round_num, r.team, int(tick)))
                if not blob:
                    continue

                payload = msgpack.unpackb(blob, raw=False, object_hook=mpnp.decode)
                gs = payload.get("game_state")
                if not gs:
                    continue
                gs = gs[0]

                # team_alive bitmask → alive slots
                alive_slots = [i for i in range(5) if (int(gs["team_alive"]) >> i) & 1]
                for slot in alive_slots:
                    alive[f, slot] = True

                rnd_state[f]   = gs["round_state"]
                enemy_pos[f]   = gs["enemy_pos"]

                pdl = payload.get("player_data")
                if pdl and len(alive_slots) == len(pdl):
                    for p_idx, p_data_arr in zip(alive_slots, pdl):
                        p = p_data_arr[0]
                        stats[f, p_idx] = [p["health"], p["armor"], p["money"]]
                        mouse[f, p_idx] = p["mouse"]
                        pos[f, p_idx]   = p["pos"]
                        kbd[f, p_idx]   = p["keyboard_bitmask"]
                        eco[f, p_idx]   = p["eco_bitmask"]
                        inv[f, p_idx]   = p["inventory_bitmask"]
                        wep[f, p_idx]   = self._bitmask_to_weapon_index(p["active_weapon_bitmask"])

        # Convert to torch tensors on CPU; move to device later in the model
        return GroundTruth(
            alive_mask        = torch.from_numpy(alive),                    # bool
            stats             = torch.from_numpy(stats),                    # float32
            mouse_delta       = torch.from_numpy(mouse),                    # float32
            position          = torch.from_numpy(pos),                      # float32
            keyboard_mask     = torch.from_numpy(kbd.astype(np.int32)),    # int32
            eco_mask          = torch.from_numpy(eco.astype(np.int64)),    # int64
            inventory_mask    = torch.from_numpy(inv.astype(np.int64)),    # int64
            active_weapon_idx = torch.from_numpy(wep),                      # int32
            round_number      = torch.from_numpy(rnd_num),                  # int32
            round_state_mask  = torch.from_numpy(rnd_state),                # uint8 → byte tensor
            enemy_positions   = torch.from_numpy(enemy_pos),                # float32
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TrainingSample:
        s = self.samples[index]
        images = self._decode_video(s)
        audio = self._decode_audio(s)
        gt = self._get_truth(s)
        return TrainingSample(images=images, audio=audio, truth=gt)

class DatasetRoot:
    dataset_path: str
    manifest: Dict[str, Any]
    train: List[Game]
    val: List[Game]
    store: LmdbCache
    config: DatasetConfig

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
    
    def __init__(self, config: DatasetConfig):
        self.store = self.LmdbCache()
        self.dataset_path = os.path.abspath(config.data_root)
        manifest_path = os.path.join(self.dataset_path, "manifest.json")
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
                          end_tick=int(r["end_tick"]), pov_video=videos, pov_audio=audio)
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

if __name__ == "__main__":
    # Simple smoke test for DatasetRoot / Epoch
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser("dataset.py smoke test")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the dataset (same as DatasetConfig.data_root)")
    parser.add_argument("--run_dir", type=str, default="./runs/smoke_test",
                        help="Run directory for DatasetConfig.run_dir")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="Which split to build an epoch for")
    parser.add_argument("--epoch_idx", type=int, default=0,
                        help="Epoch index used for window sampling")
    parser.add_argument("--num_samples", type=int, default=2,
                        help="How many samples to draw from the epoch")
    parser.add_argument("--frames_per_sample", type=int, default=16,
                        help="How many frames per sample to render into the video")
    parser.add_argument("--video_out", type=str, required=True,
                        help="Output path for the smoke-test video (e.g. /tmp/ds_smoke.mp4)")
    args = parser.parse_args()

    # Build config and dataset root
    cfg = DatasetConfig(
        data_root=args.data_root,
        run_dir=args.run_dir,
    )
    ds_root = DatasetRoot(cfg)
    epoch = ds_root.build_epoch(args.split, args.epoch_idx)

    if len(epoch) == 0:
        logging.error("Epoch is empty, nothing to smoke-test.")
        raise SystemExit(1)

    import cv2  # Only needed for the smoke test

    # Probe first sample for shape info
    first_sample = epoch[0]
    images = first_sample.images  # [T, 5, H, W, C]
    T, num_pov, H, W, C = images.shape
    frames_per_sample_default = min(args.frames_per_sample, T)

    # Prepare VideoWriter
    os.makedirs(os.path.dirname(args.video_out) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.video_out, fourcc, FRAME_RATE, (W * num_pov, H))
    if not writer.isOpened():
        logging.error(f"Could not open VideoWriter for {args.video_out}")
        raise SystemExit(1)

    logging.info(
        "Smoke test: writing up to %d samples, %d frames/sample to %s",
        args.num_samples,
        frames_per_sample_default,
        args.video_out,
    )

    def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
        """Convert [.., H, W, C] torch tensor to uint8 numpy array."""
        arr = x.detach().cpu().numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return arr

    num_samples = min(args.num_samples, len(epoch))
    for s_idx in range(num_samples):
        sample = epoch[s_idx]
        images = tensor_to_uint8(sample.images)  # [T, 5, H, W, C]
        gt = sample.truth

        frames_this = min(args.frames_per_sample, images.shape[0])

        for t in range(frames_this):
            # Take frame t from all POVs → [5, H, W, C]
            frame_stack = images[t]

            # Tile 5 POVs horizontally → [H, W*5, C]
            row = np.concatenate([frame_stack[p] for p in range(num_pov)], axis=1)
            frame_bgr = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)

            # Overlay tiny text for each POV
            for pov in range(num_pov):
                x0 = pov * W + 4
                y0 = 14  # small offset from top

                alive = bool(gt.alive_mask[t, pov].item())
                hp = float(gt.stats[t, pov, 0].item())
                weapon = int(gt.active_weapon_idx[t, pov].item())
                rnd = int(gt.round_number[t].item())
                state = int(gt.round_state_mask[t].item())

                txt1 = f"P{pov} {'A' if alive else 'D'} hp={int(hp)} w={weapon}"
                txt2 = f"r={rnd} st={state}"

                color = (0, 255, 0) if alive else (0, 0, 255)

                cv2.putText(
                    frame_bgr,
                    txt1,
                    (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame_bgr,
                    txt2,
                    (x0, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            writer.write(frame_bgr)

    writer.release()
    logging.info("Smoke test complete, video written to %s", args.video_out)


__all__ = [
    "DatasetConfig"
    "DatasetRoot",
    "TrainingSample",
    "GroundTruth",
    "Epoch",
]