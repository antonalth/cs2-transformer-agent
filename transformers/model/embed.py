#!/usr/bin/env python3
"""
embed.py — Offline Feature Precomputation

Precomputes and caches training-ready features for all recordings to enable
instantaneous data loading during training. This script processes raw video and
audio files in parallel across multiple GPUs.

- Video Features: Extracts raw, projection-less features from a DINOv3
  backbone for each frame of a video.
- Audio Features: Generates stereo log-mel spectrograms from raw .wav files.

All outputs are aligned to a common 32 fps time base and stored as memory-
mappable NumPy (.npy) files, mirroring the input directory structure. The
process is idempotent and can be resumed, skipping already-completed files.
"""
from __future__ import annotations

import os
import argparse
import logging
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm

# Reuse components from training/model scripts
from train2 import DaliConfig, FPS
from model import CS2Config, DINOv3VisualEncoder

# DALI for video decoding
try:
    from nvidia.dali import pipeline_def, fn, types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False

# torchaudio for audio processing
try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    
# decord for fast video metadata
try:
    import decord
    decord.bridge.set_bridge('torch')
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

# --- Constants & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][Rank %(rank)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# --- Helper Classes for Feature Extraction ---

class RawDINOv3Extractor(nn.Module):
    """
    A wrapper around the DINOv3VisualEncoder from model.py to extract only the
    raw, pre-projection features (e.g., CLS token) from the ViT backbone.
    """
    def __init__(self, cfg: CS2Config):
        super().__init__()
        # Instantiate the full encoder to leverage its weight loading,
        # normalization constants, and processor.
        self.encoder = DINOv3VisualEncoder(cfg)
        
        # We only need the backbone and its internal methods/buffers.
        self.backbone = self.encoder.backbone
        self.processor = self.encoder.processor
        self.compute_dtype = self.encoder.compute_dtype
        self.use_channels_last = self.encoder.use_channels_last

        # The DINOv3VisualEncoder already freezes the backbone, so we just
        # ensure the entire module is in eval mode.
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Processes a batch of raw image tensors and returns raw backbone features.
        
        Args:
            images: Tensor of shape [N, 3, H, W], uint8 or float.
        
        Returns:
            Tensor of shape [N, D_raw] where D_raw is the backbone's hidden dim.
        """
        # This forward pass is a simplified version of DINOv3VisualEncoder.forward,
        # stopping before the final projection layer.
        device = images.device
        from_uint8 = images.dtype == torch.uint8
        
        if self.processor:
            # The HF processor expects a list of PIL/numpy images or a tensor
            # on the CPU. Move to CPU for processing.
            processed = self.processor(images=images.cpu(), return_tensors="pt")
            pixel_values = processed['pixel_values'].to(device)
            if self.use_channels_last:
                pixel_values = pixel_values.to(memory_format=torch.channels_last)
        else:
            pixel_values = self.encoder._normalize_chunk(images, from_uint8=from_uint8)
        
        # Run backbone under autocast for efficiency.
        with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
            outputs = self.backbone(pixel_values=pixel_values)
            # The raw feature is the output of the pooling layer (CLS token).
            features = outputs.pooler_output
        
        return features.float() # Return as float32 for numpy saving


class AudioEmbedder(nn.Module):
    """
    Generates stereo log-mel spectrograms from .wav files, matching the
    parameters used in the DALI training pipeline.
    """
    def __init__(self, cfg: DaliConfig, device: torch.device):
        super().__init__()
        self.target_sr = cfg.sample_rate
        self.device = device
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            center=False,
            power=2.0
        ).to(device)

        self.db_transform = T.AmplitudeToDB(
            stype='power', top_db=cfg.db_cutoff
        ).to(device)

    @torch.no_grad()
    def forward(self, wav_path: str) -> torch.Tensor:
        """
        Loads, processes, and converts a wav file to a stereo log-mel tensor.
        
        Returns:
            Tensor of shape [T, n_mels, 2].
        """
        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.to(self.device)

        if sr != self.target_sr:
            resampler = T.Resample(sr, int(self.target_sr), dtype=waveform.dtype).to(self.device)
            waveform = resampler(waveform)

        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] != 2:
            raise ValueError(f"Expected mono or stereo audio, found {waveform.shape[0]} channels in {wav_path}")

        # Process each channel separately
        mel_specs = [self.mel_transform(waveform[i]) for i in range(2)]
        db_specs = [self.db_transform(mel) for mel in mel_specs]

        # Transpose from [n_mels, T] to [T, n_mels] and stack
        stereo_mel = torch.stack([db.transpose(0, 1) for db in db_specs], dim=-1)
        
        return stereo_mel


# --- DALI Pipeline for Full Video Decoding ---

def get_dali_pipeline(video_file: str, batch_size: int, num_threads: int, device_id: int):
    """
    Creates a DALI pipeline to decode an entire video file at 32 FPS.
    """
    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    def video_pipe():
        # DALI's video reader is fast but requires a file list.
        video_frames = fn.readers.video(
            device="gpu",
            filenames=[video_file],
            sequence_length=1, # Process one frame at a time to handle variable length videos easily
            step=1, # No frame skipping
            initial_fill=batch_size, # Buffer size
            normalized=False,
            image_type=types.RGB,
            dtype=types.UINT8,
            file_list_frame_num=False
        )
        return video_frames
    
    pipe = video_pipe()
    pipe.build()
    return pipe

# --- Core Processing Functions ---

def process_video(
    job: Dict,
    extractor: RawDINOv3Extractor,
    dali_batch_size: int,
    dali_threads: int,
    device_id: int,
) -> Tuple[int, int]:
    """
    Extracts features for a single video file using DALI and DINOv3.
    """
    video_in = job['video_in']
    video_out = Path(job['video_out'])
    target_T = job['target_T']
    
    if not os.path.exists(video_in):
        log.error(f"Input video not found: {video_in}")
        return 0, 0

    video_out.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup DALI pipeline for this specific video
    dali_pipe = get_dali_pipeline(video_in, dali_batch_size, dali_threads, device_id)
    
    # Use DALIGenericIterator to iterate through all frames
    dali_iter = DALIGenericIterator(
        [dali_pipe],
        output_map=["frames"],
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        dynamic_shape=True
    )
    
    all_features = []
    try:
        for batch in dali_iter:
            frames = batch[0]['frames'].squeeze(1) # DALI adds a time dimension
            
            # DALI gives [N, H, W, C], model expects [N, C, H, W]
            frames_chw = frames.permute(0, 3, 1, 2)
            features = extractor(frames_chw)
            all_features.append(features.cpu())
            
        if not all_features:
            log.warning(f"No features extracted from {video_in}")
            return 0, 0
            
        features_np = torch.cat(all_features, dim=0).numpy()
        
        # Trim to target length for alignment
        if features_np.shape[0] > target_T:
            features_np = features_np[:target_T]

        # Atomic write
        with tempfile.NamedTemporaryFile(delete=False, dir=video_out.parent, suffix=".tmp") as tmp_file:
            np.save(tmp_file, features_np)
            tmp_path = tmp_file.name
        os.replace(tmp_path, video_out)
        
        return features_np.shape[0], features_np.shape[1]
    
    finally:
        # It's good practice to clean up the iterator and its resources
        del dali_iter
        del dali_pipe


def process_audio(job: Dict, embedder: AudioEmbedder) -> int:
    """
    Extracts features for a single audio file.
    """
    audio_in = job['audio_in']
    audio_out = Path(job['audio_out'])
    target_T = job['target_T']
    
    if not os.path.exists(audio_in):
        log.error(f"Input audio not found: {audio_in}")
        return 0
        
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    
    features = embedder(audio_in)
    features_np = features.cpu().numpy()
    
    # Trim to target length
    if features_np.shape[0] > target_T:
        features_np = features_np[:target_T]

    # Atomic write
    with tempfile.NamedTemporaryFile(delete=False, dir=audio_out.parent, suffix=".tmp") as tmp_file:
        np.save(tmp_file, features_np)
        tmp_path = tmp_file.name
    os.replace(tmp_path, audio_out)
    
    return features_np.shape[0]

# --- DDP and Job Management ---

def get_media_duration(path: str) -> float:
    """Returns the duration of a media file in seconds."""
    try:
        if path.endswith('.wav'):
            info = torchaudio.info(path)
            return info.num_frames / info.sample_rate
        elif path.endswith('.mp4'):
            if not DECORD_AVAILABLE: return 0.0
            vr = decord.VideoReader(path, ctx=decord.cpu(0))
            return len(vr) / vr.get_avg_fps()
    except Exception as e:
        log.warning(f"Could not get duration for {path}: {e}")
    return 0.0


def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Patch logger to include rank
    global log
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)
    return rank, world_size, local_rank


def main(args):
    """Main script entry point."""
    if not DALI_AVAILABLE:
        raise RuntimeError("NVIDIA DALI is required for video processing.")
    if not TORCHAUDIO_AVAILABLE:
        raise RuntimeError("torchaudio is required for audio processing.")
    if not DECORD_AVAILABLE and args.mode in ['all', 'video']:
        raise RuntimeError("decord is required for video metadata.")

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    jobs = []
    if rank == 0:
        log.info("Rank 0: Discovering and preparing jobs...")
        data_root = Path(args.data_root)
        rec_root = data_root / "recordings"
        vid_root = data_root / "vit_embed"
        aud_root = data_root / "aud_embed"

        all_stems = set()
        for root, _, files in os.walk(rec_root):
            for f in files:
                if f.endswith(('.mp4', '.wav')):
                    p = Path(root) / f
                    stem = p.relative_to(rec_root).with_suffix('')
                    all_stems.add(str(stem))

        for stem_str in tqdm(sorted(list(all_stems)), desc="Scanning files"):
            stem = Path(stem_str)
            video_in = rec_root / stem.with_suffix(".mp4")
            audio_in = rec_root / stem.with_suffix(".wav")
            video_out = vid_root / stem.with_suffix(".npy")
            audio_out = aud_root / stem.with_suffix(".npy")
            
            job = {"stem": stem_str}
            
            do_video = args.mode in ['all', 'video'] and video_in.exists()
            do_audio = args.mode in ['all', 'audio'] and audio_in.exists()

            if not (do_video or do_audio):
                continue
            
            # Skip if outputs exist and not overwriting
            if not args.overwrite:
                vid_exists = do_video and video_out.exists()
                aud_exists = do_audio and audio_out.exists()
                if (not do_video or vid_exists) and (not do_audio or aud_exists):
                    continue

            # Calculate target frame count for alignment
            video_dur = get_media_duration(str(video_in)) if do_video else float('inf')
            audio_dur = get_media_duration(str(audio_in)) if do_audio else float('inf')
            min_dur = min(video_dur, audio_dur)

            if min_dur == 0 or min_dur == float('inf'):
                log.warning(f"Skipping {stem_str} due to zero or invalid duration.")
                continue
                
            job['target_T'] = int(min_dur * FPS)
            if do_video:
                job['video_in'] = str(video_in)
                job['video_out'] = str(video_out)
            if do_audio:
                job['audio_in'] = str(audio_in)
                job['audio_out'] = str(audio_out)
                
            jobs.append(job)
        
        log.info(f"Found {len(jobs)} jobs to process.")

    # Broadcast job list from rank 0 to all other ranks
    dist.barrier()
    # Use a list container so the object can be modified by broadcast_object_list
    job_container = [jobs] 
    dist.broadcast_object_list(job_container, src=0)
    if rank != 0:
        jobs = job_container[0]

    # Each rank processes its slice of the jobs
    my_jobs = jobs[rank::world_size]
    
    # Initialize models
    video_extractor = None
    if args.mode in ['all', 'video']:
        model_cfg = CS2Config() # Use defaults from model.py
        video_extractor = RawDINOv3Extractor(model_cfg).to(device)

    audio_embedder = None
    if args.mode in ['all', 'audio']:
        dali_cfg = DaliConfig() # Use defaults from train2.py
        audio_embedder = AudioEmbedder(dali_cfg, device)

    stats = {"video_ok": 0, "video_fail": 0, "audio_ok": 0, "audio_fail": 0}
    
    progress_bar = tqdm(my_jobs, desc=f"Rank {rank} Processing", position=rank)
    for job in progress_bar:
        # --- Process Video ---
        if 'video_in' in job and video_extractor:
            try:
                start_t = time.time()
                t, d = process_video(
                    job, video_extractor, args.dali_batch_size, args.dali_threads, local_rank
                )
                dt = time.time() - start_t
                if t > 0:
                    log.info(f"VIDEO OK: {job['stem']} -> [T={t}, D={d}] in {dt:.2f}s")
                    stats['video_ok'] += 1
                else:
                    stats['video_fail'] += 1
            except Exception as e:
                log.exception(f"VIDEO FAIL: {job['stem']} with error: {e}")
                stats['video_fail'] += 1
        
        # --- Process Audio ---
        if 'audio_in' in job and audio_embedder:
            try:
                start_t = time.time()
                t = process_audio(job, audio_embedder)
                dt = time.time() - start_t
                if t > 0:
                    log.info(f"AUDIO OK: {job['stem']} -> [T={t}] in {dt:.2f}s")
                    stats['audio_ok'] += 1
                else:
                    stats['audio_fail'] += 1
            except Exception as e:
                log.exception(f"AUDIO FAIL: {job['stem']} with error: {e}")
                stats['audio_fail'] += 1
    
    # Synchronize and print summary
    dist.barrier()
    
    stats_tensor = torch.tensor(
        [stats['video_ok'], stats['video_fail'], stats['audio_ok'], stats['audio_fail']],
        dtype=torch.int64,
        device=device
    )
    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        total_stats = stats_tensor.cpu().tolist()
        log.info("=" * 50)
        log.info("               PROCESSING SUMMARY")
        log.info("=" * 50)
        log.info(f"Total initial jobs: {len(jobs)}")
        log.info(f"Video Succeeded: {total_stats[0]}")
        log.info(f"Video Failed:    {total_stats[1]}")
        log.info(f"Audio Succeeded: {total_stats[2]}")
        log.info(f"Audio Failed:    {total_stats[3]}")
        log.info("=" * 50)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute video and audio embeddings.")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--mode", choices=['all', 'video', 'audio'], default='all', help="Which modalities to process.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing feature files.")
    parser.add_argument("--dali-threads", type=int, default=4, help="Number of threads for DALI video decoder.")
    parser.add_argument("--dali-batch-size", type=int, default=16, help="Batch size for DALI video decoding and ViT processing.")
    
    cli_args = parser.parse_args()
    main(cli_args)