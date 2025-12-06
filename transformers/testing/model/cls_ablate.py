#!/usr/bin/env python3
"""
ablation_test_weapon_only.py - A focused script to test if ViT patch
embeddings, processed by a mixer, contain enough information to predict the
ACTIVE WEAPON.

This version drastically simplifies the task from the previous ablation test.
By removing all other prediction heads (health, money, round number), we can
isolate the weapon prediction task. This provides the clearest possible signal
on whether the visual representation from the ViT is sufficient for this
fundamental aspect of game state.
"""

import argparse
import logging
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from tqdm import tqdm

# --- DALI Imports ---
try:
    from nvidia.dali import pipeline_def, fn, types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False

# --- Project Imports ---
try:
    from train3 import (
        Manifest, LmdbStore, build_team_rounds, EpochIndex,
        LmdbMetaFetcher, CS2Config
    )
except ImportError:
    print("Error: Could not import from train3.py.")
    exit(1)

# --- Utility functions borrowed from train3.py for robustness ---
TICK_RATE = 64
FPS = 32
TICKS_PER_FRAME = TICK_RATE // FPS
TICK_RE = re.compile(r"_([0-9]+)_([0-9]+)\.(mp4|wav)$")

def ticks_from_filename(path: str) -> tuple[int, int] | None:
    m = TICK_RE.search(os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else None

def ticks_to_frames(start_tick: int, end_tick: int) -> int:
    if end_tick < start_tick: return 0
    return ((end_tick - start_tick) // TICKS_PER_FRAME) + 1


# --- Model Components ---
class DINOv3Encoder(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m"):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError:
            raise RuntimeError("Please install transformers: `pip install transformers`")
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(pixel_values=pixel_values).last_hidden_state

class CrossAttentionMixer(nn.Module):
    def __init__(self, embed_dim: int, num_queries: int = 4, num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.query_vectors = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        B = patch_embeddings.shape[0]
        attn_output, _ = self.attention(
            query=self.query_vectors.expand(B, -1, -1),
            key=patch_embeddings,
            value=patch_embeddings,
        )
        return self.norm(self.query_vectors + attn_output)

# --- START OF CHANGE ---
class SimplePredictionHead(nn.Module):
    """A small MLP to predict ONLY the active weapon."""
    def __init__(self, input_dim: int, num_weapons: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
        )
        self.weapon_head = nn.Linear(256, num_weapons)

    def forward(self, x):
        features = self.feature_extractor(x)
        weapon_logits = self.weapon_head(features)
        return weapon_logits

class AblationModel(nn.Module):
    def __init__(self, cfg: CS2Config):
        super().__init__()
        self.vision_backbone = DINOv3Encoder()
        from transformers import AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean)
        self.register_buffer("img_std", std)
        
        embed_dim = self.vision_backbone.hidden_size
        self.mixer = CrossAttentionMixer(embed_dim=embed_dim)
        
        head_input_dim = self.mixer.num_queries * embed_dim
        self.head = SimplePredictionHead(
            input_dim=head_input_dim,
            num_weapons=cfg.weapon_dim
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32) / 255.0
        x = (x - self.img_mean) / self.img_std
        return x
        
    def forward(self, x: torch.Tensor):
        x_norm = self._normalize(x)
        patch_embeddings = self.vision_backbone(x_norm)
        mixed_vectors = self.mixer(patch_embeddings)
        final_vector = torch.flatten(mixed_vectors, start_dim=1)
        return self.head(final_vector)
# --- END OF CHANGE ---


# --- DALI Data Loading Pipeline ---
@pipeline_def
def create_dali_pipeline(file_list_path, shard_id, num_shards):
    video, labels = fn.readers.video(
        name="Reader",
        device="gpu",
        file_list=file_list_path,
        sequence_length=1, step=1, initial_fill=1024,
        normalized=False, image_type=types.RGB, dtype=types.UINT8,
        pad_last_batch=True, file_list_frame_num=True,
        file_list_include_preceding_frame=True,
        shard_id=shard_id, num_shards=num_shards,
        stick_to_shard=True, random_shuffle=True,
    )
    squeezed_video = fn.squeeze(video, axes=0)
    transposed_video = fn.transpose(squeezed_video, perm=[2, 0, 1])
    squeezed_labels = fn.squeeze(labels, axes=0)
    return transposed_video, squeezed_labels

class DaliDataloader(IterableDataset):
    def __init__(self, records, T_frames, batch_size, num_threads, device_id, split):
        if not DALI_AVAILABLE:
            raise ImportError("NVIDIA DALI is not installed or available.")
        self.batch_size, self.num_records, self.split = batch_size, len(records), split
        
        file_list_dir = os.path.join("tmp_dali_filelists", split)
        os.makedirs(file_list_dir, exist_ok=True)
        self.file_list_path = os.path.join(file_list_dir, "video_frames.txt")
        
        with open(self.file_list_path, "w") as f:
            for rec in records:
                video_path = rec.pov_videos[0]
                ticks = ticks_from_filename(video_path)
                if not ticks: continue
                total_frames = ticks_to_frames(ticks[0], ticks[1])
                if total_frames <= 0: continue
                target_idx = rec.start_f + T_frames // 2
                clamped_idx = max(0, min(target_idx, total_frames - 1))
                f.write(f"{video_path} {rec.sample_id} {clamped_idx}\n")

        self.pipeline = create_dali_pipeline(
            file_list_path=self.file_list_path,
            shard_id=0, num_shards=1,
            batch_size=batch_size, num_threads=num_threads, device_id=device_id
        )
        self.iterator = DALIGenericIterator(
            [self.pipeline], ['frames', 'sample_ids'], reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL if split == "val" else LastBatchPolicy.DROP
        )
    def __iter__(self): return self.iterator
    def __len__(self): return int(np.ceil(self.num_records / self.batch_size))

# --- Main Logic ---
def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info("Indexing dataset from manifest...")
    manifest = Manifest(args.data_root, args.manifest)
    store = LmdbStore()
    train_team_rounds = build_team_rounds(args.data_root, manifest.get_games("train"), store)
    val_team_rounds = build_team_rounds(args.data_root, manifest.get_games("val"), store)
    
    index = EpochIndex(T_frames=64, seed=42, windows_per_round=1)
    train_records, train_id_map = index.build(train_team_rounds, epoch=0)
    val_records, val_id_map = index.build(val_team_rounds, epoch=0)
    logging.info(f"Found {len(train_records)} training samples and {len(val_records)} validation samples.")
    
    train_loader = DaliDataloader(train_records, 64, args.batch_size, args.num_workers, 0, "train")
    val_loader = DaliDataloader(val_records, 64, args.batch_size, args.num_workers, 0, "val")
    
    fetcher = LmdbMetaFetcher(store)
    cfg = CS2Config()
    model = AblationModel(cfg).to(device)
    
    # --- START OF CHANGE ---
    # Loss is now only Cross-Entropy for the weapon head.
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    # --- END OF CHANGE ---
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    logging.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch in train_pbar:
            frames = batch[0]['frames'] 
            sample_ids = batch[0]['sample_ids'].cpu().numpy().flatten()
            
            # --- START OF CHANGE ---
            # Fetch only the weapon ground truth
            weapon_list = []
            for sid in sample_ids:
                rec = train_id_map[sid]
                gt = fetcher.fetch(rec)
                gt_frame_idx = min(64 // 2, gt.stats.shape[0] - 1)
                weapon_list.append(gt.active_weapon_idx[gt_frame_idx, 0])
            weapon_gt = torch.tensor(weapon_list, dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            pred_weapon_logits = model(frames)
            loss = loss_fn(pred_weapon_logits, weapon_gt)
            # --- END OF CHANGE ---

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_loader.iterator.reset()

        model.eval()
        total_val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                frames = batch[0]['frames']
                sample_ids = batch[0]['sample_ids'].cpu().numpy().flatten()

                # --- START OF CHANGE ---
                # Fetch only the weapon ground truth for validation
                weapon_list = []
                for sid in sample_ids:
                    rec = val_id_map[sid]
                    gt = fetcher.fetch(rec)
                    gt_frame_idx = min(64 // 2, gt.stats.shape[0] - 1)
                    weapon_list.append(gt.active_weapon_idx[gt_frame_idx, 0])
                weapon_gt = torch.tensor(weapon_list, dtype=torch.long, device=device)
                
                pred_weapon_logits = model(frames)
                loss = loss_fn(pred_weapon_logits, weapon_gt)
                # --- END OF CHANGE ---

                total_val_loss += loss.item()
                val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_loader.iterator.reset()
        
        logging.info(
            f"Epoch {epoch+1}/{args.epochs} -> "
            f"Avg Train Loss: {avg_train_loss:.4f}, "
            f"Avg Val Loss: {avg_val_loss:.4f}"
        )

    logging.info("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation test for ViT patch embeddings with a cross-attention mixer.")
    parser.add_argument("--data-root", type=str, default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4, help="Number of threads for DALI pipeline")
    args = parser.parse_args()
    
    if not DALI_AVAILABLE:
        logging.error("NVIDIA DALI is required for this script but not installed. Please install it.")
        exit(1)
        
    main(args)
