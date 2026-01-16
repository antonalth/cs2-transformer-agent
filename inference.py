#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import torch
import numpy as np
from accelerate import Accelerator
from tqdm import tqdm

# Add local modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), "transformers/model"))

from dataset import DatasetConfig, DatasetRoot, GroundTruth, TrainingSample
from model_novibe import ModelConfig, GamePredictorBackbone, ModelPrediction
from model_loss import CS2Loss
from visualize import visualize_frame
from config import GlobalConfig

def tensor_to_uint8(img_tensor):
    # img_tensor: [C, H, W] float or int
    # returns: [H, W, C] uint8 numpy
    if img_tensor.dtype == torch.uint8:
        x = img_tensor.float() / 255.0
    else:
        x = img_tensor
        
    x = x.permute(1, 2, 0).cpu().numpy()
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    # RGB to BGR for OpenCV
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x

class PredictionDecoder:
    def __init__(self, cfg: ModelConfig, device):
        self.cfg = cfg
        # Reconstruct bin centers for mouse
        # Same logic as CS2Loss
        self.mouse_centers = CS2Loss._generate_mu_law_bins(
            cfg.mouse_bins_count, -90.0, 90.0, 255.0
        ).to(device)
        
        # Position bins (Linear)
        self.pos_bins_x = torch.linspace(CS2Loss.MAP_MIN[0], CS2Loss.MAP_MAX[0], CS2Loss.BINS_X, device=device)
        self.pos_bins_y = torch.linspace(CS2Loss.MAP_MIN[1], CS2Loss.MAP_MAX[1], CS2Loss.BINS_Y, device=device)
        self.pos_bins_z = torch.linspace(CS2Loss.MAP_MIN[2], CS2Loss.MAP_MAX[2], CS2Loss.BINS_Z, device=device)

    def decode(self, preds: ModelPrediction) -> GroundTruth:
        # preds: Batched [1, T, ...] or [T, ...]
        # We assume batch size 1, time T.
        
        # 1. Mouse: Argmax -> Bin Center
        mx_idx = torch.argmax(preds.mouse_x, dim=-1)
        my_idx = torch.argmax(preds.mouse_y, dim=-1)
        mouse_x = self.mouse_centers[mx_idx]
        mouse_y = self.mouse_centers[my_idx]
        mouse_delta = torch.stack([mouse_x, mouse_y], dim=-1) # [T, 5, 2]

        # 2. Keyboard: Sigmoid > 0.5 -> Bitmask
        kb_probs = torch.sigmoid(preds.keyboard_logits)
        kb_bits = (kb_probs > 0.5).long()
        # Convert bits to int mask
        # kb_bits: [T, 5, 32]
        # We need to sum: sum(bit * 2^i)
        powers = 2 ** torch.arange(32, device=preds.keyboard_logits.device)
        keyboard_mask = (kb_bits * powers).sum(dim=-1).int()

        # 3. Stats: Inverse Norm
        # HP/Armor: Sigmoid -> [0, 1] -> * 100
        # Money: Sigmoid -> [0, 1] -> expm1(x * 9.68)
        s_pred = torch.sigmoid(preds.stats_logits)
        hp = s_pred[..., 0] * 100.0
        armor = s_pred[..., 1] * 100.0
        money = torch.expm1(s_pred[..., 2] * 9.68)
        stats = torch.stack([hp, armor, money], dim=-1)

        # 4. Position: Argmax -> Bin Value
        px_idx = torch.argmax(preds.player_pos_x, dim=-1)
        py_idx = torch.argmax(preds.player_pos_y, dim=-1)
        pz_idx = torch.argmax(preds.player_pos_z, dim=-1)
        
        # Mapping index to value roughly
        pos_x = self.pos_bins_x[px_idx]
        pos_y = self.pos_bins_y[py_idx]
        pos_z = self.pos_bins_z[pz_idx]
        position = torch.stack([pos_x, pos_y, pos_z], dim=-1)

        # 5. Weapon: Argmax
        active_weapon_idx = torch.argmax(preds.weapon_logits, dim=-1).int()

        # 6. Eco/Inv: Sigmoid > 0.5 -> Bitmask
        # Eco: 256 bits (4 * 64) -> [T, 5, 4] int64
        # Inv: 128 bits (2 * 64) -> [T, 5, 2] int64
        def bits_to_masks(logits, num_chunks):
            probs = torch.sigmoid(logits) # [T, 5, 64*chunks]
            bits = (probs > 0.5).long()
            chunks = []
            powers64 = 2 ** torch.arange(64, device=logits.device)
            for i in range(num_chunks):
                chunk_bits = bits[..., i*64 : (i+1)*64]
                # Ensure 64 bits matches powers
                # Summing large powers might overflow int64 in pytorch if not careful?
                # PyTorch int64 is signed. uint64 not fully supported.
                # However, dataset uses int64 for storage.
                # We do simple sum.
                val = (chunk_bits * powers64).sum(dim=-1).long()
                chunks.append(val)
            return torch.stack(chunks, dim=-1)

        eco_mask = bits_to_masks(preds.eco_logits, 4)
        inventory_mask = bits_to_masks(preds.inventory_logits, 2)

        # 7. Round State
        # Logits -> Sigmoid -> >0.5 -> Mask
        # 5 bits
        rs_probs = torch.sigmoid(preds.round_state_logits)
        rs_bits = (rs_probs > 0.5).long()
        powers5 = 2 ** torch.arange(5, device=preds.round_state_logits.device)
        round_state_mask = (rs_bits * powers5).sum(dim=-1).byte()

        # 8. Round Number
        # Sigmoid * 30
        round_number = (torch.sigmoid(preds.round_num_logit) * 30.0).int().squeeze(-1)

        # 9. Enemy Pos
        ex_idx = torch.argmax(preds.enemy_pos_x, dim=-1)
        ey_idx = torch.argmax(preds.enemy_pos_y, dim=-1)
        ez_idx = torch.argmax(preds.enemy_pos_z, dim=-1)
        e_pos_x = self.pos_bins_x[ex_idx]
        e_pos_y = self.pos_bins_y[ey_idx]
        e_pos_z = self.pos_bins_z[ez_idx]
        enemy_positions = torch.stack([e_pos_x, e_pos_y, e_pos_z], dim=-1)

        # 10. Alive Masks
        # Preds are logits for count (0-5)?
        # Actually ModelPrediction says: team_alive_logits: [B, T, 6]
        # This predicts COUNT, not mask.
        # We can't reconstruct the exact mask from count easily.
        # But we can reconstruct a "dummy" mask or just ignore it for viz if we only show count?
        # visualize.py expects a mask.
        # Let's just create an all-alive mask or base it on something else?
        # Actually, for visualization, we assume everyone we predict is alive?
        # Or we just pass the GT mask for "Alive" status to keep it clean?
        # The prompt says "overlay all predictions".
        # If we predict 3 alive, which 3? The model doesn't output per-player alive logits (except implicity in other heads?)
        # Wait, `CS2Loss.alive_count` is cross entropy on count.
        # So the model *doesn't* predict who is alive, only how many.
        # So we cannot overlay "Alive/Dead" per player prediction correctly.
        # We will use GT alive mask for the "Prediction" object to ensure panels render,
        # or we set all to True.
        
        # For now, let's copy GT alive mask to Prediction to allow rendering valid data slots.
        # This is a bit of a cheat but the model architecture limitation.
        # (Unless I missed a head).
        # We have `stats` (HP). If HP is predicted 0, they are dead.
        # Let's use Predicted HP <= 0 to determine alive mask!
        alive_mask = (hp > 0)
        enemy_alive_mask = torch.ones_like(alive_mask) # We don't have enemy HP.

        return GroundTruth(
            alive_mask=alive_mask,
            stats=stats,
            mouse_delta=mouse_delta,
            position=position,
            keyboard_mask=keyboard_mask,
            eco_mask=eco_mask,
            inventory_mask=inventory_mask,
            active_weapon_idx=active_weapon_idx,
            round_number=round_number,
            round_state_mask=round_state_mask,
            enemy_positions=enemy_positions,
            enemy_alive_mask=enemy_alive_mask
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory (containing pytorch_model.bin or similar)")
    parser.add_argument("--data_root", type=str, default="dataset0")
    parser.add_argument("--output", type=str, default="inference_output.mp4")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--epoch_idx", type=int, default=0)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--frames", type=int, default=128)
    args = parser.parse_args()

    # 1. Initialize Accelerator (Handles Device Placement)
    accelerator = Accelerator()
    device = accelerator.device

    # 2. Config & Model
    # We assume default config for now, as checkpoints usually don't store full config in this repo structure
    # If the checkpoint has config.json, we could load it, but Config is a dataclass here.
    model_cfg = ModelConfig()
    model = GamePredictorBackbone(model_cfg)
    
    # Create dummy optimizer for FSDP2 prepare requirement
    dummy_opt = torch.optim.AdamW(model.parameters(), lr=0.0)
    
    # 3. Load Checkpoint
    # accelerate.load_state expects a directory
    if os.path.isdir(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        # We must register model first.
    else:
        print(f"Checkpoint path {args.checkpoint} is not a directory.")
        return

    # Register for Accelerate
    model, dummy_opt = accelerator.prepare(model, dummy_opt)
    # NOW load state
    try:
        accelerator.load_state(args.checkpoint)
        print("Loaded accelerator state.")
    except Exception as e:
        print(f"Failed to load accelerator state: {e}")
        # Try finding pytorch_model.bin inside?
        pass

    model.eval()
    
    # 4. Dataset
    print(f"Loading dataset {args.data_root}...")
    ds_cfg = DatasetConfig(data_root=args.data_root, run_dir=".")
    ds_root = DatasetRoot(ds_cfg)
    epoch = ds_root.build_epoch(args.split, args.epoch_idx)
    
    if args.sample_idx >= len(epoch):
        print(f"Sample index {args.sample_idx} out of range (Len: {len(epoch)})")
        return

    # Use metadata directly to allow manual decoding
    sample = epoch.samples[args.sample_idx]
    print(f"Processing Sample: Round {sample.round.round_num}, Team {sample.round.team}")

    # 5. Run Inference
    # We need to slice the sample to args.frames if needed, but dataset loads chunks.
    # The sample object from `build_epoch` has a fixed length from config.
    # We can override the config length or just slice the tensor.
    
    # Manually get data to control length
    # Images: [T, 5, 3, H, W]
    images = epoch._decode_video(sample)
    audio = epoch._decode_audio(sample)
    gt = epoch._get_truth(sample)

    # Slice to requested frames
    T = min(args.frames, images.shape[0])
    images = images[:T]
    gt_sliced = GroundTruth(**{k: getattr(gt, k)[:T] if isinstance(getattr(gt, k), torch.Tensor) else getattr(gt, k) for k in vars(gt)})
    
    # Move to Device
    images_dev = images.to(device).unsqueeze(0) # [B=1, T, 5, 3, H, W]
    audio_dev = audio.to(device).unsqueeze(0)   # [B=1, 5, 2, S]
    
    print("Running forward pass...")
    with torch.no_grad():
        preds_dict = model(images_dev, audio_dev)
        # Unpack dict to dataclass
        # Squeeze batch dim [1, T, ...] -> [T, ...]
        preds_dict = {k: v.squeeze(0) for k, v in preds_dict.items()}
        preds = ModelPrediction(**preds_dict)

    # 6. Decode Predictions
    decoder = PredictionDecoder(model_cfg, device)
    pred_gt = decoder.decode(preds)

    # 7. Visualize
    if accelerator.is_main_process:
        print(f"Rendering video to {args.output}...")
        
        # Setup Video Writer
        # Images are [T, 5, 3, H, W]
        _, _, _, h, w = images.shape
        grid_w, grid_h = w * 3, h * 2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, 32.0, (grid_w, grid_h))

        for t in tqdm(range(T)):
            # Prepare frames for visualize.py (List of 5 numpy BGR images)
            frame_list = []
            for p in range(5):
                # [3, H, W] -> [H, W, 3] BGR uint8
                img_t = images[t, p]
                frame_list.append(tensor_to_uint8(img_t))
                
            # Get scalar/struct data for this frame
            # We assume visualize_frame handles slicing if we pass the whole GT object and index t
            # BUT visualize.py's convert_tensor_to_viz_data expects the object to be sliced or indexable.
            # It takes `gt_object` and `t`.
            
            # Warning: Our `pred_gt` tensors are on GPU. visualize.py might need CPU.
            # convert_tensor_to_viz_data does .tolist() or .item(), which works on GPU tensors (moves to cpu implicitly).
            # But Eco/Inventory masks are explicitly .cpu().numpy() in visualize.py.
            # So we should probably ensure everything is compatible.
            
            composite = visualize_frame(frame_list, t, gt_sliced, pred_gt)
            writer.write(composite)

        writer.release()
        print("Done.")
    
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
