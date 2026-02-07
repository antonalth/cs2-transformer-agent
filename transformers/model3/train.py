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
import os
import argparse
import logging
import math
import time
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

import wandb

from config import GlobalConfig, DatasetConfig, ModelConfig, TrainConfig
from dataset import DatasetRoot, TrainingSample, GroundTruth
from model import GamePredictorBackbone, ModelPrediction
from model_loss import ModelLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Training")

def setup_dist():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    local_rank = setup_dist()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="unnamed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_fsdp", help="Directory to save checkpoints")
    args = parser.parse_args()

    d_cfg = DatasetConfig(data_root=args.data_root, run_dir="./runs")
    t_cfg = TrainConfig(data_root=args.data_root, run_name=args.run_name, output_dir=args.output_dir)
    m_cfg = ModelConfig()
    global_cfg = GlobalConfig(dataset=d_cfg, model=m_cfg, train=t_cfg)

if __name__ == "__main__":
    main()
