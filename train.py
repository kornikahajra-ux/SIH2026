"""
Training pipeline for OceanEmbed's split thermocline / remaining-depths submodels.

Trains the two OceanUNet variants from model.py together in one loop: each
batch's full (B, 35, H, W) target/mask is split via
model.split_full_depth_profile() into the 8-level thermocline slice and the
27-level remaining-depths slice, both submodels run their own independent
forward/backward pass on the same input, and the two masked-MSE losses are
summed before a single optimizer step. The two submodels share no
parameters, so summing the losses before backward() is equivalent to
training them independently while only walking the dataloader once per
epoch - and it keeps everything in one script instead of two.

Each submodel gets its own checkpoint file, so either one can later be
swapped out, retrained, or loaded independently (e.g. by an evaluation
script that calls model.assemble_full_depth_profile() to reconstruct the
full 35-depth column from both checkpoints).

Usage:
    python train.py
    python train.py --epochs 150 --lr 5e-4 --patience 10
    python train.py --thermocline_checkpoint_name thermo_v2.pth
"""

import argparse
import itertools
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_dataloaders
from model import (
    MaskedMSELoss,
    REMAINING_NUM_LEVELS,
    THERMOCLINE_DEPTH_END,
    THERMOCLINE_DEPTH_START,
    THERMOCLINE_NUM_LEVELS,
    build_remaining_depths_model,
    build_thermocline_model,
    split_full_depth_profile,
)

# Anchor path resolution to script directory
BASE_DIR = Path(__file__).parent


def set_seed(seed: int):
    """Seed all RNGs touched by the training loop for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model_thermo, model_remaining, loader, optimizer, criterion, device, scaler=None):
    model_thermo.train()
    model_remaining.train()
    use_amp = scaler is not None

    all_params = list(itertools.chain(model_thermo.parameters(), model_remaining.parameters()))
    total_loss = total_thermo = total_remaining = 0.0

    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        y_thermo, y_remaining = split_full_depth_profile(y)
        mask_thermo, mask_remaining = split_full_depth_profile(mask)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred_thermo = model_thermo(x)
            pred_remaining = model_remaining(x)
            loss_thermo = criterion(pred_thermo, y_thermo, mask_thermo)
            loss_remaining = criterion(pred_remaining, y_remaining, mask_remaining)
            loss = loss_thermo + loss_remaining

        if use_amp:
            scaler.scale(loss).backward()
            # Unscale before clipping so max_norm is applied to true gradient magnitudes
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_thermo += loss_thermo.item() * bs
        total_remaining += loss_remaining.item() * bs

    n = len(loader.dataset)
    return total_loss / n, total_thermo / n, total_remaining / n


@torch.no_grad()
def validate_epoch(model_thermo, model_remaining, loader, criterion, device, use_amp=False):
    model_thermo.eval()
    model_remaining.eval()

    total_loss = total_thermo = total_remaining = 0.0

    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        y_thermo, y_remaining = split_full_depth_profile(y)
        mask_thermo, mask_remaining = split_full_depth_profile(mask)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred_thermo = model_thermo(x)
            pred_remaining = model_remaining(x)
            loss_thermo = criterion(pred_thermo, y_thermo, mask_thermo)
            loss_remaining = criterion(pred_remaining, y_remaining, mask_remaining)
            loss = loss_thermo + loss_remaining

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_thermo += loss_thermo.item() * bs
        total_remaining += loss_remaining.item() * bs

    n = len(loader.dataset)
    return total_loss / n, total_thermo / n, total_remaining / n


def main():
    parser = argparse.ArgumentParser(description="OceanEmbed Split (Thermocline / Remaining-Depths) Training")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (train/val)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (shared by both submodels)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--patience", type=int, default=15,
                         help="Stop early if combined val loss doesn't improve for this many epochs (0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--thermocline_checkpoint_name", type=str, default="best_ocean_unet_thermocline.pth",
                         help="Filename for the thermocline submodel's best checkpoint under checkpoints/")
    parser.add_argument("--remaining_checkpoint_name", type=str, default="best_ocean_unet_remaining.pth",
                         help="Filename for the remaining-depths submodel's best checkpoint under checkpoints/")
    parser.add_argument("--amp", action="store_true", default=None,
                         help="Force-enable mixed precision. Default: auto-enabled on CUDA, off on CPU.")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                         help="Force-disable mixed precision.")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cudnn.benchmark autotunes convolution algorithms for this fixed input
    # size (101, 241) - safe here since every batch shares that shape.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = args.amp if args.amp is not None else (device.type == "cuda")

    print(f"Using device: {device}")
    print(f"Seed: {args.seed} | Epochs: {args.epochs} | Batch size: {args.batch_size} | "
          f"LR: {args.lr} | Weight decay: {args.weight_decay} | "
          f"Early-stop patience: {args.patience} | AMP: {use_amp}")
    print(f"Thermocline submodel : depth indices [{THERMOCLINE_DEPTH_START}:{THERMOCLINE_DEPTH_END}) "
          f"-> {THERMOCLINE_NUM_LEVELS} levels")
    print(f"Remaining submodel   : depth indices [0:{THERMOCLINE_DEPTH_START}) + [{THERMOCLINE_DEPTH_END}:35) "
          f"-> {REMAINING_NUM_LEVELS} levels")

    # Load PyTorch DataLoaders - still the full 35-depth target/mask; split
    # into the two submodels' slices per-batch inside the training loop.
    train_loader, val_loader, _, stats = get_dataloaders(batch_size=args.batch_size)

    model_thermo = build_thermocline_model().to(device)
    model_remaining = build_remaining_depths_model().to(device)

    criterion = MaskedMSELoss()

    # One optimizer covering both submodels' parameters - they don't share
    # any weights, so this is equivalent to two separate optimizers while
    # only needing one .step() call per batch.
    all_params = list(itertools.chain(model_thermo.parameters(), model_remaining.parameters()))
    try:
        optimizer = AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay, fused=(device.type == "cuda"))
    except TypeError:
        optimizer = AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_dir = BASE_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    thermo_ckpt_path = checkpoint_dir / args.thermocline_checkpoint_name
    remaining_ckpt_path = checkpoint_dir / args.remaining_checkpoint_name

    history = []

    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_thermo, train_remaining = train_epoch(
            model_thermo, model_remaining, train_loader, optimizer, criterion, device, scaler=scaler
        )
        val_loss, val_thermo, val_remaining = validate_epoch(
            model_thermo, model_remaining, val_loader, criterion, device, use_amp=use_amp
        )

        scheduler.step(val_loss)
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} (thermo {train_thermo:.4f} / remaining {train_remaining:.4f}) | "
            f"Val Loss: {val_loss:.4f} (thermo {val_thermo:.4f} / remaining {val_remaining:.4f}) | "
            f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_thermo_loss": train_thermo,
            "train_remaining_loss": train_remaining,
            "val_loss": val_loss,
            "val_thermo_loss": val_thermo,
            "val_remaining_loss": val_remaining,
            "lr": current_lr,
            "elapsed_sec": elapsed,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_thermo.state_dict(),
                    "stats": stats,
                    "val_loss": val_thermo,
                },
                thermo_ckpt_path,
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_remaining.state_dict(),
                    "stats": stats,
                    "val_loss": val_remaining,
                },
                remaining_ckpt_path,
            )
            print(f"  --> Saved new best checkpoints to {thermo_ckpt_path.name} / {remaining_ckpt_path.name}")
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"\nNo val-loss improvement for {epochs_without_improvement} epochs "
                    f"(patience={args.patience}). Stopping early at epoch {epoch}."
                )
                break

    history_path = BASE_DIR / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\nTraining Complete. Best Combined Validation Loss: {best_val_loss:.4f}")
    print(f"Training history saved to: {history_path.resolve()}")


if __name__ == "__main__":
    main()
