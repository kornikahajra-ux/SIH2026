"""
Training pipeline for OceanEmbed.
Trains OceanUNet on surface input features to predict 3D subsurface ocean temperature profiles.

Usage:
    python train.py
"""

import time
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_dataloaders
from model import OceanUNet, MaskedMSELoss


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        preds = model(x)
        loss = criterion(preds, y, mask)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load PyTorch DataLoaders (batch_size=4 matches dataset setup)
    train_loader, val_loader, _, stats = get_dataloaders(batch_size=4)

    model = OceanUNet(in_channels=12, out_channels=35).to(device)
    criterion = MaskedMSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    num_epochs = 20
    best_val_loss = float("inf")
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    best_model_path = checkpoint_dir / "best_ocean_unet.pth"

    print("\nStarting Training Loop...")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "stats": stats,
                    "val_loss": val_loss,
                },
                best_model_path,
            )
            print(f"  --> Saved new best checkpoint to {best_model_path}")

    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
