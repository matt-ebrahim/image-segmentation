import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import subprocess
import os
import config
from model.unet import UNet
from data.dataset import get_data_loaders


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    return 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def train_one_epoch(loader, model, optimizer, device):
    model.train()
    epoch_loss = 0

    for batch_idx, (data, targets) in enumerate(tqdm(loader, leave=False)):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(data)
        loss = dice_loss(predictions, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def validate(loader, model, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(loader, leave=False)):
            data = data.to(device)
            targets = targets.to(device)

            predictions = model(data)
            loss = dice_loss(predictions, targets)

            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def push_checkpoint_to_github(checkpoint_path, epoch, dice_score):
    """Push the best checkpoint to GitHub using git commands"""
    try:
        # Add the checkpoint file
        subprocess.run(["git", "add", checkpoint_path], check=True)

        # Commit with a descriptive message
        commit_message = (
            f"Add best checkpoint at epoch {epoch} with Dice score {dice_score:.4f}"
        )
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Push to GitHub
        subprocess.run(["git", "push"], check=True)

        print(f"Successfully pushed {checkpoint_path} to GitHub")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pushing to GitHub: {e}")
        return False


def main():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    print("Loading data...")
    try:
        train_loader, val_loader = get_data_loaders()
    except FileNotFoundError as e:
        print(e)
        print(
            "Please ensure you have downloaded the data using 'python data/download_data.py'"
        )
        return

    # Initialize Model
    print("Initializing model...")
    model = UNet(n_channels=config.IN_CHANNELS, n_classes=config.NUM_CLASSES).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Track best validation performance
    best_dice_score = 0.0
    best_checkpoint_path = "best_checkpoint.pth"

    # Training Loop
    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")

        train_loss = train_one_epoch(train_loader, model, optimizer, device)
        val_loss = validate(val_loader, model, device)

        # Convert Dice loss to Dice score (since loss = 1 - dice_score)
        dice_score = 1 - val_loss

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice Score: {dice_score:.4f}"
        )

        # Save model checkpoint for current epoch
        torch.save(model.state_dict(), f"unet_epoch_{epoch + 1}.pth")

        # Check if this is the best model so far
        if dice_score > best_dice_score:
            best_dice_score = dice_score
            print(f"New best Dice score: {best_dice_score:.4f}")

            # Save best checkpoint
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Saved best checkpoint to {best_checkpoint_path}")

            # Push to GitHub
            print("Pushing best checkpoint to GitHub...")
            push_checkpoint_to_github(best_checkpoint_path, epoch + 1, best_dice_score)

    print(f"Training complete. Best Dice score: {best_dice_score:.4f}")


if __name__ == "__main__":
    main()
