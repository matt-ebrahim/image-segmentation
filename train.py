import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import config
from model.unet import AttentionUNet


class CombinedLoss(nn.Module):
    """Combined Dice and Cross-Entropy Loss"""

    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1e-5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        # Convert targets to long for CrossEntropyLoss
        if targets.size(1) == 1:
            targets_ce = targets.squeeze(1).long()
        else:
            targets_ce = torch.argmax(targets, dim=1).long()

        # Dice Loss calculation
        softmax_pred = F.softmax(predictions, dim=1)
        if predictions.size(1) == 1:  # Binary segmentation
            predictions_sigmoid = torch.sigmoid(predictions)
            dice_loss = 1 - (
                2 * torch.sum(predictions_sigmoid * targets) + self.smooth
            ) / (torch.sum(predictions_sigmoid) + torch.sum(targets) + self.smooth)
        else:  # Multi-class segmentation
            dice_loss_total = 0
            for c in range(softmax_pred.shape[1]):
                class_pred = softmax_pred[:, c]
                class_target = (targets_ce == c).float()
                intersection = torch.sum(class_pred * class_target)
                union = torch.sum(class_pred) + torch.sum(class_target)
                dice_loss_total += 1 - (2 * intersection + self.smooth) / (
                    union + self.smooth
                )
            dice_loss = dice_loss_total / softmax_pred.shape[1]

        # Cross Entropy Loss
        ce_loss = self.ce_loss(predictions, targets_ce)

        # Combined loss
        combined_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        return combined_loss, dice_loss, ce_loss


def calculate_dice_score(predictions, targets):
    """Calculate Dice score for evaluation"""
    if predictions.dim() == 4 and predictions.size(1) > 1:
        # Multi-class case
        predictions = torch.argmax(predictions, dim=1)
        targets = (
            torch.argmax(targets, dim=1) if targets.size(1) > 1 else targets.squeeze(1)
        )
    elif predictions.dim() == 4 and predictions.size(1) == 1:
        # Binary case with sigmoid
        predictions = (torch.sigmoid(predictions) > 0.5).float().squeeze(1)
        targets = targets.squeeze(1) if targets.size(1) == 1 else targets

    # Calculate Dice coefficient
    smooth = 1e-6
    intersect = torch.sum(predictions * targets)
    dice = (2 * intersect + smooth) / (
        torch.sum(predictions) + torch.sum(targets) + smooth
    )
    return dice.item()


def train_one_epoch(loader, model, criterion, optimizer, device, scaler, epoch):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    epoch_ce = 0

    progress_bar = tqdm(loader, desc=f"Training Epoch {epoch}")
    for batch_idx, (data, targets) in enumerate(progress_bar):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Use AMP for mixed precision training
        with autocast(enabled=config.USE_AMP):
            predictions = model(data)

            # Handle deep supervision
            if isinstance(predictions, tuple):
                # Deep supervision case
                main_pred = predictions[0]
                loss, dice_loss, ce_loss = criterion(main_pred, targets)

                # Add auxiliary losses
                aux_weight = 0.25
                for i in range(1, len(predictions)):
                    aux_loss, _, _ = criterion(predictions[i], targets)
                    loss += aux_weight * aux_loss
            else:
                # Standard case
                loss, dice_loss, ce_loss = criterion(predictions, targets)

        # Backpropagation with gradient scaling
        if config.USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        epoch_dice += dice_loss.item() if hasattr(dice_loss, "item") else dice_loss
        epoch_ce += ce_loss.item() if hasattr(ce_loss, "item") else ce_loss

        # Update progress bar
        progress_bar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Dice": f"{dice_loss.item() if hasattr(dice_loss, 'item') else dice_loss:.4f}",
                "CE": f"{ce_loss.item() if hasattr(ce_loss, 'item') else ce_loss:.4f}",
            }
        )

    avg_loss = epoch_loss / len(loader)
    avg_dice = epoch_dice / len(loader)
    avg_ce = epoch_ce / len(loader)

    return avg_loss, avg_dice, avg_ce


def validate(loader, model, criterion, device):
    """Validation function"""
    model.eval()
    epoch_loss = 0
    epoch_dice_score = 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validating")
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data = data.to(device)
            targets = targets.to(device)

            predictions = model(data)

            # Handle deep supervision during validation
            if isinstance(predictions, tuple):
                main_pred = predictions[0]
                loss, dice_loss, ce_loss = criterion(main_pred, targets)
            else:
                loss, dice_loss, ce_loss = criterion(predictions, targets)

            epoch_loss += loss.item()

            # Calculate Dice score
            dice_score = calculate_dice_score(predictions, targets)
            epoch_dice_score += dice_score

            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Dice": f"{dice_score:.4f}"}
            )

    avg_loss = epoch_loss / len(loader)
    avg_dice_score = epoch_dice_score / len(loader)

    return avg_loss, avg_dice_score


def create_checkpoint(model, optimizer, scheduler, epoch, best_val_dice, filename):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_dice": best_val_dice,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def push_checkpoint_to_github(checkpoint_path, epoch, dice_score):
    """Push the best checkpoint to GitHub using git commands"""
    try:
        # Check file size (GitHub limit is 100MB)
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

        if file_size_mb > 100:
            print(
                f"Warning: Checkpoint file is {file_size_mb:.2f}MB, exceeding GitHub's 100MB limit"
            )
            print(
                "Skipping push to GitHub. Consider using Git LFS or external storage."
            )
            return False

        # Add the checkpoint file
        subprocess.run(["git", "add", checkpoint_path], check=True)

        # Commit with a descriptive message
        commit_message = (
            f"Add best checkpoint at epoch {epoch} with Dice score {dice_score:.4f}"
        )
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Push to GitHub main branch
        subprocess.run(["git", "push", "origin", "main"], check=True)

        print(f"Successfully pushed {checkpoint_path} to GitHub")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pushing to GitHub: {e}")
        return False


def main():
    """Main training function"""
    # Setup directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Setup TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.LOG_DIR, f"training_{timestamp}")
    writer = SummaryWriter(log_dir)

    # Set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    try:
        train_loader, val_loader, test_loader = get_data_loaders()
        print(
            f"Data loaded - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches"
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Initialize model
    print("Initializing model...")
    model = AttentionUNet(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        deep_supervision=True,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model initialized - Total params: {total_params:,}, Trainable params: {trainable_params:,}"
    )

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.LR_SCHEDULER_T_MAX, eta_min=1e-6
    )

    # Initialize loss function
    criterion = CombinedLoss(
        dice_weight=config.DICE_LOSS_WEIGHT, ce_weight=config.CE_LOSS_WEIGHT
    )

    # Initialize AMP scaler for mixed precision
    scaler = GradScaler() if config.USE_AMP else None

    # Training variables
    best_val_dice = 0.0
    patience_counter = 0
    start_epoch = 0

    # Check for existing checkpoint to resume training
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"\nFound existing checkpoint: {best_model_path}")
        print("Resuming training from checkpoint...")
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scheduler_state_dict") and scheduler:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            best_val_dice = checkpoint.get("best_val_dice", 0.0)
            print(
                f"Resuming from epoch {start_epoch}, Best Dice: {best_val_dice:.4f}\n"
            )
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...\n")
            start_epoch = 0
            best_val_dice = 0.0
    else:
        print("No existing checkpoint found. Starting training from scratch.\n")

    print("Starting training...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 50)

        # Train for one epoch
        train_loss, train_dice, train_ce = train_one_epoch(
            train_loader, model, criterion, optimizer, device, scaler, epoch + 1
        )

        # Validation
        val_loss, val_dice = validate(val_loader, model, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("DiceLoss/Train", train_dice, epoch)
        writer.add_scalar("CrossEntropyLoss/Train", train_ce, epoch)
        writer.add_scalar("DiceScore/Validation", val_dice, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        # Print epoch results
        print(
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train CE: {train_ce:.4f}"
        )
        print(
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.2e}"
        )

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            create_checkpoint(
                model, optimizer, scheduler, epoch + 1, best_val_dice, best_model_path
            )
            print(f"New best model saved! Dice Score: {val_dice:.4f}")

            # Push checkpoint to GitHub
            print("Pushing best checkpoint to GitHub...")
            push_checkpoint_to_github(best_model_path, epoch + 1, best_val_dice)
        else:
            patience_counter += 1
            print(f"No improvement in {patience_counter} epochs")

        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            periodic_model_path = os.path.join(
                config.CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth"
            )
            create_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_dice, periodic_model_path
            )

    # Close TensorBoard writer
    writer.close()

    print("\nTraining complete!")
    print(f"Best validation Dice score: {best_val_dice:.4f}")

    # Load best model for final test evaluation
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print("Loading best model for final evaluation...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Final test evaluation
        print("Final evaluation on test set...")
        test_loss, test_dice = validate(test_loader, model, criterion, device)
        print(f"Final Test Results - Loss: {test_loss:.4f}, Dice: {test_dice:.4f}")


if __name__ == "__main__":
    # Import get_data_loaders here to avoid circular imports
    from data.dataset import get_data_loaders

    main()
