import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import directed_hausdorff
import config
from model.unet import AttentionUNet

def calculate_iou(predictions, targets, smooth=1e-6):
    """Calculate Intersection over Union (IoU) score"""
    if predictions.dim() == 4:
        if predictions.size(1) > 1:
            # Multi-class case
            predictions = torch.argmax(predictions, dim=1)
            targets = torch.argmax(targets, dim=1) if targets.size(1) > 1 else targets.squeeze(1)
        else:
            # Binary case
            predictions = (torch.sigmoid(predictions) > 0.5).float().squeeze(1)
            targets = targets.squeeze(1) if targets.size(1) == 1 else targets
            
    # Calculate intersection and union
    intersection = torch.sum(predictions * targets, dim=(1, 2))
    union = torch.sum(predictions, dim=(1, 2)) + torch.sum(targets, dim=(1, 2)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return torch.mean(iou).item()

def calculate_hausdorff_distance(predictions, targets):
    """Calculate Hausdorff distance"""
    if predictions.dim() == 4:
        if predictions.size(1) > 1:
            predictions = torch.argmax(predictions, dim=1).cpu().numpy()
            targets = torch.argmax(targets, dim=1).cpu().numpy() if targets.size(1) > 1 else targets.squeeze(1).cpu().numpy()
        else:
            predictions = (torch.sigmoid(predictions) > 0.5).float().squeeze(1).cpu().numpy()
            targets = targets.squeeze(1).cpu().numpy() if targets.size(1) == 1 else targets.cpu().numpy()
    else:
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
    hausdorff_distances = []
    for pred, target in zip(predictions, targets):
        # Only calculate if both have some positive pixels
        if np.sum(pred) > 0 and np.sum(target) > 0:
            hd = max(
                directed_hausdorff(pred, target)[0],
                directed_hausdorff(target, pred)[0]
            )
            hausdorff_distances.append(hd)
        elif np.sum(pred) > 0 or np.sum(target) > 0:
            # One is empty, the other is not - use maximum possible distance
            hausdorff_distances.append(max(pred.shape))
    
    return np.mean(hausdorff_distances) if hausdorff_distances else 0.0

def plot_confusion_matrix(y_true, y_pred, class_names=['Background', 'Lung']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    return plt.gcf()

def evaluate_model(model, data_loader, device, save_visualizations=True, viz_dir=None):
    """Evaluate model and compute metrics"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_dice_scores = []
    all_iou_scores = []
    hausdorff_distances = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(data_loader):
            data = data.to(device)
            targets = targets.to(device)
            
            predictions = model(data)
            
            # Handle deep supervision outputs
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            # Store predictions and targets for metric calculation
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
            # Calculate Dice score per batch
            predictions_sigmoid = torch.sigmoid(predictions) if predictions.size(1) == 1 else torch.softmax(predictions, dim=1)
            targets_binary = (targets > 0.5).float()
            
            # Vectorized Dice calculation
            intersection = torch.sum(predictions_sigmoid * targets_binary, dim=(2, 3))
            dice = (2 * intersection) / (torch.sum(predictions_sigmoid, dim=(2, 3)) + torch.sum(targets_binary, dim=(2, 3)) + 1e-6)
            all_dice_scores.extend(dice.mean(dim=1).cpu().numpy())
            
            # Calculate IoU per batch
            batch_iou = calculate_iou(predictions, targets)
            all_iou_scores.append(batch_iou)
            
            # Save visualizations for first batch
            if save_visualizations and batch_idx == 0 and viz_dir:
                save_visualization_samples(data[:4], targets[:4], predictions[:4], viz_dir)
                
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate final metrics
    avg_dice = np.mean(all_dice_scores)
    avg_iou = np.mean(all_iou_scores)
    
    # Calculate Hausdorff distance on subset for efficiency
    sample_size = min(50, len(all_predictions))
    indices = np.random.choice(len(all_predictions), sample_size, replace=False)
    hausdorff_sample_preds = all_predictions[indices]
    hausdorff_sample_targets = all_targets[indices]
    
    avg_hausdorff = calculate_hausdorff_distance(hausdorff_sample_preds, hausdorff_sample_targets)
    
    # Confusion matrix calculation
    if all_predictions.size(1) == 1:
        pred_flat = (torch.sigmoid(all_predictions) > 0.5).view(-1).cpu().numpy()
    else:
        pred_flat = torch.argmax(all_predictions, dim=1).view(-1).cpu().numpy()
        
    target_flat = all_targets.view(-1).cpu().numpy()
    if all_targets.size(1) > 1:
        target_flat = torch.argmax(all_targets, dim=1).view(-1).cpu().numpy()
    elif all_targets.size(1) == 1:
        target_flat = (all_targets > 0.5).view(-1).cpu().numpy()
        
    cm = confusion_matrix(target_flat, pred_flat)
    
    return {
        'dice_score': avg_dice,
        'iou_score': avg_iou,
        'hausdorff_distance': avg_hausdorff,
        'confusion_matrix': cm,
        'predictions': all_predictions[:10],  # Return first 10 for visualization
        'targets': all_targets[:10]
    }

def save_visualization_samples(inputs, targets, predictions, viz_dir):
    """Save visualization samples"""
    os.makedirs(viz_dir, exist_ok=True)
    
    fig, axes = plt.subplots(inputs.size(0), 3, figsize=(15, 5 * inputs.size(0)))
    if inputs.size(0) == 1:
        axes = axes.reshape(1, -1)
        
    for i in range(inputs.size(0)):
        # Input image
        input_img = inputs[i].squeeze().cpu().numpy()
        axes[i, 0].imshow(input_img, cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        if targets.size(1) == 1:
            gt_mask = targets[i].squeeze().cpu().numpy()
        else:
            gt_mask = torch.argmax(targets[i], dim=0).cpu().numpy()
        axes[i, 1].imshow(gt_mask, cmap='jet', alpha=0.7)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        if predictions.size(1) == 1:
            pred_mask = torch.sigmoid(predictions[i]).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(float)
        else:
            pred_mask = torch.softmax(predictions[i], dim=0)
            pred_mask = torch.argmax(pred_mask, dim=0).cpu().numpy()
        axes[i, 2].imshow(pred_mask, cmap='jet', alpha=0.7)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'evaluation_comparison.png'))
    plt.close()

def main():
    """Main evaluation function"""
    # Set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Create visualization directory
    viz_dir = os.path.join(config.VISUALIZATION_DIR, 'evaluation')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    try:
        from data.dataset import get_data_loaders
        train_loader, val_loader, test_loader = get_data_loaders()
        print(f"Data loaded - Test: {len(test_loader)} batches")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize model
    print("Initializing model...")
    model = AttentionUNet(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        deep_supervision=False  # Disable during evaluation
    ).to(device)
    
    # Load trained model weights
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No trained model found. Please train the model first.")
        return
        
    # Evaluate on test set
    results = evaluate_model(model, test_loader, device, save_visualizations=True, viz_dir=viz_dir)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Dice Score: {results['dice_score']:.4f}")
    print(f"IoU Score: {results['iou_score']:.4f}")
    print(f"Hausdorff Distance: {results['hausdorff_distance']:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Save results to file
    results_file = os.path.join(viz_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Dice Score: {results['dice_score']:.4f}\n")
        f.write(f"IoU Score: {results['iou_score']:.4f}\n")
        f.write(f"Hausdorff Distance: {results['hausdorff_distance']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(results['confusion_matrix']) + "\n")
    
    print(f"\nResults saved to {results_file}")
    print(f"Visualization saved to {os.path.join(viz_dir, 'evaluation_comparison.png')}")

if __name__ == "__main__":
    main()