import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from pathlib import Path
import config
from model.unet import AttentionUNet

def load_nifti_image(filepath):
    """Load NIfTI image and return as numpy array"""
    img = nib.load(filepath).get_fdata()
    return img

def preprocess_image(image_array):
    """Preprocess image for inference"""
    # Handle different dimensions
    if image_array.ndim == 3:
        # Take middle slice for 3D volumes
        depth = image_array.shape[2]
        image_array = image_array[:, :, depth // 2]
    
    # Normalize to 0-1 range
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-8)
    
    # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_array).float()
    return image_tensor

def postprocess_prediction(prediction, original_shape):
    """Postprocess model prediction"""
    # Apply softmax for multi-class or sigmoid for binary
    if prediction.size(1) == 1:
        prediction = torch.sigmoid(prediction)
    else:
        prediction = F.softmax(prediction, dim=1)
    
    # Convert to numpy
    prediction = prediction.squeeze(0).detach().cpu().numpy()
    
    # Resize to original shape if needed
    if prediction.shape[1:] != original_shape:
        # This would require interpolation, skipping for simplicity
        pass
    
    return prediction

def overlay_mask_on_image(image, mask, alpha=0.4, mask_color=[255, 0, 0]):
    """Overlay mask on image"""
    # Normalize image to 0-255 range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Convert grayscale to RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Create colored mask
    mask_colored = np.zeros_like(image_rgb)
    if mask.ndim == 2:
        # Binary mask case
        mask_colored[mask > 0.5] = mask_color
    else:
        # Multi-class mask case
        for c in range(min(mask.shape[0], 3)):  # Up to 3 classes
            color_multiplier = [0, 0, 0]
            color_multiplier[c] = 255
            mask_colored[:, :, c] = mask[c] * color_multiplier[c]
    
    # Overlay mask on image
    overlay = cv2.addWeighted(image_rgb, 1-alpha, mask_colored, alpha, 0)
    return overlay

def run_inference(model, input_tensor, device):
    """Run inference on a single input"""
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        prediction = model(input_tensor)
        
        # Handle deep supervision
        if isinstance(prediction, tuple):
            prediction = prediction[0]
            
        return prediction

def save_visualization(input_image, prediction, output_path):
    """Save visualization of input and prediction"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image
    axes[0].imshow(input_image.squeeze(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Prediction
    if prediction.ndim == 3 and prediction.shape[0] > 1:
        # Multi-class case - show argmax
        pred_class = np.argmax(prediction, axis=0)
        axes[1].imshow(pred_class, cmap='jet')
        axes[1].set_title('Prediction (Classes)')
    else:
        # Binary case
        axes[1].imshow(prediction.squeeze(), cmap='jet')
        axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Overlay
    if prediction.ndim == 3 and prediction.shape[0] > 1:
        overlay = overlay_mask_on_image(input_image.squeeze(), np.argmax(prediction, axis=0))
    else:
        overlay = overlay_mask_on_image(input_image.squeeze(), prediction.squeeze())
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main(input_path, output_dir, checkpoint_path=None):
    """Main inference function"""
    # Set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    print("Initializing model...")
    model = AttentionUNet(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        deep_supervision=False  # Disable during inference
    ).to(device)
    
    # Load trained model weights
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
        
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"No trained model found at {checkpoint_path}")
        print("Please train the model first or provide a valid checkpoint path.")
        return
    
    # Load input image
    print(f"Loading input from {input_path}")
    if input_path.endswith('.nii') or input_path.endswith('.nii.gz'):
        input_data = load_nifti_image(input_path)
    else:
        # Assume it's a regular image file
        input_data = plt.imread(input_path)
    
    # Preprocess
    input_tensor = preprocess_image(input_data)
    original_shape = input_data.shape[:2] if input_data.ndim > 2 else input_data.shape
    
    # Run inference
    print("Running inference...")
    prediction = run_inference(model, input_tensor, device)
    
    # Postprocess
    prediction_processed = postprocess_prediction(prediction, original_shape)
    
    # Generate output paths
    input_stem = Path(input_path).stem.split('.')[0]  # Remove extension
    pred_output_path = os.path.join(output_dir, f"{input_stem}_prediction.npy")
    viz_output_path = os.path.join(output_dir, f"{input_stem}_visualization.png")
    
    # Save prediction
    np.save(pred_output_path, prediction_processed)
    print(f"Prediction saved to {pred_output_path}")
    
    # Save visualization
    save_visualization(input_data, prediction_processed, viz_output_path)
    print(f"Visualization saved to {viz_output_path}")
    
    # Print prediction statistics
    print("\nPrediction Statistics:")
    print(f"Prediction shape: {prediction_processed.shape}")
    if prediction_processed.ndim == 3 and prediction_processed.shape[0] > 1:
        for c in range(prediction_processed.shape[0]):
            class_pixels = np.sum(np.argmax(prediction_processed, axis=0) == c)
            print(f"Class {c} pixels: {class_pixels}")
    else:
        pred_binary = (prediction_processed > 0.5).astype(float)
        print(f"Positive pixels: {np.sum(pred_binary):,}")
        print(f"Percentage: {np.mean(pred_binary)*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on medical images")
    parser.add_argument("input_path", help="Path to input image (NIfTI or regular image)")
    parser.add_argument("--output_dir", default="./inference_outputs", help="Output directory")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    main(args.input_path, args.output_dir, args.checkpoint)