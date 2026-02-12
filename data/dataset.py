import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import cv2
import random
import config
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MedicalImageSegmentationDataset(Dataset):
    """Dataset class for medical image segmentation with data augmentation"""
    
    def __init__(self, image_paths, mask_paths, transform=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment
        
        # Define augmentation pipeline
        if self.augment:
            self.augmentation_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load NIfTI files
        image_volume = nib.load(image_path).get_fdata()
        mask_volume = nib.load(mask_path).get_fdata()
        
        # Handle 3D volumes - take middle slice
        depth = image_volume.shape[2]
        middle_slice = depth // 2
        
        # Extract middle slice
        image = image_volume[:, :, middle_slice]
        mask = mask_volume[:, :, middle_slice]
        
        # Normalize image to 0-1 range
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        
        # Apply augmentation if needed
        if self.augment:
            augmented = self.augmentation_pipeline(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Add channel dimension for PyTorch (H, W) -> (C, H, W)
        image = np.expand_dims(image, axis=0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)
        
        return image_tensor, mask_tensor

def load_nifti_file(filepath):
    """Load a NIfTI file and return as numpy array."""
    img = nib.load(filepath).get_fdata()
    return img

def preprocess_data():
    """Preprocess and split data into train/val/test sets."""
    print("Loading and preprocessing data...")
    
    # Get all file paths
    image_files = sorted(glob.glob(os.path.join(config.IMAGES_TR_DIR, "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(config.LABELS_TR_DIR, "*.nii.gz")))
    
    print(f"Found {len(image_files)} image files and {len(label_files)} label files")
    
    # Split data into train/val/test
    # First split into train and temp (val+test)
    train_img, temp_img, train_lbl, temp_lbl = train_test_split(
        image_files, label_files, 
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        random_state=42
    )
    
    # Split temp into val and test
    val_img, test_img, val_lbl, test_lbl = train_test_split(
        temp_img, temp_lbl,
        test_size=config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
        random_state=42
    )
    
    print(f"Train: {len(train_img)}, Val: {len(val_img)}, Test: {len(test_img)}")
    
    return (train_img, train_lbl), (val_img, val_lbl), (test_img, test_lbl)

def get_data_loaders():
    """Create data loaders for train/validation/test sets."""
    # Preprocess data
    (train_img, train_lbl), (val_img, val_lbl), (test_img, test_lbl) = preprocess_data()
    
    # Create datasets
    train_dataset = MedicalImageSegmentationDataset(
        train_img, train_lbl, 
        augment=True
    )
    
    val_dataset = MedicalImageSegmentationDataset(
        val_img, val_lbl, 
        augment=False
    )
    
    test_dataset = MedicalImageSegmentationDataset(
        test_img, test_lbl, 
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# Test the dataset loading
if __name__ == "__main__":
    # Create required directories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Try to create dummy data for testing (if real data isn't available)
    if not os.path.exists(config.RAW_DATA_DIR):
        print("Warning: Real data not found. Creating dummy data for testing...")
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(config.IMAGES_TR_DIR, exist_ok=True)
        os.makedirs(config.LABELS_TR_DIR, exist_ok=True)
        
        # Create dummy files (just to test if the code runs)
        dummy_img_path = os.path.join(config.IMAGES_TR_DIR, "lung_001.nii.gz")
        dummy_lbl_path = os.path.join(config.LABELS_TR_DIR, "lung_001.nii.gz")
        
        if not os.path.exists(dummy_img_path):
            # Create dummy nibabel objects
            dummy_img = np.random.rand(256, 256, 30).astype(np.float32)
            dummy_lbl = (np.random.rand(256, 256, 30) > 0.8).astype(np.float32)
            
            nib.save(nib.Nifti1Image(dummy_img, np.eye(4)), dummy_img_path)
            nib.save(nib.Nifti1Image(dummy_lbl, np.eye(4)), dummy_lbl_path)
    
    # Try to load data
    try:
        train_loader, val_loader, test_loader = get_data_loaders()
        print("Data loaders created successfully!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test loading a batch
        for batch_idx, (images, masks) in enumerate(train_loader):
            print(f"Sample batch - Images shape: {images.shape}, Masks shape: {masks.shape}")
            break
            
    except Exception as e:
        print(f"Error loading data: {e}")