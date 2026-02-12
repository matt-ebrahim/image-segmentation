import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config

class HippocampusDataset(Dataset):
    def __init__(self, image_slices, mask_slices, transform=None):
        self.image_slices = image_slices
        self.mask_slices = mask_slices
        self.transform = transform

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]
        mask = self.mask_slices[idx]

        # Add channel dimension (H, W) -> (C, H, W)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            # Implement transforms here if needed (e.g., rotation, flip)
            pass

        # Convert to float tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask

def load_nifti_file(filepath):
    """Load a NIfTI file and return as numpy array."""
    img = nib.load(filepath).get_fdata()
    return img

def get_data_loaders():
    raw_data_dir = os.path.join(config.DATA_DIR, "Task04_Hippocampus")
    images_tr_dir = os.path.join(raw_data_dir, "imagesTr")
    labels_tr_dir = os.path.join(raw_data_dir, "labelsTr")

    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(
            f"Data not found at {raw_data_dir}. Please run data/download_data.py first."
        )

    # Get all file paths
    image_files = sorted(glob.glob(os.path.join(images_tr_dir, "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(labels_tr_dir, "*.nii.gz")))

    all_image_slices = []
    all_mask_slices = []

    print("Processing 3D volumes into 2D slices...")
    for img_path, lbl_path in zip(image_files, label_files):
        vol_img = load_nifti_file(img_path)
        vol_mask = load_nifti_file(lbl_path)

        # Normalize image volume to 0-1
        vol_img = (vol_img - np.min(vol_img)) / (np.max(vol_img) - np.min(vol_img) + 1e-8)

        # We iterate through the depth dimension (axis 2 usually)
        # Hippocampus volumes are small, we take the middle slices where the organ is
        depth = vol_img.shape[2]
        start_slice = max(0, depth // 4)
        end_slice = min(depth, 3 * depth // 4)

        for i in range(start_slice, end_slice):
            img_slice = vol_img[:, :, i]
            mask_slice = vol_mask[:, :, i]

            # Only keep slices that actually contain some of the hippocampus to avoid empty training
            if np.sum(mask_slice) > 0:
                all_image_slices.append(img_slice)
                all_mask_slices.append(mask_slice)

    # Convert lists to numpy arrays
    all_image_slices = np.array(all_image_slices)
    all_mask_slices = np.array(all_mask_slices)

    # Binarize mask
    all_mask_slices = (all_mask_slices > 0).astype(np.float32)

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        all_image_slices, all_mask_slices, test_size=1-config.TRAIN_SPLIT, random_state=42
    )

    train_dataset = HippocampusDataset(X_train, y_train)
    val_dataset = HippocampusDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    return train_loader, val_loader
