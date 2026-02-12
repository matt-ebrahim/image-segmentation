# Medical Image Segmentation with Attention U-Net

This project implements a medical image segmentation system using Attention U-Net for segmenting lung tumors in CT scans from the Medical Segmentation Decathlon dataset.

## Features

- **Attention U-Net Architecture**: Implements attention mechanisms for better feature focusing
- **Deep Supervision**: Multiple supervision signals during training for improved performance
- **Mixed Precision Training**: Efficient training using automatic mixed precision (AMP)
- **Comprehensive Data Augmentation**: Random flips, rotations, and elastic deformations
- **Combined Loss Function**: Dice + Cross-Entropy loss for robust training
- **Advanced Metrics**: Dice score, IoU, Hausdorff distance, and confusion matrix
- **Visualization**: Automatic generation of comparison plots
- **Inference Pipeline**: Easy-to-use inference script with overlay generation

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- torch>=1.9.0
- torchvision>=0.10.0
- numpy>=1.19.0
- nibabel>=3.0.0
- tqdm>=4.60.0
- scikit-learn>=0.24.0
- scikit-image>=0.18.0
- matplotlib>=3.3.0
- pandas>=1.3.0
- tensorboard>=2.5.0
- opencv-python>=4.5.0
- requests>=2.25.0
- albumentations>=1.0.0
- monai>=0.7.0

## Project Structure

```
medical-imseg-attention-unet/
├── config.py               # Configuration file with hyperparameters
├── requirements.txt        # Python dependencies
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── inference.py            # Inference script
├── data/
│   ├── download_data.py    # Script to download MSD Lung dataset
│   └── dataset.py          # Dataset class with augmentation
├── model/
│   └── unet.py             # Attention U-Net implementation
├── checkpoints/            # Saved model checkpoints
├── logs/                   # Training logs for TensorBoard
└── visualizations/         # Generated visualizations
```

## Usage

### 1. Download Data

```bash
python data/download_data.py
```

This downloads and extracts the Medical Segmentation Decathlon Task06_Lung dataset.

### 2. Training

```bash
python train.py
```

The training script will:
- Automatically split data into train/val/test sets
- Use mixed precision training for efficiency
- Apply data augmentation during training
- Save model checkpoints based on validation Dice score
- Log metrics to TensorBoard

Training configuration can be adjusted in `config.py`.

### 3. Evaluation

```bash
python evaluate.py
```

This evaluates the best model on the test set and generates:
- Dice score, IoU, Hausdorff distance metrics
- Confusion matrix visualization
- Sample visualizations comparing input, ground truth, and predictions

### 4. Inference

```bash
python inference.py /path/to/input/image.nii.gz --output_dir ./results
```

For NIfTI files:
```bash
python inference.py /path/to/Task06_Lung/imagesTs/lung_001.nii.gz
```

Results will be saved to `./inference_outputs/` by default with:
- Raw prediction masks as `.npy` files
- Visualization overlays as `.png` files

## Configuration

Key parameters can be modified in `config.py`:

- `BATCH_SIZE`: Training batch size (default: 4)
- `LEARNING_RATE`: Initial learning rate (default: 1e-3)
- `NUM_EPOCHS`: Maximum training epochs (default: 100)
- `BASE_CHANNEL_COUNT`: Base number of channels in U-Net (default: 32)
- `USE_AMP`: Enable/disable mixed precision (default: True)
- `AUGMENTATION_PROBABILITY`: Probability of applying augmentations (default: 0.5)

## Model Architecture

The Attention U-Net implementation includes:

1. **Encoder-Decoder Structure**: Standard U-Net architecture with symmetric encoder-decoder
2. **Attention Gates**: Attention blocks that focus on relevant features during upsampling
3. **Residual Connections**: Skip connections with residual learning for stable training
4. **Deep Supervision**: Auxiliary outputs at multiple decoder levels during training
5. **Batch Normalization**: For stable training dynamics

## Monitoring Training

Start TensorBoard to monitor training progress:

```bash
tensorboard --logdir logs
```

Metrics logged:
- Training/validation loss
- Dice scores
- Learning rate
- Custom metrics

## Results

Expected performance on MSD Lung dataset:
- Dice Score: ~0.75-0.85
- IoU: ~0.65-0.75
- Hausdorff Distance: Variable based on resolution

Actual results may vary based on training time and hyperparameters.

## Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**: Reduce `BATCH_SIZE` in `config.py`
2. **Data Loading Errors**: Ensure dataset is properly downloaded and extracted
3. **Module Import Errors**: Install missing dependencies from `requirements.txt`

## Citation

If you use this code in your research, please cite:

```
@article{oktay2018attention,
  title={Attention u-net: Learning where to look for the pancreas},
  author={Oktay, Ozan and Schlemper, Jo and Le Folgoc, Loic and Lee, Matthew and Heinrich, Mattias and Misawa, Kazuharu and Mori, Kensaku and McDonagh, Steven and Hammerla, Nils Y and Kainz, Bernhard and others},
  journal={arXiv preprint arXiv:1804.03999},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.