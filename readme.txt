# Modified DETR for Helmet Detection

This repository contains a modified version of DETR (DEtection TRansformer) specifically designed for motorcycle helmet detection with 5 classes. The architecture replaces the original transformer encoder with a dilated convolutional encoder and uses focal loss for classification.

## Model Architecture

![Modified DETR Architecture](model_architecture.png)

Key modifications from the original DETR:
1. **Backbone**: ResNet50/ResNet101 with output features of shape 2048×H/4×W/4
2. **Dilated Convolutional Encoder**: Uses multiple dilation factors (2, 4, 6, 8) instead of a transformer encoder
3. **Transformer Decoder**: Maintains the original transformer decoder but with a reduced number of queries (20)
4. **Classification Head**: Outputs 5+1 classes (5 helmet classes + background)
5. **Loss Function**: Uses focal loss for classification to handle class imbalance

## Classes

The model detects the following 5 classes:
1. **rider_with_helmet**: Single rider wearing a helmet
2. **rider_without_helmet**: Single rider not wearing a helmet
3. **rider_and_passenger_with_helmet**: Both rider and passenger wearing helmets
4. **rider_and_passenger_without_helmet**: Neither rider nor passenger wearing a helmet
5. **rider_with_helmet_and_passenger_without_helmet**: Rider wearing a helmet but passenger not wearing a helmet

## Repository Structure

```
.
├── backbone.py                 # Backbone network implementation
├── dilated_encoder.py          # Dilated convolutional encoder implementation
├── helmet_detr.py              # Main model implementation
├── inference.py                # Script for running inference
├── loss.py                     # Implementation of focal loss and other losses
├── mlp.py                      # MLP implementation for prediction heads
├── positional_encoding.py      # Positional encoding implementation
├── README.md                   # This README file
├── train.py                    # Training script
└── transformer_decoder.py      # Transformer decoder implementation
```

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- Pillow
- matplotlib
- numpy

Install requirements with:
```bash
pip install torch torchvision pillow matplotlib numpy
```

## Dataset Preparation

The model expects data in COCO format. Prepare your dataset as follows:

```
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── val/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    ├── train.json
    └── val.json
```

The annotation file should follow the COCO format with the 5 classes defined above.

## Training

To train the model, run:

```bash
python train.py --data_path ./data --train_ann_file annotations/train.json --val_ann_file annotations/val.json --output_dir ./output
```

### Training Arguments

- `--data_path`: