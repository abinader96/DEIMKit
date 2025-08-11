# Training DEIMKit with Pretrained Weights on Custom Datasets

This guide explains how to fine-tune DEIMKit models using pretrained weights on your custom object detection dataset.

## Prerequisites

1. **Install DEIMKit and dependencies**:
```bash
pip install mlflow  # For experiment tracking
```

2. **Prepare your dataset in COCO format**:
   - Training images in a folder
   - Validation images in a folder  
   - COCO format annotation JSON files for both train and validation sets

## Dataset Preparation

Your dataset structure should look like this:
```
my_dataset/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── valid/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── train_annotations.json  # COCO format
└── valid_annotations.json   # COCO format
```

### COCO Format Requirements
Your annotation files should follow the COCO format with:
- `images`: List of image metadata
- `annotations`: List of bounding box annotations
- `categories`: List of object classes

## Basic Training Command

Here's a basic example to train with pretrained weights:

```bash
python scripts/azure_train.py \
    --train-ann-file /path/to/my_dataset/train_annotations.json \
    --train-img-folder /path/to/my_dataset/train \
    --val-ann-file /path/to/my_dataset/valid_annotations.json \
    --val-img-folder /path/to/my_dataset/valid \
    --model-name deim_hgnetv2_s \
    --pretrained \
    --num-classes 10 \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 0.0001 \
    --output-dir ./outputs/my_custom_model
```

## Key Parameters Explained

### Model Selection
- `--model-name`: Choose from:
  - `deim_hgnetv2_n`: Nano model (fastest, least accurate)
  - `deim_hgnetv2_s`: Small model (good balance)
  - `deim_hgnetv2_m`: Medium model
  - `deim_hgnetv2_l`: Large model
  - `deim_hgnetv2_x`: Extra large model (slowest, most accurate)

### Dataset Configuration
- `--num-classes`: Number of object classes in your dataset (excluding background)
- `--image-size`: Input image dimensions, e.g., `--image-size 640 640` for 640x640

### Training Configuration
- `--pretrained`: Use this flag to load pretrained COCO weights
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size per GPU (reduce if you run out of memory)
- `--learning-rate`: Initial learning rate (0.0001 is a good starting point)

## Advanced Training Example

Here's a more advanced example with all the bells and whistles:

```bash
python scripts/azure_train.py \
    --train-ann-file /data/custom_dataset/annotations/train.json \
    --train-img-folder /data/custom_dataset/images/train \
    --val-ann-file /data/custom_dataset/annotations/val.json \
    --val-img-folder /data/custom_dataset/images/val \
    --model-name deim_hgnetv2_m \
    --pretrained \
    --num-classes 5 \
    --image-size 800 800 \
    --epochs 100 \
    --batch-size 16 \
    --val-batch-size 32 \
    --learning-rate 0.00005 \
    --lr-gamma 0.1 \
    --weight-decay 0.0001 \
    --patience 15 \
    --num-workers 8 \
    --num-queries 300 \
    --freeze-at 2 \
    --mlflow-experiment "custom-object-detection" \
    --mlflow-run-name "deim-m-800x800-frozen-backbone" \
    --save-best-only \
    --output-dir ./outputs/custom_model_v2
```

### Advanced Parameters:
- `--patience 15`: Stop training if mAP doesn't improve for 15 epochs
- `--freeze-at 2`: Freeze backbone up to stage 2 (useful for small datasets)
- `--num-queries 300`: Number of object queries (increase for dense scenes)
- `--lr-gamma 0.1`: Learning rate decay factor
- `--save-best-only`: Only save the best checkpoint (saves disk space)

## Resume Training from Checkpoint

To resume training from a previous run:

```bash
python scripts/azure_train.py \
    --train-ann-file /path/to/train.json \
    --train-img-folder /path/to/train/images \
    --val-ann-file /path/to/val.json \
    --val-img-folder /path/to/val/images \
    --model-name deim_hgnetv2_s \
    --checkpoint ./outputs/previous_run/checkpoint0050.pth \
    --num-classes 10 \
    --epochs 100 \
    --output-dir ./outputs/resumed_training
```

## Tips for Custom Datasets

### 1. Start with a Smaller Model
For initial experiments, use `deim_hgnetv2_n` or `deim_hgnetv2_s`:
```bash
--model-name deim_hgnetv2_n --batch-size 16
```

### 2. Adjust Learning Rate
- For small datasets (< 1000 images): Use lower learning rate `--learning-rate 0.00001`
- For large datasets (> 10000 images): Can use higher learning rate `--learning-rate 0.0001`

### 3. Freeze Backbone for Small Datasets
If you have < 1000 images, freeze early backbone layers:
```bash
--freeze-at 3  # Freezes first 3 stages of backbone
```

### 4. Use Early Stopping
Prevent overfitting with early stopping:
```bash
--patience 20  # Stop if no improvement for 20 epochs
```

### 5. Adjust Batch Size for GPU Memory
- 8GB GPU: `--batch-size 4` or `--batch-size 8`
- 16GB GPU: `--batch-size 16`
- 24GB+ GPU: `--batch-size 32` or higher

### 6. Image Size Considerations
- Smaller images (416x416): Faster training, lower accuracy
- Medium images (640x640): Good balance (default)
- Larger images (800x800, 1024x1024): Slower training, higher accuracy

## Monitoring Training with MLflow

The script automatically logs to MLflow. To view the results:

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

You'll see:
- Training/validation loss curves
- mAP metrics over time
- Model hyperparameters
- Best model checkpoints

## Azure ML Integration

For Azure ML deployment, the script automatically detects the Azure ML environment:

```bash
# The script will automatically:
# 1. Use Azure ML's MLflow tracking
# 2. Save outputs to Azure ML's output directory
# 3. Register models in Azure ML model registry
```

## Common Issues and Solutions

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch-size 4

# Use smaller image size
--image-size 512 512

# Use gradient accumulation (if implemented)
```

### Slow Training
```bash
# Increase batch size (if GPU memory allows)
--batch-size 32

# Use more workers for data loading
--num-workers 16

# Use smaller model
--model-name deim_hgnetv2_n
```

### Poor Performance on Custom Dataset
```bash
# Use pretrained weights
--pretrained

# Freeze fewer layers
--freeze-at 1  # or --freeze-at -1 for no freezing

# Train longer
--epochs 200

# Adjust learning rate
--learning-rate 0.00005
```

## Example: Training on a 3-Class Dataset

Let's say you have a custom dataset with 3 classes: "car", "person", "bicycle"

```bash
python scripts/azure_train.py \
    --train-ann-file ./my_data/train.json \
    --train-img-folder ./my_data/train_images \
    --val-ann-file ./my_data/val.json \
    --val-img-folder ./my_data/val_images \
    --model-name deim_hgnetv2_s \
    --pretrained \
    --num-classes 3 \
    --image-size 640 640 \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 0.00005 \
    --patience 10 \
    --freeze-at 2 \
    --mlflow-experiment "3-class-detection" \
    --output-dir ./outputs/3_class_model
```

## Next Steps

After training:
1. Find your best model at `./outputs/your_model/best.pth`
2. Use it for inference with the Predictor class
3. Export to ONNX for deployment: Set `EXPORT_ONNX=true` environment variable
4. Evaluate on test set using the evaluation scripts

## Additional Resources

- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
