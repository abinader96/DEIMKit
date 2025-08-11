#!/usr/bin/env python
"""
Azure ML Training Script for DEIMKit

This script provides a comprehensive training pipeline for DEIM models
with MLflow integration and early stopping support, designed for Azure ML.

Usage:
    python azure_train.py \
        --dataset-path /path/to/dataset \
        --model-name deim_hgnetv2_s \
        --epochs 100 \
        --patience 15 \
        --batch-size 16
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
# sys.path.append(str(Path(__file__).parent.parent))

from deimkit import (
    Config, 
    Trainer, 
    configure_dataset, 
    configure_model,
    MLflowCallback
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DEIM models with MLflow tracking and early stopping"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="deim_hgnetv2_s",
        choices=["deim_hgnetv2_n", "deim_hgnetv2_s", "deim_hgnetv2_m", 
                 "deim_hgnetv2_l", "deim_hgnetv2_x"],
        help="Model architecture to use"
    )
    parser.add_argument(
        "--pretrained", 
        action="store_true", 
        default=True,
        help="Use pretrained weights"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--train-ann-file", 
        type=str, 
        required=True,
        help="Path to training annotations file (COCO format JSON)"
    )
    parser.add_argument(
        "--train-img-folder", 
        type=str, 
        required=True,
        help="Path to training images folder"
    )
    parser.add_argument(
        "--val-ann-file", 
        type=str, 
        required=True,
        help="Path to validation annotations file (COCO format JSON)"
    )
    parser.add_argument(
        "--val-img-folder", 
        type=str, 
        required=True,
        help="Path to validation images folder"
    )
    parser.add_argument(
        "--image-size", 
        type=int, 
        nargs=2, 
        default=[640, 640],
        help="Input image size as height and width (e.g., --image-size 640 640 for 640x640 images)"
    )
    parser.add_argument(
        "--num-classes", 
        type=int, 
        required=True,
        help="Number of classes in dataset (excluding background)"
    )
    
    # Training configuration
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./outputs",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16,
        help="Training batch size per GPU"
    )
    parser.add_argument(
        "--val-batch-size", 
        type=int, 
        default=None,
        help="Validation batch size (defaults to batch-size)"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=4,
        help="Number of data loading workers"
    )
    
    # Optimization configuration
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.0001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--lr-gamma", 
        type=float, 
        default=0.5,
        help="Learning rate annealing factor"
    )
    parser.add_argument(
        "--weight-decay", 
        type=float, 
        default=0.0001,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--warmup-epochs", 
        type=int, 
        default=None,
        help="Number of warmup epochs (auto-calculated if not set)"
    )
    
    # Early stopping configuration
    parser.add_argument(
        "--patience", 
        type=int, 
        default=None,
        help="Early stopping patience (disabled if not set)"
    )
    
    # MLflow configuration
    parser.add_argument(
        "--mlflow-experiment", 
        type=str, 
        default="deim-object-detection",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--mlflow-run-name", 
        type=str, 
        default=None,
        help="MLflow run name (auto-generated if not set)"
    )
    parser.add_argument(
        "--no-mlflow", 
        action="store_true",
        help="Disable MLflow tracking"
    )
    
    # Other configuration
    parser.add_argument(
        "--save-best-only", 
        action="store_true",
        help="Only save best model checkpoint"
    )
    parser.add_argument(
        "--num-queries", 
        type=int, 
        default=300,
        help="Number of object queries"
    )
    parser.add_argument(
        "--freeze-at", 
        type=int, 
        default=-1,
        help="Freeze backbone at this stage (-1 for no freezing)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def setup_azure_ml_environment():
    """Setup Azure ML specific environment variables and paths."""
    # Azure ML sets these environment variables
    if "AZUREML_RUN_ID" in os.environ:
        print("Running in Azure ML environment")
        
        # Set MLflow tracking URI (handled by azureml-mlflow)
        # No need to set explicitly when using azureml-mlflow
        
        # Update output directory to Azure ML outputs
        if os.path.exists("/mnt/azureml/outputs"):
            return "/mnt/azureml/outputs"
    
    return None


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup Azure ML environment if running in Azure
    azure_output_dir = setup_azure_ml_environment()
    if azure_output_dir:
        args.output_dir = azure_output_dir
    
    # Set random seed for reproducibility
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create configuration
    print(f"Loading model configuration: {args.model_name}")
    conf = Config.from_model_name(args.model_name)
    
    # Configure model
    conf = configure_model(
        conf, 
        num_queries=args.num_queries,
        freeze_at=args.freeze_at,
        pretrained=args.pretrained
    )
    
    # Configure dataset
    train_ann_file = Path(args.train_ann_file)
    train_img_folder = Path(args.train_img_folder)
    val_ann_file = Path(args.val_ann_file)
    val_img_folder = Path(args.val_img_folder)
    
    # Validate dataset paths
    if not train_ann_file.exists():
        raise FileNotFoundError(f"Training annotations not found: {train_ann_file}")
    if not val_ann_file.exists():
        raise FileNotFoundError(f"Validation annotations not found: {val_ann_file}")
    if not train_img_folder.exists():
        raise FileNotFoundError(f"Training images folder not found: {train_img_folder}")
    if not val_img_folder.exists():
        raise FileNotFoundError(f"Validation images folder not found: {val_img_folder}")
    
    print(f"Configuring dataset:")
    print(f"  Training annotations: {train_ann_file}")
    print(f"  Training images: {train_img_folder}")
    print(f"  Validation annotations: {val_ann_file}")
    print(f"  Validation images: {val_img_folder}")
    print(f"  Image size: {args.image_size[0]}x{args.image_size[1]}")
    print(f"  Number of classes: {args.num_classes} (+ 1 background)")
    
    conf = configure_dataset(
        config=conf,
        image_size=tuple(args.image_size),
        train_ann_file=str(train_ann_file),
        train_img_folder=str(train_img_folder),
        val_ann_file=str(val_ann_file),
        val_img_folder=str(val_img_folder),
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size or args.batch_size,
        num_classes=args.num_classes + 1,  # Add 1 for background class
        output_dir=args.output_dir
    )
    
    # Set num_workers in the config
    conf.yaml_cfg["train_dataloader"]["num_workers"] = args.num_workers
    conf.yaml_cfg["val_dataloader"]["num_workers"] = args.num_workers
    
    # Setup callbacks
    callbacks = []
    
    if not args.no_mlflow:
        print(f"Setting up MLflow tracking: {args.mlflow_experiment}")
        mlflow_callback = MLflowCallback(
            experiment_name=args.mlflow_experiment,
            run_name=args.mlflow_run_name,
            log_models=True
        )
        callbacks.append(mlflow_callback)
    
    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(conf, callbacks=callbacks)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    print("\nStarting training...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Early stopping patience: {args.patience if args.patience else 'Disabled'}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)
    
    trainer.fit(
        epochs=args.epochs,
        patience=args.patience,
        lr=args.learning_rate,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        warmup_iter=args.warmup_epochs,
        save_best_only=args.save_best_only
    )
    
    print("\nTraining completed!")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_stats = trainer.evaluate()
    
    if 'coco_eval_bbox' in eval_stats:
        mAP = eval_stats['coco_eval_bbox'][0]
        print(f"Final mAP@0.50:0.95: {mAP:.4f}")
    
    # Export model to ONNX if requested (useful for deployment)
    if os.environ.get("EXPORT_ONNX", "false").lower() == "true":
        print("\nExporting model to ONNX...")
        from deimkit import Exporter
        
        exporter = Exporter(conf)
        onnx_path = Path(args.output_dir) / "model.onnx"
        
        # Use best checkpoint if available
        best_checkpoint = Path(args.output_dir) / "best.pth"
        if best_checkpoint.exists():
            exporter.to_onnx(
                checkpoint_path=str(best_checkpoint),
                output_path=str(onnx_path)
            )
            print(f"Model exported to: {onnx_path}")


if __name__ == "__main__":
    main()
