#!/usr/bin/env python3
"""
yolo_to_coco.py - Convert YOLOv8 annotations to COCO format

This script converts a dataset with YOLOv8 annotations (*.txt files) to COCO JSON format.
It handles the conversion of bounding box coordinates and creates the required COCO structure.
Supports datasets with train/val splits and includes a supercategory.
If no splits are found, it automatically creates an 80:20 train/val split.
"""

import os
import json
import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import glob
from datetime import datetime
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("yolo_to_coco")


class YoloToCOCOConverter:
    """Convert YOLOv8 format annotations to COCO format."""

    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        class_file: Optional[str] = None,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        """
        Initialize the converter with paths to data directories.

        Args:
            dataset_dir: Root directory containing images and labels folders
            output_dir: Directory to save the COCO JSON outputs
            class_file: Optional path to a file containing class names (one per line)
            train_ratio: Ratio of data to use for training if auto-splitting (default: 0.8)
            seed: Random seed for reproducible splits
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.class_file = class_file
        self.train_ratio = train_ratio
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load class names if provided, otherwise use indices
        self.categories = self._load_categories()
        
    def _create_info(self) -> Dict[str, Any]:
        """Create the info section of COCO JSON."""
        return {
            "description": "Converted from YOLOv8 format",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "YOLOv8 to COCO converter",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _create_license(self) -> Dict[str, Any]:
        """Create a default license entry for COCO JSON."""
        return {
            "id": 1,
            "name": "Unknown License",
            "url": "",
        }

    def _load_categories(self) -> List[str]:
        """
        Load category names from file or create default numbered categories.
        
        Returns:
            List of category names
        """
        if self.class_file and os.path.exists(self.class_file):
            try:
                with open(self.class_file, "r") as f:
                    categories = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(categories)} categories from {self.class_file}")
                return categories
            except Exception as e:
                logger.error(f"Error loading class file: {e}")
                logger.info("Using default numbered categories")
        
        # If no class file or error loading, create default categories
        # Assuming 80 classes as in COCO, but this can be adjusted
        return [f"class_{i}" for i in range(80)]

    def _get_image_info(self, image_path: Path, image_id: int) -> Dict[str, Any]:
        """
        Extract image information needed for COCO format.
        
        Args:
            image_path: Path to the image file
            image_id: Unique ID for the image
            
        Returns:
            Dictionary with image information
        """
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            return {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "license": 1,
                "date_captured": "",
            }
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            # Return default values if image can't be processed
            return {
                "id": image_id,
                "file_name": image_path.name,
                "width": 0,
                "height": 0,
                "license": 1,
                "date_captured": "",
            }

    def _convert_bbox(
        self, x_center: float, y_center: float, width: float, height: float, img_width: int, img_height: int
    ) -> List[float]:
        """
        Convert YOLOv8 bbox format (x_center, y_center, width, height) to COCO format (x, y, width, height).
        
        Args:
            x_center: Normalized x center coordinate (0-1)
            y_center: Normalized y center coordinate (0-1)
            width: Normalized width (0-1)
            height: Normalized height (0-1)
            img_width: Width of the image in pixels
            img_height: Height of the image in pixels
            
        Returns:
            List of [x, y, width, height] in COCO format (absolute pixels)
        """
        # Convert normalized coordinates to absolute pixel values
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height
        
        # Convert center coordinates to top-left coordinates (COCO format)
        x = x_center_abs - (width_abs / 2)
        y = y_center_abs - (height_abs / 2)
        
        return [x, y, width_abs, height_abs]

    def _initialize_coco_data(self) -> Dict[str, Any]:
        """Initialize a new COCO data structure with supercategory."""
        coco_data = {
            "info": self._create_info(),
            "licenses": [self._create_license()],
            "images": [],
            "annotations": [],
            "categories": [],
        }
        
        # Add supercategory at index 0
        coco_data["categories"].append({
            "id": 0,
            "name": "objects",
            "supercategory": "none"
        })
        
        # Add other categories starting from index 1
        for idx, name in enumerate(self.categories):
            coco_data["categories"].append({
                "id": idx + 1,  # Shift all category IDs by 1
                "name": name,
                "supercategory": "objects"  # All categories belong to "objects" supercategory
            })
            
        return coco_data

    def _create_auto_split(self) -> Tuple[List[Path], List[Path]]:
        """
        Create an automatic train/val split if no explicit splits are found.
        
        Returns:
            Tuple of (train_image_files, val_image_files)
        """
        logger.info("No explicit train/val splits found. Creating automatic 80:20 split.")
        
        # Check for images directly in the images folder
        image_dir = self.dataset_dir / "images"
        label_dir = self.dataset_dir / "labels"
        
        if not image_dir.exists() or not label_dir.exists():
            # Try to find images and labels directly in the dataset directory
            image_dir = self.dataset_dir
            label_dir = self.dataset_dir
            
        # Get all image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        all_image_files = []
        for ext in image_extensions:
            all_image_files.extend([Path(p) for p in glob.glob(str(image_dir / ext))])
        
        if not all_image_files:
            logger.error(f"No image files found in {image_dir}")
            return [], []
        
        # Filter to only include images that have corresponding label files
        valid_image_files = []
        for img_path in all_image_files:
            base_name = img_path.stem
            label_file = label_dir / f"{base_name}.txt"
            if label_file.exists():
                valid_image_files.append(img_path)
            else:
                logger.warning(f"No label file found for {img_path.name}, skipping")
        
        if not valid_image_files:
            logger.error("No valid image files with corresponding labels found")
            return [], []
        
        # Shuffle the files for random split
        random.shuffle(valid_image_files)
        
        # Split into train and validation sets
        split_idx = int(len(valid_image_files) * self.train_ratio)
        train_files = valid_image_files[:split_idx]
        val_files = valid_image_files[split_idx:]
        
        logger.info(f"Auto-split created: {len(train_files)} training images, {len(val_files)} validation images")
        
        return train_files, val_files

    def _process_image_files(self, image_files: List[Path], label_dir: Path, split: str) -> Dict[str, Any]:
        """
        Process a list of image files and their corresponding labels.
        
        Args:
            image_files: List of image file paths
            label_dir: Directory containing label files
            split: Name of the split (train/val)
            
        Returns:
            COCO data dictionary
        """
        coco_data = self._initialize_coco_data()
        
        annotation_id = 0
        image_id = 0
        
        logger.info(f"Processing {len(image_files)} images for {split} split")
        
        for img_path in image_files:
            base_name = img_path.stem
            label_file = label_dir / f"{base_name}.txt"
            
            # Skip if no label file exists (should not happen as we filtered earlier)
            if not label_file.exists():
                logger.warning(f"No label file found for {img_path.name}, skipping")
                continue
            
            # Get image info
            img_info = self._get_image_info(img_path, image_id)
            if img_info["width"] == 0 or img_info["height"] == 0:
                logger.warning(f"Invalid image dimensions for {img_path.name}, skipping")
                continue
                
            coco_data["images"].append(img_info)
            
            # Process annotations
            try:
                with open(label_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split()
                        if len(parts) < 5:
                            logger.warning(f"Invalid annotation format in {label_file}: {line}")
                            continue
                            
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Skip invalid annotations
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                            logger.warning(f"Invalid bbox values in {label_file}: {line}")
                            continue
                            
                        # Convert bbox to COCO format
                        bbox = self._convert_bbox(
                            x_center, y_center, width, height, 
                            img_info["width"], img_info["height"]
                        )
                        
                        # Calculate area
                        area = bbox[2] * bbox[3]
                        
                        # Create annotation with adjusted category_id (+1 to account for supercategory)
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id + 1,  # Shift category ID by 1
                            "bbox": bbox,
                            "area": area,
                            "segmentation": [],
                            "iscrowd": 0,
                        }
                        
                        coco_data["annotations"].append(annotation)
                        annotation_id += 1
            except Exception as e:
                logger.error(f"Error processing label file {label_file}: {e}")
                
            image_id += 1
            
        return coco_data

    def convert_split(self, split: str) -> None:
        """
        Convert YOLOv8 annotations for a specific split (train/val) to COCO format.
        
        Args:
            split: Dataset split name (e.g., 'train', 'val')
        """
        image_dir = self.dataset_dir / "images" / split
        label_dir = self.dataset_dir / "labels" / split
        output_file = self.output_dir / f"{split}_coco.json"
        
        if not image_dir.exists() or not label_dir.exists():
            logger.error(f"Image or label directory for split '{split}' not found")
            return
            
        logger.info(f"Processing {split} split")
        
        # Get all image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend([Path(p) for p in glob.glob(str(image_dir / ext))])
        
        if not image_files:
            logger.error(f"No image files found in {image_dir}")
            return
        
        # Process the images and create COCO data
        coco_data = self._process_image_files(image_files, label_dir, split)
        
        # Save COCO JSON
        try:
            with open(output_file, "w") as f:
                json.dump(coco_data, f, indent=2)
            logger.info(f"Conversion complete for {split} split. COCO annotations saved to {output_file}")
            logger.info(f"Processed {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
        except Exception as e:
            logger.error(f"Error saving COCO JSON for {split} split: {e}")

    def convert_auto_split(self) -> None:
        """
        Create and convert an automatic train/val split.
        """
        # Create automatic split
        train_files, val_files = self._create_auto_split()
        
        if not train_files and not val_files:
            logger.error("Failed to create automatic split")
            return
        
        # Determine label directory
        label_dir = self.dataset_dir / "labels"
        if not label_dir.exists():
            label_dir = self.dataset_dir
        
        # Process train split
        train_coco = self._process_image_files(train_files, label_dir, "train")
        train_output = self.output_dir / "train_coco.json"
        
        # Process val split
        val_coco = self._process_image_files(val_files, label_dir, "val")
        val_output = self.output_dir / "val_coco.json"
        
        # Save COCO JSONs
        try:
            with open(train_output, "w") as f:
                json.dump(train_coco, f, indent=2)
            logger.info(f"Train split saved to {train_output}")
            logger.info(f"Processed {len(train_coco['images'])} images and {len(train_coco['annotations'])} annotations")
            
            with open(val_output, "w") as f:
                json.dump(val_coco, f, indent=2)
            logger.info(f"Validation split saved to {val_output}")
            logger.info(f"Processed {len(val_coco['images'])} images and {len(val_coco['annotations'])} annotations")
        except Exception as e:
            logger.error(f"Error saving COCO JSON files: {e}")

    def convert(self) -> None:
        """
        Convert all dataset splits to COCO format.
        If no explicit splits are found, create an automatic split.
        """
        # Check for train and val splits
        has_train = (self.dataset_dir / "images" / "train").exists() and (self.dataset_dir / "labels" / "train").exists()
        has_val = (self.dataset_dir / "images" / "val").exists() and (self.dataset_dir / "labels" / "val").exists()
        
        if has_train and has_val:
            # Process explicit splits
            logger.info("Found explicit train and val splits")
            self.convert_split("train")
            self.convert_split("val")
        else:
            # Create and process automatic split
            logger.info("No explicit splits found, creating automatic 80:20 split")
            self.convert_auto_split()


def main():
    """Parse arguments and run the converter."""
    parser = argparse.ArgumentParser(description="Convert YOLOv8 annotations to COCO format")
    parser.add_argument("--dataset-dir", required=True, help="Root directory containing images and labels folders")
    parser.add_argument("--output-dir", required=True, help="Directory to save COCO JSON files")
    parser.add_argument("--classes", help="Path to file with class names (one per line)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of data to use for training if auto-splitting (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits (default: 42)")
    
    args = parser.parse_args()
    
    converter = YoloToCOCOConverter(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        class_file=args.classes,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    
    converter.convert()


if __name__ == "__main__":
    main()