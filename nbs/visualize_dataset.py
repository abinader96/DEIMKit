#!/usr/bin/env python3
"""
visualize_annotations.py - Visualize bounding box annotations from COCO JSON files

This script randomly selects images from a dataset and displays them with their
bounding box annotations overlaid. It supports both COCO JSON format and YOLO format.
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import cv2
import numpy as np
import glob
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("visualize_annotations")


class AnnotationVisualizer:
    """Visualize bounding box annotations on images."""

    def __init__(
        self,
        dataset_dir: str,
        annotation_path: Optional[str] = None,
        format_type: str = "coco",
        class_file: Optional[str] = None,
    ):
        """
        Initialize the visualizer with paths to data.

        Args:
            dataset_dir: Directory containing the images
            annotation_path: Path to COCO JSON file (if format_type is 'coco')
            format_type: 'coco' or 'yolo'
            class_file: Optional path to a file containing class names (one per line)
        """
        self.dataset_dir = Path(dataset_dir)
        self.annotation_path = Path(annotation_path) if annotation_path else None
        self.format_type = format_type.lower()
        self.class_file = class_file
        
        # Validate format type
        if self.format_type not in ["coco", "yolo"]:
            raise ValueError("format_type must be either 'coco' or 'yolo'")
        
        # Load class names if provided
        self.categories = self._load_categories()
        
        # Generate random colors for each class
        random.seed(42)  # For reproducible colors
        self.colors = self._generate_colors(len(self.categories) + 1)  # +1 for supercategory
        
        # Load annotations if COCO format
        if self.format_type == "coco" and self.annotation_path:
            self.coco_data = self._load_coco_annotations()
        else:
            self.coco_data = None

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
        return [f"class_{i}" for i in range(80)]

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """
        Generate n distinct colors for visualization.
        
        Args:
            n: Number of colors to generate
            
        Returns:
            List of BGR color tuples
        """
        colors = []
        for i in range(n):
            # Generate vibrant colors with good contrast
            hue = i / n * 360
            saturation = 0.8 + random.random() * 0.2  # 0.8-1.0
            value = 0.8 + random.random() * 0.2  # 0.8-1.0
            
            # Convert HSV to RGB
            h = hue / 60
            c = value * saturation
            x = c * (1 - abs(h % 2 - 1))
            m = value - c
            
            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            # Convert to 0-255 range and BGR format (for OpenCV)
            bgr = (
                int((b + m) * 255),
                int((g + m) * 255),
                int((r + m) * 255)
            )
            colors.append(bgr)
        
        return colors

    def _load_coco_annotations(self) -> Dict[str, Any]:
        """
        Load COCO annotations from JSON file.
        
        Returns:
            COCO data dictionary
        """
        try:
            with open(self.annotation_path, "r") as f:
                coco_data = json.load(f)
            logger.info(f"Loaded COCO annotations with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
            return coco_data
        except Exception as e:
            logger.error(f"Error loading COCO annotations: {e}")
            return None

    def _get_random_image_coco(self) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        """
        Get a random image and its annotations from COCO data.
        
        Returns:
            Tuple of (image_path, annotations)
        """
        if not self.coco_data:
            logger.error("No COCO data loaded")
            return None, None
        
        # Select a random image
        if not self.coco_data["images"]:
            logger.error("No images found in COCO data")
            return None, None
            
        random_image = random.choice(self.coco_data["images"])
        image_id = random_image["id"]
        image_filename = random_image["file_name"]
        
        # Find the image file
        image_path = None
        for root, _, files in os.walk(self.dataset_dir):
            if image_filename in files:
                image_path = os.path.join(root, image_filename)
                break
        
        if not image_path:
            logger.error(f"Image file {image_filename} not found in {self.dataset_dir}")
            return None, None
        
        # Get annotations for this image
        annotations = [
            ann for ann in self.coco_data["annotations"] 
            if ann["image_id"] == image_id
        ]
        
        return image_path, annotations

    def _get_random_image_yolo(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get a random image and its annotation file path from YOLO format dataset.
        
        Returns:
            Tuple of (image_path, label_path)
        """
        # Find all image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_files = []
        
        # Check if we have a train/val structure
        image_dirs = [self.dataset_dir / "images" / "train", self.dataset_dir / "images" / "val"]
        if not any(d.exists() for d in image_dirs):
            # Try images directory
            image_dirs = [self.dataset_dir / "images"]
            if not any(d.exists() for d in image_dirs):
                # Try root directory
                image_dirs = [self.dataset_dir]
        
        # Collect all image files
        for image_dir in image_dirs:
            if image_dir.exists():
                for ext in image_extensions:
                    image_files.extend(glob.glob(str(image_dir / ext)))
        
        if not image_files:
            logger.error(f"No image files found in {self.dataset_dir}")
            return None, None
        
        # Select a random image
        random_image_path = random.choice(image_files)
        random_image_path = Path(random_image_path)
        
        # Find corresponding label file
        base_name = random_image_path.stem
        label_dirs = [
            self.dataset_dir / "labels" / "train",
            self.dataset_dir / "labels" / "val",
            self.dataset_dir / "labels",
            random_image_path.parent.parent / "labels" / random_image_path.parent.name,
            self.dataset_dir
        ]
        
        label_path = None
        for label_dir in label_dirs:
            potential_label = label_dir / f"{base_name}.txt"
            if potential_label.exists():
                label_path = potential_label
                break
        
        if not label_path:
            logger.error(f"No label file found for {random_image_path}")
            return str(random_image_path), None
        
        return str(random_image_path), str(label_path)

    def _parse_yolo_annotations(self, label_path: str, img_width: int, img_height: int) -> List[Dict[str, Any]]:
        """
        Parse YOLO format annotations.
        
        Args:
            label_path: Path to YOLO label file
            img_width: Width of the image
            img_height: Height of the image
            
        Returns:
            List of annotation dictionaries in COCO-like format
        """
        annotations = []
        
        try:
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) < 5:
                        logger.warning(f"Invalid annotation format in {label_path}: {line}")
                        continue
                        
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Skip invalid annotations
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                        logger.warning(f"Invalid bbox values in {label_path}: {line}")
                        continue
                    
                    # Convert to COCO format (x, y, width, height)
                    x = (x_center - width/2) * img_width
                    y = (y_center - height/2) * img_height
                    w = width * img_width
                    h = height * img_height
                    
                    annotations.append({
                        "category_id": class_id + 1,  # +1 to match COCO format with supercategory
                        "bbox": [x, y, w, h]
                    })
        except Exception as e:
            logger.error(f"Error parsing YOLO annotations from {label_path}: {e}")
        
        return annotations

    def _draw_annotations(
        self, 
        image: np.ndarray, 
        annotations: List[Dict[str, Any]],
        category_mapping: Optional[Dict[int, str]] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image: OpenCV image array
            annotations: List of annotation dictionaries
            category_mapping: Optional mapping from category_id to category name
            
        Returns:
            Image with annotations drawn
        """
        annotated_image = image.copy()
        
        for ann in annotations:
            # Get bbox and category
            bbox = ann["bbox"]
            category_id = ann["category_id"]
            
            # Convert bbox to integers
            x, y, w, h = [int(coord) for coord in bbox]
            
            # Get color for this category
            color = self.colors[category_id % len(self.colors)]
            
            # Draw rectangle
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
            
            # Get category name
            if category_mapping and category_id in category_mapping:
                category_name = category_mapping[category_id]
            elif category_id - 1 < len(self.categories):  # Adjust for supercategory offset
                category_name = self.categories[category_id - 1]
            else:
                category_name = f"class_{category_id}"
            
            # Draw label background
            text_size, _ = cv2.getTextSize(category_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                annotated_image, 
                (x, y - text_size[1] - 5), 
                (x + text_size[0], y), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_image, 
                category_name, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
        
        return annotated_image

    def visualize_random_image(self) -> None:
        """
        Select a random image, draw its annotations, and display it.
        """
        if self.format_type == "coco":
            image_path, annotations = self._get_random_image_coco()
            
            if not image_path or not annotations:
                logger.error("Failed to get random image and annotations")
                return
            
            # Create category mapping
            category_mapping = {}
            if self.coco_data and "categories" in self.coco_data:
                for cat in self.coco_data["categories"]:
                    category_mapping[cat["id"]] = cat["name"]
        else:  # YOLO format
            image_path, label_path = self._get_random_image_yolo()
            
            if not image_path:
                logger.error("Failed to get random image")
                return
            
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return
                
            img_height, img_width = image.shape[:2]
            
            # Parse YOLO annotations
            if label_path:
                annotations = self._parse_yolo_annotations(label_path, img_width, img_height)
            else:
                annotations = []
                
            category_mapping = None  # Use default category names
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        # Draw annotations
        annotated_image = self._draw_annotations(image, annotations, category_mapping)
        
        # Display image
        window_name = "Annotation Visualization"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, annotated_image)
        
        logger.info(f"Displaying image: {os.path.basename(image_path)} with {len(annotations)} annotations")
        logger.info("Press 'q' to quit, any other key to view another random image")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            else:
                # Show another random image
                cv2.destroyAllWindows()
                self.visualize_random_image()
                break
        
        cv2.destroyAllWindows()


def main():
    """Parse arguments and run the visualizer."""
    parser = argparse.ArgumentParser(description="Visualize bounding box annotations")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing the dataset")
    parser.add_argument("--annotation", help="Path to COCO JSON file (required for COCO format)")
    parser.add_argument("--format", choices=["coco", "yolo"], default="coco", help="Annotation format (default: coco)")
    parser.add_argument("--classes", help="Path to file with class names (one per line)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.format == "coco" and not args.annotation:
        parser.error("--annotation is required when format is 'coco'")
    
    visualizer = AnnotationVisualizer(
        dataset_dir=args.dataset_dir,
        annotation_path=args.annotation,
        format_type=args.format,
        class_file=args.classes,
    )
    
    visualizer.visualize_random_image()


if __name__ == "__main__":
    main()