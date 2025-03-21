import argparse
from pathlib import Path
from deimkit.exporter import Exporter
from deimkit.config import Config
from loguru import logger

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yml",
        help="Path to configuration file (default: config.yml)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="model.pth",
        help="Path to model checkpoint (default: model.pth)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="model.onnx",
        help="Output path for ONNX model (default: model.onnx)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize config and exporter
    config = Config(args.config)
    exporter = Exporter(config)
    
    # Export model to ONNX
    output_path = exporter.to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output
    )
    
    logger.info(f"Model successfully exported to: {output_path}")

if __name__ == "__main__":
    main()