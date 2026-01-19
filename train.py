#!/usr/bin/env python3
"""
Main training script for VisDrone DETR object detection.
"""
import sys
from pathlib import Path

print("🚀 Starting VisDrone DETR Training...")

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import main module
import main as main_module

if __name__ == "__main__":
    # Parse arguments and run
    args = main_module.get_args_parser().parse_args()
    print(f"✅ Arguments parsed: {args}")
    main_module.main(args)
