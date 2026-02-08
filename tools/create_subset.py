#!/usr/bin/env python3
"""
Create a balanced subset of VisDrone dataset for faster training.
Can use symlinks (fast, space-efficient) or real file copies (portable).
"""
import json
import random
import os
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse

def create_balanced_subset(
    annotation_file,
    output_annotation_file,
    images_dir,
    output_images_dir,
    sample_ratio=0.3,
    min_objects_per_image=3,
    seed=42,
    image_transfer="symlink"
):
    """
    Create a balanced subset of the dataset.
    
    Args:
        annotation_file: Input COCO annotation JSON
        output_annotation_file: Output annotation JSON
        images_dir: Input images directory
        output_images_dir: Output images directory
        sample_ratio: Fraction of dataset to keep (0.0-1.0)
        min_objects_per_image: Minimum objects to keep image
        seed: Random seed for reproducibility
        image_transfer: How to materialize subset images: "symlink" or "copy"
    """
    random.seed(seed)
    
    print(f"📂 Loading annotations from {annotation_file}")
    with open(annotation_file) as f:
        coco_data = json.load(f)
    
    print(f"📊 Original dataset:")
    print(f"   - Images: {len(coco_data['images'])}")
    print(f"   - Annotations: {len(coco_data['annotations'])}")
    print(f"   - Categories: {len(coco_data['categories'])}")
    
    # Count objects per image
    print("\n🔍 Analyzing dataset...")
    image_obj_count = defaultdict(int)
    for ann in coco_data['annotations']:
        image_obj_count[ann['image_id']] += 1
    
    # Filter images with enough objects
    good_images = [
        img for img in coco_data['images']
        if image_obj_count[img['id']] >= min_objects_per_image
    ]
    
    print(f"   - Images with >={min_objects_per_image} objects: {len(good_images)}/{len(coco_data['images'])}")
    
    # Sample images
    n_samples = max(1, int(len(good_images) * sample_ratio))
    sampled_images = random.sample(good_images, n_samples)
    sampled_ids = {img['id'] for img in sampled_images}
    
    # Filter annotations
    sampled_annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] in sampled_ids
    ]
    
    # Count objects per category in subset
    category_counts = defaultdict(int)
    for ann in sampled_annotations:
        category_counts[ann['category_id']] += 1
    
    print(f"\n✂️ Subset statistics:")
    print(f"   - Images: {len(sampled_images)} ({sample_ratio*100:.1f}% of good images)")
    print(f"   - Annotations: {len(sampled_annotations)}")
    print(f"   - Avg objects/image: {len(sampled_annotations)/len(sampled_images):.1f}")
    
    # Create subset annotation file
    subset_data = {
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': coco_data['categories']
    }
    
    # Save annotations
    print(f"\n💾 Saving annotations to {output_annotation_file}")
    output_annotation_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_annotation_file, 'w') as f:
        json.dump(subset_data, f, indent=2)
    
    # Create links or copies for subset images
    mode_label = "symlinks" if image_transfer == "symlink" else "copies"
    print(f"🖼️  Materializing {len(sampled_images)} images as {mode_label}...")
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Get absolute path of source directory
    images_dir = images_dir.resolve()
    
    transferred = 0
    missing = 0
    for img in tqdm(sampled_images):
        src = images_dir / img['file_name']
        dst = output_images_dir / img['file_name']
        
        if src.exists():
            # Remove existing destination (file/symlink)
            if dst.exists() or dst.is_symlink():
                dst.unlink()

            if image_transfer == "symlink":
                os.symlink(src, dst)
            elif image_transfer == "copy":
                shutil.copy2(src, dst)
            else:
                raise ValueError(f"Unsupported image_transfer: {image_transfer}")
            transferred += 1
        else:
            print(f"⚠️  Missing: {src}")
            missing += 1
    
    print(f"\n✅ Subset created successfully!")
    if image_transfer == "symlink":
        print(f"   - Linked: {transferred} images (using symlinks - no extra disk space)")
    else:
        print(f"   - Copied: {transferred} images (portable dataset)")
    if missing > 0:
        print(f"   - Missing: {missing} images")
    
    # Print category distribution
    print(f"\n📊 Object distribution by category:")
    for cat in coco_data['categories']:
        count = category_counts.get(cat['id'], 0)
        print(f"   - {cat['name']:20s}: {count:5d} objects")
    
    return subset_data


def main():
    parser = argparse.ArgumentParser(description='Create dataset subset using symlinks')
    parser.add_argument('--input-annotations', required=True, help='Input COCO JSON')
    parser.add_argument('--output-annotations', required=True, help='Output COCO JSON')
    parser.add_argument('--input-images', required=True, help='Input images directory')
    parser.add_argument('--output-images', required=True, help='Output images directory')
    parser.add_argument('--ratio', type=float, default=0.3, help='Sampling ratio (default: 0.3)')
    parser.add_argument('--min-objects', type=int, default=3, help='Min objects per image (default: 3)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--image-transfer', choices=['symlink', 'copy'], default='symlink',
                        help='How to materialize subset images (default: symlink)')
    
    args = parser.parse_args()
    
    create_balanced_subset(
        annotation_file=Path(args.input_annotations),
        output_annotation_file=Path(args.output_annotations),
        images_dir=Path(args.input_images),
        output_images_dir=Path(args.output_images),
        sample_ratio=args.ratio,
        min_objects_per_image=args.min_objects,
        seed=args.seed,
        image_transfer=args.image_transfer
    )


if __name__ == '__main__':
    main()
