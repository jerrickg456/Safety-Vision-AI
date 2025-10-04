#!/usr/bin/env python3
"""
Dataset augmentation script to add more diverse safety equipment images
"""
import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def augment_existing_images():
    """
    Create augmented versions of existing training images
    """
    train_dir = Path('data/train/images')
    labels_dir = Path('data/train/labels')
    
    if not train_dir.exists():
        print("Training directory not found!")
        return
        
    # Create augmented directories
    aug_dir = train_dir.parent / 'images_augmented'
    aug_labels_dir = labels_dir.parent / 'labels_augmented'
    aug_dir.mkdir(exist_ok=True)
    aug_labels_dir.mkdir(exist_ok=True)
    
    # Copy original files first
    for img_file in train_dir.glob('*.png'):
        shutil.copy2(img_file, aug_dir / img_file.name)
        
        # Copy corresponding label
        label_file = labels_dir / (img_file.stem + '.txt')
        if label_file.exists():
            shutil.copy2(label_file, aug_labels_dir / label_file.name)
    
    # Apply augmentations
    augmentation_count = 0
    for img_file in train_dir.glob('*.png'):
        if augmentation_count > 500:  # Limit to prevent too much data
            break
            
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        label_file = labels_dir / (img_file.stem + '.txt')
        if not label_file.exists():
            continue
            
        # Read labels
        with open(label_file, 'r') as f:
            labels = f.read().strip()
        
        # Apply various augmentations
        augmented_images = []
        
        # 1. Brightness adjustment
        bright_img = cv2.convertScaleAbs(img, alpha=1.3, beta=20)
        dark_img = cv2.convertScaleAbs(img, alpha=0.7, beta=-20)
        
        # 2. Horizontal flip
        flipped_img = cv2.flip(img, 1)
        
        # 3. Gaussian blur
        blurred_img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # 4. Add noise
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        noisy_img = cv2.add(img, noise)
        
        augmented_images = [
            (bright_img, 'bright'),
            (dark_img, 'dark'), 
            (flipped_img, 'flip'),
            (blurred_img, 'blur'),
            (noisy_img, 'noise')
        ]
        
        # Save augmented images
        for aug_img, suffix in augmented_images:
            new_name = f"{img_file.stem}_{suffix}{img_file.suffix}"
            cv2.imwrite(str(aug_dir / new_name), aug_img)
            
            # Copy labels (adjust for horizontal flip if needed)
            new_label_name = f"{img_file.stem}_{suffix}.txt"
            if suffix == 'flip' and labels:
                # Adjust horizontal flip coordinates
                adjusted_labels = []
                for line in labels.split('\n'):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # YOLO format: class x_center y_center width height
                            cls, x, y, w, h = parts[:5]
                            x_flipped = 1.0 - float(x)  # Flip x coordinate
                            adjusted_labels.append(f"{cls} {x_flipped} {y} {w} {h}")
                
                with open(aug_labels_dir / new_label_name, 'w') as f:
                    f.write('\n'.join(adjusted_labels))
            else:
                # Copy original labels
                shutil.copy2(label_file, aug_labels_dir / new_label_name)
            
            augmentation_count += 1
    
    print(f"✅ Created {augmentation_count} augmented images")
    print(f"📁 Augmented dataset: {aug_dir}")
    print(f"📁 Augmented labels: {aug_labels_dir}")

def create_diverse_examples():
    """
    Create text file with suggestions for diverse safety equipment images
    """
    examples_file = Path('data/diverse_safety_equipment_examples.txt')
    
    examples = [
        "Fire Extinguisher Examples:",
        "- Red ABC fire extinguisher mounted on wall",
        "- Portable CO2 fire extinguisher in office",
        "- Foam fire extinguisher in industrial setting",
        "- Different brands: Kidde, Amerex, First Alert",
        "- Various sizes: 2.5lb, 5lb, 10lb, 20lb",
        "- Different angles: front view, side view, angled",
        "- Different lighting: bright, dim, fluorescent, natural",
        "- Different backgrounds: white wall, brick wall, industrial",
        "",
        "Oxygen Tank Examples:",
        "- Medical oxygen cylinder with gauge",
        "- Industrial oxygen tank with regulator",
        "- Portable oxygen concentrator",
        "- Different sizes and colors",
        "- With and without wheeled cart",
        "",
        "First Aid Box Examples:",
        "- Wall-mounted green first aid box",
        "- Portable first aid kit",
        "- Industrial first aid station",
        "- Different sizes and manufacturers",
        "- Open and closed positions",
        "",
        "Fire Alarm Examples:", 
        "- Smoke detector on ceiling",
        "- Fire alarm pull station",
        "- Heat detector",
        "- Different brands and models",
        "- Various mounting positions",
        "",
        "Emergency Phone Examples:",
        "- Red emergency telephone",
        "- Blue emergency phone box",
        "- Industrial emergency communication device",
        "- Wall-mounted and standalone versions",
        "",
        "Safety Switch Panel Examples:",
        "- Emergency stop button (red mushroom button)",
        "- Safety disconnect switch",
        "- Emergency power cutoff",
        "- Industrial safety control panel",
        "- Different sizes and configurations"
    ]
    
    with open(examples_file, 'w') as f:
        f.write('\n'.join(examples))
    
    print(f"📝 Created examples guide: {examples_file}")

if __name__ == "__main__":
    print("🔄 Starting dataset augmentation...")
    augment_existing_images()
    create_diverse_examples()
    print("✅ Dataset augmentation complete!")
    print("\n💡 To improve real-world performance:")
    print("1. Check data/diverse_safety_equipment_examples.txt for image ideas")
    print("2. Add more diverse images to data/train/images/")
    print("3. Label them with your annotation tool")
    print("4. Re-run training")