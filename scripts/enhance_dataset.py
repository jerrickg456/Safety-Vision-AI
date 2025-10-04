#!/usr/bin/env python3
"""
Advanced Dataset Enhancement for Ultra-Precision Safety Detection
"""
import os
import cv2
import numpy as np
import random
from pathlib import Path
import shutil
from PIL import Image, ImageEnhance, ImageFilter
import yaml

class DatasetEnhancer:
    def __init__(self, base_dir='data'):
        self.base_dir = Path(base_dir)
        self.train_images = self.base_dir / 'train' / 'images'
        self.train_labels = self.base_dir / 'train' / 'labels'
        
    def enhance_dataset(self):
        """Create enhanced dataset with advanced augmentations"""
        print("🔧 Creating Ultra-Precision Dataset...")
        
        # Create enhanced directories
        enhanced_dir = self.base_dir / 'enhanced'
        enhanced_images = enhanced_dir / 'images'
        enhanced_labels = enhanced_dir / 'labels'
        
        enhanced_images.mkdir(parents=True, exist_ok=True)
        enhanced_labels.mkdir(parents=True, exist_ok=True)
        
        # Copy original files
        print("📁 Copying original dataset...")
        for img_file in self.train_images.glob('*.png'):
            shutil.copy2(img_file, enhanced_images / img_file.name)
            
            label_file = self.train_labels / (img_file.stem + '.txt')
            if label_file.exists():
                shutil.copy2(label_file, enhanced_labels / label_file.name)
        
        # Advanced augmentations
        augmentation_count = 0
        max_augmentations = 3000  # Create 3000 additional high-quality images
        
        print("🎨 Applying advanced augmentations...")
        
        for img_file in list(self.train_images.glob('*.png'))[:500]:  # Limit base images
            if augmentation_count >= max_augmentations:
                break
                
            label_file = self.train_labels / (img_file.stem + '.txt')
            if not label_file.exists():
                continue
                
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Read labels
            with open(label_file, 'r') as f:
                labels = f.read().strip()
            
            # Apply multiple high-quality augmentations
            augmented_images = self.create_quality_augmentations(img, str(img_file))
            
            for i, (aug_img, aug_name) in enumerate(augmented_images):
                if augmentation_count >= max_augmentations:
                    break
                    
                new_img_name = f"{img_file.stem}_aug_{aug_name}_{i}.png"
                new_label_name = f"{img_file.stem}_aug_{aug_name}_{i}.txt"
                
                # Save augmented image
                cv2.imwrite(str(enhanced_images / new_img_name), aug_img)
                
                # Copy labels (augment if needed)
                augmented_labels = self.augment_labels(labels, aug_name)
                with open(enhanced_labels / new_label_name, 'w') as f:
                    f.write(augmented_labels)
                
                augmentation_count += 1
        
        print(f"✅ Created {augmentation_count} enhanced images")
        
        # Create enhanced config
        self.create_enhanced_config()
        
        return enhanced_dir
    
    def create_quality_augmentations(self, img, img_path):
        """Create high-quality augmentations focusing on real-world variations"""
        augmentations = []
        
        # 1. Lighting variations (most important for real images)
        # Bright lighting
        bright = cv2.convertScaleAbs(img, alpha=1.4, beta=30)
        augmentations.append((bright, 'bright_light'))
        
        # Low lighting
        dark = cv2.convertScaleAbs(img, alpha=0.6, beta=-30)
        augmentations.append((dark, 'low_light'))
        
        # Contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        augmentations.append((enhanced, 'contrast'))
        
        # 2. Realistic blur variations
        # Motion blur
        kernel_motion = np.zeros((15, 15))
        kernel_motion[int((15-1)/2), :] = np.ones(15)
        kernel_motion = kernel_motion / 15
        motion_blur = cv2.filter2D(img, -1, kernel_motion)
        augmentations.append((motion_blur, 'motion_blur'))
        
        # Gaussian blur (camera focus)
        gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)
        augmentations.append((gaussian_blur, 'focus_blur'))
        
        # 3. Color variations (for different cameras/lighting)
        # Warm tone
        warm = img.copy()
        warm[:, :, 0] = np.clip(warm[:, :, 0] * 0.9, 0, 255)  # Reduce blue
        warm[:, :, 2] = np.clip(warm[:, :, 2] * 1.1, 0, 255)  # Increase red
        augmentations.append((warm, 'warm_tone'))
        
        # Cool tone
        cool = img.copy()
        cool[:, :, 0] = np.clip(cool[:, :, 0] * 1.1, 0, 255)  # Increase blue
        cool[:, :, 2] = np.clip(cool[:, :, 2] * 0.9, 0, 255)  # Reduce red
        augmentations.append((cool, 'cool_tone'))
        
        # 4. Noise variations (real camera noise)
        # Gaussian noise
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        noisy = cv2.add(img, noise)
        augmentations.append((noisy, 'camera_noise'))
        
        # Salt and pepper noise
        sp_noise = img.copy()
        salt_pepper = np.random.random(img.shape[:2])
        sp_noise[salt_pepper < 0.01] = 0
        sp_noise[salt_pepper > 0.99] = 255
        augmentations.append((sp_noise, 'sp_noise'))
        
        # 5. Geometric variations (minimal to preserve labels)
        h, w = img.shape[:2]
        
        # Small rotation
        center = (w//2, h//2)
        angle = random.choice([-2, -1, 1, 2])  # Very small angles
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmentations.append((rotated, f'rot_{angle}'))
        
        # Horizontal flip (most common real variation)
        flipped = cv2.flip(img, 1)
        augmentations.append((flipped, 'hflip'))
        
        return augmentations
    
    def augment_labels(self, labels, aug_name):
        """Augment labels based on augmentation type"""
        if not labels.strip():
            return labels
            
        # Only horizontal flip needs label adjustment
        if aug_name == 'hflip':
            adjusted_labels = []
            for line in labels.split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x, y, w, h = parts[:5]
                        x_flipped = 1.0 - float(x)  # Flip x coordinate
                        adjusted_labels.append(f"{cls} {x_flipped} {y} {w} {h}")
            return '\n'.join(adjusted_labels)
        
        # For other augmentations, keep original labels
        return labels
    
    def create_enhanced_config(self):
        """Create enhanced config file"""
        config = {
            'path': str(Path.cwd() / 'data'),
            'train': 'enhanced/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': 7,
            'names': [
                'OxygenTank',
                'NitrogenTank', 
                'FirstAidBox',
                'FireAlarm',
                'SafetySwitchPanel',
                'EmergencyPhone',
                'FireExtinguisher'
            ]
        }
        
        config_path = self.base_dir / 'enhanced_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"📝 Created enhanced config: {config_path}")
        return config_path

def main():
    enhancer = DatasetEnhancer()
    enhanced_dir = enhancer.enhance_dataset()
    
    print("\n🎯 ULTRA-PRECISION DATASET READY!")
    print(f"📁 Enhanced dataset: {enhanced_dir}")
    print(f"📝 Config file: data/enhanced_config.yaml")
    print("\n💡 Next steps:")
    print("1. Run ultra-precision training with enhanced dataset")
    print("2. Use YOLOv8x model for maximum accuracy")
    print("3. Train for 100+ epochs for best results")

if __name__ == "__main__":
    main()