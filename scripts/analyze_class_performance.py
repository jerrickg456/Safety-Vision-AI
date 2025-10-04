#!/usr/bin/env python3
"""
Analyze per-class performance to identify weak detection classes
"""
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def analyze_class_performance():
    """Analyze performance for each safety equipment class"""
    print("🔍 Analyzing Class-Specific Performance...")
    
    # Load the high-precision model
    model_path = 'runs/high_precision/weights/best.pt'
    if not Path(model_path).exists():
        print("❌ High-precision model not found!")
        return
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    
    # Safety equipment classes
    class_names = [
        'OxygenTank',      # 0
        'NitrogenTank',    # 1 
        'FirstAidBox',     # 2
        'FireAlarm',       # 3
        'SafetySwitchPanel', # 4
        'EmergencyPhone',  # 5
        'FireExtinguisher' # 6
    ]
    
    print(f"✅ Model loaded. Classes: {class_names}")
    
    # Test images
    test_images = list(Path('data/test/images').glob('*.png'))
    print(f"📸 Testing on {len(test_images)} images...")
    
    # Class statistics
    class_stats = defaultdict(lambda: {'total_detections': 0, 'high_conf_detections': 0, 'confidences': []})
    
    total_detections = 0
    
    for img_path in test_images[:20]:  # Test on first 20 images
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Run multiple inference variants for better detection
        variants = [
            ('original', img),
            ('enhanced', enhance_contrast(img)),
            ('bright', cv2.convertScaleAbs(img, alpha=1.3, beta=25)),
            ('dark', cv2.convertScaleAbs(img, alpha=0.7, beta=-15))
        ]
        
        for variant_name, variant_img in variants:
            try:
                results = model(
                    variant_img,
                    conf=0.15,  # Lower threshold to catch more detections
                    iou=0.4,
                    verbose=False
                )
                
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    for conf, cls_id in zip(confidences, class_ids):
                        if 0 <= cls_id < len(class_names):
                            class_name = class_names[cls_id]
                            class_stats[class_name]['total_detections'] += 1
                            class_stats[class_name]['confidences'].append(float(conf))
                            
                            if conf >= 0.5:  # High confidence
                                class_stats[class_name]['high_conf_detections'] += 1
                            
                            total_detections += 1
                            
            except Exception as e:
                continue
    
    print(f"\n📊 CLASS PERFORMANCE ANALYSIS:")
    print("=" * 60)
    
    for class_name in class_names:
        stats = class_stats[class_name]
        if stats['total_detections'] > 0:
            avg_conf = np.mean(stats['confidences'])
            max_conf = max(stats['confidences'])
            min_conf = min(stats['confidences'])
            high_conf_ratio = stats['high_conf_detections'] / stats['total_detections']
            
            grade = get_class_grade(avg_conf, high_conf_ratio)
            
            print(f"\n🎯 {class_name}:")
            print(f"   Total Detections: {stats['total_detections']}")
            print(f"   High Confidence (50%+): {stats['high_conf_detections']} ({high_conf_ratio:.1%})")
            print(f"   Average Confidence: {avg_conf:.1%}")
            print(f"   Confidence Range: {min_conf:.1%} - {max_conf:.1%}")
            print(f"   Performance Grade: {grade}")
            
            if grade in ['C', 'D', 'F']:
                print(f"   ⚠️  NEEDS IMPROVEMENT!")
        else:
            print(f"\n❌ {class_name}: No detections found")
            print(f"   ⚠️  CRITICAL: Class not being detected at all!")
    
    # Recommendations
    print(f"\n🚀 IMPROVEMENT RECOMMENDATIONS:")
    print("=" * 60)
    
    weak_classes = []
    for class_name in class_names:
        stats = class_stats[class_name]
        if stats['total_detections'] == 0:
            weak_classes.append(class_name)
            print(f"❌ {class_name}: No detections - needs more training data")
        elif stats['total_detections'] > 0:
            avg_conf = np.mean(stats['confidences'])
            if avg_conf < 0.4:
                weak_classes.append(class_name)
                print(f"⚠️  {class_name}: Low confidence ({avg_conf:.1%}) - needs better examples")
    
    if weak_classes:
        print(f"\n📝 SOLUTION STEPS:")
        print(f"1. Add more diverse training images for: {', '.join(weak_classes)}")
        print(f"2. Lower confidence threshold for these classes")
        print(f"3. Use class-specific augmentation")
        print(f"4. Consider class balancing in dataset")
    
    return class_stats

def enhance_contrast(img):
    """Enhanced contrast preprocessing"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def get_class_grade(avg_conf, high_conf_ratio):
    """Grade class performance"""
    if avg_conf >= 0.8 and high_conf_ratio >= 0.7:
        return "A+"
    elif avg_conf >= 0.7 and high_conf_ratio >= 0.6:
        return "A"
    elif avg_conf >= 0.6 and high_conf_ratio >= 0.5:
        return "B+"
    elif avg_conf >= 0.5 and high_conf_ratio >= 0.4:
        return "B"
    elif avg_conf >= 0.4:
        return "C"
    elif avg_conf >= 0.3:
        return "D"
    else:
        return "F"

if __name__ == "__main__":
    analyze_class_performance()