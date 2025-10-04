#!/usr/bin/env python3
"""
Precision Test Script - Test model accuracy on specific safety equipment images
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def test_model_precision():
    """Test current model precision on validation dataset"""
    
    print("🎯 PRECISION TESTING - Duality Detector")
    print("=" * 50)
    
    # Load best available model
    models_to_test = [
        ('runs/high_precision/weights/best.pt', 'High-Precision Model'),
        ('runs/baseline6/weights/best.pt', 'GPU-Trained Model'),
        ('runs/baseline4/weights/best.pt', 'CPU-Trained Model'),
    ]
    
    for model_path, model_name in models_to_test:
        if os.path.exists(model_path):
            print(f"\n🔍 Testing: {model_name}")
            print(f"📁 Path: {model_path}")
            
            model = YOLO(model_path)
            
            # Test on validation dataset
            results = model.val(
                data='data/config.yaml',
                imgsz=640,
                batch=1,
                conf=0.25,
                iou=0.4,
                verbose=False
            )
            
            print(f"📊 Overall mAP50: {results.box.map50:.3f} ({results.box.map50*100:.1f}%)")
            print(f"📊 Overall mAP50-95: {results.box.map:.3f} ({results.box.map*100:.1f}%)")
            
            # Per-class results
            print("\n📋 Per-Class Accuracy (mAP50):")
            class_names = list(model.names.values())
            for i, class_name in enumerate(class_names):
                if hasattr(results.box, 'maps50') and i < len(results.box.maps50):
                    accuracy = results.box.maps50[i]
                    status = "🎯" if accuracy > 0.8 else "✅" if accuracy > 0.6 else "⚠️" if accuracy > 0.4 else "❌"
                    print(f"  {status} {class_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            break
    else:
        print("❌ No trained models found!")
        return
    
    # Test individual images
    print(f"\n🔍 INDIVIDUAL IMAGE TESTS")
    print("-" * 30)
    
    test_images = [
        'data/test/images/000000000_light_unclutter.png',
        'data/test/images/000000001_dark_clutter.png',
        'data/test/images/000000002_vdark_unclutter.png',
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n📸 Testing: {os.path.basename(img_path)}")
            
            results = model.predict(
                source=img_path,
                conf=0.25,
                iou=0.4,
                verbose=False,
                save=False
            )
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        if hasattr(box, 'cls') and hasattr(box, 'conf'):
                            class_id = int(box.cls.item())
                            confidence = float(box.conf.item())
                            class_name = model.names.get(class_id, 'unknown')
                            
                            precision = "🎯 Very High" if confidence >= 0.8 else \
                                       "✅ High" if confidence >= 0.6 else \
                                       "⚠️ Medium" if confidence >= 0.4 else \
                                       "❓ Low"
                            
                            print(f"  {precision}: {class_name} ({confidence:.3f})")
                else:
                    print("  ❌ No detections")
            else:
                print("  ❌ No results")

def benchmark_models():
    """Benchmark different models for comparison"""
    print("\n🏆 MODEL BENCHMARK COMPARISON")
    print("=" * 50)
    
    models = {
        'YOLOv8s (Small)': 'runs/baseline6/weights/best.pt',
        'YOLOv8m (Medium)': 'runs/high_precision/weights/best.pt',
    }
    
    for model_name, model_path in models.items():
        if os.path.exists(model_path):
            print(f"\n🔬 Benchmarking: {model_name}")
            model = YOLO(model_path)
            
            # Quick validation
            results = model.val(data='data/config.yaml', verbose=False)
            
            print(f"  📊 mAP50: {results.box.map50*100:.1f}%")
            print(f"  📊 mAP50-95: {results.box.map*100:.1f}%")
            print(f"  ⚡ Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")

if __name__ == "__main__":
    test_model_precision()
    benchmark_models()
    
    print(f"\n💡 RECOMMENDATIONS FOR HIGHER PRECISION:")
    print("1. Use confidence threshold >= 0.6 for high precision")
    print("2. Use YOLOv8m model for better accuracy than YOLOv8s") 
    print("3. Train for 50+ epochs with advanced hyperparameters")
    print("4. Add more diverse training data for weak classes")
    print("5. Use ensemble predictions for critical applications")