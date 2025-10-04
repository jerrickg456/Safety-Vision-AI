#!/usr/bin/env python3
"""
Multi-Model Ensemble for Ultra-Precision Safety Detection
Combines multiple models for maximum accuracy
"""
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from typing import List, Tuple, Dict
from collections import defaultdict
import time

class EnsembleDetector:
    def __init__(self):
        self.models = []
        self.model_weights = []
        
    def load_models(self):
        """Load multiple trained models for ensemble"""
        model_paths = [
            'runs/high_precision/weights/best.pt',  # YOLOv8m high precision
            'runs/ultra_precision/weights/best.pt',  # YOLOv8x ultra precision (when available)
            'yolov8l.pt'  # YOLOv8l as backup
        ]
        
        weights = [0.4, 0.5, 0.1]  # Weight importance
        
        print("🔧 Loading ensemble models...")
        
        for i, (path, weight) in enumerate(zip(model_paths, weights)):
            if Path(path).exists():
                try:
                    model = YOLO(path)
                    self.models.append(model)
                    self.model_weights.append(weight)
                    print(f"✅ Loaded model {i+1}: {path}")
                except Exception as e:
                    print(f"❌ Failed to load {path}: {e}")
            else:
                print(f"⚠️ Model not found: {path}")
        
        # Normalize weights
        total_weight = sum(self.model_weights)
        self.model_weights = [w/total_weight for w in self.model_weights]
        
        print(f"🎯 Ensemble ready with {len(self.models)} models")
        return len(self.models) > 0
    
    def ensemble_predict(self, image_path: str, 
                        conf_threshold: float = 0.15,
                        iou_threshold: float = 0.5,
                        use_tta: bool = True) -> Dict:
        """
        Perform ensemble prediction with test-time augmentation
        """
        if not self.models:
            raise ValueError("No models loaded!")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        all_predictions = []
        
        # Get predictions from each model
        for i, (model, weight) in enumerate(zip(self.models, self.model_weights)):
            print(f"🔍 Running model {i+1}...")
            
            if use_tta:
                # Test-time augmentation
                predictions = self._test_time_augmentation(model, img, conf_threshold)
            else:
                results = model(img, conf=conf_threshold, verbose=False)
                predictions = self._extract_predictions(results[0])
            
            # Weight predictions
            for pred in predictions:
                pred['confidence'] *= weight
                pred['model_id'] = i
            
            all_predictions.extend(predictions)
        
        # Ensemble NMS
        final_predictions = self._ensemble_nms(all_predictions, iou_threshold)
        
        # Calculate ensemble confidence
        ensemble_stats = self._calculate_ensemble_stats(all_predictions, final_predictions)
        
        return {
            'predictions': final_predictions,
            'ensemble_stats': ensemble_stats,
            'model_count': len(self.models),
            'total_raw_predictions': len(all_predictions)
        }
    
    def _test_time_augmentation(self, model, img, conf_threshold):
        """Apply test-time augmentation for robust predictions"""
        augmentations = [
            ('original', lambda x: x),
            ('flip_h', lambda x: cv2.flip(x, 1)),
            ('bright', lambda x: cv2.convertScaleAbs(x, alpha=1.2, beta=20)),
            ('dark', lambda x: cv2.convertScaleAbs(x, alpha=0.8, beta=-20)),
            ('contrast', self._enhance_contrast),
        ]
        
        all_preds = []
        h, w = img.shape[:2]
        
        for aug_name, aug_func in augmentations:
            aug_img = aug_func(img.copy())
            
            # Run inference
            results = model(aug_img, conf=conf_threshold, verbose=False)
            preds = self._extract_predictions(results[0])
            
            # Reverse augmentation on coordinates
            if aug_name == 'flip_h':
                for pred in preds:
                    pred['x'] = w - pred['x']
            
            # Add augmentation info
            for pred in preds:
                pred['augmentation'] = aug_name
                pred['confidence'] *= 0.9 if aug_name != 'original' else 1.0  # Slight penalty for augmented
            
            all_preds.extend(preds)
        
        return all_preds
    
    def _enhance_contrast(self, img):
        """Enhance image contrast"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def _extract_predictions(self, results):
        """Extract predictions from YOLO results"""
        predictions = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                predictions.append({
                    'x': x_center,
                    'y': y_center,
                    'width': width,
                    'height': height,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'class_name': results.names[cls_id]
                })
        
        return predictions
    
    def _ensemble_nms(self, predictions, iou_threshold):
        """Apply Non-Maximum Suppression across ensemble predictions"""
        if not predictions:
            return []
        
        # Group by class
        class_groups = defaultdict(list)
        for pred in predictions:
            class_groups[pred['class_id']].append(pred)
        
        final_predictions = []
        
        # Apply NMS per class
        for class_id, class_preds in class_groups.items():
            # Sort by confidence
            class_preds.sort(key=lambda x: x['confidence'], reverse=True)
            
            keep = []
            while class_preds:
                # Take highest confidence prediction
                best = class_preds.pop(0)
                keep.append(best)
                
                # Remove overlapping predictions
                remaining = []
                for pred in class_preds:
                    iou = self._calculate_iou(best, pred)
                    if iou < iou_threshold:
                        remaining.append(pred)
                    else:
                        # Merge overlapping predictions (weighted average)
                        best = self._merge_predictions(best, pred)
                
                class_preds = remaining
            
            final_predictions.extend(keep)
        
        return final_predictions
    
    def _calculate_iou(self, pred1, pred2):
        """Calculate Intersection over Union"""
        x1_inter = max(pred1['x1'], pred2['x1'])
        y1_inter = max(pred1['y1'], pred2['y1'])
        x2_inter = min(pred1['x2'], pred2['x2'])
        y2_inter = min(pred1['y2'], pred2['y2'])
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (pred1['x2'] - pred1['x1']) * (pred1['y2'] - pred1['y1'])
        area2 = (pred2['x2'] - pred2['x1']) * (pred2['y2'] - pred2['y1'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_predictions(self, pred1, pred2):
        """Merge overlapping predictions with weighted average"""
        w1 = pred1['confidence']
        w2 = pred2['confidence']
        total_weight = w1 + w2
        
        merged = pred1.copy()
        merged['confidence'] = max(w1, w2)  # Take max confidence
        
        # Weighted average of coordinates
        merged['x'] = (pred1['x'] * w1 + pred2['x'] * w2) / total_weight
        merged['y'] = (pred1['y'] * w1 + pred2['y'] * w2) / total_weight
        merged['width'] = (pred1['width'] * w1 + pred2['width'] * w2) / total_weight
        merged['height'] = (pred1['height'] * w1 + pred2['height'] * w2) / total_weight
        
        # Recalculate bounding box
        merged['x1'] = merged['x'] - merged['width'] / 2
        merged['y1'] = merged['y'] - merged['height'] / 2
        merged['x2'] = merged['x'] + merged['width'] / 2
        merged['y2'] = merged['y'] + merged['height'] / 2
        
        return merged
    
    def _calculate_ensemble_stats(self, all_preds, final_preds):
        """Calculate ensemble statistics"""
        stats = {
            'agreement_rate': 0.0,
            'confidence_boost': 0.0,
            'detection_stability': 0.0
        }
        
        if not all_preds or not final_preds:
            return stats
        
        # Calculate agreement rate (how many models detected each object)
        model_votes = defaultdict(int)
        for pred in all_preds:
            key = f"{pred['class_id']}_{int(pred['x'])}_{int(pred['y'])}"
            model_votes[key] += 1
        
        if model_votes:
            avg_votes = sum(model_votes.values()) / len(model_votes)
            stats['agreement_rate'] = avg_votes / len(self.models)
        
        # Calculate confidence boost
        single_model_conf = np.mean([p['confidence'] for p in all_preds[:len(final_preds)]])
        ensemble_conf = np.mean([p['confidence'] for p in final_preds])
        stats['confidence_boost'] = ensemble_conf / single_model_conf if single_model_conf > 0 else 0
        
        # Detection stability (lower variance = more stable)
        confidences_by_object = defaultdict(list)
        for pred in all_preds:
            key = f"{pred['class_id']}_{int(pred['x']//50)*50}_{int(pred['y']//50)*50}"
            confidences_by_object[key].append(pred['confidence'])
        
        variances = [np.var(confs) for confs in confidences_by_object.values() if len(confs) > 1]
        stats['detection_stability'] = 1.0 - (np.mean(variances) if variances else 0.0)
        
        return stats

def test_ensemble():
    """Test ensemble detector on sample images"""
    detector = EnsembleDetector()
    
    if not detector.load_models():
        print("❌ No models available for ensemble")
        return
    
    # Test on sample images
    test_images = list(Path('data/test/images').glob('*.png'))[:5]
    
    print(f"\n🧪 Testing ensemble on {len(test_images)} images...\n")
    
    for img_path in test_images:
        print(f"📸 Testing: {img_path.name}")
        
        try:
            start_time = time.time()
            results = detector.ensemble_predict(
                str(img_path),
                conf_threshold=0.15,
                use_tta=True
            )
            inference_time = time.time() - start_time
            
            predictions = results['predictions']
            stats = results['ensemble_stats']
            
            print(f"   🎯 Detections: {len(predictions)}")
            print(f"   📊 Agreement Rate: {stats['agreement_rate']:.2%}")
            print(f"   🚀 Confidence Boost: {stats['confidence_boost']:.2f}x")
            print(f"   ⚡ Stability: {stats['detection_stability']:.2%}")
            print(f"   ⏱️  Time: {inference_time:.2f}s")
            
            # Show top detections
            for pred in sorted(predictions, key=lambda x: x['confidence'], reverse=True)[:3]:
                print(f"      {pred['class_name']}: {pred['confidence']:.1%}")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}\n")

if __name__ == "__main__":
    test_ensemble()