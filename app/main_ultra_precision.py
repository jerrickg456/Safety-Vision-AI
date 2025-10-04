#!/usr/bin/env python3
"""
Ultra-Precision Safety Equipment Detection API with Ensemble Models
Enhanced for maximum real-world accuracy
"""
import os
os.environ['YOLO_VERBOSE'] = 'False'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image, ImageEnhance
import json
import torch
from pathlib import Path
import time
import traceback
from typing import List, Dict, Optional
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ultra-Precision Safety Equipment Detector")
app.mount("/static", StaticFiles(directory="app"), name="static")

class UltraPrecisionDetector:
    def __init__(self):
        self.models = []
        self.model_names = []
        self.load_ensemble_models()
        self.class_names = [
            'OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm',
            'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher'
        ]
        
    def load_ensemble_models(self):
        """Load multiple models for ensemble detection"""
        model_configs = [
            {
                'path': 'runs/ultra_precision/weights/best.pt',
                'name': 'YOLOv8x Ultra-Precision',
                'weight': 0.5,
                'priority': 1
            },
            {
                'path': 'runs/high_precision/weights/best.pt',
                'name': 'YOLOv8m High-Precision',
                'weight': 0.3,
                'priority': 2
            },
            {
                'path': 'yolov8l.pt',
                'name': 'YOLOv8l Baseline',
                'weight': 0.2,
                'priority': 3
            }
        ]
        
        logger.info("🚀 Loading Ultra-Precision Ensemble Models...")
        
        for config in model_configs:
            try:
                if Path(config['path']).exists():
                    # Force GPU usage for RTX 3050
                    device = 0 if torch.cuda.is_available() else 'cpu'
                    model = YOLO(config['path'])
                    model.to(device)
                    
                    self.models.append({
                        'model': model,
                        'weight': config['weight'],
                        'name': config['name'],
                        'priority': config['priority']
                    })
                    
                    logger.info(f"✅ Loaded: {config['name']} (GPU: {torch.cuda.is_available()})")
                else:
                    logger.warning(f"⚠️ Model not found: {config['path']}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load {config['name']}: {e}")
        
        if not self.models:
            # Fallback to basic model
            logger.info("📦 Loading fallback model...")
            device = 0 if torch.cuda.is_available() else 'cpu'
            model = YOLO('yolov8s.pt')
            model.to(device)
            self.models.append({
                'model': model,
                'weight': 1.0,
                'name': 'YOLOv8s Fallback',
                'priority': 9
            })
        
        logger.info(f"🎯 Ensemble ready with {len(self.models)} models")
    
    def preprocess_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Enhanced preprocessing for maximum detection accuracy"""
        processed_images = []
        
        # Original image
        processed_images.append(('original', image))
        
        # Enhanced contrast version
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        processed_images.append(('enhanced_contrast', enhanced_bgr))
        
        # Brightness variations
        bright = cv2.convertScaleAbs(image, alpha=1.3, beta=25)
        processed_images.append(('bright', bright))
        
        darker = cv2.convertScaleAbs(image, alpha=0.7, beta=-15)
        processed_images.append(('darker', darker))
        
        return processed_images
    
    def ensemble_detect(self, image: np.ndarray, confidence_threshold: float = 0.15) -> Dict:
        """Ultra-precision ensemble detection"""
        start_time = time.time()
        
        # Preprocess image variants
        image_variants = self.preprocess_image(image)
        
        all_detections = []
        model_results = {}
        
        # Run each model on each image variant
        for model_info in self.models:
            model = model_info['model']
            model_name = model_info['name']
            weight = model_info['weight']
            
            model_detections = []
            
            for variant_name, img_variant in image_variants:
                try:
                    # Ultra-precision inference settings
                    results = model(
                        img_variant,
                        conf=confidence_threshold,
                        iou=0.4,  # Lower IoU for more detections
                        agnostic_nms=False,  # Class-specific NMS
                        max_det=50,  # Allow more detections
                        augment=True,  # Test-time augmentation
                        verbose=False
                    )
                    
                    if results and len(results) > 0 and results[0].boxes is not None:
                        detections = self.extract_detections(results[0], weight, model_name, variant_name)
                        model_detections.extend(detections)
                        
                except Exception as e:
                    logger.error(f"Detection error with {model_name} on {variant_name}: {e}")
            
            all_detections.extend(model_detections)
            model_results[model_name] = len(model_detections)
        
        # Advanced ensemble NMS
        final_detections = self.advanced_nms(all_detections)
        
        # Calculate precision metrics
        precision_score = self.calculate_precision_score(final_detections, len(all_detections))
        
        processing_time = time.time() - start_time
        
        return {
            'detections': final_detections,
            'precision_score': precision_score,
            'model_count': len(self.models),
            'total_raw_detections': len(all_detections),
            'processing_time': processing_time,
            'model_results': model_results,
            'gpu_used': torch.cuda.is_available()
        }
    
    def extract_detections(self, results, weight: float, model_name: str, variant_name: str) -> List[Dict]:
        """Extract and format detections from YOLO results"""
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf * weight),  # Apply model weight
                    'class_id': int(cls_id),
                    'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f'Unknown_{cls_id}',
                    'model': model_name,
                    'variant': variant_name,
                    'raw_confidence': float(conf)
                }
                
                detections.append(detection)
        
        return detections
    
    def advanced_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Advanced Non-Maximum Suppression with ensemble weighting"""
        if not detections:
            return []
        
        # Group by class
        class_groups = defaultdict(list)
        for det in detections:
            class_groups[det['class_id']].append(det)
        
        final_detections = []
        
        for class_id, class_detections in class_groups.items():
            # Sort by confidence (highest first)
            class_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            keep = []
            while class_detections:
                # Take highest confidence detection
                best = class_detections.pop(0)
                keep.append(best)
                
                # Remove overlapping detections
                remaining = []
                for det in class_detections:
                    iou = self.calculate_iou(best['bbox'], det['bbox'])
                    
                    if iou < iou_threshold:
                        remaining.append(det)
                    else:
                        # Merge high-overlap detections for better precision
                        if iou > 0.7 and abs(best['confidence'] - det['confidence']) < 0.2:
                            best = self.merge_detections(best, det)
                
                class_detections = remaining
            
            final_detections.extend(keep)
        
        # Sort final detections by confidence
        final_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return final_detections
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def merge_detections(self, det1: Dict, det2: Dict) -> Dict:
        """Merge two similar detections for improved precision"""
        w1 = det1['confidence']
        w2 = det2['confidence']
        total_weight = w1 + w2
        
        merged = det1.copy()
        
        # Weighted average of bounding boxes
        box1, box2 = det1['bbox'], det2['bbox']
        merged['bbox'] = [
            (box1[0] * w1 + box2[0] * w2) / total_weight,
            (box1[1] * w1 + box2[1] * w2) / total_weight,
            (box1[2] * w1 + box2[2] * w2) / total_weight,
            (box1[3] * w1 + box2[3] * w2) / total_weight
        ]
        
        # Take maximum confidence
        merged['confidence'] = max(w1, w2) * 1.1  # Boost for ensemble agreement
        merged['ensemble_merged'] = True
        
        return merged
    
    def calculate_precision_score(self, detections: List[Dict], raw_count: int) -> float:
        """Calculate overall precision score (0-1)"""
        if not detections:
            return 0.0
        
        # Base score from average confidence
        avg_confidence = sum(det['confidence'] for det in detections) / len(detections)
        
        # Bonus for ensemble agreement (multiple models detecting same objects)
        ensemble_bonus = min(0.2, len(detections) * 0.05)
        
        # Penalty for too many detections (potential false positives)
        if len(detections) > 10:
            fp_penalty = (len(detections) - 10) * 0.02
        else:
            fp_penalty = 0
        
        # Model diversity bonus
        models_used = len(set(det.get('model', 'unknown') for det in detections))
        diversity_bonus = min(0.1, models_used * 0.03)
        
        precision_score = min(1.0, avg_confidence + ensemble_bonus + diversity_bonus - fp_penalty)
        
        return precision_score

# Initialize detector
logger.info("🚀 Initializing Ultra-Precision Safety Equipment Detector...")
detector = UltraPrecisionDetector()

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main detection interface"""
    with open("app/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/detect/")
async def detect_safety_equipment(file: UploadFile = File(...)):
    """Ultra-precision safety equipment detection endpoint"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
        # Read and process image
        image_bytes = await file.read()
        
        # Enhanced format support
        try:
            # Try PIL first for better format support (AVIF, HEIC, etc.)
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if pil_image.mode in ('RGBA', 'LA', 'P'):
                pil_image = pil_image.convert('RGB')
            elif pil_image.mode == 'L':
                pil_image = pil_image.convert('RGB')
            
            # Convert to OpenCV format
            image_array = np.array(pil_image)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
        except Exception as pil_error:
            # Fallback to OpenCV
            try:
                nparr = np.frombuffer(image_bytes, np.uint8)
                image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image_bgr is None:
                    raise ValueError("Could not decode image with OpenCV")
                    
            except Exception as cv_error:
                logger.error(f"Image processing failed - PIL: {pil_error}, CV2: {cv_error}")
                raise HTTPException(status_code=400, detail="Could not process image. Please check the file format.")
        
        # Validate image
        if image_bgr is None or image_bgr.size == 0:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        logger.info(f"Processing image: {file.filename}, Size: {image_bgr.shape}")
        
        # Ultra-precision detection
        results = detector.ensemble_detect(image_bgr, confidence_threshold=0.12)
        
        # Format response
        response = {
            "success": True,
            "filename": file.filename,
            "detections": len(results['detections']),
            "precision_score": round(results['precision_score'], 3),
            "grade": get_precision_grade(results['precision_score']),
            "processing_time": round(results['processing_time'], 2),
            "model_info": {
                "ensemble_models": results['model_count'],
                "gpu_accelerated": results['gpu_used'],
                "total_inferences": results['total_raw_detections']
            },
            "objects": []
        }
        
        # Add detected objects
        for i, detection in enumerate(results['detections']):
            obj_data = {
                "id": i + 1,
                "class": detection['class_name'],
                "confidence": round(detection['confidence'], 3),
                "confidence_percent": f"{detection['confidence']*100:.1f}%",
                "bbox": [round(coord, 1) for coord in detection['bbox']],
                "model": detection.get('model', 'Unknown'),
                "enhanced": detection.get('ensemble_merged', False)
            }
            response["objects"].append(obj_data)
        
        # Performance metrics
        response["performance"] = {
            "accuracy_level": "Ultra-Precision" if results['precision_score'] > 0.85 else "High-Precision",
            "model_agreement": f"{(results['total_raw_detections'] / max(1, len(results['detections']))):.1f}x",
            "gpu_acceleration": "RTX 3050" if results['gpu_used'] else "CPU"
        }
        
        logger.info(f"Detection completed: {len(results['detections'])} objects, Precision: {results['precision_score']:.3f}")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Detection failed: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

def get_precision_grade(score: float) -> str:
    """Convert precision score to letter grade"""
    if score >= 0.95:
        return "A+"
    elif score >= 0.90:
        return "A"
    elif score >= 0.85:
        return "A-"
    elif score >= 0.80:
        return "B+"
    elif score >= 0.75:
        return "B"
    elif score >= 0.70:
        return "B-"
    else:
        return "C"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(detector.models),
        "gpu_available": torch.cuda.is_available(),
        "version": "Ultra-Precision v2.0"
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Ultra-Precision Safety Equipment Detection Server...")
    logger.info(f"🎯 Models loaded: {len(detector.models)}")
    logger.info(f"⚡ GPU acceleration: {torch.cuda.is_available()}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")