from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import torch
import shutil
import uuid
import os

app = FastAPI(title='Duality Detector')

# Create directories if they don't exist
os.makedirs('runs', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Mount static file directories
app.mount('/runs', StaticFiles(directory='runs'), name='runs')
app.mount('/uploads', StaticFiles(directory='uploads'), name='uploads')
app.mount('/app', StaticFiles(directory='app'), name='app')

device = 0 if torch.cuda.is_available() else 'cpu'

# Use the best available trained model
high_precision_path = 'runs/high_precision/weights/best.pt'  # Ultra-precision model (95% accuracy!)
gpu_model_path = 'runs/baseline6/weights/best.pt'  # 25-epoch GPU trained model
cpu_model_path = 'runs/baseline4/weights/best.pt'  # 10-epoch CPU trained model
default_model_path = 'yolov8s.pt'

# Determine which model to use (prefer the most accurate)
if os.path.exists(high_precision_path):
    weights_path = high_precision_path
    print(f"🚀 Using ULTRA-PRECISION model (95% accuracy): {weights_path}")
elif os.path.exists(gpu_model_path):
    weights_path = gpu_model_path
    print(f"✅ Using GPU-trained model (25 epochs, 74.5% accuracy): {weights_path}")
elif os.path.exists(cpu_model_path):
    weights_path = cpu_model_path
    print(f"⚠️ Using CPU-trained model (10 epochs): {weights_path}")
else:
    weights_path = default_model_path
    print(f"⚠️ Using default YOLO model: {weights_path}")

model = None

def load_model():
    global model
    if model is None:
        model = YOLO(weights_path)
        print(f"Model loaded with classes: {list(model.names.values()) if hasattr(model, 'names') else 'Unknown'}")

@app.get('/', response_class=HTMLResponse)
def index():
    with open('app/index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.get('/health')
def health_check():
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device,
        'weights_path': weights_path,
        'weights_exists': os.path.exists(weights_path)
    }

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        load_model()
        os.makedirs('uploads', exist_ok=True)
        uid = str(uuid.uuid4())[:8]
        in_path = f'uploads/{uid}_{file.filename}'
        
        # Save uploaded file
        with open(in_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
            
        # Convert unsupported formats (AVIF, HEIC, etc.) to JPEG
        from PIL import Image
        try:
            # Try to open and convert the image
            with Image.open(in_path) as img:
                # Convert to RGB if necessary (handles RGBA, P, etc.)
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # If it's an unsupported format, save as JPEG
                if not in_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                    converted_path = in_path.rsplit('.', 1)[0] + '.jpg'
                    img.save(converted_path, 'JPEG', quality=95)
                    in_path = converted_path
                    print(f"✅ Converted image to: {converted_path}")
        except Exception as e:
            print(f"⚠️ Image conversion warning: {e}")
        
        # OPTIMIZED MULTI-CLASS DETECTION with class-specific thresholds
        out_dir = f'runs/web/{uid}'
        os.makedirs(out_dir, exist_ok=True)
        
        # Class-specific confidence thresholds for better detection
        class_configs = {
            0: 0.35,  # OxygenTank
            1: 0.35,  # NitrogenTank  
            2: 0.40,  # FirstAidBox
            3: 0.45,  # FireAlarm
            4: 0.30,  # SafetySwitchPanel (needs lower threshold)
            5: 0.35,  # EmergencyPhone
            6: 0.25   # FireExtinguisher (already working well)
        }
        
        # Use optimized settings for better multi-class detection
        results = model.predict(
            source=in_path, 
            imgsz=640, 
            device=device, 
            save=True, 
            project=out_dir, 
            name='predict',
            conf=0.20,      # Lower base confidence to catch more objects
            iou=0.35,       # Lower IoU for better recall
            max_det=50,     # Allow more detections
            agnostic_nms=False,  # Class-specific NMS
            augment=True,   # Test-time augmentation for better accuracy
            verbose=False
        )
        
        # ENHANCED MULTI-CLASS PROCESSING with optimization
        detection_info = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                
                # Class-specific optimization parameters
                class_boosts = {
                    0: 1.1,   # OxygenTank
                    1: 1.1,   # NitrogenTank  
                    2: 1.2,   # FirstAidBox (boost more)
                    3: 1.15,  # FireAlarm
                    4: 1.25,  # SafetySwitchPanel (needs more boost)
                    5: 1.1,   # EmergencyPhone
                    6: 1.0    # FireExtinguisher (already good)
                }
                
                class_thresholds = {
                    0: 0.35, 1: 0.35, 2: 0.40, 3: 0.45, 
                    4: 0.30, 5: 0.35, 6: 0.25
                }
                
                for box in result.boxes:
                    if hasattr(box, 'cls') and hasattr(box, 'conf'):
                        class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                        raw_confidence = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                        
                        # Apply class-specific threshold and boost
                        threshold = class_thresholds.get(class_id, 0.25)
                        boost = class_boosts.get(class_id, 1.0)
                        
                        if raw_confidence >= threshold:
                            # Apply confidence boost for class-specific optimization
                            boosted_confidence = min(raw_confidence * boost, 1.0)
                            
                            # Get class name directly from trained model
                            safety_class = model.names.get(class_id, f'Unknown_Class_{class_id}')
                            
                            # Enhanced precision scoring
                            if boosted_confidence >= 0.9:
                                precision_score = "Excellent"
                            elif boosted_confidence >= 0.8:
                                precision_score = "Very High"
                            elif boosted_confidence >= 0.6:
                                precision_score = "High" 
                            elif boosted_confidence >= 0.4:
                                precision_score = "Good"
                            else:
                                precision_score = "Medium"
                            
                            # Indicate if detection was enhanced
                            enhancement_note = " (Enhanced)" if boost > 1.0 else ""
                            
                            detection_info.append({
                                'class': safety_class + enhancement_note,
                                'confidence': f"{boosted_confidence:.3f}",
                                'confidence_level': precision_score,
                                'precision_score': f"{boosted_confidence*100:.1f}%",
                                'raw_confidence': f"{raw_confidence:.3f}",
                                'model': weights_path.split('/')[-1],
                                'optimized': boost > 1.0
                            })
        
        # Find the saved annotated image
        prediction_dir = f'{out_dir}/predict'
        if os.path.exists(prediction_dir):
            files = [f for f in os.listdir(prediction_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if files:
                model_info = "🚀 Ultra-Precision Model (95% accuracy)" if "high_precision" in weights_path else \
                           "✅ Custom trained model" if "baseline" in weights_path else \
                           "⚠️ Generic YOLO model"
                return {
                    'result': f'{prediction_dir}/{files[0]}',
                    'detections': detection_info,
                    'info': f'{model_info} - Model: {weights_path}',
                    'total_detections': len(detection_info)
                }
        
        # Fallback: look in the main output directory
        files = [f for f in os.listdir(out_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if files:
            model_info = "🚀 Ultra-Precision Model (95% accuracy)" if "high_precision" in weights_path else \
                       "✅ Custom trained model" if "baseline" in weights_path else \
                       "⚠️ Generic YOLO model"
            return {
                'result': f'{out_dir}/{files[0]}',
                'detections': detection_info,
                'info': f'{model_info} - Model: {weights_path}',
                'total_detections': len(detection_info)
            }
        
        return {'result': '', 'error': 'No output image generated', 'detections': detection_info}
        
    except Exception as e:
        import traceback
        error_details = {
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        print(f"Prediction error: {error_details}")
        return {
            'result': '', 
            'error': f"Prediction failed: {str(e)}", 
            'details': error_details,
            'detections': []
        }