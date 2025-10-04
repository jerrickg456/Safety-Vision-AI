import argparse
import os
import torch
from ultralytics import YOLO

def train_ultra_precision_model():
    """
    Train an ultra-precision model with YOLOv8x and advanced techniques
    """
    # Change to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    print("🚀 Starting ULTRA-PRECISION training with YOLOv8x...")
    print(f"Device: {'GPU (RTX 3050)' if device == 0 else 'CPU'}")
    
    # Use YOLOv8x (extra-large) for maximum accuracy
    model = YOLO('yolov8x.pt')  # Extra-large model - best accuracy
    
    model.train(
        data='data/enhanced_config.yaml',
        imgsz=832,                    # Higher resolution for better detail detection
        epochs=100,                   # Much more epochs for convergence
        batch=2,                     # Smaller batch for RTX 3050 with XL model
        workers=4,                   
        name='ultra_precision',
        project='runs',
        pretrained=True,
        device=device,
        
        # ULTRA-PRECISION HYPERPARAMETERS
        optimizer='AdamW',           
        lr0=0.00001,                # Very low learning rate for stability
        lrf=0.000001,               # Ultra-low final learning rate  
        momentum=0.95,              # Higher momentum for stability
        weight_decay=0.001,         # More regularization
        warmup_epochs=10,           # Longer warmup
        warmup_momentum=0.9,        # Warm momentum
        warmup_bias_lr=0.01,        # Warmup bias learning rate
        
        # MINIMAL AUGMENTATION FOR PRECISION
        mosaic=0.0,                 # No mosaic - cleaner training
        mixup=0.0,                  # No mixup
        copy_paste=0.0,             # No copy-paste
        degrees=2,                  # Minimal rotation
        translate=0.02,             # Minimal translation
        scale=0.1,                  # Minimal scaling
        shear=1,                    # Minimal shearing
        perspective=0.00001,        # Almost no perspective
        flipud=0.0,                 # No vertical flip
        fliplr=0.3,                 # Reduced horizontal flip
        hsv_h=0.005,               # Minimal hue variation
        hsv_s=0.3,                 # Minimal saturation variation
        hsv_v=0.1,                 # Minimal value variation
        erasing=0.0,               # No random erasing
        
        # ULTRA-QUALITY SETTINGS
        patience=50,                # Much more patience
        save_period=10,             # Save every 10 epochs
        val=True,                   
        plots=True,                 
        verbose=True,
        
        # PRECISION FOCUSED
        single_cls=False,           
        rect=True,                  # Rectangular training for efficiency
        cos_lr=True,                # Cosine learning rate
        close_mosaic=50,            # Stop mosaic very early
        amp=False,                  # Disable mixed precision for stability
        fraction=1.0,               # Use full dataset
        
        # ULTRA-STRICT DETECTION
        box=10.0,                   # Higher box loss weight
        cls=0.2,                    # Lower class loss for precision
        dfl=2.0,                    # Higher distribution focal loss
        label_smoothing=0.01,       # Minimal label smoothing
        nbs=64,                     
        overlap_mask=True,          
        mask_ratio=4,               
        dropout=0.2,                # More dropout for regularization
        
        # VALIDATION SETTINGS
        iou=0.7,                    # Very high IoU threshold
        conf=None,                  
        max_det=200,                # Fewer max detections
        half=False,                 # Full precision
        dnn=False,                  
        save=True,                  
        save_txt=True,              # Save results
        save_conf=True,             # Save confidences
        save_crop=False,            
        show_conf=True,             
        show_labels=True,           
        visualize=False,            
        augment=False,              # No test augmentation for consistency
        agnostic_nms=False,         # Class-specific NMS
        retina_masks=False,         
        
        # MULTI-SCALE TRAINING
        multi_scale=True,           # Enable multi-scale training
        
        # ADVANCED LOSS FUNCTIONS
        # fl_gamma=0.0,              # Focal loss gamma (deprecated)
        
        # ENSEMBLE TECHNIQUES
        exist_ok=False,            # Don't overwrite
    )

if __name__ == '__main__':
    train_ultra_precision_model()