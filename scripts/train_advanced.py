import argparse
import os
import torch
from ultralytics import YOLO

def train_high_precision_model():
    """
    Train a high-precision model with advanced hyperparameters
    """
    # Change to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    print("🚀 Starting HIGH-PRECISION training with advanced hyperparameters...")
    print(f"Device: {'GPU' if device == 0 else 'CPU'}")
    
    # Use YOLOv8m (medium) for better accuracy than YOLOv8s (small)
    model = YOLO('yolov8m.pt')  # Medium model - better accuracy
    
    model.train(
        data='data/config.yaml',
        imgsz=640,
        epochs=50,                    # More epochs for better convergence
        batch=4,                     # Smaller batch for RTX 3050 with medium model
        workers=4,                   # More workers
        name='high_precision',
        project='runs',
        pretrained=True,
        device=device,
        
        # ADVANCED HYPERPARAMETERS FOR HIGH PRECISION
        optimizer='AdamW',           # Best optimizer
        lr0=0.00005,                # Lower learning rate for precision
        lrf=0.00001,                # Lower final learning rate  
        momentum=0.937,             # Optimized momentum
        weight_decay=0.0005,        # Regularization
        
        # AUGMENTATION - More conservative for precision
        mosaic=0.2,                 # Less mosaic (reduces confusion)
        mixup=0.0,                  # No mixup for cleaner training
        copy_paste=0.0,             # No copy-paste
        degrees=5,                  # Less rotation
        translate=0.05,             # Less translation
        scale=0.2,                  # Less scaling
        shear=2,                    # Less shearing
        perspective=0.0001,         # Minimal perspective
        flipud=0.0,                 # No vertical flip
        fliplr=0.5,                 # Keep horizontal flip
        hsv_h=0.010,               # Less hue variation
        hsv_s=0.5,                 # Less saturation variation
        hsv_v=0.2,                 # Less value variation
        
        # PRECISION FOCUSED SETTINGS
        patience=25,                # More patience for convergence
        save_period=5,              # Save every 5 epochs
        val=True,                   # Validate every epoch
        plots=True,                 # Generate plots
        verbose=True,
        
        # QUALITY SETTINGS
        single_cls=False,           # Multi-class detection
        rect=False,                 # Square images for consistency
        cos_lr=True,                # Cosine learning rate schedule
        close_mosaic=15,            # Stop mosaic early
        amp=True,                   # Mixed precision training
        fraction=1.0,               # Use full dataset
        
        # DETECTION SPECIFIC
        box=7.5,                    # Box loss weight
        cls=0.3,                    # Lower class loss weight for precision
        dfl=1.5,                    # Distribution focal loss
        pose=12.0,                  # Pose loss weight
        kobj=1.0,                   # Keypoint obj loss weight
        label_smoothing=0.05,       # Label smoothing for generalization
        nbs=64,                     # Nominal batch size
        overlap_mask=True,          # Mask overlapping
        mask_ratio=4,               # Mask downsample ratio
        dropout=0.1,                # Dropout for regularization
        
        # VALIDATION SETTINGS
        iou=0.6,                    # Higher IoU threshold for stricter evaluation
        conf=None,                  # Let model decide confidence
        max_det=300,                # Max detections per image
        half=False,                 # Full precision
        dnn=False,                  # Use PyTorch
        save=True,                  # Save checkpoints
        save_txt=False,             # Don't save labels
        save_conf=False,            # Don't save confidences
        save_crop=False,            # Don't save crops
        show_conf=True,             # Show confidence
        show_labels=True,           # Show labels
        visualize=False,            # Don't visualize features
        augment=False,              # No test-time augmentation
        agnostic_nms=False,         # Class-specific NMS
        retina_masks=False,         # Standard masks
        embed=None,                 # No embedding
    )

if __name__ == '__main__':
    train_high_precision_model()