import argparse
import os
import torch
from ultralytics import YOLO

EPOCHS = 30
MOSAIC = 0.4
OPTIMIZER = 'AdamW'
MOMENTUM = 0.9
LR0 = 1e-4
LRF = 1e-4
BATCH = 16
WORKERS = 2
SINGLE_CLS = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch', type=int, default=BATCH)
    args = parser.parse_args()
    
    # Change to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    model = YOLO('yolov8s.pt')
    model.train(
        data='data/config.yaml',
        imgsz=640,
        epochs=args.epochs,
        batch=args.batch,
        workers=WORKERS,
        name='baseline',
        project='runs',
        pretrained=True,
        device=device,
        single_cls=SINGLE_CLS,
        mosaic=MOSAIC,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM
    )