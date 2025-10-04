import argparse
import os
import torch
import glob
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='runs/baseline/weights/best.pt')
    parser.add_argument('--source', default='data/test/images')
    parser.add_argument('--out', default='runs/predictions')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(args.weights)
    results = model.predict(source=args.source, imgsz=640, device=device, save=True, project=args.out, name='.')

if __name__ == '__main__':
    main()