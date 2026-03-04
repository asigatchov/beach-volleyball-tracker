#!/usr/bin/env python3
"""
finetune_combined.py

Fine-tune a YOLO11 model on a prepared two-class dataset (ball + player).
Make sure:
  1. `ultralytics` is installed.
  2. `data.yaml` exists under `data_dir`, for example:
     train: train/images
     val:   val/images
     nc: 2
     names: ['ball','player']
  3. You have placed a suitable COCO pretrained weight file, e.g. `yolo11s.pt`.

Usage:
  python finetune_combined.py

The script loads `yolo11s.pt` from COCO pretrained weights and fine-tunes on `combined_dataset`.
"""
from ultralytics import YOLO

# ====== Configuration ======
# Pretrained weight file
base_weights    = 'yolo11s.pt'                # or 'yolov8s.pt'
# Dataset config file
data_yaml       = 'combined_dataset/data.yaml'
# Training parameters
epochs          = 50                          # total training epochs
imgsz           = 960                         # input image size
batch_size      = 16                          # batch size
device_id       = '0'                         # GPU id, e.g. '0' or 'cpu'
# =====================

def main():
    device = f"cuda:{device_id}" if device_id.isdigit() else device_id

    # 1. Load COCO pretrained model
    model = YOLO(base_weights)

    # 2. Start fine-tuning
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        name='finetune_ball_player',
        exist_ok=True  # overwrite if the same runs directory already exists
    )

    print('Fine-tuning completed. Weights are saved to runs/train/finetune_ball_player/weights/')

if __name__ == '__main__':
    main()
