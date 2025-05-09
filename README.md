# Multi-Modal Object Detection under Missing Modalities

This project focuses on robust object detection under the missing modality scenario, where multi-modal inputs (e.g., visible and infrared images) are not always available. We propose a knowledge distillation and transfer learning-based framework to improve detection accuracy when only a single modality is accessible.

## 🔍 Project Highlights

- **Task**: Multi-modal object detection with missing modality handling.
- **Main Techniques**:
  - **Knowledge Distillation**: KL Divergence and Prototype Distillation Loss.
  - **Transfer Learning**: Cross-modality feature initialization and fine-tuning.
- **Backbones**: ResNet18 / ResNet34 / ResNet50.
- **Base Detector**: Deformable DETR R50.
- **Dataset**: [M3FD Dataset](https://github.com/JinyuanLiu-CV/TarDAL) (publicly available).

## 🧠 Method Overview

- **KL Divergence Loss**: Aligns prediction distributions of teacher (multi-modal) and student (single-modal) models.
- **Prototype Distillation Loss**: Transfers semantic structure by matching category-level feature prototypes.
- **Transfer Learning**: Trains on one modality and adapts to another via fine-tuning.

## 🚀  How to use
- **Notes**: Make sure in the mmdetection directory 
- **Training the model**: python tools/train.py work_dirs/deformable-detr_r50_16xb2-50e_coco.py
- **Change model config file**: Change the file in work_dirs/deformable-detr_r50_16xb2-50e_coco.py
- **Results**: Traning results are in the directory of work_dirs/deformable-detr_r50_16xb2-50e_coco.py

## 📊 Trainng log and pth file are on Baidu netdisk
- **Location**: https://pan.baidu.com/s/1MjHzUOWfnpkQ2Dxu33Bw8A extraction code: 4dii

## ⚙️ Environment

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.1+
- Other dependencies listed in `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt
