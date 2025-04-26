# A-YOLOv8-Based-Framework-for-Automated-Detection-of-Surface-Defects-in-Printed-Circuit-Boards
An end-to-end YOLOv8-based system for detecting six types of PCB surface defects. It includes dataset conversion, automated train/val split, model training, and real-time inference, offering a fast and accurate solution for PCB quality inspection.

---

# PCB Defect Detection using YOLOv8

Detect and classify surface defects on Printed Circuit Boards (PCBs) using a deep learning pipeline built with YOLOv8.  
The project covers dataset preparation, model training, validation, and real-time inference.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 🚀 Overview

This project automates the detection of six common PCB surface defects using the YOLOv8 object detection model.  
It includes:

- Data extraction and formatting
- XML to YOLO annotation conversion
- Dataset splitting (train/val)
- YOLOv8 model training
- Inference and result visualization

---

## 📦 Dataset

- Format: Pascal VOC (XML annotations + images)
- Size: ~1386 images
- Classes:
  - Missing Hole
  - Mouse Bite
  - Open Circuit
  - Short
  - Spur
  - Spurious Copper

---

## 🏗️ Project Structure

```bash
.
├── pcbarchive.zip         # Original dataset zip
├── dataset/               # Processed dataset in YOLO format
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
├── pcb.yaml               # YOLOv8 configuration file
├── runs/                  # Training runs and model weights
├── test.jpg               # Sample test image
├── result.jpg             # Output image after detection
└── main.py                # Full project script
```

---



## 🧠 Model Details

- **Model Used**: YOLOv8n (Nano Version)
- **Epochs**: 20
- **Image Size**: 640x640
- **Optimization**: Automatically handles missing labels and skipped annotations.

---

## 📊 Results

- High accuracy in detecting all six types of defects.
- Real-time detection enabled.
- Lightweight model suitable for deployment on edge devices.

---

## 📄 License


---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Kaggle PCB Dataset](https://www.kaggle.com/datasets)

---

## 👨‍💻 Author
- **Name**: ABHINAV K R
- **Contact**: abhinavnowkr@gmailcom
- **LinkedIn**: [abhinavkravi](https://www.linkedin.com/in/abhinavkravi/)
