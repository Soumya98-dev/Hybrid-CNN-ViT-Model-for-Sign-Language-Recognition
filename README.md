# Hybrid-CNN-ViT-Model-for-Sign-Language-Recognition

CSC 7760 Introduction to Deep Learning Final Project

# Hand Gesture Recognition with Hybrid CNN-ViT Model

## Overview

This repository contains the source code and implementation for our final project in CSC 7760 - Introduction to Deep Learning, Winter 2025 at Wayne State University. The project focuses on building a robust and efficient hand gesture recognition system using a **Hybrid CNN-ViT architecture** on the **HaGRID dataset**.

## Project Objective

To design and implement a deep learning model that leverages the power of **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** for recognizing hand gestures with high accuracy and real-world applicability.

## Key Features

- **Hybrid CNN-ViT Model:** Sequential architecture where CNN extracts local features and ViT captures global dependencies.
- **Dynamic Hand Cropping:** Uses HaGRID-provided bounding boxes to crop hand regions and improve training efficiency.
- **Efficient Training:** Downsampled dataset (50k images) used for quicker experimentation on Google Colab.
- **Attention Layer:** Optional self-attention layer to refine feature learning.

## Repository Structure

```
├── data/
│   └── hagrid/                 # Cropped hand images after preprocessing
├── notebooks/
│   └── HandGestureRecognition.ipynb  # Training pipeline (Colab-ready)
├── scripts/
│   └── preprocess.py          # Hand cropping script
├── requirements.txt
├── README.md
└── results/
    ├── training_accuracy.png
    └── confusion_matrix.png
```

## Dataset

**HAnd Gesture Recognition Image Dataset (HaGRID):**
[https://www.kaggle.com/datasets/kapitanov/hagrid](https://www.kaggle.com/datasets/kapitanov/hagrid)

The dataset consists of over **550,000+ images** across 18 gesture classes. For this project, we used a **50,000 image subset**.

## Setup

Install the required dependencies:

```
pip install -r requirements.txt
```

## How to Use

1. **Preprocess Data:**

```
python scripts/preprocess.py
```

2. **Train Model:**
   Open and run the Jupyter notebook:

```
notebooks/HandGestureRecognition.ipynb
```

3. **Results:**
   Training graphs and confusion matrix are stored in `/results`.

## Future Scope

- Scale training to full HaGRID dataset.
- Integrate MediaPipe keypoints for multi-modal learning.
- Optimize model for mobile/edge deployment.

## References

1. Gupta et al., "Hand Gestures Recognition using Edge Computing System based on Vision Transformer and Lightweight CNN," JAIHC, 2023.
2. Cheng et al., "Lightweight hybrid model based on MobileNet-v2 and Vision Transformer," EAAI, 2024.
3. Kothadiya et al., "SIGNFORMER: DeepVision Transformer for Sign Language Recognition," IEEE Access, 2023.
4. HaGRID Dataset - https://www.kaggle.com/datasets/kapitanov/hagrid
5. Dataset Link - https://drive.google.com/drive/folders/1frhpUFlOzsyoQFVVg69zloenhiiO_UTj?usp=sharing

---

> _This project is submitted as part of the CSC 7760: Introduction to Deep Learning course at Wayne State University._
