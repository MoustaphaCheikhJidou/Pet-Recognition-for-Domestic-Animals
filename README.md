# Pet-Recognition-for-Domestic-Animals
# Domestic Animal Recognition

**Deep Learning Project – SID47**  
École Supérieure Polytechnique, April 17, 2025  
Participants:  
- Mohamed Yahya YAHYA (23084)  
- EL Moustapha Cheikh JIDDOU (23231)

---

## 🎯 Project Objective

Automatically classify three species of domestic animals—cats, dogs, and rabbits—from user-supplied photographs. We implement a Convolutional Neural Network (CNN) pipeline with data augmentation, train/test evaluation, and provide a simple REST API for deployment.

---

## 📂 Repository Structure

```text
Animaux_domestiques/
├── Data/                         # Raw and processed image data
│   ├── raw/                      # Original downloads (Unsplash & Google)
│   └── processed/                # Resized (128×128), augmented & split (train/val/test)
├── notebooks/                    # Jupyter notebooks for exploration & experiments
│   ├── Projet_DL.ipynb           # Baseline CNN model & EDA
│   └── Projet_deep_2.ipynb       # Final training with full augmentation pipeline
├── src/                          # Python modules
│   ├── data_pipeline.py         # Download, preprocess & augment images
│   ├── train_cnn.py             # Define, compile & train the CNN
│   └── evaluate.py              # Evaluate on test set & generate metrics/plots
├── models/                       # Saved models
│   └── best_pet_cnn.h5          # Best‐performing CNN weights
├── output/                       # Generated artifacts
│   ├── figures/                 # Loss/accuracy curves, confusion matrix, example predictions
│   └── reports/                 # Final PDF/HTML report
├── Deployment/                   # Deployment scripts & Docker configs
│   ├── app.py                   # Flask API for image prediction
│   └── Dockerfile               # Containerization setup
├── requirements.txt              # Python dependencies
└── README.md                     # ← You are here

