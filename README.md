# ECG-Based Arrhythmia Classification  
### *A Comparative Study of Machine Learning and Deep Learning Models*

---

## Overview

This project presents a complete pipeline for **ECG-based arrhythmia classification** using traditional **machine learning algorithms** and **deep learning models**.  
The study compares multiple algorithms on both **binary** and **multiclass** classification tasks using features extracted from ECG signals (PTB Diagnostic dataset).

The goal is to understand the performance differences between ML and DL approaches and evaluate how dataset imbalance affects various models.

---

## Features

- **Preprocessing of ECG signals and feature extraction**
- **Training and evaluation of:**
  - K-Means Clustering  
  - Logistic Regression  
  - Random Forest  
  - Support Vector Machines (Linear, Polynomial, RBF)  
  - Gaussian Naïve Bayes  
  - Artificial Neural Networks  
- **Performance comparison on:**
  - Unbalanced dataset  
  - Balanced dataset  
- **Evaluation using:**
  - Confusion matrix  
  - Sensitivity (Recall)  
  - Specificity  

---

## Models & Results

### Binary Classification Summary

| Model | Dataset | Sensitivity | Specificity |
|-------|---------|-------------|-------------|
| **K-Means Clustering** | Unbalanced | 0.7323 | 0.2500 |
| **K-Means Clustering** | Balanced | 0.9222 | 0.8215 |
| **Logistic Regression** | Unbalanced | 0.9341 | 0.9231 |
| **Logistic Regression** | Balanced | 0.9529 | 0.9889 |
| **Random Forest** | Unbalanced | **1.0000** | 0.4000 |
| **Random Forest** | Balanced | 0.9765 | 0.9667 |
| **SVM (Linear)** | Unbalanced | 0.8977 | 0.9375 |
| **SVM (Linear)** | Balanced | 0.9655 | 0.9773 |
| **SVM (Polynomial)** | Unbalanced | 0.9545 | 0.8750 |
| **SVM (Polynomial)** | Balanced | 0.9080 | 0.9886 |
| **SVM (RBF)** | Unbalanced | 0.9432 | **1.0000** |
| **SVM (RBF)** | Balanced | **1.0000** | 0.9886 |
| **Gaussian Naïve Bayes** | Unbalanced | 0.9451 | 0.9231 |
| **Gaussian Naïve Bayes** | Balanced | 0.9412 | 0.9222 |
| **Neural Network** | Unbalanced | 0.9890 | 0.9231 |
| **Neural Network** | Balanced | 0.9765 | 0.9889 |

---

### Multiclass Classification Summary

- Cluster-based multiclass mapping was not feasible for many classes in K-Means due to missing cluster associations.
- All ML and DL models were trained on both unbalanced and balanced datasets.
- Performance varies significantly across classes due to class imbalance.


---

## Installation

### Clone the repository
```bash
git clone https://github.com/Rajith-07/ECG-Based-Arrhythmia-Classification.git
cd ECG-Based-Arrhythmia-Classification
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage
### Preprocessing
```bash
python src/preprocessing.py
```
### Train ML Models
```bash
python src/train_ml_models.py
```
### Train Neural Network
```bash
python src/train_nn.py
```

---

## Dataset
This project uses the PTB Diagnostic ECG dataset, with extracted features saved as CSV for reproducibility.

If using raw ECG signals:

- Feature extraction must be repeated

- Preprocessing should match the original workflow

---

## Key Concepts
- Arrhythmia classification

- Handling imbalanced data

- Comparison of ML vs DL

- Confusion matrix analysis

- Sensitivity & specificity evaluation

---

## Tech Stack
### Python

- NumPy, Pandas

- Scikit-learn

- TensorFlow / Keras

- Matplotlib, Seaborn

---

## Authors
- Rajith S (CB.EN.U4CCE23039)

- S.P. Darshan (CB.EN.U4CCE23043)

- Sriharish V.J. (CB.EN.U4CCE23051)

---

## License
This project is licensed under the **MIT License**.

---

## Acknowledgments
Special thanks to the PTB Diagnostic ECG dataset contributors and all open-source libraries used in this work.
