# Handwritten Digit Classifier

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F79A3E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An end-to-end Supervised Machine Learning pipeline built with **Scikit-Learn**, **OpenCV**, and **SciPy** designed to classify handwritten digits ($0 \text{--} 9$) with high precision. Trained on the full **MNIST 784** dataset (28x28 resolution), this project features Center-of-Mass image normalization, Multi-Layer Perceptron neural network classification, real-time 28x28 input thumbnail preview, and binary model serialization.

---

## 1. Architecture & ML Lifecycle Pipeline

The system executes a deterministic machine learning lifecycle pipeline designed for reproducibility, robustness, and clean separation of concerns:

```
+-------------------+     +-----------------------+     +-----------------------+
|  Dataset          |     |  Feature              |     |  Dataset              |
|  Ingestion        | --> |  Engineering          | --> |  Splitting            |
|  (MNIST 70,000)   |     |  (28x28 -> 784 Vector)|     |  (60k Train / 10k Test|
+-------------------+     +-----------------------+     +-----------------------+
                                                                    |
                                                                    v
+-------------------+     +-----------------------+     +-----------------------+
|  Model            |     |  Performance          |     |  Supervised Training  |
|  Serialization    | <-- |  Evaluation           | <-- |  (MLPClassifier       |
|  (digit_model)    |     |  (Test Accuracy ~97%) |     |   128x64 Architecture)|
+-------------------+     +-----------------------+     +-----------------------+
```

1. **Dataset Ingestion**: Automatically fetches OpenML's `mnist_784` dataset comprising 70,000 normalized $28 \times 28$ pixel grayscale images spanning 10 digit classes ($0 \text{--} 9$).
2. **Feature Engineering**: Scales pixel intensity values to $[0.0, 1.0]$ float range and flattens spatial matrices into dense 784-element feature vectors ($\mathbf{x}_i \in \mathbb{R}^{784}$).
3. **Dataset Splitting**: Partitions the dataset into 60,000 training samples and 10,000 test samples for out-of-sample performance evaluation.
4. **Supervised Model Training**: Trains a Multi-Layer Perceptron neural network (`MLPClassifier`, hidden layers: 128x64, `max_iter=20`, `random_state=42`).
5. **Model Serialization**: Exports trained network weights to `digit_model.pkl` via `joblib` for real-time inference.

---

## 2. Performance & Evaluation Metrics

### Out-of-Sample Performance Summary

| Metric | Target Value |
| :--- | :--- |
| **Dataset Resolution** | $28 \times 28$ pixels ($784$ features) |
| **Train / Test Partition** | 60,000 Training Samples / 10,000 Test Samples |
| **Model Architecture** | MLPClassifier `(128, 64)`, `max_iter=20`, `random_state=42` |
| **Pre-processing Normalization** | Bounding box crop (20x20), 28x28 padding, Center-of-Mass shift to (14, 14), float [0.0, 1.0] scaling |
| **Live UI Debug Preview** | Real-time $28 \times 28$ model input thumbnail display |

### Detailed Classification Breakdown

```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        33
           1       1.00      1.00      1.00        28
           2       1.00      1.00      1.00        33
           3       1.00      0.97      0.99        34
           4       1.00      1.00      1.00        46
           5       0.98      0.98      0.98        47
           6       0.97      1.00      0.99        35
           7       0.97      1.00      0.99        34
           8       1.00      0.97      0.98        30
           9       0.97      0.97      0.97        40

    accuracy                           0.99       360
   macro avg       0.99      0.99      0.99       360
weighted avg       0.99      0.99      0.99       360
```

> [!NOTE]
> **Technical Note on Out-of-Sample Validation**: Evaluated on strictly un-seen test instances, holdout evaluation is critical to ensure the decision boundaries generalize effectively to out-of-sample data distributions. Evaluating on training data alone risks masking overfitting and high-variance optimization errors.

---

## 3. Repository Structure

```text
handwritten-digit-classifier/
├── .gitignore          # Filters out python bytecode, model binaries, and virtual environments
├── LICENSE             # Open-source MIT License terms
├── README.md           # Comprehensive technical documentation
├── app.py              # Interactive Tkinter GUI application for live digit classification
├── digit_model.pkl     # Serialized Support Vector Machine binary model weights
├── requirements.txt    # Production dependency declarations
└── train.py            # Machine learning pipeline script (data loading, training, evaluation)
```

---

## 4. Quick Start & Execution Guide

### Prerequisites

- **Python 3.9+** installed on your system.
- `git` version control tool.

### Setup & Pipeline Execution

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Hasara-269/handwritten-digit-classifier.git
   cd handwritten-digit-classifier
   ```

2. **Create & Activate a Virtual Environment**:
   - **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install Project Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the Machine Learning Training Pipeline**:
   ```bash
   python train.py
   ```

5. **Launch the Interactive GUI Application**:
   ```bash
   python app.py
   ```

---

## 5. Tech Stack

| Library / Tool | Role in Project |
| :--- | :--- |
| **[Scikit-Learn](https://scikit-learn.org/)** | Model selection, dataset loading, SVC training algorithm, and classification metrics. |
| **[NumPy](https://numpy.org/)** | N-dimensional array processing and vector space matrix flattening operations. |
| **[OpenCV](https://opencv.org/)** | Computer vision utilities for image pre-processing and dynamic array transformations. |
| **[Joblib](https://joblib.readthedocs.io/)** | Lightweight, high-efficiency Python object serialization for model persistence. |
| **[Matplotlib](https://matplotlib.org/)** | Plotting and graphical rendering for data distribution diagnostics. |

---

## 6. License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for complete licensing information.
