# Handwritten Digit Classifier

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F79A3E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An end-to-end Supervised Machine Learning pipeline built with **Scikit-Learn** and **OpenCV** designed to classify handwritten digits ($0 \text{--} 9$) with high precision. This project demonstrates production-grade ML engineering patterns including feature matrix transformation, hyperparameter-tuned Support Vector Classification (SVC), model evaluation on unseen test distributions, and binary model serialization.

---

## 1. Architecture & ML Lifecycle Pipeline

The system executes a deterministic machine learning lifecycle pipeline designed for reproducibility, robustness, and clean separation of concerns:

```
+------------------+     +-----------------------+     +-----------------------+
|  Dataset         |     |  Feature              |     |  Dataset              |
|  Ingestion       | --> |  Engineering          | --> |  Splitting            |
|  (1,797 samples) |     |  (8x8 -> 64D Vector)  |     |  (80% Train / 20% Test|
+------------------+     +-----------------------+     +-----------------------+
                                                                   |
                                                                   v
+------------------+     +-----------------------+     +-----------------------+
|  Model           |     |  Performance          |     |  Supervised           |
|  Serialization   | <-- |  Evaluation           | <-- |  Training             |
|  (digit_model)   |     |  (Accuracy ~98.89%)   |     |  (SVC with RBF Kernel)|
+------------------+     +-----------------------+     +-----------------------+
```

1. **Dataset Ingestion**: Automatically ingests Scikit-Learn's standard Digits dataset comprising $1,797$ normalized $8 \times 8$ pixel grayscale images spanning 10 distinct digit classes ($0 \text{--} 9$).
2. **Feature Engineering**: Flattens 2D spatial pixel matrices ($8 \times 8$) into dense 64-element 1D feature vectors ($\mathbf{x}_i \in \mathbb{R}^{64}$), scaling intensity values for mathematical optimization.
3. **Dataset Splitting**: Executes an 80/20 train-test split (`test_size=0.2`, `random_state=42`) yielding 1,437 training samples and 360 test samples to ensure an unbiased out-of-sample performance estimation.
4. **Supervised Model Training**: Trains a Support Vector Classifier (`sklearn.svm.SVC`) employing a Radial Basis Function (RBF) kernel with targeted hyperparameter selection ($\gamma = 0.001$) to construct non-linear decision boundary hyperplanes.
5. **Model Serialization**: Exports the optimized model state into a lightweight binary weight file (`digit_model.pkl`) using `joblib` serialization for low-latency inference in production.

---

## 2. Performance & Evaluation Metrics

### Out-of-Sample Performance Summary

| Metric | Target Value |
| :--- | :--- |
| **Overall Accuracy** | **98.89%** ($356 / 360$ test instances correctly classified) |
| **Train / Test Split Ratio** | 80% Train (1,437 samples) / 20% Test (360 samples) |
| **Kernel Function** | Radial Basis Function (RBF) |
| **Gamma Hyperparameter ($\gamma$)** | `0.001` |

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

4. **Run the Machine Learning Pipeline**:
   ```bash
   python train.py
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
