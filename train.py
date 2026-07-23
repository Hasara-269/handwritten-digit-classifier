"""
Supervised Learning Pipeline for Handwritten Digit Classification.

This script loads the Scikit-Learn Digits dataset, preprocesses feature matrices,
trains and compares advanced classifier architectures (GridSearch hyperparameter-tuned
Support Vector Classifier and Multi-Layer Perceptron Neural Network), evaluates out-of-sample
performance metrics, and serializes the top-performing model to disk using Joblib.
"""

import logging
import joblib
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Set up clean terminal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logger = logging.getLogger(__name__)


def main():
    logger.info("==================================================")
    logger.info(" Starting ML Pipeline Training & Optimization ")
    logger.info("==================================================")

    # 1. Dataset Ingestion
    logger.info("Loading Scikit-Learn Digits dataset (8x8 grayscale images)...")
    digits = datasets.load_digits()
    n_samples = len(digits.images)

    # Reshape 8x8 images into 1D feature vectors of length 64
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    logger.info(f"Dataset Ingested: {n_samples} total samples, {X.shape[1]} features per sample.")

    # 2. Train-Test Split (80/20)
    logger.info("Splitting dataset into 80% training and 20% testing sets (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training partition: {len(X_train)} samples | Testing partition: {len(X_test)} samples")

    # 3. Model Architecture 1: Multi-Layer Perceptron (MLPClassifier)
    logger.info("\n--- Training Multi-Layer Perceptron (MLPClassifier) ---")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20
    )
    mlp_model.fit(X_train, y_train)
    mlp_preds = mlp_model.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_preds)
    logger.info(f"MLPClassifier Accuracy: {mlp_acc * 100:.2f}%")

    # 4. Model Architecture 2: Hyperparameter-Tuned Support Vector Classifier (SVC)
    logger.info("\n--- Tuning Support Vector Classifier (SVC with GridSearchCV) ---")
    param_grid = {
        'C': [1, 10, 100],
        'gamma': [0.001, 0.0001],
        'kernel': ['rbf']
    }
    grid_search = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    svc_best_model = grid_search.best_estimator_
    logger.info(f"Optimal SVC Parameters: {grid_search.best_params_}")

    svc_preds = svc_best_model.predict(X_test)
    svc_acc = accuracy_score(y_test, svc_preds)
    logger.info(f"Hyperparameter-Tuned SVC Accuracy: {svc_acc * 100:.2f}%")

    # 5. Model Selection & Final Evaluation
    if mlp_acc >= svc_acc:
        best_model = mlp_model
        best_name = "Multi-Layer Perceptron (MLPClassifier)"
        best_acc = mlp_acc
        best_preds = mlp_preds
    else:
        best_model = svc_best_model
        best_name = "Hyperparameter-Tuned Support Vector Classifier (SVC)"
        best_acc = svc_acc
        best_preds = svc_preds

    logger.info(f"\nSelected Best Model Architecture: {best_name} ({best_acc * 100:.2f}% Accuracy)")
    logger.info("\nFinal Out-of-Sample Classification Report:\n" + classification_report(y_test, best_preds))

    # 6. Model Serialization
    model_filename = 'digit_model.pkl'
    logger.info(f"Serializing best model weights to '{model_filename}' via Joblib...")
    joblib.dump(best_model, model_filename)
    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
