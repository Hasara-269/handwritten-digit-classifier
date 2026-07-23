"""
Supervised Learning Pipeline for Handwritten Digit Classification (MNIST 28x28).

This script fetches the OpenML MNIST dataset (784 features, 28x28 resolution),
scales feature matrices to [0.0, 1.0], partitions data into 60,000 training and
10,000 test samples, trains a Multi-Layer Perceptron (MLPClassifier), evaluates out-of-sample
performance metrics, and serializes the trained model weights into 'digit_model.pkl'.
"""

import logging
import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Set up clean terminal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("==================================================")
    logger.info(" Starting MNIST 28x28 ML Training Pipeline ")
    logger.info("==================================================")

    # 1. Dataset Ingestion (MNIST 784)
    logger.info("Fetching OpenML 'mnist_784' dataset (784 features, 28x28 grayscale images)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')

    # Convert to numeric arrays and scale pixel values to [0.0, 1.0] float range
    logger.info("Scaling pixel intensity values to float range [0.0, 1.0]...")
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)

    logger.info(f"Dataset Ingested: {X.shape[0]} total samples, {X.shape[1]} features per sample.")

    # 2. Partition into 60,000 Training and 10,000 Test Samples
    logger.info("Partitioning dataset into 60,000 training samples and 10,000 testing samples...")
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    logger.info(f"Training partition: {len(X_train)} samples | Testing partition: {len(X_test)} samples")

    # 3. Supervised Model Training (MLPClassifier)
    logger.info("Initializing and training Multi-Layer Perceptron (MLPClassifier)...")
    logger.info("Architecture: hidden_layer_sizes=(128, 64), max_iter=20, random_state=42")

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=20,
        random_state=42,
        verbose=True
    )

    model.fit(X_train, y_train)
    logger.info("Model training completed.")

    # 4. Out-of-Sample Performance Evaluation
    logger.info("Evaluating model accuracy on 10,000 test samples...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Overall Model Test Accuracy: {accuracy * 100:.2f}%")

    logger.info("\nDetailed Out-of-Sample Classification Report:\n" + classification_report(y_test, predictions))

    # 5. Model Serialization
    model_filename = 'digit_model.pkl'
    logger.info(f"Serializing trained MNIST model weights to '{model_filename}' via Joblib...")
    joblib.dump(model, model_filename)
    logger.info("MNIST Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
