"""
Supervised Learning Pipeline for Handwritten Digit Classification.

This script loads the Scikit-Learn Digits dataset, preprocesses the data,
splits it into training and testing sets, trains a Support Vector Machine (SVM)
classifier, evaluates its accuracy, and saves the trained model to disk.
"""

import logging
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Set up terminal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading the Scikit-Learn Digits dataset...")
    # The digits dataset consists of 8x8 pixel images of digits
    digits = datasets.load_digits()
    
    # Extract images and labels
    logger.info("Reshaping image arrays from 2D matrices into 1D feature vectors...")
    n_samples = len(digits.images)
    
    # Reshape the 8x8 images into 1D feature vectors of length 64
    X = digits.images.reshape((n_samples, -1))
    y = digits.target
    
    logger.info(f"Dataset loaded. Total samples: {n_samples}, Features per sample: {X.shape[1]}")
    
    # Split the dataset into 80% training and 20% testing sets
    logger.info("Splitting dataset into 80% training and 20% testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Training set size: {len(X_train)} samples")
    logger.info(f"Testing set size: {len(X_test)} samples")
    
    # Train a Support Vector Machine (SVM) classifier with SVC
    logger.info("Initializing and training the Support Vector Machine (SVC)...")
    # Gamma is set to 0.001 which is typical for this dataset to achieve good performance
    model = SVC(gamma=0.001, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Model training completed.")
    
    # Evaluate model accuracy on unseen test data
    logger.info("Evaluating model on test data...")
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Output classification report
    logger.info("\nClassification Report:\n" + classification_report(y_test, predictions))
    
    # Serialize and save the trained model to disk
    model_filename = 'digit_model.pkl'
    logger.info(f"Saving the trained model to '{model_filename}'...")
    joblib.dump(model, model_filename)
    logger.info("Pipeline execution finished successfully.")

if __name__ == "__main__":
    main()
