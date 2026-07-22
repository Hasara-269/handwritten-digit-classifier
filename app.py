"""
Interactive GUI Testing Application for Handwritten Digit Classifier.

This application provides a graphical user interface (Tkinter) allowing users
to draw handwritten digits (0-9) on an interactive canvas. The drawn input is
preprocessed using OpenCV and NumPy, then passed to a pre-trained Support Vector
Classifier (SVC) serialized with Joblib to render real-time digit predictions
and confidence metrics.
"""

import os
import sys
import logging
import tkinter as tk
from tkinter import messagebox
import cv2
import joblib
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
CANVAS_SIZE = 280
MODEL_INPUT_SIZE = (8, 8)
MODEL_FILENAME = "digit_model.pkl"
LINE_WIDTH = 18  # Width of white drawing stroke for optimal 8x8 downscaling


class DigitClassifierApp:
    """Tkinter-based Interactive Handwritten Digit Classifier Application."""

    def __init__(self, root: tk.Tk, model_path: str = MODEL_FILENAME):
        self.root = root
        self.root.title("Handwritten Digit Classifier")
        self.root.geometry("400x530")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")

        self.model_path = model_path
        self.model = None
        self.last_x = None
        self.last_y = None

        # In-memory grayscale canvas buffer for direct OpenCV matrix processing
        self.image_buffer = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

        # Load serialized model weights
        self._load_model()

        # Build UI layout
        self._setup_ui()

    def _load_model(self) -> None:
        """Loads the pre-trained Joblib classifier model file."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file '{self.model_path}' not found.")
            messagebox.showerror(
                "Model File Error",
                f"Could not find serialized model file '{self.model_path}'.\n\n"
                "Please run 'python train.py' to generate the model artifact first."
            )
            return

        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Successfully loaded classifier model from '{self.model_path}'.")
        except Exception as e:
            logger.error(f"Failed to load model file '{self.model_path}': {e}")
            messagebox.showerror(
                "Model Load Error",
                f"Failed to deserialize model artifact '{self.model_path}':\n{e}"
            )

    def _setup_ui(self) -> None:
        """Constructs the Tkinter graphical interface components."""
        # Header Label
        title_label = tk.Label(
            self.root,
            text="Handwritten Digit Classifier",
            font=("Helvetica", 16, "bold"),
            fg="#cdd6f4",
            bg="#1e1e2e",
            pady=10
        )
        title_label.pack()

        # Instruction Sub-header
        subtitle_label = tk.Label(
            self.root,
            text="Draw a single digit (0-9) below:",
            font=("Helvetica", 11),
            fg="#a6adc8",
            bg="#1e1e2e"
        )
        subtitle_label.pack(pady=(0, 10))

        # Canvas Frame (Outer border)
        canvas_frame = tk.Frame(self.root, bg="#89b4fa", bd=2)
        canvas_frame.pack()

        # Interactive Drawing Canvas (280x280)
        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            cursor="crosshair",
            highlightthickness=0
        )
        self.canvas.pack()

        # Mouse Event Bindings
        self.canvas.bind("<Button-1>", self._on_button_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_button_release)

        # Dynamic Results Display Card
        self.result_frame = tk.Frame(self.root, bg="#313244", bd=1, relief="solid")
        self.result_frame.pack(fill="x", padx=30, pady=15)

        self.prediction_label = tk.Label(
            self.result_frame,
            text="Prediction: --",
            font=("Helvetica", 14, "bold"),
            fg="#a6e3a1",
            bg="#313244"
        )
        self.prediction_label.pack(pady=4)

        self.confidence_label = tk.Label(
            self.result_frame,
            text="Confidence: --%",
            font=("Helvetica", 11),
            fg="#cdd6f4",
            bg="#313244"
        )
        self.confidence_label.pack(pady=(0, 4))

        # Action Buttons Container
        btn_frame = tk.Frame(self.root, bg="#1e1e2e")
        btn_frame.pack(pady=5)

        # Predict Button
        self.predict_btn = tk.Button(
            btn_frame,
            text="Predict",
            font=("Helvetica", 11, "bold"),
            bg="#89b4fa",
            fg="#11111b",
            activebackground="#b4befe",
            activeforeground="#11111b",
            padx=15,
            pady=5,
            bd=0,
            cursor="hand2",
            command=self.predict_digit
        )
        self.predict_btn.grid(row=0, column=0, padx=8)

        # Clear Button
        self.clear_btn = tk.Button(
            btn_frame,
            text="Clear",
            font=("Helvetica", 11, "bold"),
            bg="#f38ba8",
            fg="#11111b",
            activebackground="#f5e0dc",
            activeforeground="#11111b",
            padx=15,
            pady=5,
            bd=0,
            cursor="hand2",
            command=self.clear_canvas
        )
        self.clear_btn.grid(row=0, column=1, padx=8)

        # Exit Button
        self.exit_btn = tk.Button(
            btn_frame,
            text="Exit",
            font=("Helvetica", 11, "bold"),
            bg="#6c7086",
            fg="#11111b",
            activebackground="#9399b2",
            activeforeground="#11111b",
            padx=15,
            pady=5,
            bd=0,
            cursor="hand2",
            command=self.root.quit
        )
        self.exit_btn.grid(row=0, column=2, padx=8)

    def _on_button_press(self, event: tk.Event) -> None:
        """Handles initial mouse click on the drawing canvas."""
        self.last_x = event.x
        self.last_y = event.y

    def _on_mouse_drag(self, event: tk.Event) -> None:
        """Draws stroke on both Tkinter Canvas and in-memory NumPy OpenCV image buffer."""
        if self.last_x is not None and self.last_y is not None:
            x1, y1 = self.last_x, self.last_y
            x2, y2 = event.x, event.y

            # Render smooth line on Tkinter canvas UI
            self.canvas.create_line(
                x1, y1, x2, y2,
                fill="white",
                width=LINE_WIDTH,
                capstyle=tk.ROUND,
                smooth=True
            )

            # Mirror line onto grayscale NumPy array buffer
            cv2.line(
                self.image_buffer,
                (x1, y1),
                (x2, y2),
                color=255,
                thickness=LINE_WIDTH,
                lineType=cv2.LINE_AA
            )

            self.last_x = x2
            self.last_y = y2

    def _on_button_release(self, event: tk.Event) -> None:
        """Resets stroke tracking coordinates upon mouse release."""
        self.last_x = None
        self.last_y = None

    def clear_canvas(self) -> None:
        """Resets the UI canvas elements and clears the image buffer matrix."""
        self.canvas.delete("all")
        self.image_buffer.fill(0)
        self.prediction_label.config(text="Prediction: --", fg="#a6e3a1")
        self.confidence_label.config(text="Confidence: --%")
        logger.info("Canvas and drawing buffer cleared.")

    def preprocess_image(self) -> np.ndarray | None:
        """
        Preprocesses the drawn 280x280 canvas array to match Scikit-Learn Digits dataset format.

        Steps:
            1. Bounding Box Crop: Finds non-zero pixels and crops to drawing limits.
            2. Aspect Ratio Preserving Pad & Center: Pads cropped digit into a centered square frame.
            3. Smooth & Resize: Applies Gaussian blur and resizes down to 8x8 pixels.
            4. Normalization: Rescales pixel values from [0, 255] to [0, 16] and flattens to 64D vector.

        Returns:
            np.ndarray | None: A 1D feature vector of shape (1, 64) with pixel values scaled [0, 16],
                               or None if the canvas is completely blank.
        """
        # Find coordinates of all non-zero (white drawing) pixels
        non_zero_pts = cv2.findNonZero(self.image_buffer)

        # Fallback check for empty canvas
        if non_zero_pts is None:
            return None

        # 1. Bounding Box Crop
        x, y, w, h = cv2.boundingRect(non_zero_pts)
        cropped = self.image_buffer[y:y + h, x:x + w]

        # 2. Aspect Ratio Preserving Pad & Center
        # Calculate maximum dimension and add padding margin (20%) to keep digit centered
        max_dim = max(w, h)
        margin = int(max_dim * 0.20)
        square_size = max_dim + (2 * margin)

        # Create a black square canvas frame
        square_canvas = np.zeros((square_size, square_size), dtype=np.uint8)

        # Compute offsets to center the cropped digit inside the square frame
        offset_x = (square_size - w) // 2
        offset_y = (square_size - h) // 2
        square_canvas[offset_y:offset_y + h, offset_x:offset_x + w] = cropped

        # 3. Gaussian Blur to smooth stroke edges from mouse drawing
        blurred = cv2.GaussianBlur(square_canvas, (3, 3), 0)

        # 4. Downscale centered square image to 8x8 pixels using area interpolation
        resized_image = cv2.resize(
            blurred,
            MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_AREA
        )

        # 5. Rescale pixel intensity values from [0, 255] to Scikit-Learn Digits range [0, 16]
        scaled_features = (resized_image.astype(np.float64) / 255.0) * 16.0

        # 6. Flatten 8x8 matrix into 1D 64-element feature vector
        feature_vector = scaled_features.reshape(1, -1)
        return feature_vector

    def predict_digit(self) -> None:
        """Executes preprocessing and model inference on the current canvas drawing."""
        if self.model is None:
            messagebox.showwarning(
                "Model Missing",
                "Model is not loaded. Please ensure 'digit_model.pkl' exists."
            )
            return

        try:
            # 1. Execute advanced preprocessing pipeline
            feature_vector = self.preprocess_image()

            # Fallback handling if canvas is blank
            if feature_vector is None:
                messagebox.showinfo(
                    "Empty Canvas",
                    "Please draw a digit on the canvas before clicking Predict."
                )
                return

            # 2. Predict digit class
            prediction = int(self.model.predict(feature_vector)[0])

            # 3. Calculate prediction confidence probability or score
            confidence = 100.0
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(feature_vector)[0]
                confidence = float(probabilities[prediction] * 100.0)
            elif hasattr(self.model, "decision_function"):
                decision_scores = self.model.decision_function(feature_vector)[0]
                # Apply Softmax normalization over decision function outputs
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                softmax_probs = exp_scores / np.sum(exp_scores)
                confidence = float(softmax_probs[prediction] * 100.0)

            # 4. Render results to UI labels
            self.prediction_label.config(text=f"Prediction: {prediction}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
            logger.info(f"Inference completed - Predicted Digit: {prediction}, Confidence: {confidence:.2f}%")

        except Exception as e:
            logger.error(f"Error during inference execution: {e}")
            messagebox.showerror("Inference Error", f"An error occurred during prediction:\n{e}")


def main():
    root = tk.Tk()
    app = DigitClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
