"""
Interactive GUI Testing Application for Handwritten Digit Classifier (MNIST 28x28).

This application provides a Tkinter GUI allowing users to draw digits (0-9) on a 280x280 canvas.
The drawn image is preprocessed into MNIST format:
  1. Cropped tightly around drawing and resized to fit within a 20x20 frame.
  2. Placed inside a 28x28 canvas matrix.
  3. Shifted via scipy.ndimage.center_of_mass so its center of mass aligns at (14, 14).
  4. Scaled to [0.0, 1.0] and flattened into a 784-element feature vector.
A live 28x28 debug preview thumbnail renders the exact feature matrix fed into the MLP model.
"""

from __future__ import annotations

import os
import sys
import logging
import tkinter as tk
from tkinter import messagebox
import cv2
import joblib
import numpy as np
from scipy.ndimage import center_of_mass, shift

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
CANVAS_SIZE = 280
MODEL_INPUT_SIZE = (28, 28)
DEBUG_PREVIEW_SCALE = 3  # 28x28 pixels scaled by 3 = 84x84 preview canvas
MODEL_FILENAME = "digit_model.pkl"
LINE_WIDTH = 18  # Drawing pen stroke width


class DigitClassifierApp:
    """Tkinter-based Interactive Handwritten Digit Classifier Application (MNIST 28x28)."""

    def __init__(self, root: tk.Tk, model_path: str = MODEL_FILENAME):
        self.root = root
        self.root.title("Handwritten Digit Classifier (MNIST 28x28)")
        self.root.geometry("440x640")
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
            pady=8
        )
        title_label.pack()

        # Instruction Sub-header
        subtitle_label = tk.Label(
            self.root,
            text="Draw a digit (0-9) on the canvas below:",
            font=("Helvetica", 10),
            fg="#a6adc8",
            bg="#1e1e2e"
        )
        subtitle_label.pack(pady=(0, 8))

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

        # Dynamic Results & Debug Preview Container Frame
        self.result_frame = tk.Frame(self.root, bg="#313244", bd=1, relief="solid")
        self.result_frame.pack(fill="x", padx=25, pady=12)

        # Left Column: Debug Preview Thumbnail (28x28 Model View)
        preview_container = tk.Frame(self.result_frame, bg="#313244")
        preview_container.pack(side="left", padx=12, pady=10)

        preview_lbl = tk.Label(
            preview_container,
            text="MNIST View\n(28x28 Input)",
            font=("Helvetica", 8, "bold"),
            fg="#a6adc8",
            bg="#313244"
        )
        preview_lbl.pack(pady=(0, 2))

        # 84x84 Debug Canvas (Renders 28x28 image with 3x3 pixel blocks)
        self.debug_canvas = tk.Canvas(
            preview_container,
            width=84,
            height=84,
            bg="#11111b",
            highlightthickness=1,
            highlightbackground="#585b70"
        )
        self.debug_canvas.pack()

        # Right Column: Prediction & Confidence Labels
        info_container = tk.Frame(self.result_frame, bg="#313244")
        info_container.pack(side="left", fill="both", expand=True, padx=(5, 12), pady=10)

        self.prediction_label = tk.Label(
            info_container,
            text="Prediction: --",
            font=("Helvetica", 14, "bold"),
            fg="#a6e3a1",
            bg="#313244",
            anchor="w"
        )
        self.prediction_label.pack(fill="x", pady=4)

        self.confidence_label = tk.Label(
            info_container,
            text="Confidence: --%",
            font=("Helvetica", 11),
            fg="#cdd6f4",
            bg="#313244",
            anchor="w"
        )
        self.confidence_label.pack(fill="x", pady=2)

        # Action Buttons Container
        btn_frame = tk.Frame(self.root, bg="#1e1e2e")
        btn_frame.pack(pady=8)

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
        """Resets the UI canvas elements, debug preview, and clears the image buffer matrix."""
        self.canvas.delete("all")
        self.debug_canvas.delete("all")
        self.image_buffer.fill(0)
        self.prediction_label.config(text="Prediction: --", fg="#a6e3a1")
        self.confidence_label.config(text="Confidence: --%")
        logger.info("Canvas, debug preview thumbnail, and drawing buffer cleared.")

    def _update_debug_preview(self, normalized_28x28: np.ndarray) -> None:
        """
        Renders a 28x28 normalized pixel matrix [0.0, 1.0] onto the 84x84 debug canvas preview.

        Args:
            normalized_28x28 (np.ndarray): 28x28 matrix with pixel values scaled [0.0, 1.0].
        """
        self.debug_canvas.delete("all")

        # Map [0.0, 1.0] range to [0, 255] grayscale integer levels
        uint8_matrix = np.clip(normalized_28x28 * 255.0, 0, 255).astype(np.uint8)

        cell_size = DEBUG_PREVIEW_SCALE  # 3 pixels per cell (28 * 3 = 84)
        for row in range(28):
            for col in range(28):
                val = int(uint8_matrix[row, col])
                if val == 0:
                    continue  # Optimization: canvas background is already dark
                hex_color = f"#{val:02x}{val:02x}{val:02x}"
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                self.debug_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=hex_color,
                    outline=""
                )

    def preprocess_image(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Preprocesses the drawn canvas array into standard 28x28 MNIST format with Center of Mass alignment.

        Pipeline Steps:
            1. Bounding Box Crop: Isolate drawn digit from 280x280 canvas.
            2. Fit into 20x20 Frame: Resize drawing while preserving aspect ratio inside 20x20 pixel bounding box.
            3. Center in 28x28 Canvas: Place 20x20 digit inside a 28x28 pixel frame.
            4. Center of Mass Shift: Compute center of mass using scipy.ndimage.center_of_mass and shift digit to (14, 14).
            5. Normalization: Scale pixel intensities to [0.0, 1.0] float range and flatten to 784-element vector.

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]:
                - 1D feature vector of shape (1, 784) for model inference.
                - 2D matrix of shape (28, 28) for debug UI preview rendering.
                Returns (None, None) if canvas is blank.
        """
        # Find coordinates of all non-zero (white drawing) pixels
        non_zero_pts = cv2.findNonZero(self.image_buffer)

        # Fallback check for empty canvas
        if non_zero_pts is None:
            return None, None

        # 1. Bounding Box Crop
        x, y, w, h = cv2.boundingRect(non_zero_pts)
        cropped = self.image_buffer[y:y + h, x:x + w]

        # 2. Resize cropped drawing to fit inside a 20x20 bounding box (preserving aspect ratio)
        if w > h:
            new_w = 20
            new_h = max(1, int(round((h * 20.0) / w)))
        else:
            new_h = 20
            new_w = max(1, int(round((w * 20.0) / h)))

        resized_20x20 = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 3. Place 20x20 digit inside a centered 28x28 canvas matrix
        padded_28x28 = np.zeros((28, 28), dtype=np.float32)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        padded_28x28[start_y:start_y + new_h, start_x:start_x + new_w] = resized_20x20.astype(np.float32)

        # 4. Center of Mass Shift to exact center (14, 14)
        cy, cx = center_of_mass(padded_28x28)
        if not np.isnan(cy) and not np.isnan(cx):
            shift_y = 14.0 - cy
            shift_x = 14.0 - cx
            centered_28x28 = shift(padded_28x28, [shift_y, shift_x], cval=0.0)
        else:
            centered_28x28 = padded_28x28

        # 5. Normalization to [0.0, 1.0] float range matching MNIST
        img_max = centered_28x28.max()
        if img_max > 0:
            normalized_28x28 = np.clip(centered_28x28 / img_max, 0.0, 1.0)
        else:
            normalized_28x28 = np.zeros((28, 28), dtype=np.float32)

        # 6. Flatten 28x28 matrix into 1D 784-element feature vector
        feature_vector = normalized_28x28.reshape(1, -1)
        return feature_vector, normalized_28x28

    def predict_digit(self) -> None:
        """Executes preprocessing, debug thumbnail update, and model inference on current canvas drawing."""
        if self.model is None:
            messagebox.showwarning(
                "Model Missing",
                "Model is not loaded. Please ensure 'digit_model.pkl' exists."
            )
            return

        try:
            # 1. Execute advanced MNIST 28x28 preprocessing pipeline
            feature_vector, normalized_28x28 = self.preprocess_image()

            # Fallback handling if canvas is blank
            if feature_vector is None or normalized_28x28 is None:
                messagebox.showinfo(
                    "Empty Canvas",
                    "Please draw a digit on the canvas before clicking Predict."
                )
                return

            # 2. Update Live 28x28 Debug Preview Thumbnail in UI
            self._update_debug_preview(normalized_28x28)

            # 3. Predict digit class
            prediction = int(self.model.predict(feature_vector)[0])

            # 4. Calculate prediction confidence probability or score
            confidence = 100.0
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(feature_vector)[0]
                confidence = float(probabilities[prediction] * 100.0)
            elif hasattr(self.model, "decision_function"):
                decision_scores = self.model.decision_function(feature_vector)[0]
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                softmax_probs = exp_scores / np.sum(exp_scores)
                confidence = float(softmax_probs[prediction] * 100.0)

            # 5. Render results to UI labels
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
