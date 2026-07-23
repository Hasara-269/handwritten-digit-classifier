"""
Interactive GUI Testing Application for Handwritten Digit Classifier.

This application provides a graphical user interface (Tkinter) allowing users
to draw handwritten digits (0-9) on an interactive canvas. The drawn input is
preprocessed using OpenCV (bounding box crop, centering, dilation, Gaussian blur,
and min-max scaling) and passed to a pre-trained ML model serialized with Joblib.
A live 8x8 debug preview thumbnail renders the exact feature matrix fed into the model.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
CANVAS_SIZE = 280
MODEL_INPUT_SIZE = (8, 8)
DEBUG_PREVIEW_SCALE = 8  # 8x8 pixels scaled by 8 = 64x64 preview canvas
MODEL_FILENAME = "digit_model.pkl"
LINE_WIDTH = 20  # Width of white drawing stroke


class DigitClassifierApp:
    """Tkinter-based Interactive Handwritten Digit Classifier Application."""

    def __init__(self, root: tk.Tk, model_path: str = MODEL_FILENAME):
        self.root = root
        self.root.title("Handwritten Digit Classifier")
        self.root.geometry("420x620")
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
        self.result_frame.pack(fill="x", padx=30, pady=12)

        # Left Column: Debug Preview Thumbnail (8x8 Model View)
        preview_container = tk.Frame(self.result_frame, bg="#313244")
        preview_container.pack(side="left", padx=15, pady=10)

        preview_lbl = tk.Label(
            preview_container,
            text="Model View\n(8x8 Input)",
            font=("Helvetica", 8, "bold"),
            fg="#a6adc8",
            bg="#313244"
        )
        preview_lbl.pack(pady=(0, 2))

        # 64x64 Debug Canvas (Renders 8x8 image with 8x8 pixel blocks)
        self.debug_canvas = tk.Canvas(
            preview_container,
            width=64,
            height=64,
            bg="#11111b",
            highlightthickness=1,
            highlightbackground="#585b70"
        )
        self.debug_canvas.pack()

        # Right Column: Prediction & Confidence Labels
        info_container = tk.Frame(self.result_frame, bg="#313244")
        info_container.pack(side="left", fill="both", expand=True, padx=(5, 15), pady=10)

        self.prediction_label = tk.Label(
            info_container,
            text="Prediction: --",
            font=("Helvetica", 14, "bold"),
            fg="#a6e3a1",
            bg="#313244",
            anchor="w"
        )
        self.prediction_label.pack(fill="x", pady=2)

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

    def _update_debug_preview(self, normalized_8x8: np.ndarray) -> None:
        """
        Renders an 8x8 normalized pixel array [0, 16] onto the 64x64 debug canvas preview.

        Args:
            normalized_8x8 (np.ndarray): 8x8 matrix with pixel values scaled [0, 16].
        """
        self.debug_canvas.delete("all")

        # Map [0, 16] range to [0, 255] grayscale integer levels
        uint8_matrix = np.clip((normalized_8x8 / 16.0) * 255.0, 0, 255).astype(np.uint8)

        cell_size = DEBUG_PREVIEW_SCALE  # 8 pixels per cell
        for row in range(8):
            for col in range(8):
                val = int(uint8_matrix[row, col])
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
        Preprocesses the drawn 280x280 canvas array to match Scikit-Learn Digits dataset format.

        Pipeline Steps:
            1. Bounding Box Crop: Finds non-zero pixels and crops tightly around drawing.
            2. Aspect Ratio Preserving Pad & Center: Pads cropped digit into a centered square frame.
            3. Dilation & Gaussian Blur: Morphological dilation preserves thin strokes; blur anti-aliases.
            4. Resize: Downscales centered square to 8x8 matrix using area interpolation.
            5. Min-Max Scaling: Directly maps dynamic range to [0, 16] float intensity scale.

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]:
                - 1D feature vector of shape (1, 64) for model inference.
                - 2D matrix of shape (8, 8) for debug UI preview rendering.
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

        # 2. Aspect Ratio Preserving Pad & Center
        max_dim = max(w, h)
        margin = int(max_dim * 0.20)
        square_size = max_dim + (2 * margin)

        # Create a black square canvas frame
        square_canvas = np.zeros((square_size, square_size), dtype=np.uint8)

        # Compute offsets to center the cropped digit inside the square frame
        offset_x = (square_size - w) // 2
        offset_y = (square_size - h) // 2
        square_canvas[offset_y:offset_y + h, offset_x:offset_x + w] = cropped

        # 3. Canvas Dilation & Gaussian Blur (Prevents thin strokes from vanishing upon 8x8 downscaling)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(square_canvas, kernel, iterations=1)
        blurred = cv2.GaussianBlur(dilated, (3, 3), 0)

        # 4. Downscale centered square image to 8x8 pixels using area interpolation
        resized_image = cv2.resize(
            blurred,
            MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_AREA
        )

        # 5. Min-Max Scaling directly mapped to [0, 16] range
        img_min, img_max = resized_image.min(), resized_image.max()
        if img_max > img_min:
            scaled_2d = ((resized_image.astype(np.float64) - img_min) / (img_max - img_min)) * 16.0
        else:
            scaled_2d = np.zeros_like(resized_image, dtype=np.float64)

        # 6. Flatten 8x8 matrix into 1D 64-element feature vector
        feature_vector = scaled_2d.reshape(1, -1)
        return feature_vector, scaled_2d

    def predict_digit(self) -> None:
        """Executes preprocessing, debug thumbnail update, and model inference on current canvas drawing."""
        if self.model is None:
            messagebox.showwarning(
                "Model Missing",
                "Model is not loaded. Please ensure 'digit_model.pkl' exists."
            )
            return

        try:
            # 1. Execute advanced preprocessing pipeline
            feature_vector, scaled_2d = self.preprocess_image()

            # Fallback handling if canvas is blank
            if feature_vector is None or scaled_2d is None:
                messagebox.showinfo(
                    "Empty Canvas",
                    "Please draw a digit on the canvas before clicking Predict."
                )
                return

            # 2. Update Live 8x8 Debug Preview Thumbnail in UI
            self._update_debug_preview(scaled_2d)

            # 3. Predict digit class
            prediction = int(self.model.predict(feature_vector)[0])

            # 4. Calculate prediction confidence probability or score
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
