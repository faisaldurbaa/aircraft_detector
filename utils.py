### utils.py
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import logging
from PIL import Image
import io
import zipfile
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)

@st.cache_resource
def load_model(model_path: Path) -> YOLO:
    """Loads a YOLOv8 model from the specified path."""
    try:
        logging.info(f"Attempting to load model from: {model_path}")
        model = YOLO(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        error_msg = f"An error occurred while loading the model: {e}"
        logging.error(error_msg)
        st.error(error_msg)
        st.stop()

def image_to_bytes(img: Image.Image) -> bytes:
    """Converts a PIL Image to a byte stream for downloading."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def create_zip(images_dict: dict) -> bytes:
    """Creates a ZIP archive from a dictionary of processed images."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for filename, data in images_dict.items():
            processed_img_bytes = image_to_bytes(data['processed'])
            zip_file.writestr(f"detected_{filename}.png", processed_img_bytes)
    return zip_buffer.getvalue()

def process_image(model: YOLO, image: Image.Image, confidence: float, iou: float) -> tuple[Image.Image, int]:
    """Processes a single image with the YOLO model."""
    results = model.predict(source=image, conf=confidence, iou=iou)
    result = results[0]
    detection_count = len(result.boxes)
    processed_array = result.plot()
    processed_image = Image.fromarray(cv2.cvtColor(processed_array, cv2.COLOR_BGR2RGB))
    return processed_image, detection_count