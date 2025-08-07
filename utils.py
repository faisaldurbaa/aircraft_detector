### utils.py
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import logging
from PIL import Image
import io
import zipfile
import cv2
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)

@st.cache_resource
def load_model(model_path: Path) -> YOLO:
    """Loads a YOLOv8 model from the specified path."""
    try:
        model = YOLO(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

def resize_image(image: Image.Image, max_size: tuple[int, int] = (1920, 1080)) -> Image.Image:
    """
    Resizes a PIL Image if its dimensions exceed the max size, preserving aspect ratio.
    This prevents memory issues and speeds up processing for very large images.

    Args:
        image (Image.Image): The input image.
        max_size (tuple[int, int]): A tuple containing (max_width, max_height).

    Returns:
        Image.Image: The resized (or original) image.
    """
    if image.width > max_size[0] or image.height > max_size[1]:
        st.toast(f"Image is large ({image.width}x{image.height}), resizing to fit within {max_size[0]}x{max_size[1]}...")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

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

def process_image(model: YOLO, image: Image.Image, confidence: float, iou: float) -> tuple[Image.Image, int, dict]:
    """
    Processes a single image, returning the annotated image, detection count, and speed metrics.
    """
    # Resize image before processing
    image = resize_image(image)
    
    results = model.predict(source=image, conf=confidence, iou=iou)
    result = results[0]
    
    detection_count = len(result.boxes)
    speed_metrics = result.speed  # e.g., {'preprocess': ms, 'inference': ms, 'postprocess': ms}
    
    processed_array = result.plot()
    processed_image = Image.fromarray(cv2.cvtColor(processed_array, cv2.COLOR_BGR2RGB))
    
    return processed_image, detection_count, speed_metrics

def get_video_info(video_bytes: bytes) -> dict:
    """Validates a video from bytes and extracts its properties using OpenCV."""
    # ... (This function remains unchanged from Phase 4) ...
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_bytes)
        video_path = tfile.name
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        Path(video_path).unlink()
        return {"error": "Could not open video file."}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    Path(video_path).unlink()
    if fps == 0 or frame_count == 0:
        return {"error": "Could not read video properties."}
    duration_seconds = frame_count / fps
    return {"width": width, "height": height, "fps": fps, "frame_count": frame_count, "duration": duration_seconds}