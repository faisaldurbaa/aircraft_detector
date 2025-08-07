### utils.py
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

@st.cache_resource
def load_model(model_path: Path) -> YOLO:
    """
    Loads a YOLOv8 model from the specified path.
    Uses @st.cache_resource to load the model only once per session.

    Args:
        model_path (Path): The path to the YOLOv8 model file.

    Returns:
        YOLO: The loaded YOLOv8 model object.
    """
    try:
        logging.info(f"Attempting to load model from: {model_path}")
        model = YOLO(model_path)
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        error_msg = f"Model file not found at {model_path}. Please ensure the file exists."
        logging.error(error_msg)
        st.error(error_msg)
        st.stop()
    except Exception as e:
        error_msg = f"An error occurred while loading the model: {e}"
        logging.error(error_msg)
        st.error(error_msg)
        st.stop()