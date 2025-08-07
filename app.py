### app.py
import streamlit as st
from PIL import Image
from pathlib import Path
from utils import load_model # Import the helper function

# --- PATHS ---
ASSETS_DIR = Path("assets")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "best.pt"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aircraft Detection | YOLOv8",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SESSION STATE INITIALIZATION ---
# This is to store results and control the flow of the app
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {}
if 'current_video' not in st.session_state:
    st.session_state.current_video = None
if 'processing_status' not in st.session_state:
    # Can be 'idle', 'processing', 'done'
    st.session_state.processing_status = 'idle'

# --- MODEL LOADING ---
# This is the core of our application. The model is loaded once and cached.
with st.spinner("Loading the aircraft detection model, please wait..."):
    model = load_model(MODEL_PATH)

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("⚙️ Control Panel")
app_mode = st.sidebar.radio(
    "Choose Input Mode",
    ["Images", "Video"],
    help="Select whether you want to process still images or a video file."
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum probability to consider a detection valid. Lower values detect more objects, but with more potential false positives."
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="Intersection over Union (IoU). Lower values result in fewer overlapping boxes being detected."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Example Gallery")
st.sidebar.markdown("Click to use an example asset.")

# --- EXAMPLE ASSETS ---
# Replace with your actual asset filenames
EXAMPLE_IMAGE_FILES = ["img1.jpeg", "img2.jpg"]
EXAMPLE_VIDEO_FILE = "f16.mp4" # Add your video file here

# Create paths for example images
example_image_paths = [ASSETS_DIR / f for f in EXAMPLE_IMAGE_FILES if (ASSETS_DIR / f).exists()]

# Display example image buttons
for image_path in example_image_paths:
    if image_path.exists():
        if st.sidebar.button(f"Use Example: {image_path.name}", use_container_width=True):
            # This logic will be fully implemented in Phase 3
            st.toast(f"Loading example: {image_path.name}")
    else:
        st.sidebar.warning(f"Asset not found: {image_path.name}")


# --- MAIN INTERFACE ---
st.title("✈️ Aircraft Detection using YOLOv8")
st.markdown(
    """
    This application leverages a fine-tuned **YOLOv8m** model to perform real-time aircraft detection on images and videos.
    Use the sidebar to configure the detection parameters and choose your input source.
    """
)
st.markdown("---")

# --- UI LOGIC FOR IMAGE/VIDEO MODE ---
if app_mode == 'Images':
    st.header("🖼️ Multi-Image Processing")
    st.info("This section will be built in Phase 3. For now, you can see the interface layout.")
    # Placeholder for file uploader and results display
    st.empty() # We will populate this later

elif app_mode == 'Video':
    st.header("🎥 Video Analysis")
    st.info("This section will be built in Phase 4. For now, you can see the interface layout.")
    # Placeholder for video uploader and results display
    st.empty() # We will populate this later


# --- EMPTY STATE MESSAGE ---
if st.session_state.processing_status == 'idle':
    st.info("👆 Upload files or use an example from the sidebar to get started.")