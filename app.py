### app.py
import streamlit as st
from PIL import Image
from pathlib import Path
import time

# Import helper functions from utils.py
from utils import load_model, image_to_bytes, create_zip, process_image

# --- PATHS ---
ASSETS_DIR = Path("assets")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "best.pt"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aircraft Detection",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SESSION STATE INITIALIZATION ---
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {}
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = 'idle'

# --- MODEL LOADING ---
with st.spinner("Loading the detection model, please wait..."):
    model = load_model(MODEL_PATH)

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("⚙️ Control Panel")
app_mode = st.sidebar.radio(
    "Choose Input Mode", ["Images", "Video"],
    help="Select whether you want to process still images or a video file."
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
    help="Minimum probability to consider a detection valid."
)
iou_threshold = st.sidebar.slider(
    "IoU Threshold", 0.0, 1.0, 0.45, 0.05,
    help="Lower values result in fewer overlapping boxes."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Example Gallery")
st.sidebar.markdown("Click a button to try an example.")

# --- EXAMPLE GALLERY WITH VISIBLE BUTTONS ---
EXAMPLE_IMAGE_FILES = ["img1.jpeg", "img2.jpg"]
example_image_paths = [ASSETS_DIR / f for f in EXAMPLE_IMAGE_FILES if (ASSETS_DIR / f).exists()]

for image_path in example_image_paths:
    with st.sidebar.container(border=True):
        st.image(str(image_path), use_container_width=True)
        if st.button("Click to Try this Example", key=f"try_{image_path.name}", use_container_width=True):
            st.session_state.processing_status = 'processing'
            original_image = Image.open(image_path).convert("RGB")
            processed_image, detection_count = process_image(model, original_image, confidence_threshold, iou_threshold)
            st.session_state.processed_images[image_path.name] = {
                'original': original_image, 'processed': processed_image, 'detection_count': detection_count
            }
            st.session_state.processing_status = 'done'
            st.toast(f"Processed example '{image_path.name}' with {detection_count} detections!")
            st.rerun()

# --- MAIN INTERFACE ---
st.title("✈️ Aircraft Detection")
st.markdown("This application performs aircraft detection in images and videos. Configure parameters in the sidebar.")
st.markdown("---")

if app_mode == 'Images':
    st.header("🖼️ Multi-Image Processing")

    # The file uploader with the label hidden, as requested.
    uploaded_files = st.file_uploader(
        "Upload Images",
        label_visibility="collapsed",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )

    if uploaded_files:
        # The subheader below the uploader has been removed, as requested.
        new_files_to_show = [f for f in uploaded_files if f.name not in st.session_state.processed_images]
        if new_files_to_show:
            cols = st.columns(4)
            for idx, uploaded_file in enumerate(new_files_to_show):
                with cols[idx % 4]:
                    st.image(uploaded_file, caption=f"{uploaded_file.name[:20]}...", use_container_width=True)

        if st.button("Process All Uploaded Images", type="primary", use_container_width=True):
            files_to_process = [f for f in uploaded_files if f.name not in st.session_state.processed_images]
            if files_to_process:
                st.session_state.processing_status = 'processing'
                progress_bar = st.progress(0, text="Starting processing...")
                for i, uploaded_file in enumerate(files_to_process):
                    progress_text = f"Processing file {i + 1}/{len(files_to_process)}: {uploaded_file.name}"
                    progress_bar.progress((i + 1) / len(files_to_process), text=progress_text)
                    original_image = Image.open(uploaded_file).convert("RGB")
                    proc_img, count = process_image(model, original_image, confidence_threshold, iou_threshold)
                    st.session_state.processed_images[uploaded_file.name] = {
                        'original': original_image, 'processed': proc_img, 'detection_count': count
                    }
                progress_bar.empty()
                st.success(f"✅ Processed {len(files_to_process)} new images.")
                st.rerun()
            else:
                st.info("All uploaded images have already been processed.")

    if st.session_state.processed_images:
        st.markdown("---")
        st.header("🔍 Detection Results")

        col1, col2, _ = st.columns([0.3, 0.3, 0.4])
        with col1:
             if st.button("🗑️ Clear All Results", use_container_width=True):
                st.session_state.processed_images = {}
                st.session_state.processing_status = 'idle'
                st.rerun()
        with col2:
             if st.session_state.processed_images:
                zip_bytes = create_zip(st.session_state.processed_images)
                st.download_button("📥 Download All as ZIP", zip_bytes, "detected_images.zip", "application/zip", use_container_width=True)

        view_mode = st.radio("View Mode:", ["Side-by-side", "Grid View"], horizontal=True, label_visibility="collapsed")
        st.markdown("---")

        if view_mode == "Side-by-side":
            for filename, data in reversed(list(st.session_state.processed_images.items())):
                st.markdown(f"#### {filename} (Detections: {data['detection_count']})")
                c1, c2 = st.columns(2)
                c1.image(data['original'], "Original", use_container_width=True)
                c2.image(data['processed'], "Processed", use_container_width=True)
                st.download_button("Download Image", image_to_bytes(data['processed']), f"detected_{filename}.png", "image/png")
                st.markdown("---")
        else:
            st.subheader("Grid of Processed Images")
            cols = st.columns(3)
            for idx, (filename, data) in enumerate(reversed(list(st.session_state.processed_images.items()))):
                with cols[idx % 3]:
                    st.image(data['processed'], f"{filename} ({data['detection_count']} detections)", use_container_width=True)

elif app_mode == 'Video':
    st.header("🎥 Video Analysis")
    st.info("This section will be built in Phase 4.")