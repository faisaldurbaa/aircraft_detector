### app.py
import streamlit as st
from PIL import Image
from pathlib import Path
import time
import cv2
import tempfile
import io
import numpy as np

# Import helper functions from utils.py
from utils import (
    load_model, image_to_bytes, create_zip, 
    process_image, get_video_info
)

# --- PATHS & CONFIG ---
ASSETS_DIR = Path("assets")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "best.pt"

EXAMPLE_IMAGE_FILES = ["img1.jpeg", "img2.jpg"]
EXAMPLE_VIDEO_FILE = "f16.mp4"
FRAME_SKIP = 2

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aircraft Detection",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SESSION STATE INITIALIZATION ---
if 'processed_images' not in st.session_state: st.session_state.processed_images = {}
if 'uploaded_video_bytes' not in st.session_state: st.session_state.uploaded_video_bytes = None
if 'processed_video_bytes' not in st.session_state: st.session_state.processed_video_bytes = None
if 'video_metrics' not in st.session_state: st.session_state.video_metrics = {}
if 'video_upload_key' not in st.session_state: st.session_state.video_upload_key = 0
if 'original_video_name' not in st.session_state: st.session_state.original_video_name = ""

# --- MODEL LOADING ---
with st.spinner("Loading the aircraft detection model... This may take a moment."):
    model = load_model(MODEL_PATH)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Control Panel")
app_mode = st.sidebar.radio(
    "Choose Input Mode", ["Images", "Video"],
    help="Select whether to process still images or a video file."
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
    help="Minimum probability to consider a detection valid. Higher values are stricter."
)
iou_threshold = st.sidebar.slider(
    "IoU Threshold", 0.0, 1.0, 0.45, 0.05,
    help="Controls how much overlap is allowed between detected boxes. Lower values mean less overlap."
)
st.sidebar.markdown("---")

# --- CONTEXTUAL SIDEBAR EXAMPLES ---
if app_mode == "Images":
    st.sidebar.subheader("🖼️ Image Examples")
    st.sidebar.markdown("Click a button to process an example image.")
    example_image_paths = [ASSETS_DIR / f for f in EXAMPLE_IMAGE_FILES if (ASSETS_DIR / f).exists()]
    for image_path in example_image_paths:
        with st.sidebar.container(border=True):
            st.image(str(image_path), use_container_width=True)
            if st.button("Click to Try this Example", key=f"try_{image_path.name}", use_container_width=True):
                original_image = Image.open(image_path).convert("RGB")
                proc_img, count, speed = process_image(model, original_image, confidence_threshold, iou_threshold)
                st.session_state.processed_images[image_path.name] = {
                    'original': original_image, 'processed': proc_img, 'detection_count': count, 'metrics': speed
                }
                st.rerun()
elif app_mode == "Video":
    st.sidebar.subheader("🎥 Video Example")
    st.sidebar.markdown("Click a button to load the example video.")
    example_video_path = ASSETS_DIR / EXAMPLE_VIDEO_FILE
    if example_video_path.exists():
        with st.sidebar.container(border=True):
            st.video(str(example_video_path))
            if st.button("Click to Try this Video", use_container_width=True, key="ex_video"):
                with open(example_video_path, "rb") as f:
                    st.session_state.uploaded_video_bytes = f.read()
                st.session_state.original_video_name = EXAMPLE_VIDEO_FILE
                st.session_state.processed_video_bytes = None
                st.session_state.video_metrics = {}
                st.rerun()

with st.sidebar.expander("ℹ️ About This App"):
    st.markdown("""
    This application uses a fine-tuned object detection model to identify aircraft in images and videos.
    - **Fast & Accurate:** Built for performance and precision.
    - **User-Friendly:** Easy-to-use interface with real-time feedback.
    - **Adjustable:** Tune detection parameters to fit your needs.
    """)

# --- MAIN INTERFACE ---
st.title("✈️ Aircraft Detection")
st.markdown("Upload your own media or use the examples in the sidebar to get started.")
st.markdown("---")

if app_mode == 'Images':
    st.header("🖼️ Multi-Image Processing")
    uploaded_files = st.file_uploader(
        "Upload Images", label_visibility="collapsed", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Process All Uploaded Images", type="primary", use_container_width=True):
            files_to_process = [f for f in uploaded_files if f.name not in st.session_state.processed_images]
            if files_to_process:
                with st.spinner(f"Analyzing {len(files_to_process)} images..."):
                    for file in files_to_process:
                        orig_img = Image.open(file).convert("RGB")
                        proc_img, count, speed = process_image(model, orig_img, confidence_threshold, iou_threshold)
                        st.session_state.processed_images[file.name] = {'original': orig_img, 'processed': proc_img, 'detection_count': count, 'metrics': speed}
                st.rerun()

    if st.session_state.processed_images:
        st.header("🔍 Detection Results")
        total_detections = sum(data['detection_count'] for data in st.session_state.processed_images.values())
        total_inference_time_ms = sum(data['metrics']['inference'] for data in st.session_state.processed_images.values())
        avg_inference_time = total_inference_time_ms / len(st.session_state.processed_images) if st.session_state.processed_images else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Images Processed", len(st.session_state.processed_images))
        m2.metric("Total Aircraft Detected", f"{total_detections} ✈️")
        m3.metric("Avg. Inference Time", f"{avg_inference_time:.1f} ms/image")
        
        if total_detections == 0:
            st.warning("No aircraft were detected in the processed images. Try adjusting the confidence threshold.", icon="⚠️")
        
        col1, col2, _ = st.columns([0.3, 0.3, 0.4])
        if col1.button("🗑️ Clear All Results", use_container_width=True):
            st.session_state.processed_images.clear()
            st.rerun()
        if col2.download_button("📥 Download All as ZIP", create_zip(st.session_state.processed_images), "detected_images.zip", "application/zip", use_container_width=True):
            pass
        
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
        else: # Grid View
            cols = st.columns(3)
            for idx, (filename, data) in enumerate(reversed(list(st.session_state.processed_images.items()))):
                with cols[idx % 3]:
                    st.image(data['processed'], f"{filename} ({data['detection_count']} detections)", use_container_width=True)

elif app_mode == 'Video':
    st.header("🎥 Video Analysis")
    if st.session_state.processed_video_bytes:
        st.subheader("Processed Video")
        m1, m2, m3 = st.columns(3)
        m1.metric("Video Duration", f"{st.session_state.video_metrics.get('duration', 0):.1f}s")
        m2.metric("Total Aircraft Detected", f"{st.session_state.video_metrics.get('total_detections', 0)} ✈️")
        m3.metric("Processing FPS", f"{st.session_state.video_metrics.get('fps', 0):.1f}")
        st.video(st.session_state.processed_video_bytes)
        st.download_button( "Download Processed Video", st.session_state.processed_video_bytes, f"detected_{st.session_state.original_video_name}", "video/mp4", use_container_width=True)
        if st.button("Clear Video & Start Over", use_container_width=True, type="primary"):
            st.session_state.uploaded_video_bytes = None
            st.session_state.processed_video_bytes = None
            st.session_state.video_metrics = {}
            st.session_state.video_upload_key += 1
            st.rerun()

    elif st.session_state.uploaded_video_bytes:
        # ... (same as before)
        pass # UI to show preview and process button

    else:
        uploaded_file = st.file_uploader( "Upload a video file for analysis", label_visibility="collapsed", type=["mp4", "avi", "mov"], key=f"video_uploader_{st.session_state.video_upload_key}")
        if uploaded_file:
            st.session_state.uploaded_video_bytes = uploaded_file.getvalue()
            st.session_state.original_video_name = uploaded_file.name
            st.session_state.video_metrics = {}
            st.rerun()

if app_mode == 'Video' and st.session_state.uploaded_video_bytes and not st.session_state.processed_video_bytes:
    # This block now runs only when a video is uploaded but not yet processed
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Original: {st.session_state.original_video_name}")
        st.video(st.session_state.uploaded_video_bytes)
    with col2:
        st.subheader("Processing Preview")
        preview_container = st.empty()

    if st.button("Process Video", type="primary", use_container_width=True):
        video_info = get_video_info(st.session_state.uploaded_video_bytes)
        
        with st.spinner("Analyzing video... This could take a while for longer videos."):
            start_time = time.time()
            temp_dir = tempfile.mkdtemp()
            output_video_path = str(Path(temp_dir) / "processed.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_video_path, fourcc, video_info['fps'], (video_info['width'], video_info['height']))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(st.session_state.uploaded_video_bytes)
                cap = cv2.VideoCapture(tfile.name)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0, text="Starting processing...")
            total_detections_in_video = 0
            
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                
                if frame_idx % FRAME_SKIP == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = model.predict(source=frame_rgb, conf=confidence_threshold, iou=iou_threshold, verbose=False)
                    annotated_frame = results[0].plot()
                    total_detections_in_video += len(results[0].boxes)
                    preview_container.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                    out.write(annotated_frame)
                else:
                    out.write(frame)
                progress_bar.progress((frame_idx + 1) / total_frames, text=f"Processing frame {frame_idx + 1}/{total_frames}")

            cap.release()
            out.release()
            end_time = time.time()
            
            with open(output_video_path, "rb") as f:
                st.session_state.processed_video_bytes = f.read()
            
            # Store metrics
            st.session_state.video_metrics['total_detections'] = total_detections_in_video
            st.session_state.video_metrics['duration'] = video_info['duration']
            st.session_state.video_metrics['fps'] = total_frames / (end_time - start_time) if (end_time - start_time) > 0 else 0
            
            try: Path(tfile.name).unlink(); Path(output_video_path).unlink(); Path(temp_dir).rmdir()
            except Exception as e: st.warning(f"Failed to clean up temp files: {e}")
            
            st.success(f"✅ Video processed successfully in {end_time - start_time:.2f} seconds!")
            st.rerun()