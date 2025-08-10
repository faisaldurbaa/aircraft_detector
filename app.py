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
    process_image, get_video_info, validate_file_upload,
    sanitize_filename, generate_file_hash, check_rate_limit,
    get_rate_limit_stats, create_session, validate_session, get_session_stats,
    get_system_health, is_system_healthy, performance_monitor, app_logger
)
import logging
import uuid

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- PATHS & CONFIG ---
ASSETS_DIR = Path("assets")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "best.pt"

EXAMPLE_IMAGE_FILES = ["img1.jpeg", "img2.jpeg"]
EXAMPLE_VIDEO_FILE = "f18.mp4"
FRAME_SKIP = 2

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aircraft Detection",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SESSION STATE INITIALIZATION ---
# Initialize secure session
# Initialize secure session with monitoring
session = create_session()
session_valid, session_message = validate_session()

# Log session creation
app_logger.info("User session initialized",
               session_id=session['session_id'],
               client_id=session['client_id'])

if not session_valid:
    st.error(f"⚠️ Session Error: {session_message}")
    st.info("Please refresh the page to start a new session.")
    st.stop()

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
                try:
                    original_image = Image.open(image_path).convert("RGB")
                    proc_img, count, speed = process_image(model, original_image, confidence_threshold, iou_threshold)
                    st.session_state.processed_images[image_path.name] = {
                        'original': original_image, 'processed': proc_img, 'detection_count': count, 'metrics': speed
                    }
                    st.rerun()
                except (ValueError, OSError) as e:
                    st.error(f"Failed to process example image: {e}")
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

# System status in sidebar
with st.sidebar.expander("🔒 System Status", expanded=False):
    # Session status
    session_stats = get_session_stats()
    st.subheader("Session Info")
    col1, col2 = st.columns(2)
    col1.metric("Session Age", f"{session_stats['session_age']}s")
    col2.metric("Files Processed", session_stats['files_processed'])
    
    # Rate limiting status
    st.subheader("Rate Limiting")
    rate_stats = get_rate_limit_stats()
    col1, col2 = st.columns(2)
    col1.metric("Active Clients", rate_stats['active_clients'])
    col2.metric("Blocked Clients", rate_stats['blocked_clients'])
    
    # Show current client's status
    rate_allowed, rate_message, remaining = check_rate_limit()
    if rate_allowed:
        st.success(f"✅ Status: Active ({remaining} requests remaining)")
    else:
        st.error(f"❌ Status: Rate Limited")
    
    st.caption(f"Session ID: {session['session_id'][:8]}...")
    st.caption("Rate limiting: 10 requests per 5 minutes")

with st.sidebar.expander("ℹ️ About This App"):
    st.markdown("""
    This application uses a fine-tuned object detection model to identify aircraft in images and videos.
    - **Fast & Accurate:** Built for performance and precision.
    - **User-Friendly:** Easy-to-use interface with real-time feedback.
    - **Adjustable:** Tune detection parameters to fit your needs.
    - **Secure:** File validation, rate limiting, and session management for safe operation.
    
    **Developer:** [Faisal Durbaa](https://github.com/faisaldurbaa)  
    **Repository:** [aircraft_detector](https://github.com/faisaldurbaa/aircraft_detector)
    """)

# --- MAIN INTERFACE ---
# Add health status indicator
health = get_system_health()
is_healthy, health_issues = is_system_healthy()

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("✈️ Aircraft Detection")
with col2:
    if is_healthy:
        st.success("✅ System OK")
    else:
        st.error("❌ System Issues")
with col3:
    uptime_hours = health['uptime_seconds'] / 3600
    st.metric("Uptime", f"{uptime_hours:.1f}h")

# Show critical issues prominently
if not is_healthy and len(health_issues) > 0:
    with st.expander("⚠️ System Issues - Click to expand", expanded=False):
        for issue in health_issues:
            st.warning(issue)

st.markdown("Upload your own media or use the examples in the sidebar to get started.")
st.markdown("---")

if app_mode == 'Images':
    st.header("🖼️ Multi-Image Processing")
    uploaded_files = st.file_uploader(
        "Upload Images", label_visibility="collapsed", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'], accept_multiple_files=True
    )
    if uploaded_files:
        # Validate all uploaded files first
        valid_files = []
        for file in uploaded_files:
            file_bytes = file.getvalue()
            is_valid, result = validate_file_upload(file_bytes, file.name, 'image')
            
            if not is_valid:
                st.error(f"❌ {file.name}: {result}")
                continue
            
            # result is the sanitized filename when valid
            sanitized_name = result
            file.name = sanitized_name  # Update file name to sanitized version
            valid_files.append(file)
            
            # Generate file hash for integrity
            file_hash = generate_file_hash(file_bytes)
            logging.info(f"Validated file: {sanitized_name} (hash: {file_hash[:16]}...)")
        
        if valid_files:
            st.success(f"✅ {len(valid_files)} of {len(uploaded_files)} files validated successfully")
            
            # Check rate limit before processing
            rate_allowed, rate_message, remaining = check_rate_limit()
            
            if not rate_allowed:
                st.error(f"⚠️ {rate_message}")
                st.info("Rate limiting helps prevent system overload and ensures fair usage for all users.")
            else:
                st.info(f"🔄 {rate_message}")
                
            if st.button("Process All Uploaded Images", type="primary", use_container_width=True, disabled=not rate_allowed):
                files_to_process = [f for f in valid_files if f.name not in st.session_state.processed_images]
                if files_to_process:
                    with st.spinner(f"Analyzing {len(files_to_process)} images..."):
                        for file in files_to_process:
                            try:
                                orig_img = Image.open(file).convert("RGB")
                                proc_img, count, speed = process_image(model, orig_img, confidence_threshold, iou_threshold)
                                st.session_state.processed_images[file.name] = {'original': orig_img, 'processed': proc_img, 'detection_count': count, 'metrics': speed}
                            except (ValueError, OSError) as e:
                                st.error(f"Failed to process {file.name}: {e}")
                                continue
                    st.rerun()
        elif uploaded_files:
            st.warning("⚠️ No valid image files to process. Please check the error messages above.")

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
        m2.metric("Total Aircraft Detected on All Frames", f"{st.session_state.video_metrics.get('total_detections', 0)} ✈️")
        m3.metric("Processing FPS", f"{st.session_state.video_metrics.get('fps', 0):.1f}")
        # Add format specification to help Streamlit handle the video
        st.video(st.session_state.processed_video_bytes, format='video/mp4', start_time=0)
        st.download_button( "Download Processed Video", st.session_state.processed_video_bytes, f"detected_{st.session_state.original_video_name}", "video/mp4", use_container_width=True)
        if st.button("Clear Video & Start Over", use_container_width=True, type="primary"):
            st.session_state.uploaded_video_bytes = None
            st.session_state.processed_video_bytes = None
            st.session_state.video_metrics = {}
            st.session_state.video_upload_key += 1
            st.rerun()

    elif st.session_state.uploaded_video_bytes:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Original: {st.session_state.original_video_name}")
            st.video(st.session_state.uploaded_video_bytes)
        with col2:
            st.subheader("Processing Preview")
            preview_container = st.empty()

        # Check rate limit before video processing
        rate_allowed, rate_message, remaining = check_rate_limit()
        
        if not rate_allowed:
            st.error(f"⚠️ {rate_message}")
            st.info("Rate limiting helps prevent system overload. Video processing is resource-intensive.")
        else:
            st.info(f"🔄 {rate_message}")
            
        if st.button("Process Video", type="primary", use_container_width=True, disabled=not rate_allowed):
            video_info = get_video_info(st.session_state.uploaded_video_bytes)
            
            # Validate video info before processing
            if 'error' in video_info:
                st.error(f"Cannot process video: {video_info['error']}")
            else:
                with st.spinner("Analyzing video... This could take a while for longer videos."):
                    start_time = time.time()
                    temp_dir = tempfile.mkdtemp()
                    output_video_path = str(Path(temp_dir) / "processed.mp4")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(st.session_state.uploaded_video_bytes)
                        cap = cv2.VideoCapture(tfile.name)
                    
                    if not cap.isOpened():
                        st.error("Failed to open video file for processing")
                    else:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Determine actual output dimensions (handle resizing)
                        output_width = min(1920, video_info['width'])
                        output_height = min(1080, video_info['height'])
                        
                        # Try multiple codecs for web compatibility
                        codecs_to_try = [
                            ('avc1', 'H.264/AVC'),  # Best for web compatibility
                            ('mp4v', 'MPEG-4'),     # Fallback option
                            ('XVID', 'XVID')        # Alternative fallback
                        ]
                        
                        out = None
                        for codec_fourcc, codec_name in codecs_to_try:
                            fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
                            out = cv2.VideoWriter(output_video_path, fourcc, video_info['fps'], (output_width, output_height))
                            if out.isOpened():
                                logging.info(f"Using {codec_name} codec for video output")
                                break
                            else:
                                logging.warning(f"{codec_name} codec failed, trying next...")
                                out.release()
                        
                        if out is None or not out.isOpened():
                            # Final attempt with default codec
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(output_video_path, fourcc, video_info['fps'], (output_width, output_height))
                            if out.isOpened():
                                logging.info("Using default mp4v codec as last resort")
                            else:
                                logging.error("All video codecs failed")
                    
                        if not out.isOpened():
                            st.error("Failed to initialize video writer. Check codec support.")
                            cap.release()
                        else:
                            progress_bar = st.progress(0, text="Starting processing...")
                            total_detections_in_video = 0
                            last_good_annotated_frame = None
                            logging.info(f"Starting video processing: {total_frames} frames, output size: {output_width}x{output_height}")

                            for frame_idx in range(total_frames):
                                ret, frame = cap.read()
                                if not ret: 
                                    logging.warning(f"Failed to read frame {frame_idx}")
                                    break
                                
                                # Ensure consistent frame dimensions throughout
                                if frame.shape[0] != output_height or frame.shape[1] != output_width:
                                    frame = cv2.resize(frame, (output_width, output_height))
                                
                                # Initialize frame_to_write with current frame as fallback
                                frame_to_write = frame.copy()
                        
                                if frame_idx % FRAME_SKIP == 0:
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    try:
                                        results = model.predict(source=frame_rgb, conf=confidence_threshold, iou=iou_threshold, verbose=False)
                                        if results and len(results) > 0:
                                            result = results[0]
                                            annotated_frame = result.plot()  # This returns BGR format
                                            detection_count = len(result.boxes) if result.boxes is not None else 0
                                            total_detections_in_video += detection_count
                                            
                                            # Ensure annotated frame has correct dimensions
                                            if annotated_frame.shape[0] != output_height or annotated_frame.shape[1] != output_width:
                                                annotated_frame = cv2.resize(annotated_frame, (output_width, output_height))
                                            
                                            last_good_annotated_frame = annotated_frame.copy()
                                            frame_to_write = annotated_frame
                                            
                                            # Update preview less frequently to reduce UI overhead
                                            if frame_idx % (FRAME_SKIP * 5) == 0:
                                                preview_container.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                                            
                                            logging.debug(f"Frame {frame_idx}: {detection_count} detections")
                                    except Exception as e:
                                        logging.error(f"Error processing frame {frame_idx}: {e}")
                                        # Use last good frame if available, otherwise original
                                        if last_good_annotated_frame is not None:
                                            frame_to_write = last_good_annotated_frame.copy()
                                elif last_good_annotated_frame is not None:
                                    frame_to_write = last_good_annotated_frame.copy()
                        
                                # Ensure frame_to_write has correct dimensions before writing
                                if frame_to_write.shape[0] != output_height or frame_to_write.shape[1] != output_width:
                                    frame_to_write = cv2.resize(frame_to_write, (output_width, output_height))
                                
                                # Write frame (already in BGR format for VideoWriter)
                                success = out.write(frame_to_write)
                                if not success:
                                    logging.error(f"Failed to write frame {frame_idx}")
                                
                                # Update progress less frequently to improve performance
                                if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                                    progress_bar.progress((frame_idx + 1) / total_frames, text=f"Processing frame {frame_idx + 1}/{total_frames}")

                            cap.release()
                            out.release()
                            end_time = time.time()
                    
                            # Verify output video was created and has content
                            if not Path(output_video_path).exists():
                                st.error("Failed to create output video file")
                            else:
                                output_size = Path(output_video_path).stat().st_size
                                if output_size == 0:
                                    st.error("Output video file is empty")
                                else:
                                    logging.info(f"Video processing completed. Output file size: {output_size} bytes")
                                    
                                    # Additional validation for web compatibility
                                    try:
                                        test_cap = cv2.VideoCapture(output_video_path)
                                        if test_cap.isOpened():
                                            test_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                            test_duration = test_frame_count / video_info['fps'] if video_info['fps'] > 0 else 0
                                            logging.info(f"Output video validation: {test_frame_count} frames, {test_duration:.1f}s duration")
                                        test_cap.release()
                                    except Exception as e:
                                        logging.warning(f"Video validation failed: {e}")
                    
                                    with open(output_video_path, "rb") as f:
                                        st.session_state.processed_video_bytes = f.read()
                                    
                                    st.session_state.video_metrics['total_detections'] = total_detections_in_video
                                    st.session_state.video_metrics['duration'] = video_info['duration']
                                    processing_duration = end_time - start_time
                                    st.session_state.video_metrics['fps'] = total_frames / processing_duration if processing_duration > 0 else 0
                    
                                    # Comprehensive cleanup with better error handling
                                    cleanup_errors = []
                                    try:
                                        if Path(tfile.name).exists():
                                            Path(tfile.name).unlink()
                                            logging.debug(f"Cleaned up input temp file: {tfile.name}")
                                    except (FileNotFoundError, OSError) as e:
                                        cleanup_errors.append(f"input file: {e}")
                                    
                                    try:
                                        if Path(output_video_path).exists():
                                            Path(output_video_path).unlink()
                                            logging.debug(f"Cleaned up output temp file: {output_video_path}")
                                    except (FileNotFoundError, OSError) as e:
                                        cleanup_errors.append(f"output file: {e}")
                                    
                                    try:
                                        if Path(temp_dir).exists():
                                            Path(temp_dir).rmdir()
                                            logging.debug(f"Cleaned up temp directory: {temp_dir}")
                                    except (FileNotFoundError, OSError) as e:
                                        cleanup_errors.append(f"temp directory: {e}")
                                    
                                    if cleanup_errors:
                                        logging.warning(f"Cleanup issues: {'; '.join(cleanup_errors)}")
                                        st.warning("Some temporary files could not be cleaned up, but processing completed successfully.")
                    
                                    st.success(f"✅ Video processed successfully in {processing_duration:.2f} seconds!")
                                    st.rerun()
    else:
        uploaded_file = st.file_uploader( "Upload a video file for analysis (Max duration: 30 seconds)", label_visibility="collapsed", type=["mp4", "avi", "mov", "mkv", "webm"], key=f"video_uploader_{st.session_state.video_upload_key}")
        if uploaded_file:
            video_bytes = uploaded_file.getvalue()
            
            # Validate uploaded video file
            is_valid, result = validate_file_upload(video_bytes, uploaded_file.name, 'video')
            
            if not is_valid:
                st.error(f"❌ Video validation failed: {result}")
            else:
                sanitized_name = result
                file_hash = generate_file_hash(video_bytes)
                logging.info(f"Validated video: {sanitized_name} (hash: {file_hash[:16]}...)")
                
                # Validate video duration (30 seconds max)
                video_info = get_video_info(video_bytes)
                if 'error' in video_info:
                    st.error(f"Invalid video file: {video_info['error']}")
                elif video_info['duration'] > 30:
                    st.error(f"Video duration ({video_info['duration']:.1f}s) exceeds maximum allowed duration of 30 seconds.")
                else:
                    st.success(f"✅ Video validated: {sanitized_name}")
                    st.session_state.uploaded_video_bytes = video_bytes
                    st.session_state.original_video_name = sanitized_name
                    st.session_state.video_metrics = {}
                    st.rerun()