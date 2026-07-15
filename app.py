### app.py
import streamlit as st
from PIL import Image
from pathlib import Path
import time
import cv2
import tempfile

# Import helper functions from utils.py
from utils import (
    load_model, image_to_bytes, create_zip,
    process_image, get_video_info, validate_file_upload,
    generate_file_hash, check_rate_limit,
    get_rate_limit_stats, create_session, validate_session, get_session_stats,
    get_system_health, is_system_healthy, app_logger
)
import logging

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
    page_icon=":material/flight:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def _load_hangar_css(mtime: float) -> str:
    """Cache CSS text so Cloud free-tier skips full reads every rerun."""
    css_path = ASSETS_DIR / "hangar_theme.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""


def apply_hangar_theme() -> None:
    """Inject Hangar Briefing Console CSS (lean, no external fonts)."""
    css_path = ASSETS_DIR / "hangar_theme.css"
    mtime = css_path.stat().st_mtime if css_path.exists() else 0.0
    css = _load_hangar_css(mtime)
    if css:
        # st.html keeps <style> intact; st.markdown can leak CSS as visible text.
        st.html(f"<style>{css}</style>")


apply_hangar_theme()

# --- SESSION STATE INITIALIZATION ---
session = create_session()
session_valid, session_message = validate_session()

app_logger.info(
    "User session initialized",
    session_id=session["session_id"],
    client_id=session["client_id"],
)

if not session_valid:
    st.error(f"Session error: {session_message}")
    st.info("Refresh the page to start a new session.")
    st.stop()

if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}
if "uploaded_video_bytes" not in st.session_state:
    st.session_state.uploaded_video_bytes = None
if "processed_video_bytes" not in st.session_state:
    st.session_state.processed_video_bytes = None
if "video_metrics" not in st.session_state:
    st.session_state.video_metrics = {}
if "video_upload_key" not in st.session_state:
    st.session_state.video_upload_key = 0
if "original_video_name" not in st.session_state:
    st.session_state.original_video_name = ""
if "_results_zip" not in st.session_state:
    st.session_state._results_zip = None
if "_zip_fingerprint" not in st.session_state:
    st.session_state._zip_fingerprint = None
if "_annotated_bytes" not in st.session_state:
    st.session_state._annotated_bytes = {}


def _clear_image_export_cache() -> None:
    st.session_state._results_zip = None
    st.session_state._zip_fingerprint = None
    st.session_state._annotated_bytes = {}


def _store_processed_image(
    name: str,
    original: Image.Image,
    processed: Image.Image,
    count: int,
    metrics: dict,
) -> None:
    st.session_state.processed_images[name] = {
        "original": original,
        "processed": processed,
        "detection_count": count,
        "metrics": metrics,
    }
    st.session_state._annotated_bytes[name] = image_to_bytes(processed)
    st.session_state._zip_fingerprint = None
    st.session_state._results_zip = None


def _ensure_results_zip() -> bytes:
    """Build ZIP once per results set — not on every Streamlit rerun."""
    fingerprint = tuple(
        (name, data["detection_count"], id(data["processed"]))
        for name, data in st.session_state.processed_images.items()
    )
    if st.session_state._zip_fingerprint != fingerprint or st.session_state._results_zip is None:
        st.session_state._results_zip = create_zip(st.session_state.processed_images)
        st.session_state._zip_fingerprint = fingerprint
        for name, data in st.session_state.processed_images.items():
            if name not in st.session_state._annotated_bytes:
                st.session_state._annotated_bytes[name] = image_to_bytes(data["processed"])
    return st.session_state._results_zip


def _annotated_download_bytes(filename: str) -> bytes:
    cached = st.session_state._annotated_bytes.get(filename)
    if cached is not None:
        return cached
    data = st.session_state.processed_images[filename]
    buf = image_to_bytes(data["processed"])
    st.session_state._annotated_bytes[filename] = buf
    return buf


def _example_image_paths() -> list[Path]:
    return [ASSETS_DIR / f for f in EXAMPLE_IMAGE_FILES if (ASSETS_DIR / f).exists()]


def _run_sample_image(image_path: Path) -> None:
    original_image = Image.open(image_path).convert("RGB")
    proc_img, count, speed = process_image(
        model, original_image, confidence_threshold, iou_threshold
    )
    _store_processed_image(image_path.name, original_image, proc_img, count, speed)


def _load_sample_video() -> None:
    example_video_path = ASSETS_DIR / EXAMPLE_VIDEO_FILE
    with open(example_video_path, "rb") as f:
        st.session_state.uploaded_video_bytes = f.read()
    st.session_state.original_video_name = EXAMPLE_VIDEO_FILE
    st.session_state.processed_video_bytes = None
    st.session_state.video_metrics = {}


# --- MODEL LOADING ---
with st.spinner("Loading detection model…"):
    model = load_model(MODEL_PATH)

# --- SIDEBAR ---
st.sidebar.markdown('<p class="hangar-side-label">Detection</p>', unsafe_allow_html=True)
app_mode = st.sidebar.radio(
    "Input mode",
    ["Images", "Video"],
    help="Still images or a single video file.",
)

confidence_threshold = st.sidebar.slider(
    "Confidence",
    0.0,
    1.0,
    0.25,
    0.05,
    help="Minimum score to keep a box. Higher is stricter — fewer boxes.",
)
iou_threshold = st.sidebar.slider(
    "Overlap (IoU)",
    0.0,
    1.0,
    0.45,
    0.05,
    help="How much boxes may overlap. Lower reduces duplicate boxes.",
)
st.sidebar.markdown('<hr class="hangar-divider">', unsafe_allow_html=True)

# --- SIDEBAR EXAMPLES (primary place to pick mode samples) ---
if app_mode == "Images":
    st.sidebar.markdown('<p class="hangar-side-label">Examples</p>', unsafe_allow_html=True)
    st.sidebar.caption("One click runs a sample detection.")
    for image_path in _example_image_paths():
        with st.sidebar.container(border=True):
            st.image(str(image_path), width="stretch")
            if st.button("Run example", key=f"try_{image_path.name}", width="stretch"):
                try:
                    _run_sample_image(image_path)
                    st.rerun()
                except (ValueError, OSError) as e:
                    st.error(f"Failed to process example: {e}")
elif app_mode == "Video":
    st.sidebar.markdown('<p class="hangar-side-label">Examples</p>', unsafe_allow_html=True)
    st.sidebar.caption("Load sample, then process in the main view.")
    example_video_path = ASSETS_DIR / EXAMPLE_VIDEO_FILE
    if example_video_path.exists():
        with st.sidebar.container(border=True):
            st.video(str(example_video_path))
            if st.button("Load example", width="stretch", key="ex_video"):
                _load_sample_video()
                st.rerun()

health = get_system_health()
is_healthy, health_issues = is_system_healthy()

with st.sidebar.expander("System status", expanded=False):
    session_stats = get_session_stats()
    col1, col2 = st.columns(2)
    col1.metric("Session age", f"{session_stats['session_age']}s")
    col2.metric("Files processed", session_stats["files_processed"])

    uptime_hours = health["uptime_seconds"] / 3600
    st.metric("Uptime", f"{uptime_hours:.1f}h")
    if is_healthy:
        st.caption("System: nominal")
    else:
        st.caption("System: issues detected")
        for issue in health_issues:
            st.warning(issue)

    rate_stats = get_rate_limit_stats()
    col1, col2 = st.columns(2)
    col1.metric("Active clients", rate_stats["active_clients"])
    col2.metric("Blocked clients", rate_stats["blocked_clients"])

    rate_allowed, rate_message, remaining = check_rate_limit()
    if rate_allowed:
        st.caption(f"Rate limit: {remaining} requests remaining")
    else:
        st.error("Rate limited")

    st.caption(f"Session {session['session_id'][:8]} · 10 req / 5 min")

with st.sidebar.expander("About"):
    st.markdown(
        """
Fine-tuned YOLOv8 for military aircraft detection in images and video — upload media, tune thresholds, export annotated results.

**Developer:** [Faisal Durbaa](https://github.com/faisaldurbaa)  
**Repository:** [aircraft_detector](https://github.com/faisaldurbaa/aircraft_detector)
        """
    )

# --- MAIN INTERFACE ---
_has_payload = bool(st.session_state.processed_images) or bool(
    st.session_state.processed_video_bytes
)
_ready_class = "hangar-ready" if is_healthy else "hangar-ready is-err"
_ready_label = "Model ready" if is_healthy else "Model degraded"

if _has_payload:
    st.html(
        f"""
        <div class="hangar-masthead">
          <p class="hangar-kicker">YOLOv8 · aircraft detection</p>
          <h1>Aircraft Detection</h1>
          <p class="{_ready_class}">{_ready_label}</p>
        </div>
        """
    )
else:
    st.html(
        f"""
        <div class="hangar-masthead">
          <p class="hangar-kicker">YOLOv8 · aircraft detection</p>
          <h1>Aircraft Detection</h1>
          <p class="hangar-brief">Upload media or run a sidebar example. Annotated boxes and inference timing land below.</p>
          <p class="{_ready_class}">{_ready_label}</p>
        </div>
        """
    )

if not is_healthy and health_issues:
    with st.expander("System issues", expanded=True):
        for issue in health_issues:
            st.warning(issue)

st.html('<hr class="hangar-divider">')


def _render_image_uploader(key: str = "image_uploader") -> bool:
    """Upload + validate + run detection for still images. Returns True if files are staged."""
    uploaded_files = st.file_uploader(
        "Drop images here",
        label_visibility="collapsed",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        key=key,
    )
    if not uploaded_files:
        return False

    valid_files = []
    for file in uploaded_files:
        file_bytes = file.getvalue()
        is_valid, result = validate_file_upload(file_bytes, file.name, "image")

        if not is_valid:
            st.error(f"{file.name}: {result}")
            continue

        sanitized_name = result
        file.name = sanitized_name
        valid_files.append(file)

        file_hash = generate_file_hash(file_bytes)
        logging.info(f"Validated file: {sanitized_name} (hash: {file_hash[:16]}...)")

    if not valid_files:
        st.warning("No valid image files to process.")
        return True

    st.caption(f"{len(valid_files)} of {len(uploaded_files)} files validated")

    rate_allowed, rate_message, remaining = check_rate_limit()

    if not rate_allowed:
        st.error(rate_message)
        st.caption("Rate limiting protects shared Cloud capacity.")
    else:
        st.caption(rate_message)

    if st.button(
        "Run detection",
        type="primary",
        width="stretch",
        disabled=not rate_allowed,
        key=f"run_detection_{key}",
    ):
        files_to_process = [f for f in valid_files if f.name not in st.session_state.processed_images]
        if files_to_process:
            with st.spinner(f"Analyzing {len(files_to_process)} images…"):
                for file in files_to_process:
                    try:
                        orig_img = Image.open(file).convert("RGB")
                        proc_img, count, speed = process_image(
                            model, orig_img, confidence_threshold, iou_threshold
                        )
                        _store_processed_image(file.name, orig_img, proc_img, count, speed)
                    except (ValueError, OSError) as e:
                        st.error(f"Failed to process {file.name}: {e}")
                        continue
            st.rerun()
    return True


if app_mode == "Images":
    has_results = bool(st.session_state.processed_images)

    # Results lead when present — detections are the product.
    if has_results:
        st.header("Results")
        total_detections = sum(
            data["detection_count"] for data in st.session_state.processed_images.values()
        )
        total_inference_time_ms = sum(
            data["metrics"]["inference"] for data in st.session_state.processed_images.values()
        )
        avg_inference_time = (
            total_inference_time_ms / len(st.session_state.processed_images)
            if st.session_state.processed_images
            else 0
        )
        image_count = len(st.session_state.processed_images)

        st.html(
            f"""
            <div class="hangar-readouts">
              <div class="hangar-readout">
                <span class="label">Images</span>
                <span class="value">{image_count}</span>
              </div>
              <div class="hangar-readout">
                <span class="label">Aircraft</span>
                <span class="value">{total_detections}</span>
              </div>
              <div class="hangar-readout">
                <span class="label">Avg inference</span>
                <span class="value">{avg_inference_time:.1f}<span class="unit">ms</span></span>
              </div>
            </div>
            """
        )

        if total_detections == 0:
            st.warning("No aircraft detected. Lower confidence in the sidebar and re-run.")

        zip_bytes = _ensure_results_zip()
        col1, col2, _ = st.columns([0.3, 0.3, 0.4])
        if col1.button("Clear results", width="stretch"):
            st.session_state.processed_images.clear()
            _clear_image_export_cache()
            st.rerun()
        if col2.download_button(
            "Download ZIP",
            zip_bytes,
            "detected_images.zip",
            "application/zip",
            width="stretch",
        ):
            pass

        view_mode = st.radio(
            "View mode",
            ["Side-by-side", "Grid"],
            horizontal=True,
            label_visibility="collapsed",
        )
        st.html('<div class="hangar-toolbar-gap"></div>')
        if view_mode == "Side-by-side":
            for filename, data in reversed(list(st.session_state.processed_images.items())):
                det = data["detection_count"]
                det_label = "detection" if det == 1 else "detections"
                st.html(
                    f"""
                    <div class="hangar-result-meta">
                      <span class="name">{filename}</span>
                      <span class="count">{det} {det_label}</span>
                    </div>
                    """
                )
                c1, c2 = st.columns(2)
                c1.image(data["original"], "Original", width="stretch")
                c2.image(data["processed"], "Detected", width="stretch")
                st.download_button(
                    "Download annotated",
                    _annotated_download_bytes(filename),
                    f"detected_{filename}.png",
                    "image/png",
                    key=f"dl_{filename}",
                )
                st.html('<hr class="hangar-divider">')
        else:
            cols = st.columns(3)
            for idx, (filename, data) in enumerate(
                reversed(list(st.session_state.processed_images.items()))
            ):
                with cols[idx % 3]:
                    st.image(
                        data["processed"],
                        f"{filename} · {data['detection_count']}",
                        width="stretch",
                    )

        with st.expander("Add more images", expanded=False):
            _render_image_uploader(key="image_uploader_more")
    else:
        st.header("Images")
        sample_paths = _example_image_paths()
        has_upload = _render_image_uploader(key="image_uploader")
        if not has_upload:
            if sample_paths:
                if st.button("Run sample", type="primary", width="stretch", key="main_run_sample"):
                    try:
                        with st.spinner("Running sample detection…"):
                            _run_sample_image(sample_paths[0])
                        st.rerun()
                    except (ValueError, OSError) as e:
                        st.error(f"Failed to process sample: {e}")
            st.html(
                """
                <div class="hangar-empty">
                  <strong>No images loaded</strong>
                  <p>Drop images above, or use Examples in the sidebar to finish a detection in under a minute.</p>
                  <p class="hangar-empty-hint">Next: sidebar <em>Run example</em>, or upload and hit <em>Run detection</em>.</p>
                </div>
                """
            )

elif app_mode == "Video":
    if st.session_state.processed_video_bytes:
        st.header("Results")
        duration = st.session_state.video_metrics.get("duration", 0)
        total_det = st.session_state.video_metrics.get("total_detections", 0)
        proc_fps = st.session_state.video_metrics.get("fps", 0)
        st.html(
            f"""
            <div class="hangar-readouts">
              <div class="hangar-readout">
                <span class="label">Duration</span>
                <span class="value">{duration:.1f}<span class="unit">s</span></span>
              </div>
              <div class="hangar-readout">
                <span class="label">Detections</span>
                <span class="value">{total_det}</span>
              </div>
              <div class="hangar-readout">
                <span class="label">Process rate</span>
                <span class="value">{proc_fps:.1f}<span class="unit">fps</span></span>
              </div>
            </div>
            """
        )
        st.video(st.session_state.processed_video_bytes, format="video/mp4", start_time=0)
        col1, col2, _ = st.columns([0.3, 0.3, 0.4])
        col1.download_button(
            "Download",
            st.session_state.processed_video_bytes,
            f"detected_{st.session_state.original_video_name}",
            "video/mp4",
            width="stretch",
        )
        if col2.button("Clear results", width="stretch"):
            st.session_state.uploaded_video_bytes = None
            st.session_state.processed_video_bytes = None
            st.session_state.video_metrics = {}
            st.session_state.video_upload_key += 1
            st.rerun()

    elif st.session_state.uploaded_video_bytes:
        st.header("Video")
        col1, col2 = st.columns(2)
        with col1:
            st.html(
                f"""
                <div class="hangar-result-meta">
                  <span class="name">{st.session_state.original_video_name}</span>
                  <span class="count">Source</span>
                </div>
                """
            )
            st.video(st.session_state.uploaded_video_bytes)
        with col2:
            st.html(
                """
                <div class="hangar-result-meta">
                  <span class="name">Live preview</span>
                  <span class="count">During run</span>
                </div>
                """
            )
            preview_container = st.empty()

        rate_allowed, rate_message, remaining = check_rate_limit()

        if not rate_allowed:
            st.error(rate_message)
            st.caption("Video processing is resource-intensive on shared Cloud capacity.")
        else:
            st.caption(rate_message)

        if st.button("Run detection", type="primary", width="stretch", disabled=not rate_allowed):
            video_info = get_video_info(st.session_state.uploaded_video_bytes)

            if "error" in video_info:
                st.error(f"Cannot process video: {video_info['error']}")
            else:
                with st.spinner("Analyzing video… longer clips take more time."):
                    start_time = time.time()
                    temp_dir = tempfile.mkdtemp()
                    output_video_path = str(Path(temp_dir) / "processed.mp4")

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                        tfile.write(st.session_state.uploaded_video_bytes)
                        cap = cv2.VideoCapture(tfile.name)

                    if not cap.isOpened():
                        st.error("Failed to open video file for processing")
                    else:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                        output_width = min(1920, video_info["width"])
                        output_height = min(1080, video_info["height"])

                        codecs_to_try = [
                            ("avc1", "H.264/AVC"),
                            ("mp4v", "MPEG-4"),
                            ("XVID", "XVID"),
                        ]

                        out = None
                        for codec_fourcc, codec_name in codecs_to_try:
                            fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
                            out = cv2.VideoWriter(
                                output_video_path, fourcc, video_info["fps"], (output_width, output_height)
                            )
                            if out.isOpened():
                                logging.info(f"Using {codec_name} codec for video output")
                                break
                            else:
                                logging.warning(f"{codec_name} codec failed, trying next...")
                                out.release()

                        if out is None or not out.isOpened():
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            out = cv2.VideoWriter(
                                output_video_path, fourcc, video_info["fps"], (output_width, output_height)
                            )
                            if out.isOpened():
                                logging.info("Using default mp4v codec as last resort")
                            else:
                                logging.error("All video codecs failed")

                        if not out.isOpened():
                            st.error("Failed to initialize video writer. Check codec support.")
                            cap.release()
                        else:
                            progress_bar = st.progress(0, text="Starting processing…")
                            total_detections_in_video = 0
                            last_good_annotated_frame = None
                            logging.info(
                                f"Starting video processing: {total_frames} frames, output size: {output_width}x{output_height}"
                            )

                            for frame_idx in range(total_frames):
                                ret, frame = cap.read()
                                if not ret:
                                    logging.warning(f"Failed to read frame {frame_idx}")
                                    break

                                if frame.shape[0] != output_height or frame.shape[1] != output_width:
                                    frame = cv2.resize(frame, (output_width, output_height))

                                frame_to_write = frame.copy()

                                if frame_idx % FRAME_SKIP == 0:
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    try:
                                        results = model.predict(
                                            source=frame_rgb,
                                            conf=confidence_threshold,
                                            iou=iou_threshold,
                                            verbose=False,
                                        )
                                        if results and len(results) > 0:
                                            result = results[0]
                                            annotated_frame = result.plot()
                                            detection_count = (
                                                len(result.boxes) if result.boxes is not None else 0
                                            )
                                            total_detections_in_video += detection_count

                                            if (
                                                annotated_frame.shape[0] != output_height
                                                or annotated_frame.shape[1] != output_width
                                            ):
                                                annotated_frame = cv2.resize(
                                                    annotated_frame, (output_width, output_height)
                                                )

                                            last_good_annotated_frame = annotated_frame.copy()
                                            frame_to_write = annotated_frame

                                            if frame_idx % (FRAME_SKIP * 5) == 0:
                                                preview_container.image(
                                                    cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                                                )

                                            logging.debug(f"Frame {frame_idx}: {detection_count} detections")
                                    except Exception as e:
                                        logging.error(f"Error processing frame {frame_idx}: {e}")
                                        if last_good_annotated_frame is not None:
                                            frame_to_write = last_good_annotated_frame.copy()
                                elif last_good_annotated_frame is not None:
                                    frame_to_write = last_good_annotated_frame.copy()

                                if (
                                    frame_to_write.shape[0] != output_height
                                    or frame_to_write.shape[1] != output_width
                                ):
                                    frame_to_write = cv2.resize(
                                        frame_to_write, (output_width, output_height)
                                    )

                                success = out.write(frame_to_write)
                                if not success:
                                    logging.error(f"Failed to write frame {frame_idx}")

                                if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                                    progress_bar.progress(
                                        (frame_idx + 1) / total_frames,
                                        text=f"Frame {frame_idx + 1}/{total_frames}",
                                    )

                            cap.release()
                            out.release()
                            end_time = time.time()

                            if not Path(output_video_path).exists():
                                st.error("Failed to create output video file")
                            else:
                                output_size = Path(output_video_path).stat().st_size
                                if output_size == 0:
                                    st.error("Output video file is empty")
                                else:
                                    logging.info(
                                        f"Video processing completed. Output file size: {output_size} bytes"
                                    )

                                    try:
                                        test_cap = cv2.VideoCapture(output_video_path)
                                        if test_cap.isOpened():
                                            test_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                            test_duration = (
                                                test_frame_count / video_info["fps"]
                                                if video_info["fps"] > 0
                                                else 0
                                            )
                                            logging.info(
                                                f"Output video validation: {test_frame_count} frames, {test_duration:.1f}s duration"
                                            )
                                        test_cap.release()
                                    except Exception as e:
                                        logging.warning(f"Video validation failed: {e}")

                                    with open(output_video_path, "rb") as f:
                                        st.session_state.processed_video_bytes = f.read()

                                    st.session_state.video_metrics["total_detections"] = (
                                        total_detections_in_video
                                    )
                                    st.session_state.video_metrics["duration"] = video_info["duration"]
                                    processing_duration = end_time - start_time
                                    st.session_state.video_metrics["fps"] = (
                                        total_frames / processing_duration if processing_duration > 0 else 0
                                    )

                                    cleanup_errors = []
                                    try:
                                        if Path(tfile.name).exists():
                                            Path(tfile.name).unlink()
                                    except (FileNotFoundError, OSError) as e:
                                        cleanup_errors.append(f"input file: {e}")

                                    try:
                                        if Path(output_video_path).exists():
                                            Path(output_video_path).unlink()
                                    except (FileNotFoundError, OSError) as e:
                                        cleanup_errors.append(f"output file: {e}")

                                    try:
                                        if Path(temp_dir).exists():
                                            Path(temp_dir).rmdir()
                                    except (FileNotFoundError, OSError) as e:
                                        cleanup_errors.append(f"temp directory: {e}")

                                    if cleanup_errors:
                                        logging.warning(f"Cleanup issues: {'; '.join(cleanup_errors)}")

                                    st.success(f"Video processed in {processing_duration:.2f}s")
                                    st.rerun()
    else:
        st.header("Video")
        example_video_path = ASSETS_DIR / EXAMPLE_VIDEO_FILE
        uploaded_file = st.file_uploader(
            "Upload a video (max 30 seconds)",
            label_visibility="collapsed",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            key=f"video_uploader_{st.session_state.video_upload_key}",
        )
        if uploaded_file:
            video_bytes = uploaded_file.getvalue()

            is_valid, result = validate_file_upload(video_bytes, uploaded_file.name, "video")

            if not is_valid:
                st.error(f"Video validation failed: {result}")
            else:
                sanitized_name = result
                file_hash = generate_file_hash(video_bytes)
                logging.info(f"Validated video: {sanitized_name} (hash: {file_hash[:16]}...)")

                video_info = get_video_info(video_bytes)
                if "error" in video_info:
                    st.error(f"Invalid video file: {video_info['error']}")
                elif video_info["duration"] > 30:
                    st.error(
                        f"Duration ({video_info['duration']:.1f}s) exceeds the 30s limit."
                    )
                else:
                    st.caption(f"Validated · {sanitized_name}")
                    st.session_state.uploaded_video_bytes = video_bytes
                    st.session_state.original_video_name = sanitized_name
                    st.session_state.video_metrics = {}
                    st.rerun()
        else:
            if example_video_path.exists():
                if st.button("Load sample", type="primary", width="stretch", key="main_load_sample"):
                    _load_sample_video()
                    st.rerun()
            st.html(
                """
                <div class="hangar-empty">
                  <strong>No video loaded</strong>
                  <p>Upload a short clip (≤30s), or use Examples in the sidebar, then run detection.</p>
                  <p class="hangar-empty-hint">Next: sidebar <em>Load example</em>, or upload and hit <em>Run detection</em>.</p>
                </div>
                """
            )
