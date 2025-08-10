### utils.py
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import logging
import json
import psutil
import threading
from datetime import datetime
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from PIL import Image
import io
import zipfile
import cv2
import tempfile
import warnings
import mimetypes
import hashlib
import time
import secrets
import hmac
from typing import Tuple, Optional, Dict, List
from collections import defaultdict, deque

# Suppress specific streamlit warnings about ScriptRunContext
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

# Performance monitoring storage
_performance_metrics = {
    'requests_total': 0,
    'requests_failed': 0,
    'processing_times': deque(maxlen=100),  # Keep last 100 processing times
    'memory_usage': deque(maxlen=50),
    'cpu_usage': deque(maxlen=50),
    'model_predictions': 0,
    'files_processed': 0,
    'errors_logged': 0,
    'startup_time': time.time()
}

# Thread lock for metrics
_metrics_lock = threading.Lock()

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    message: str
    module: str
    session_id: str = ""
    client_id: str = ""
    processing_time: float = 0.0
    file_hash: str = ""
    error_type: str = ""
    stack_trace: str = ""
    metrics: dict = None
    extra_data: dict = None  # Store additional dynamic data

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.extra_data is None:
            self.extra_data = {}

    def to_dict(self) -> dict:
        data = asdict(self)
        # Flatten extra_data into the main dict
        if data['extra_data']:
            data.update(data['extra_data'])
        del data['extra_data']  # Remove the extra_data key itself
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

class StructuredLogger:
    """Enhanced logging with structured output and metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Set up structured logging format
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _create_log_entry(self, level: str, message: str, **kwargs) -> LogEntry:
        """Creates a structured log entry."""
        # Separate known LogEntry fields from extra data
        known_fields = {
            'session_id', 'client_id', 'processing_time', 'file_hash', 
            'error_type', 'stack_trace', 'metrics'
        }
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'module': self.name
        }
        
        extra_data = {}
        
        # Separate known fields from extra data
        for key, value in kwargs.items():
            if key in known_fields:
                log_data[key] = value
            else:
                extra_data[key] = value
        
        log_data['extra_data'] = extra_data if extra_data else None
        
        return LogEntry(**log_data)
    
    def info(self, message: str, **kwargs):
        log_entry = self._create_log_entry('INFO', message, **kwargs)
        self.logger.info(f"{message} | {json.dumps(kwargs) if kwargs else ''}")
        self._update_metrics('info')
    
    def error(self, message: str, **kwargs):
        log_entry = self._create_log_entry('ERROR', message, **kwargs)
        self.logger.error(f"{message} | {json.dumps(kwargs) if kwargs else ''}")
        self._update_metrics('error')
    
    def warning(self, message: str, **kwargs):
        log_entry = self._create_log_entry('WARNING', message, **kwargs)
        self.logger.warning(f"{message} | {json.dumps(kwargs) if kwargs else ''}")
        self._update_metrics('warning')
    
    def debug(self, message: str, **kwargs):
        log_entry = self._create_log_entry('DEBUG', message, **kwargs)
        self.logger.debug(f"{message} | {json.dumps(kwargs) if kwargs else ''}")
    
    def _update_metrics(self, level: str):
        """Updates internal metrics."""
        with _metrics_lock:
            if level == 'error':
                _performance_metrics['errors_logged'] += 1

# Initialize structured loggers
app_logger = StructuredLogger('aircraft_detector.app')
utils_logger = StructuredLogger('aircraft_detector.utils')
security_logger = StructuredLogger('aircraft_detector.security')
performance_logger = StructuredLogger('aircraft_detector.performance')

# Configure root logging (backwards compatibility)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Security configuration
MAX_FILE_SIZE_MB = 100  # Maximum file size in MB
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
ALLOWED_IMAGE_MIMES = {'image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'}
ALLOWED_VIDEO_MIMES = {'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 'video/webm'}
MAX_FILENAME_LENGTH = 255
BLOCKED_PATTERNS = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com', '.js', '.vbs', '.jar']

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 10  # Max requests per time window
RATE_LIMIT_WINDOW = 300   # Time window in seconds (5 minutes)
RATE_LIMIT_COOLDOWN = 60  # Cooldown period in seconds after hitting limit

# Global rate limiting storage (in production, use Redis or database)
_rate_limiter_storage: Dict[str, deque] = defaultdict(deque)
_rate_limiter_blocked: Dict[str, float] = defaultdict(float)

# Session security configuration
SESSION_TOKEN_LENGTH = 32
CSRF_TOKEN_LENGTH = 16
SESSION_TIMEOUT = 3600  # 1 hour in seconds

# Session storage (in production, use Redis or database)
_session_storage: Dict[str, Dict] = {}
_csrf_tokens: Dict[str, str] = {}

def sanitize_filename(filename: str) -> str:
    """Sanitizes a filename to prevent path traversal and dangerous characters."""
    if not filename:
        return "unnamed_file"
    
    # Remove path components
    filename = Path(filename).name
    
    # Check length
    if len(filename) > MAX_FILENAME_LENGTH:
        name, ext = Path(filename).stem[:MAX_FILENAME_LENGTH-10], Path(filename).suffix
        filename = f"{name}{ext}"
    
    # Remove dangerous characters
    dangerous_chars = '<>:"/\\|?*'
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Check for blocked patterns
    filename_lower = filename.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in filename_lower:
            filename = filename_lower.replace(pattern, '_blocked_')
    
    return filename

def validate_file_upload(file_bytes: bytes, filename: str, file_type: str = 'image') -> Tuple[bool, str]:
    """Validates uploaded files for security and format compliance.
    
    Args:
        file_bytes: The file content as bytes
        filename: Original filename
        file_type: 'image' or 'video'
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_bytes:
        return False, "Empty file uploaded"
    
    # Check file size
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f}MB). Maximum allowed: {MAX_FILE_SIZE_MB}MB"
    
    # Sanitize and validate filename
    clean_filename = sanitize_filename(filename)
    if not clean_filename or clean_filename == 'unnamed_file':
        return False, "Invalid filename"
    
    # Check file extension
    file_ext = Path(clean_filename).suffix.lower()
    allowed_extensions = ALLOWED_IMAGE_EXTENSIONS if file_type == 'image' else ALLOWED_VIDEO_EXTENSIONS
    if file_ext not in allowed_extensions:
        return False, f"File extension '{file_ext}' not allowed. Allowed: {', '.join(allowed_extensions)}"
    
    # Check MIME type from file content
    try:
        mime_type, _ = mimetypes.guess_type(clean_filename)
        allowed_mimes = ALLOWED_IMAGE_MIMES if file_type == 'image' else ALLOWED_VIDEO_MIMES
        
        if mime_type and mime_type not in allowed_mimes:
            return False, f"MIME type '{mime_type}' not allowed"
    except Exception as e:
        logging.warning(f"MIME type detection failed: {e}")
    
    # Check file signature (magic bytes) for basic validation
    if file_type == 'image':
        if not _validate_image_signature(file_bytes):
            return False, "File does not appear to be a valid image"
    elif file_type == 'video':
        if not _validate_video_signature(file_bytes):
            return False, "File does not appear to be a valid video"
    
    # Check for suspicious content
    if _contains_suspicious_content(file_bytes):
        return False, "File contains suspicious content"
    
    logging.info(f"File validation passed: {clean_filename} ({file_size_mb:.1f}MB)")
    return True, clean_filename

def _validate_image_signature(file_bytes: bytes) -> bool:
    """Validates image file signatures (magic bytes)."""
    if len(file_bytes) < 12:
        return False
    
    # Common image signatures
    signatures = {
        b'\xff\xd8\xff': 'JPEG',
        b'\x89PNG\r\n\x1a\n': 'PNG', 
        b'BM': 'BMP',
        b'GIF87a': 'GIF',
        b'GIF89a': 'GIF',
        b'RIFF': 'WEBP',  # Simplified check
        b'II*\x00': 'TIFF',
        b'MM\x00*': 'TIFF'
    }
    
    for sig in signatures:
        if file_bytes.startswith(sig):
            return True
    
    return False

def _validate_video_signature(file_bytes: bytes) -> bool:
    """Validates video file signatures (magic bytes)."""
    if len(file_bytes) < 12:
        return False
    
    # Common video signatures
    signatures = [
        b'\x00\x00\x00\x14ftyp',  # MP4
        b'\x00\x00\x00\x18ftyp',  # MP4
        b'\x00\x00\x00\x1cftyp',  # MP4
        b'\x00\x00\x00\x20ftyp',  # MP4
        b'ftyp',  # MP4 (at offset 4)
        b'RIFF',  # AVI
        b'\x1a\x45\xdf\xa3',  # MKV
    ]
    
    for sig in signatures:
        if sig in file_bytes[:32]:  # Check first 32 bytes
            return True
    
    return False

def _contains_suspicious_content(file_bytes: bytes) -> bool:
    """Checks for suspicious content patterns in files."""
    # Check for executable signatures
    executable_sigs = [
        b'MZ',  # DOS/Windows executable
        b'\x7fELF',  # Linux executable
        b'\xca\xfe\xba\xbe',  # Java class file
        b'PK\x03\x04',  # ZIP/JAR (could contain executables)
    ]
    
    for sig in executable_sigs:
        if file_bytes.startswith(sig):
            return True
    
    # Check for script patterns
    script_patterns = [
        b'<script',
        b'javascript:',
        b'eval(',
        b'exec(',
    ]
    
    file_start = file_bytes[:1024].lower()  # Check first 1KB
    for pattern in script_patterns:
        if pattern in file_start:
            return True
    
    return False

def generate_file_hash(file_bytes: bytes) -> str:
    """Generates SHA-256 hash of file content for integrity checking."""
    return hashlib.sha256(file_bytes).hexdigest()

def get_client_id() -> str:
    """Gets a unique identifier for the current client session."""
    try:
        import streamlit as st
        # Use Streamlit session state to create a unique client ID
        if 'client_id' not in st.session_state:
            import uuid
            st.session_state.client_id = str(uuid.uuid4())[:8]
        return st.session_state.client_id
    except ImportError:
        return "default_client"

def check_rate_limit(client_id: str = None) -> Tuple[bool, str, int]:
    """Checks if client is within rate limits.
    
    Args:
        client_id: Unique client identifier (auto-generated if None)
    
    Returns:
        Tuple of (is_allowed, message, remaining_requests)
    """
    if client_id is None:
        client_id = get_client_id()
    
    current_time = time.time()
    
    # Check if client is in cooldown period
    if client_id in _rate_limiter_blocked:
        blocked_until = _rate_limiter_blocked[client_id]
        if current_time < blocked_until:
            remaining_cooldown = int(blocked_until - current_time)
            return False, f"Rate limit exceeded. Try again in {remaining_cooldown} seconds.", 0
        else:
            # Cooldown period expired, remove from blocked list
            del _rate_limiter_blocked[client_id]
    
    # Clean up old requests outside the time window
    client_requests = _rate_limiter_storage[client_id]
    cutoff_time = current_time - RATE_LIMIT_WINDOW
    
    while client_requests and client_requests[0] < cutoff_time:
        client_requests.popleft()
    
    # Check if under rate limit
    if len(client_requests) < RATE_LIMIT_REQUESTS:
        client_requests.append(current_time)
        remaining = RATE_LIMIT_REQUESTS - len(client_requests)
        return True, f"Request allowed. {remaining} requests remaining.", remaining
    else:
        # Rate limit exceeded, add to blocked list
        _rate_limiter_blocked[client_id] = current_time + RATE_LIMIT_COOLDOWN
        logging.warning(f"Rate limit exceeded for client {client_id}")
        return False, f"Rate limit exceeded ({RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW//60} minutes). Cooldown: {RATE_LIMIT_COOLDOWN} seconds.", 0

def reset_rate_limit(client_id: str = None) -> None:
    """Resets rate limit for a client (admin function)."""
    if client_id is None:
        client_id = get_client_id()
    
    if client_id in _rate_limiter_storage:
        del _rate_limiter_storage[client_id]
    if client_id in _rate_limiter_blocked:
        del _rate_limiter_blocked[client_id]
    
    logging.info(f"Rate limit reset for client {client_id}")

def get_rate_limit_stats() -> Dict[str, int]:
    """Gets current rate limiting statistics."""
    return {
        'active_clients': len(_rate_limiter_storage),
        'blocked_clients': len(_rate_limiter_blocked),
        'total_requests': sum(len(requests) for requests in _rate_limiter_storage.values())
    }

def create_session() -> Dict[str, any]:
    """Creates a new secure session."""
    try:
        import streamlit as st
        
        if 'session_data' not in st.session_state:
            session_id = secrets.token_urlsafe(SESSION_TOKEN_LENGTH)
            csrf_token = secrets.token_urlsafe(CSRF_TOKEN_LENGTH)
            
            session_data = {
                'session_id': session_id,
                'csrf_token': csrf_token,
                'created_at': time.time(),
                'last_activity': time.time(),
                'file_hashes': set(),  # Track processed files to prevent duplicates
                'processing_count': 0,
                'client_id': get_client_id()
            }
            
            st.session_state.session_data = session_data
            _session_storage[session_id] = session_data
            _csrf_tokens[session_id] = csrf_token
            
            logging.info(f"Created new session: {session_id[:8]}...")
        
        return st.session_state.session_data
    
    except ImportError:
        # Fallback for non-Streamlit environments
        return {'session_id': 'fallback', 'csrf_token': 'fallback', 'created_at': time.time()}

def validate_session() -> Tuple[bool, str]:
    """Validates the current session."""
    try:
        import streamlit as st
        
        if 'session_data' not in st.session_state:
            return False, "No active session"
        
        session = st.session_state.session_data
        current_time = time.time()
        
        # Check session timeout
        if current_time - session['created_at'] > SESSION_TIMEOUT:
            cleanup_session(session['session_id'])
            return False, "Session expired"
        
        # Update last activity
        session['last_activity'] = current_time
        
        return True, "Session valid"
    
    except ImportError:
        return True, "Fallback session"

def cleanup_session(session_id: str) -> None:
    """Cleans up expired or invalid sessions."""
    try:
        import streamlit as st
        
        if session_id in _session_storage:
            del _session_storage[session_id]
        if session_id in _csrf_tokens:
            del _csrf_tokens[session_id]
        
        if 'session_data' in st.session_state:
            if st.session_state.session_data.get('session_id') == session_id:
                del st.session_state.session_data
        
        security_logger.info(f"Cleaned up session: {session_id[:8]}...",
                           session_id=session_id)
    
    except ImportError:
        pass

def get_session_stats() -> Dict[str, any]:
    """Gets current session statistics."""
    session = create_session()
    current_time = time.time()
    
    return {
        'session_age': int(current_time - session['created_at']),
        'processing_count': session['processing_count'],
        'files_processed': len(session['file_hashes']),
        'active_sessions': len(_session_storage),
        'last_activity': int(current_time - session['last_activity'])
    }

def get_system_health() -> Dict[str, any]:
    """Gets comprehensive system health metrics."""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Update rolling metrics
        with _metrics_lock:
            _performance_metrics['cpu_usage'].append(cpu_percent)
            _performance_metrics['memory_usage'].append(memory.percent)
            
            # Calculate averages
            avg_processing_time = (sum(_performance_metrics['processing_times']) / 
                                 len(_performance_metrics['processing_times'])) if _performance_metrics['processing_times'] else 0
        
        uptime = time.time() - _performance_metrics['startup_time']
        
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': int(uptime),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': round(memory.available / 1024**3, 2),
                'disk_free_gb': round(disk.free / 1024**3, 2),
            },
            'application': {
                'requests_total': _performance_metrics['requests_total'],
                'requests_failed': _performance_metrics['requests_failed'],
                'avg_processing_time': round(avg_processing_time, 3),
                'model_predictions': _performance_metrics['model_predictions'],
                'files_processed': _performance_metrics['files_processed'],
            },
            'sessions': {
                'active_sessions': len(_session_storage),
                'rate_limited_clients': len(_rate_limiter_blocked)
            }
        }
    
    except Exception as e:
        utils_logger.error(f"Health check failed: {str(e)}", error_type=type(e).__name__)
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def is_system_healthy() -> Tuple[bool, List[str]]:
    """Checks if system is healthy and returns issues."""
    issues = []
    
    try:
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > 90:
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            issues.append(f"High memory usage: {memory.percent:.1f}%")
        
        # Check error rate
        total_requests = _performance_metrics['requests_total']
        failed_requests = _performance_metrics['requests_failed']
        if total_requests > 0:
            error_rate = (failed_requests / total_requests) * 100
            if error_rate > 20:  # More than 20% error rate
                issues.append(f"High error rate: {error_rate:.1f}%")
        
        return len(issues) == 0, issues
    
    except Exception as e:
        issues.append(f"Health check error: {str(e)}")
        return False, issues

@st.cache_resource
def load_model(model_path: Path) -> YOLO:
    """Loads a YOLOv8 model from the specified path."""
    try:
        model = YOLO(model_path)
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
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

@contextmanager
def performance_monitor(operation: str, session_id: str = ""):
    """Context manager to monitor operation performance."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        with _metrics_lock:
            _performance_metrics['requests_total'] += 1
        
        performance_logger.info(f"Started {operation}", 
                              session_id=session_id,
                              start_memory=start_memory)
        yield
        
    except Exception as e:
        with _metrics_lock:
            _performance_metrics['requests_failed'] += 1
        
        performance_logger.error(f"Failed {operation}: {str(e)}",
                               session_id=session_id,
                               error_type=type(e).__name__)
        raise
    
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        processing_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        with _metrics_lock:
            _performance_metrics['processing_times'].append(processing_time)
            _performance_metrics['memory_usage'].append(end_memory)
        
        performance_logger.info(f"Completed {operation}",
                              session_id=session_id,
                              processing_time=processing_time,
                              memory_used=memory_delta,
                              total_memory=end_memory)

def process_image(model: YOLO, image: Image.Image, confidence: float, iou: float) -> tuple[Image.Image, int, dict]:
    """
    Processes a single image, returning the annotated image, detection count, and speed metrics.
    """
    # Validate image format and size
    if image is None:
        raise ValueError("Invalid image: Image is None")
    if image.size[0] < 32 or image.size[1] < 32:
        raise ValueError(f"Image too small: {image.size}. Minimum size is 32x32 pixels")
    
    # Resize image before processing
    image = resize_image(image)
    
    try:
        results = model.predict(source=image, conf=confidence, iou=iou)
        result = results[0]
        
        # Safe null check for detection count
        detection_count = len(result.boxes) if result.boxes is not None else 0
        speed_metrics = result.speed if hasattr(result, 'speed') else {'preprocess': 0, 'inference': 0, 'postprocess': 0}
        
        processed_array = result.plot()
        processed_image = Image.fromarray(cv2.cvtColor(processed_array, cv2.COLOR_BGR2RGB))
        
        utils_logger.info(f"Processed image: {detection_count} detections",
                         processing_time=speed_metrics.get('inference', 0),
                         detection_count=detection_count,
                         image_size=f"{image.size[0]}x{image.size[1]}")
        
        with _metrics_lock:
            _performance_metrics['model_predictions'] += 1
            _performance_metrics['files_processed'] += 1
        return processed_image, detection_count, speed_metrics
        
    except Exception as e:
        utils_logger.error(f"Error during image processing: {str(e)}",
                          error_type=type(e).__name__,
                          image_size=f"{image.size[0]}x{image.size[1]}" if hasattr(image, 'size') else 'unknown')
        # Return original image with zero detections on error
        return image, 0, {'preprocess': 0, 'inference': 0, 'postprocess': 0}

def get_video_info(video_bytes: bytes) -> dict:
    """Validates a video from bytes and extracts its properties using OpenCV."""
    if not video_bytes or len(video_bytes) == 0:
        logging.warning("Received empty video data")
        return {"error": "Empty video data"}
    
    video_path = None
    cap = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(video_bytes)
            video_path = tfile.name
        
        logging.info(f"Created temporary video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return {"error": "Could not open video file. File may be corrupted or in unsupported format."}
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            logging.error(f"Invalid video properties: fps={fps}, frames={frame_count}, size={width}x{height}")
            return {"error": "Invalid video properties detected"}
        
        duration_seconds = frame_count / fps
        logging.info(f"Video info: {width}x{height}, {fps}fps, {frame_count} frames, {duration_seconds:.1f}s")
        return {
            "width": width, 
            "height": height, 
            "fps": fps, 
            "frame_count": frame_count, 
            "duration": duration_seconds
        }
    
    except Exception as e:
        logging.error(f"Error analyzing video: {str(e)}")
        return {"error": f"Error analyzing video: {str(e)}"}
    
    finally:
        # Ensure proper cleanup with better error handling
        try:
            if cap is not None:
                cap.release()
                logging.debug("VideoCapture released")
        except Exception as e:
            logging.warning(f"Error releasing VideoCapture: {e}")
            
        try:
            if video_path and Path(video_path).exists():
                Path(video_path).unlink()
                logging.debug(f"Temporary video file deleted: {video_path}")
        except OSError as e:
            logging.warning(f"Could not delete temporary file {video_path}: {e}")