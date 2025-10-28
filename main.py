import streamlit as st
import json
import cv2
from ultralytics import YOLO 
import numpy as np
import math
import re
import os
import sqlite3
from datetime import datetime
from PIL import Image
import tempfile
import pandas as pd
import io
import matplotlib.pyplot as plt

# --- TESSERACT OCR LIBRARIES ---
import pytesseract
# NOTE: Manual TESSERACT_PATH assignment has been REMOVED. 
# The script now relies entirely on Tesseract being in the system's PATH.
# ---------------------------------

# NOTE: The TrafficDB class definition is assumed to be in 'traffic_db.py'
# The user's prompt did not include this file, so a placeholder class is used
# to prevent execution errors, but *you must provide the actual implementation*
# of TrafficDB for the second mode to function correctly.
class TrafficDB:
    """Placeholder for the required TrafficDB class from traffic_db.py."""
    def __init__(self, db_name='traffic_analysis.db'):
        self.db_name = db_name
        self.setup_traffic_database()

    def setup_traffic_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                source_type TEXT,
                vehicle_class TEXT,
                count INTEGER,
                traffic_level TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_result(self, timestamp, source_type, vehicle_class, count, traffic_level):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analysis_results 
            (timestamp, source_type, vehicle_class, count, traffic_level)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, source_type, vehicle_class, count, traffic_level))
        conn.commit()
        conn.close()

    def fetch_all_data(self):
        conn = sqlite3.connect(self.db_name)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()
        return df

    def clear_db(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM analysis_results')
        conn.commit()
        conn.close()
        
# --- Global Configuration and Initialization ---

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs("json", exist_ok=True)

# Model paths for the two distinct features
LP_CUSTOM_WEIGHTS_PATH = "weights/best.pt" 
ATCC_MODEL_PATH = "yolo11n.pt" 

# Class Names for LP Detector
LP_CLASS_NAMES = ["licence", "licenseplate"] 

# Check Tesseract availability once (relying on system PATH)
try:
    pytesseract.image_to_string(Image.new('RGB', (10, 10)), config='--psm 10')
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False
    
# --- Common/Cached YOLO Model Loader ---

@st.cache_resource
def initialize_yolo_model(weights_path):
    """Initializes and caches the YOLO model."""
    try:
        if not os.path.exists(weights_path):
            st.error(f"YOLO model not found at path: {weights_path}")
            return None
        model = YOLO(weights_path)
        return model
    except Exception as e:
        st.error(f"An error occurred during model loading from {weights_path}: {e}")
        return None

# --- DATABASE SETUP (License Plate) ---

def setup_license_plate_database():
    """Sets up the SQLite database and table for License Plates."""
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS LicensePlates (
            id INTEGER PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            license_plate TEXT
        )
    ''')
    conn.commit()
    conn.close()

setup_license_plate_database()

# --- LICENSE PLATE MODE FUNCTIONS ---

def tesseract_ocr_process(frame, x1, y1, x2, y2):
    """Performs Tesseract OCR on a cropped license plate."""
    if not TESSERACT_AVAILABLE:
        return "OCR_PATH_ERROR"
        
    h, w, _ = frame.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return "INVALID_CROP" 
        
    cropped_frame = frame[y1:y2, x1:x2].copy()
    
    try:
        # Pre-processing: Grayscale -> Threshold (Otsu) -> Blur
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.medianBlur(thresh, 3) 
        
        pil_image = Image.fromarray(thresh)
        
        # PSM 7: Treat as a single text line/block (ideal for plates)
        ocr_config = '--psm 7 -l eng'
        raw_text = pytesseract.image_to_string(pil_image, config=ocr_config)
    except Exception:
        return "OCR_EXEC_FAIL"
    
    # --- Cleanup and Formatting ---
    pattern = re.compile(r'[^A-Z0-9\s]')
    cleaned_text = pattern.sub('', raw_text.upper()).strip()
    final_text = cleaned_text.replace(" ", "") 

    # --- Enforced Saving Logic (Guaranteed result) ---
    if not final_text:
        return f"NO_CLEAN_TEXT({raw_text.strip() or 'BLANK'})"
        
    return final_text

def save_lp_json(license_plates, startTime, endTime):
    """Saves license plate data to individual and cumulative JSON files."""
    if not license_plates:
        return
        
    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plates": list(license_plates)
    }
    
    interval_file_path = f"json/output_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent=2)

    cummulative_file_path = "json/LicensePlateData.json"
    existing_data = []
    if os.path.exists(cummulative_file_path):
        try:
            with open(cummulative_file_path, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            st.warning("Cumulative JSON file corrupted. Starting a new one.")

    existing_data.append(interval_data)

    with open(cummulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)

    save_to_lp_database(license_plates, startTime, endTime)
    st.success(f"Saved data for {len(license_plates)} detected entries to JSON/DB.")


def save_to_lp_database(license_plates, start_time, end_time):
    """Saves license plate data to the SQLite database (LicensePlates table)."""
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    for plate in license_plates:
        cursor.execute('''
            INSERT INTO LicensePlates(start_time, end_time, license_plate)
            VALUES (?, ?, ?)
        ''', (start_time.isoformat(), end_time.isoformat(), plate))
            
    conn.commit()
    conn.close()

def process_lp_frame(frame, license_plates_set, model):
    """Runs YOLO detection and Tesseract OCR on a single frame."""
    if model is None:
        return frame
        
    results = model.predict(frame, conf=0.45, verbose=False)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            conf = math.ceil(box.conf[0].item() * 100) / 100
            classNameInt = int(box.cls[0].item())
            
            if classNameInt < len(LP_CLASS_NAMES):
                clsName = LP_CLASS_NAMES[classNameInt]
            else:
                clsName = "Unknown" 

            # Execute OCR function
            label = tesseract_ocr_process(frame.copy(), x1, y1, x2, y2)
            
            # Label will always be non-empty due to enforced placeholders
            license_plates_set.add(label)
                
            display_label = label if label else f'{clsName}:{conf:.2f}'

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw text background
            textSize = cv2.getTextSize(display_label, 0, fontScale=0.5, thickness=2)[0]
            c2 = x1 + textSize[0] + 5, y1 - textSize[1] - 8
            cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, display_label, (x1, y1 - 4), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    return frame

def lp_video_processing_loop(cap, model):
    """Processes video from a capture object (file or camera) for License Plate Detection."""
    st.subheader("Processing Video Feed... 🚗")
    
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    plate_placeholder = st.empty()
    
    startTime = datetime.now()
    license_plates = set()
    frame_count = 0
    
    is_file = cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 
    max_frames = 600 if not is_file else cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret or frame is None:
            break

        frame_count += 1
        
        h, w, _ = frame.shape
        if w > 800:
            frame = cv2.resize(frame, (800, int(800 * h / w)))
        
        processed_frame = process_lp_frame(frame, license_plates, model)
        
        frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"Frame {frame_count}/{int(max_frames) if is_file else 'live'}")

        # Time-based saving logic (every 20 seconds)
        currentTime = datetime.now()
        if (currentTime - startTime).seconds >= 20:
            endTime = currentTime
            save_lp_json(license_plates, startTime, endTime)
            startTime = currentTime
            license_plates.clear()

        status_placeholder.text(f"Frames processed: {frame_count} | Unique Entries: {len(license_plates)} (since last save)")
        plate_placeholder.json({"Detected Entries (since last save)": list(license_plates)})
        
        if not is_file and frame_count >= 600:
             break 

        cv2.waitKey(1) 

    if license_plates:
        save_lp_json(license_plates, startTime, datetime.now())
        
    cap.release()
    frame_placeholder.empty()
    st.success("Video processing finished.")

# --- ATCC (VEHICLE ANALYZER) MODE FUNCTIONS ---

def calculate_traffic_level(total_count):
    """Classifies traffic density based on total vehicle count."""
    if total_count == 0:
        return "No Traffic"
    elif total_count <= 5:
        return "Low Traffic"
    elif total_count <= 15:
        return "Medium Traffic"
    else:
        return "High Traffic"

def process_atcc_detection(results, db: TrafficDB, source_type="Image/Video"):
    """
    Processes YOLO detection results, logs to DB, and returns summary data.
    Assumes `results` is an ultralytics 'Results' object or a list of such objects.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not isinstance(results, list):
        results = [results]

    total_vehicles = 0
    class_counts = {}

    for res in results:
        if hasattr(res.boxes, 'cls') and res.boxes.cls is not None:
            detection_classes = res.boxes.cls.cpu().numpy()
            
            try:
                class_names = [results[0].names[int(cls_id)] for cls_id in detection_classes]
            except (AttributeError, KeyError):
                class_names = [f"Class {int(cls_id)}" for cls_id in detection_classes]

            for class_name in class_names:
                total_vehicles += 1
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    traffic_level = calculate_traffic_level(total_vehicles)

    for vehicle_class, count in class_counts.items():
        db.save_result(timestamp, source_type, vehicle_class, count, traffic_level)
    
    if not class_counts:
          db.save_result(timestamp, source_type, "N/A", 0, "No Traffic")


    summary = {
        'timestamp': timestamp,
        'total_vehicles': total_vehicles,
        'traffic_level': traffic_level,
        'class_counts': class_counts
    }
    return summary

def annotate_atcc_image(result):
    """Annotates a single YOLO result image and returns it as a PNG byte buffer."""
    annotated_img = result.plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    buf = io.BytesIO()
    plt.figure(figsize=(8, 8))
    plt.imshow(annotated_img_rgb)
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return buf

def display_raw_data(db: TrafficDB):
    """Displays the raw contents of the SQLite database in a new expander."""
    with st.expander("Raw Database Table (analysis_results)", expanded=True):
        raw_df = db.fetch_all_data()
        st.dataframe(raw_df, use_container_width=True)
        st.markdown(f"Total Records: **{len(raw_df)}**")

# --- CUSTOM CSS STYLES (FROM ATCC APP) ---

def apply_custom_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #3B82F6; /* Blue-500 */
            text-align: center;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #60A5FA; /* Blue-400 */
        }
        .stButton>button {
            background-color: #10B981; /* Emerald-500 */
            color: white;
            font-weight: 600;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            transition: background-color 0.2s;
        }
        .stButton>button:hover {
            background-color: #059669; /* Emerald-600 */
        }
        .traffic-low { color: #10B981; font-weight: 700; }
        .traffic-medium { color: #F59E0B; font-weight: 700; }
        .traffic-high { color: #EF4444; font-weight: 700; }
        .traffic-none { color: #6B7280; font-weight: 700; }
        .stFileUploader {
            border: 2px dashed #9CA3AF;
            border-radius: 0.5rem;
            padding: 1rem;
        }
        .caption {
            font-size: 0.8rem;
            color: #6B7280;
            text-align: center;
            margin-top: -0.5rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


# --- LICENSE PLATE MODE LAYOUT ---

def license_plate_mode(model):
    st.title("License Plate Detector & Tesseract OCR 🏷️")
    st.sidebar.title("LP Detector Options")

    # Display Tesseract status
    if TESSERACT_AVAILABLE:
        st.sidebar.success("Tesseract OCR Active.")
    else:
        st.sidebar.error("Tesseract Error: Using placeholders for OCR results.")
        st.warning("Tesseract is unavailable. Please ensure it's installed and added to your system PATH.")
    
    if model is None:
        st.error("License Plate YOLO model did not load. Detection is disabled.")
        return

    source_option = st.sidebar.radio(
        "Select Input Source:",
        ('Upload Video', 'Upload Photo', 'Use Webcam (Experimental)')
    )
    
    st.markdown("---")

    # --- 1. Video Upload ---
    if source_option == 'Upload Video':
        st.header("Video File Upload")
        uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            st.video(uploaded_file)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1] or ".mp4") as tfile:
                tfile.write(uploaded_file.read())
                temp_video_path = tfile.name
                
            if st.button("Start Processing Video 🎬"):
                with st.spinner('Initializing video stream...'):
                    cap = cv2.VideoCapture(temp_video_path)
                lp_video_processing_loop(cap, model)
                
            try:
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
            except Exception as e:
                st.warning(f"Could not clean up temporary file: {e}")

    # --- 2. Photo Upload ---
    elif source_option == 'Upload Photo':
        st.header("Image File Upload")
        uploaded_image = st.file_uploader("Choose a photo...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert('RGB')
            img_array = np.array(image)
            frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption='Original Image', use_container_width=True)
            
            with col2:
                if st.button("Analyze Photo 🖼️"):
                    with st.spinner('Analyzing image...'):
                        license_plates = set()
                        h, w, _ = frame.shape
                        if w > 800:
                            frame = cv2.resize(frame, (800, int(800 * h / w)))
                        
                        processed_frame = process_lp_frame(frame, license_plates, model)
                        
                        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption='Processed Image', use_container_width=True)
                        
                    if license_plates:
                        st.success("Analysis Complete! Detected entries saved to JSON/DB.")
                        st.json(list(license_plates))
                        
                        current_time = datetime.now()
                        save_lp_json(license_plates, current_time, current_time)
                    else:
                        st.info("No license plate objects were detected by YOLO.")


    # --- 3. Webcam Input ---
    elif source_option == 'Use Webcam (Experimental)':
        st.header("Webcam Input (Experimental)")
        st.warning("Webcam capture can be inconsistent in Streamlit. This mode will attempt to run for ~600 frames.")
        
        if st.button("Start Camera 📸"):
            with st.spinner('Attempting to open camera...'):
                cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open camera. Check permissions or if another application is using it.")
            else:
                lp_video_processing_loop(cap, model)

# --- ATCC MODE LAYOUT ---

def atcc_mode(model, db: TrafficDB):
    
    apply_custom_styles()
    st.markdown('<div class="main-header">ATCC YOLOv11 Vehicle Analyzer</div>', unsafe_allow_html=True)
    st.markdown("---")

    if model is None:
        st.error("ATCC YOLO model did not load. Detection is disabled.")
        return

    # --- Sidebar for Settings ---
    with st.sidebar:
        st.header("ATCC Analyzer Options")
        analysis_mode = st.radio("Choose Input Source",
            ('Upload Image/Video', 'Webcam Capture'),
            index=0,
            key='atcc_analysis_mode')
        
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
        
        st.markdown("---")
        st.subheader("Model Information")
        st.markdown(f"Model: `{ATCC_MODEL_PATH}`")
        class_names = model.names.values() if hasattr(model, 'names') else ["(Classes not loaded)"]
        st.markdown(f"Classes Detected ({len(class_names)}): {', '.join(class_names)}")
        st.markdown("---")
        st.markdown("### Data Storage")
        
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            if st.button("View Raw DB"):
                st.session_state['view_raw_db'] = True
        with col_db2:
            if st.button("Clear DB"):
                db.clear_db()
                st.success("Database cleared! Reloading...")
                st.rerun() 

        if st.session_state.get('view_raw_db', False):
            st.session_state['view_raw_db'] = False
            display_raw_data(db)

    # --- Main Content Area ---
    col_input, col_results = st.columns([1, 1])

    uploaded_file = None
    process_button = False

    with col_input:
        st.subheader("Input Source")
        
        if analysis_mode == 'Upload Image/Video':
            uploaded_file = st.file_uploader(
                "Upload an Image (jpg, png) or Video (mp4, mov, avi)",
                type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'],
                key='atcc_uploader'
            )
            process_button = st.button("Start Analysis (Upload)", key='atcc_process_upload')

        elif analysis_mode == 'Webcam Capture':
            st.warning("Webcam analysis is resource-intensive.")
            webcam_image = st.camera_input("Capture an image or video segment.", key='atcc_webcam')
            
            uploaded_file = webcam_image
            process_button = st.button("Start Analysis (Webcam)", disabled=not webcam_image, key='atcc_process_webcam')
        
        media_container = st.container()
        if uploaded_file and analysis_mode == 'Upload Image/Video':
            media_type = uploaded_file.type.split('/')[0]
            if media_type == 'image':
                media_container.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            elif media_type == 'video':
                media_container.markdown('<p class="caption">Uploaded Video</p>', unsafe_allow_html=True)
                media_container.video(uploaded_file)

    # --- Analysis Execution ---
    results_summary = None
    annotated_media = None
    
    if uploaded_file and process_button:
        
        media_type = uploaded_file.type.split('/')[0]
        
        with col_results:
            st.subheader("Detection Results")
            progress_bar = st.progress(0)
            
            temp_path = None
            try:
                # --- Temporary File Handling ---
                file_extension = os.path.splitext(uploaded_file.name)[1] if uploaded_file.name else f".{uploaded_file.type.split('/')[1]}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file.flush()
                    temp_path = tmp_file.name

                args = {
                    'conf': confidence_threshold,
                    'iou': iou_threshold,
                    'save': False,
                    'verbose': False
                }
                
                # --- Run Detection ---
                if media_type == 'image' or (analysis_mode == 'Webcam Capture' and uploaded_file):
                    st.info("Processing image...")
                    results = model.predict(temp_path, **args)
                    annotated_media_buffer = annotate_atcc_image(results[0])
                    results_summary = process_atcc_detection(results, db, source_type="Image" if media_type == 'image' else "Webcam Snapshot")
                    
                    st.markdown('<p class="caption">Annotated Image</p>', unsafe_allow_html=True)
                    st.image(annotated_media_buffer, use_container_width=True)
                    progress_bar.progress(100)
                    
                elif media_type == 'video' and analysis_mode == 'Upload Image/Video':
                    st.info("Processing video feed...")
                    
                    cap = cv2.VideoCapture(temp_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
                    
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames == 0: total_frames = 100

                    frame_idx = 0
                    all_results = []
                    video_placeholder = st.empty()

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        results = model.predict(frame, **args)
                        annotated_frame = results[0].plot()
                        all_results.append(results[0])
                        out.write(annotated_frame)
                        
                        if frame_idx % 5 == 0:
                            video_placeholder.image(annotated_frame, channels="BGR", caption=f"Processing Frame {frame_idx + 1}", use_container_width=True)

                        frame_idx += 1
                        progress_bar.progress(min(int(frame_idx / total_frames * 100), 100))

                    cap.release()
                    out.release()
                    
                    results_summary = process_atcc_detection(all_results, db, source_type="Video")
                    annotated_media = out_path
                    
                    video_placeholder.empty()
                    st.markdown('<p class="caption">Annotated Video Output</p>', unsafe_allow_html=True)
                    st.video(annotated_media, format="video/mp4", start_time=0)
                    
            except Exception as e:
                st.error(f"An error occurred during detection: {e}")
                results_summary = None
            finally:
                # Clean up temporary files
                if temp_path and os.path.exists(temp_path): os.remove(temp_path)
                if annotated_media and os.path.exists(annotated_media) and media_type == 'video': os.remove(annotated_media)
                progress_bar.empty()

    # --- Display Results and Visualization ---
    st.markdown("---")
    
    if results_summary:
        st.subheader("Analysis Summary")
        
        traffic_level_class = results_summary['traffic_level'].lower().replace(' ', '-')
        
        st.markdown(f"""
            <div style="background-color: #F3F4F6; padding: 15px; border-radius: 0.5rem;">
                <p style="font-size: 1.1rem; margin-bottom: 5px;">
                    <strong>Timestamp:</strong> {results_summary['timestamp']}
                </p>
                <p style="font-size: 1.1rem; margin-bottom: 5px;">
                    <strong>Total Vehicles Detected:</strong> <span style="color:#3B82F6; font-weight:700;">{results_summary['total_vehicles']}</span>
                </p>
                <p style="font-size: 1.1rem; margin-bottom: 0;">
                    <strong>Traffic Level:</strong>
                    <span class="traffic-{traffic_level_class}">{results_summary['traffic_level']}</span>
                </p>
            </div>
            <br>
        """, unsafe_allow_html=True)

        st.subheader("Vehicle Breakdown")
        
        class_counts_dict = results_summary.get('class_counts', {})
        if class_counts_dict:
            class_df = pd.DataFrame(
                list(class_counts_dict.items()),
                columns=['Vehicle Class', 'Count']
            )
            class_df.set_index('Vehicle Class', inplace=True)
            st.table(class_df)
        else:
            st.info("No vehicles were detected in the media.")


    st.markdown("---")
    st.header("Traffic History Data")
    
    db_df = db.fetch_all_data()
    
    if not db_df.empty:
        st.subheader("Raw Analysis Log")
        st.dataframe(db_df, use_container_width=True)
        st.markdown(f"Total Records: **{len(db_df)}**")
        
        traffic_summary = db_df['traffic_level'].value_counts().reset_index()
        traffic_summary.columns = ['Traffic Level', 'Count']
        st.subheader("Traffic Level Totals")
        st.dataframe(traffic_summary)

    else:
        st.info("No historical analysis data is available yet. Run an analysis to populate the database.")


# --- MAIN APPLICATION ENTRY POINT ---

def main():
    st.set_page_config(page_title="Combined YOLO App", layout="wide", initial_sidebar_state="expanded")
    
    st.sidebar.title("App Selection")
    
    # Main selector for the two application modes
    app_mode = st.sidebar.radio(
        "Select Application Mode:",
        ('License Plate Detector (LP) / OCR', 'Vehicle Traffic Analyzer (ATCC)'),
        key='app_mode_select'
    )
    
    st.sidebar.markdown("---")

    # Initialize DB (Traffic) in session state
    if 'atcc_db' not in st.session_state:
        st.session_state['atcc_db'] = TrafficDB()
    atcc_db = st.session_state['atcc_db']

    if app_mode == 'License Plate Detector (LP) / OCR':
        # Load the LP model
        lp_model = initialize_yolo_model(LP_CUSTOM_WEIGHTS_PATH)
        license_plate_mode(lp_model)
    elif app_mode == 'Vehicle Traffic Analyzer (ATCC)':
        # Load the ATCC model
        atcc_model = initialize_yolo_model(ATCC_MODEL_PATH)
        atcc_mode(atcc_model, atcc_db)

if __name__ == '__main__':
    main()