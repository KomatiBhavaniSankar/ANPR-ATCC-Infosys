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

# --- TESSERACT OCR LIBRARIES ---
import pytesseract
# NOTE: Manual TESSERACT_PATH assignment has been REMOVED. 
# The script now relies entirely on Tesseract being in the system's PATH.
# If it's not, TESSERACT_AVAILABLE will be False, and it will use placeholders.
# ---------------------------------

# --- Configuration and Initialization ---

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs("json", exist_ok=True)
CUSTOM_WEIGHTS_PATH = "weights/best.pt" 

@st.cache_resource
def initialize_yolo_model(weights_path):
    """Initializes and caches the YOLO model."""
    try:
        if not os.path.exists(weights_path):
            return None
        model = YOLO(weights_path)
        return model
    except Exception:
        return None

# Class Names: Match your trained dataset classes
CLASS_NAMES = ["licence", "licenseplate"] 

# Check Tesseract availability once (relying on system PATH)
try:
    # Attempt a simple check to ensure Tesseract runs
    pytesseract.image_to_string(Image.new('RGB', (10, 10)), config='--psm 10')
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False
    
# Database setup
def setup_database():
    """Sets up the SQLite database and table if they don't exist."""
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

setup_database()

# --- OCR Processing Function (Tesseract) ---

def tesseract_ocr_process(frame, x1, y1, x2, y2):
    """
    Performs Tesseract OCR. Returns recognized text or an informative placeholder
    to ensure data is saved for every detected plate.
    """
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
        # Placeholder containing the raw text found (before cleanup) if final text is blank
        return f"NO_CLEAN_TEXT({raw_text.strip() or 'BLANK'})"
        
    return final_text

# --- Database and File Saving Functions ---

def save_json(license_plates, startTime, endTime):
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

    save_to_database(license_plates, startTime, endTime)
    st.success(f"Saved data for {len(license_plates)} detected entries to JSON/DB.")


def save_to_database(license_plates, start_time, end_time):
    """Saves license plate data to the SQLite database."""
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    for plate in license_plates:
        cursor.execute('''
            INSERT INTO LicensePlates(start_time, end_time, license_plate)
            VALUES (?, ?, ?)
        ''', (start_time.isoformat(), end_time.isoformat(), plate))
            
    conn.commit()
    conn.close()

# --- Core Processing Logic ---

def process_frame(frame, license_plates_set, model):
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
            
            if classNameInt < len(CLASS_NAMES):
                clsName = CLASS_NAMES[classNameInt]
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

def video_processing_loop(cap, model):
    """Processes video from a capture object (file or camera)."""
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
        
        processed_frame = process_frame(frame, license_plates, model)
        
        frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"Frame {frame_count}/{int(max_frames) if is_file else 'live'}")

        # Time-based saving logic (every 20 seconds)
        currentTime = datetime.now()
        if (currentTime - startTime).seconds >= 20:
            endTime = currentTime
            save_json(license_plates, startTime, endTime)
            startTime = currentTime
            license_plates.clear()

        status_placeholder.text(f"Frames processed: {frame_count} | Unique Entries: {len(license_plates)} (since last save)")
        plate_placeholder.json({"Detected Entries (since last save)": list(license_plates)})
        
        if not is_file and frame_count >= 600:
             break 

        cv2.waitKey(1) 

    if license_plates:
        save_json(license_plates, startTime, datetime.now())
        
    cap.release()
    frame_placeholder.empty()
    st.success("Video processing finished.")


# --- Streamlit App Layout ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Custom License Plate Detector & OCR", layout="wide")
    st.title("Custom License Plate Detector & Tesseract OCR 🏷️")
    st.sidebar.title("Input Options")

    # Display Tesseract status
    if TESSERACT_AVAILABLE:
        st.sidebar.success("Tesseract OCR Active.")
    else:
        st.sidebar.error("Tesseract Error: Using placeholders for OCR results.")
        st.warning("Tesseract is unavailable. Please ensure it's installed and added to your system PATH.")
    
    model = initialize_yolo_model(CUSTOM_WEIGHTS_PATH)
    
    if model is None:
        st.warning("YOLO model did not load. Object detection is disabled.")

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
                video_processing_loop(cap, model)
                
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
                        
                        processed_frame = process_frame(frame, license_plates, model)
                        
                        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption='Processed Image', use_container_width=True)
                        
                    if license_plates:
                        st.success("Analysis Complete! Detected entries saved to JSON/DB.")
                        st.json(list(license_plates))
                        
                        current_time = datetime.now()
                        save_json(license_plates, current_time, current_time)
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
                video_processing_loop(cap, model)
                

if __name__ == '__main__':
    main()