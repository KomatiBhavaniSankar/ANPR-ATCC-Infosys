# --- 1. SETTING UP THE TOOLS (IMPORTING LIBRARIES) ---
# This section is like gathering all the necessary tools and supplies before starting a project.

# Streamlit is the tool that turns our Python code into a simple, interactive website/app.
import streamlit as st
# 'json' is used for saving the results (like license plates) in a structured, standard file format.
import json
# 'cv2' (OpenCV) is a powerful library for handling images and videos, essential for video feeds and drawing boxes.
import cv2
# 'ultralytics' is the company behind the specific AI models (YOLO) we use for object detection.
from ultralytics import YOLO 
# 'numpy' is for high-speed mathematical calculations, especially with images (which are just big numbers).
import numpy as np
# 'math' is for basic math operations, like rounding or calculating percentages.
import math
# 're' is for checking and cleaning up text, like making sure a license plate has only letters and numbers.
import re
# 'os' helps the program talk to the computer's operating system, like checking if a file exists or creating folders.
import os
# 'sqlite3' is used to create and manage a simple database file to store our analysis results long-term.
import sqlite3
# 'datetime' is for keeping track of when the analysis was run (the timestamp).
from datetime import datetime
# 'PIL' (Pillow) is another library for handling image files, used here to prepare images for text reading.
from PIL import Image
# 'tempfile' is for creating temporary files that are automatically deleted later, especially useful when handling uploaded videos.
import tempfile
# 'pandas' is used for handling data in tables (like spreadsheets), which is great for summarizing the traffic analysis results.
import pandas as pd
# 'io' helps handle data streams, especially when saving images without needing a physical file.
import io
# 'matplotlib.pyplot' is used to create simple charts and graphs for visualizing the traffic data.
import matplotlib.pyplot as plt

# --- TESSERACT OCR LIBRARIES ---
# pytesseract is the specific tool we use to read text from an image (Optical Character Recognition - OCR).
import pytesseract
# NOTE: Manual TESSERACT_PATH assignment has been REMOVED. 
# The program now expects Tesseract (the text-reading software) to be already installed and ready on the computer.
# ---------------------------------

# --- TRAFFIC DATABASE CLASS (ATCC MODE) ---
# This class is like the 'logbook' for the Vehicle Traffic Analyzer (ATCC) mode.
class TrafficDB:
    """Placeholder for the required TrafficDB class from traffic_db.py."""
    # When the logbook starts, it sets up the file where data will be stored.
    def __init__(self, db_name='traffic_analysis.db'):
        self.db_name = db_name
        self.setup_traffic_database()

    # This function creates a table in the database to store the vehicle counts, traffic levels, and time.
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

    # This function saves a single observation (like "3 cars, Low Traffic") into the database.
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

    # This function retrieves ALL the stored traffic analysis data to show the user.
    def fetch_all_data(self):
        conn = sqlite3.connect(self.db_name)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()
        return df

    # This function clears the entire logbook, removing all past traffic analysis results.
    def clear_db(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM analysis_results')
        conn.commit()
        conn.close()
        
# --- 2. GLOBAL SETTINGS AND MODEL PATHS ---
# These are like the master settings for the entire application.

# This line fixes a specific technical issue that can happen when using these AI libraries.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Make sure a folder named 'json' exists to save the license plate data.
os.makedirs("json", exist_ok=True)

# File locations for the two different AI brains (YOLO models).
# LP_CUSTOM_WEIGHTS_PATH: The AI model specifically trained to find license plates    (ANPR).
LP_CUSTOM_WEIGHTS_PATH = "weights/best.pt" 
# ATCC_MODEL_PATH: The general AI model trained to find cars, trucks, buses, etc  .
ATCC_MODEL_PATH = "yolo11n.pt" 

# The specific names (classes) the LP AI model looks for (a license plate is a 'licence' or 'licenseplate').
LP_CLASS_NAMES = ["licence", "licenseplate"] 

# Check if the text-reading tool (Tesseract) is available on the system.
try:
    pytesseract.image_to_string(Image.new('RGB', (10, 10)), config='--psm 10')
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False
    
# --- 3. CACHED AI MODEL LOADER ---

# @st.cache_resource tells Streamlit to load the AI model only once, saving time and computer resources.
def initialize_yolo_model(weights_path):
    """Initializes and caches the YOLO model (the AI brain)."""
    try:
        if not os.path.exists(weights_path):
            st.error(f"YOLO model not found at path: {weights_path}")
            return None
        # This line loads the AI model from the specified file.
        model = YOLO(weights_path)
        return model
    except Exception as e:
        st.error(f"An error occurred during model loading from {weights_path}: {e}")
        return None

# --- 4. LICENSE PLATE DATABASE SETUP ---
# This sets up the separate logbook specifically for license plate records.

def setup_license_plate_database():
    """Sets up the SQLite database and table for License Plates."""
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    # Create the table to store the plate number and the time it was seen.
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

# Run the setup function to make sure the license plate database is ready before the program starts.
setup_license_plate_database()

# --- 5. LICENSE PLATE MODE CORE FUNCTIONS ---

def tesseract_ocr_process(frame, x1, y1, x2, y2):
    """Performs Tesseract OCR (text reading) on a cropped license plate."""
    if not TESSERACT_AVAILABLE:
        # If the text-reading tool isn't found, return an error message.
        return "OCR_PATH_ERROR"
        
    # 'Crop' the image to focus only on the area where the license plate was detected.
    h, w, _ = frame.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return "INVALID_CROP" 
        
    cropped_frame = frame[y1:y2, x1:x2].copy()
    
    try:
        # Pre-processing: Tries to improve the image quality (grayscale, contrast adjustment, blurring) 
        # to make it easier for the text-reading tool (Tesseract) to work.
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.medianBlur(thresh, 3) 
        
        pil_image = Image.fromarray(thresh)
        
        # 'PSM 7' tells Tesseract to look for a single line of text, which is ideal for a license plate.
        ocr_config = '--psm 7 -l eng'
        raw_text = pytesseract.image_to_string(pil_image, config=ocr_config)
    except Exception:
        return "OCR_EXEC_FAIL"
    
    # --- Cleanup and Formatting ---
    # This section removes any strange symbols or characters that the text-reading tool might mistake for letters or numbers.
    pattern = re.compile(r'[^A-Z0-9\s]')
    cleaned_text = pattern.sub('', raw_text.upper()).strip()
    # It also removes spaces so the final plate is just a string of characters (e.g., 'MH12AB1234').
    final_text = cleaned_text.replace(" ", "") 

    # If the cleanup fails, save a placeholder result so we know the image was processed.
    if not final_text:
        return f"NO_CLEAN_TEXT({raw_text.strip() or 'BLANK'})"
        
    return final_text

def save_lp_json(license_plates, startTime, endTime):
    """Saves license plate data to individual and cumulative JSON files."""
    if not license_plates:
        return
        
    # This prepares the data to be saved, including the time period it was collected.
    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plates": list(license_plates)
    }
    
    # Save a file for this specific time interval.
    interval_file_path = f"json/output_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent=2)

    # Save all results into one master file for easy reference.
    cummulative_file_path = "json/LicensePlateData.json"
    existing_data = []
    if os.path.exists(cummulative_file_path):
        try:
            # Try to read the old data first.
            with open(cummulative_file_path, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            st.warning("Cumulative JSON file corrupted. Starting a new one.")

    # Add the new data to the old data.
    existing_data.append(interval_data)

    # Write the complete list back to the master file.
    with open(cummulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)

    # Also save the data into the structured database table.
    save_to_lp_database(license_plates, startTime, endTime)
    st.success(f"Saved data for {len(license_plates)} detected entries to JSON/DB.")


def save_to_lp_database(license_plates, start_time, end_time):
    """Saves license plate data to the SQLite database (LicensePlates table)."""
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    # Saves each unique license plate found into the database.
    for plate in license_plates:
        cursor.execute('''
            INSERT INTO LicensePlates(start_time, end_time, license_plate)
            VALUES (?, ?, ?)
        ''', (start_time.isoformat(), end_time.isoformat(), plate))
            
    conn.commit()
    conn.close()

def process_lp_frame(frame, license_plates_set, model):
    """Runs YOLO detection and Tesseract OCR on a single image frame."""
    if model is None:
        return frame
        
    # Run the AI model (YOLO) to predict where the license plates are in the image.
    results = model.predict(frame, conf=0.45, verbose=False)
    
    for result in results:
        # Get the coordinates (x1, y1, x2, y2) of the detected box.
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Get the confidence score (how sure the AI is about the detection).
            conf = math.ceil(box.conf[0].item() * 100) / 100
            classNameInt = int(box.cls[0].item())
            
            # Execute the text-reading function (OCR) on the cropped plate image.
            label = tesseract_ocr_process(frame.copy(), x1, y1, x2, y2)
            
            # Add the unique plate number to the set of collected plates.
            license_plates_set.add(label)
                
            display_label = label if label else f'{clsName}:{conf:.2f}'

            # Draw a blue box around the detected license plate on the image.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw a box for the text label above the plate.
            textSize = cv2.getTextSize(display_label, 0, fontScale=0.5, thickness=2)[0]
            c2 = x1 + textSize[0] + 5, y1 - textSize[1] - 8
            cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
            
            # Write the detected license plate number onto the image.
            cv2.putText(frame, display_label, (x1, y1 - 4), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    return frame

def lp_video_processing_loop(cap, model):
    """Processes video from a capture object (file or camera) for License Plate Detection."""
    st.subheader("Processing Video Feed... 🚗")
    
    # Placeholders are used to update the image and text on the web app in real-time.
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
            # Stop if the video ends or the camera closes.
            break

        frame_count += 1
        
        h, w, _ = frame.shape
        # Resize the frame if it's too big, to speed up processing.
        if w > 800:
            frame = cv2.resize(frame, (800, int(800 * h / w)))
        
        # Process the frame (detect plates, read text, draw boxes).
        processed_frame = process_lp_frame(frame, license_plates, model)
        
        # Display the processed image on the web app.
        frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"Frame {frame_count}/{int(max_frames) if is_file else 'live'}")

        # Time-based saving logic: saves all unique plates found every 20 seconds.
        currentTime = datetime.now()
        if (currentTime - startTime).seconds >= 20:
            endTime = currentTime
            save_lp_json(license_plates, startTime, endTime)
            startTime = currentTime
            license_plates.clear() # Clear the list after saving and start collecting new plates.

        # Update the status information displayed to the user.
        status_placeholder.text(f"Frames processed: {frame_count} | Unique Entries: {len(license_plates)} (since last save)")
        plate_placeholder.json({"Detected Entries (since last save)": list(license_plates)})
        
        # Stop webcam mode after 600 frames.
        if not is_file and frame_count >= 600:
             break 

        cv2.waitKey(1) # Necessary small delay for video processing.

    # Save any remaining plates if the video stopped mid-interval.
    if license_plates:
        save_lp_json(license_plates, startTime, datetime.now())
        
    cap.release()
    frame_placeholder.empty()
    st.success("Video processing finished.")

# --- 6. ATCC (VEHICLE ANALYZER) MODE FUNCTIONS ---

def calculate_traffic_level(total_count):
    """Classifies traffic density (No, Low, Medium, High) based on total vehicle count."""
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
    This counts how many vehicles of each type (car, truck, bus, etc.) were found.
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
                # Find the name (e.g., 'car') for each detected class ID.
                class_names = [results[0].names[int(cls_id)] for cls_id in detection_classes]
            except (AttributeError, KeyError):
                class_names = [f"Class {int(cls_id)}" for cls_id in detection_classes]

            # Count the total number of vehicles and the count for each type.
            for class_name in class_names:
                total_vehicles += 1
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Determine the overall traffic density.
    traffic_level = calculate_traffic_level(total_vehicles)

    # Save the results (counts and traffic level) into the traffic database.
    for vehicle_class, count in class_counts.items():
        db.save_result(timestamp, source_type, vehicle_class, count, traffic_level)
    
    if not class_counts:
          db.save_result(timestamp, source_type, "N/A", 0, "No Traffic")

    # Prepare a simple summary to show the user immediately.
    summary = {
        'timestamp': timestamp,
        'total_vehicles': total_vehicles,
        'traffic_level': traffic_level,
        'class_counts': class_counts
    }
    return summary

def annotate_atcc_image(result):
    """Annotates a single YOLO result image (draws the boxes and labels) and prepares it for display."""
    # This uses the YOLO model's built-in function to draw the bounding boxes and labels (e.g., 'car: 0.95') on the image.
    annotated_img = result.plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Use Matplotlib to save the image as a byte buffer (an in-memory file) which Streamlit can display.
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
        # Fetches all data and displays it in a table format.
        raw_df = db.fetch_all_data()
        st.dataframe(raw_df, use_container_width=True)
        st.markdown(f"Total Records: **{len(raw_df)}**")

# --- CUSTOM CSS STYLES (FROM ATCC APP) ---

def apply_custom_styles():
    """Applies custom look and feel (CSS styles) to the Streamlit app for a better user interface."""
    st.markdown("""
    <style>
    /* This section contains the rules for how the web app looks (fonts, colors, button styles, etc.) */
    </style>
    """, unsafe_allow_html=True)


# --- LICENSE PLATE MODE LAYOUT ---

def license_plate_mode(model):
    """Sets up the layout and functionality for the License Plate Detector (LP) application mode."""
    st.title("License Plate Detector & Tesseract OCR 🏷️")
    st.sidebar.title("LP Detector Options")

    # Display Tesseract status (tells the user if the text-reading tool is working).
    if TESSERACT_AVAILABLE:
        st.sidebar.success("Tesseract OCR Active.")
    else:
        st.sidebar.error("Tesseract Error: Using placeholders for OCR results.")
        st.warning("Tesseract is unavailable. Please ensure it's installed and added to your system PATH.")
    
    if model is None:
        st.error("License Plate YOLO model did not load. Detection is disabled.")
        return

    # Let the user choose where the video/image comes from (file or camera).
    source_option = st.sidebar.radio(
        "Select Input Source:",
        ('Upload Video', 'Upload Photo', 'Use Webcam (Experimental)')
    )
    
    st.markdown("---")

    # Handles the process if the user selects to upload a video file.
    if source_option == 'Upload Video':
        st.header("Video File Upload")
        uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            st.video(uploaded_file)
            
            # Save the uploaded video temporarily to a local file so OpenCV can read it.
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1] or ".mp4") as tfile:
                tfile.write(uploaded_file.read())
                temp_video_path = tfile.name
                
            if st.button("Start Processing Video 🎬"):
                with st.spinner('Initializing video stream...'):
                    cap = cv2.VideoCapture(temp_video_path)
                    # Start the main loop to process the video frame by frame.
                    lp_video_processing_loop(cap, model)
                
            # Clean up the temporary video file after processing.
            try:
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
            except Exception as e:
                st.warning(f"Could not clean up temporary file: {e}")

    # Handles the process if the user selects to upload a single photo.
    elif source_option == 'Upload Photo':
        st.header("Image File Upload")
        uploaded_image = st.file_uploader("Choose a photo...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image is not None:
            # Convert the uploaded image into a format OpenCV can use.
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
                        
                        # Process the image once.
                        processed_frame = process_lp_frame(frame, license_plates, model)
                        
                        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption='Processed Image', use_container_width=True)
                        
                    if license_plates:
                        st.success("Analysis Complete! Detected entries saved to JSON/DB.")
                        st.json(list(license_plates))
                        
                        current_time = datetime.now()
                        # Save the results immediately since it's a single image.
                        save_lp_json(license_plates, current_time, current_time)
                    else:
                        st.info("No license plate objects were detected by YOLO.")


    # Handles the process if the user selects to use the computer's webcam.
    elif source_option == 'Use Webcam (Experimental)':
        st.header("Webcam Input (Experimental)")
        st.warning("Webcam capture can be inconsistent in Streamlit. This mode will attempt to run for ~600 frames.")
        
        if st.button("Start Camera 📸"):
            with st.spinner('Attempting to open camera...'):
                # Open the default camera (ID 0).
                cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open camera. Check permissions or if another application is using it.")
            else:
                # Start the video processing loop for the live camera feed.
                lp_video_processing_loop(cap, model)

# --- ATCC MODE LAYOUT ---

def atcc_mode(model, db: TrafficDB):
    
    apply_custom_styles()
    # Display the main header for the ATCC mode.
    st.markdown('<div class="main-header">ATCC YOLOv11 Vehicle Analyzer</div>', unsafe_allow_html=True)
    st.markdown("---")

    if model is None:
        st.error("ATCC YOLO model did not load. Detection is disabled.")
        return

    # --- Sidebar for Settings ---
    with st.sidebar:
        st.header("ATCC Analyzer Options")
        # Let the user choose between uploading a file or using the webcam.
        analysis_mode = st.radio("Choose Input Source",
            ('Upload Image/Video', 'Webcam Capture'),
            index=0,
            key='atcc_analysis_mode')
        
        # Sliders allow the user to fine-tune the AI's detection settings.
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
        
        st.markdown("---")
        st.subheader("Model Information")
        st.markdown(f"Model: `{ATCC_MODEL_PATH}`")
        # Show what types of objects the AI is trained to detect (cars, buses, etc.).
        class_names = model.names.values() if hasattr(model, 'names') else ["(Classes not loaded)"]
        st.markdown(f"Classes Detected ({len(class_names)}): {', '.join(class_names)}")
        st.markdown("---")
        st.markdown("### Data Storage")
        
        # Buttons to view the raw data or clear the database logbook.
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
        
        # File upload area.
        if analysis_mode == 'Upload Image/Video':
            uploaded_file = st.file_uploader(
                "Upload an Image (jpg, png) or Video (mp4, mov, avi)",
                type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'],
                key='atcc_uploader'
            )
            process_button = st.button("Start Analysis (Upload)", key='atcc_process_upload')

        # Webcam capture area.
        elif analysis_mode == 'Webcam Capture':
            st.warning("Webcam analysis is resource-intensive.")
            # Streamlit's built-in camera function.
            webcam_image = st.camera_input("Capture an image or video segment.", key='atcc_webcam')
            
            uploaded_file = webcam_image
            process_button = st.button("Start Analysis (Webcam)", disabled=not webcam_image, key='atcc_process_webcam')
        
        # Display the uploaded file in the main area.
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
                # Save the uploaded file to a temporary location for the AI model to access.
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
                
                # --- Run Detection (Image/Snapshot) ---
                if media_type == 'image' or (analysis_mode == 'Webcam Capture' and uploaded_file):
                    st.info("Processing image...")
                    # Run the AI model on the image.
                    results = model.predict(temp_path, **args)
                    # Draw the boxes on the image.
                    annotated_media_buffer = annotate_atcc_image(results[0])
                    # Count the results and save them to the database.
                    results_summary = process_atcc_detection(results, db, source_type="Image" if media_type == 'image' else "Webcam Snapshot")
                    
                    st.markdown('<p class="caption">Annotated Image</p>', unsafe_allow_html=True)
                    # Display the final image with all the boxes drawn.
                    st.image(annotated_media_buffer, use_container_width=True)
                    progress_bar.progress(100)
                    
                # --- Run Detection (Video) ---
                elif media_type == 'video' and analysis_mode == 'Upload Image/Video':
                    st.info("Processing video feed...")
                    
                    # Set up to read the video file frame by frame.
                    cap = cv2.VideoCapture(temp_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Set up to save a new video file with the detection boxes drawn on it.
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
                        
                        # Run the AI model on the current frame.
                        results = model.predict(frame, **args)
                        # Draw the boxes and save the frame to the new video file.
                        annotated_frame = results[0].plot()
                        all_results.append(results[0])
                        out.write(annotated_frame)
                        
                        # Update the preview image on the web app occasionally.
                        if frame_idx % 5 == 0:
                            video_placeholder.image(annotated_frame, channels="BGR", caption=f"Processing Frame {frame_idx + 1}", use_container_width=True)

                        frame_idx += 1
                        # Update the progress bar for the user.
                        progress_bar.progress(min(int(frame_idx / total_frames * 100), 100))

                    cap.release()
                    out.release()
                    
                    # Process the overall results of the entire video and save to DB.
                    results_summary = process_atcc_detection(all_results, db, source_type="Video")
                    annotated_media = out_path
                    
                    video_placeholder.empty()
                    st.markdown('<p class="caption">Annotated Video Output</p>', unsafe_allow_html=True)
                    # Display the final processed video.
                    st.video(annotated_media, format="video/mp4", start_time=0)
                    
            except Exception as e:
                st.error(f"An error occurred during detection: {e}")
                results_summary = None
            finally:
                # Clean up the temporary files after the analysis is done.
                if temp_path and os.path.exists(temp_path): os.remove(temp_path)
                if annotated_media and os.path.exists(annotated_media) and media_type == 'video': os.remove(annotated_media)
                progress_bar.empty()

    # --- Display Results and Visualization ---
    st.markdown("---")
    
    if results_summary:
        st.subheader("Analysis Summary")
        
        traffic_level_class = results_summary['traffic_level'].lower().replace(' ', '-')
        
        # Display the key findings (total vehicles, traffic level) in a nice box.
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
        
        # Show a table of how many of each vehicle type were counted (Car, Truck, Bus).
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
    
    # Retrieve and display all past analysis data from the database.
    db_df = db.fetch_all_data()
    
    if not db_df.empty:
        st.subheader("Raw Analysis Log")
        st.dataframe(db_df, use_container_width=True)
        st.markdown(f"Total Records: **{len(db_df)}**")
        
        # Summarize how many times each traffic level (Low, Medium, High) has been recorded.
        traffic_summary = db_df['traffic_level'].value_counts().reset_index()
        traffic_summary.columns = ['Traffic Level', 'Count']
        st.subheader("Traffic Level Totals")
        st.dataframe(traffic_summary)

    else:
        st.info("No historical analysis data is available yet. Run an analysis to populate the database.")


# --- 7. MAIN APPLICATION ENTRY POINT ---

def main():
    """This is the starting point of the whole application, controlling the main layout and mode switching."""
    # Set up the main web page configuration (title, layout).
    st.set_page_config(page_title="Combined YOLO App", layout="wide", initial_sidebar_state="expanded")
    
    st.sidebar.title("App Selection")
    
    # Main selector for the two application modes (LP or ATCC).
    app_mode = st.sidebar.radio(
        "Select Application Mode:",
        ('License Plate Detector (LP) / OCR', 'Vehicle Traffic Analyzer (ATCC)'),
        key='app_mode_select'
    )
    
    st.sidebar.markdown("---")

    # Initialize DB (Traffic) in session state
    if 'atcc_db' not in st.session_state:
        # Create an instance of the traffic logbook (TrafficDB) and keep it running for the session.
        st.session_state['atcc_db'] = TrafficDB()
    atcc_db = st.session_state['atcc_db']

    if app_mode == 'License Plate Detector (LP) / OCR':
        # Load the LP AI model and run the LP mode function.
        lp_model = initialize_yolo_model(LP_CUSTOM_WEIGHTS_PATH)
        license_plate_mode(lp_model)
    elif app_mode == 'Vehicle Traffic Analyzer (ATCC)':
        # Load the ATCC AI model and run the ATCC mode function.
        atcc_model = initialize_yolo_model(ATCC_MODEL_PATH)
        atcc_mode(atcc_model, atcc_db)

# Start the application when the Python file is run.
if __name__ == '__main__':
    main()