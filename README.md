

# ANPR-ATCC: Advanced Automatic Number Plate Recognition & Traffic Classification System

## Project Overview

ANPR-ATCC is a powerful, unified traffic monitoring platform combining:

  - **Automatic Number Plate Recognition (ANPR):** Real-time detection and OCR extraction of vehicle license plates from images, videos, and camera streams.
  - **Automatic Traffic Classification (ATCC):** Multi-class vehicle detection supporting traffic density classification from live or recorded media.

This solution leverages the best of modern computer vision and OCR technologies, providing a **scalable, flexible, and easy-to-use** system ready for deployment in smart city projects, traffic analytics, law enforcement, and academic research.

-----

## 🔑 Key Features

### **1. Robust License Plate Detection & OCR**
Accurate **YOLOv10** detection of diverse license plates combined with **Tesseract OCR** for high-fidelity text extraction.

---

### **2. Intelligent ANPR Whitelist/Blacklist System 🚨**
This feature enables real-time, prioritized security and access control based on the detected license plate against two managed lists:

#### ✅ **Whitelisting for VIP and Emergency Vehicles**
The system instantly recognizes plates belonging to:
- VIP personnel (e.g., government officials)
- Emergency services (ambulances, police, fire trucks)

Detection of a whitelisted plate triggers:
- Immediate, high-priority alert  
- Optional integration with access control systems (e.g., boom barriers)  
- Swift automated clearance through checkpoints or controlled zones  

#### 🛑 **Blacklisting for Stolen Vehicles**
A continuously updated list of stolen, flagged, or wanted vehicles.

When a blacklisted plate is detected:
- A critical violation alarm is triggered  
- Location and timestamp are logged  
- Optional real-time notification to law enforcement for interception  

---

### **3. Multi-class Vehicle Traffic Analyzer**
**YOLOv11n**-based vehicle detection that classifies vehicle types, counts them, and estimates traffic congestion in real time.

---

### **4. Flexible Input Sources**
Supports:
- Video files  
- Images  
- Real-time camera streams  
- *Experimental*: Webcam streaming  

---

### **5. Comprehensive Data Logging & Advanced Visualization 📊**
Outputs are persistently stored in:
- SQLite databases  
- Cumulative JSON files  

Enhanced Streamlit dashboards provide:
- ATCC traffic visualizations  
- Traffic flow analysis  
- Peak-hour heatmaps  
- Trend analysis  
- Historical detections overview  

---

### **6. Interactive and Customizable UI**
Streamlit-powered interface with:
- Confidence and IoU threshold sliders  
- Real-time detection previews  
- Built-in database management tools  

---

### **7. Error Handling & Resilience**
Includes:
- Graceful fallback if Tesseract OCR is unavailable  
- Temporary file cleanup  
- Reliable database synchronization  

---

### **8. Cross-platform Compatibility**
Tested on:
- Linux  
- Windows  
- macOS  

Comes with detailed environment setup instructions.

---

### **9. Modular Architecture**
Separate pipelines for:
- **ANPR**  
- **ATCC**

This ensures:
- Easy maintenance  
- Clean codebase  
- High extensibility  



-----

## Technology Stack with Rationale

| Technology | Description | Justification |
|---|---|---|
| **YOLOv10 & YOLOv11n (Ultralytics)** | Ultra-fast, state-of-the-art object detection architectures. | Proven accuracy and speed, customizable weights for license plates and vehicle types. |
| **Tesseract OCR** | Open-source text recognition engine supporting multiple languages. | Lightweight, widely supported, best integration with Python workflows. |
| **Streamlit** | Python framework for building interactive data apps. | Rapid prototyping with minimal code for highly interactive UIs. |
| **SQLite** | Lightweight, serverless SQL database. | Perfect for embedded analytics, portable, zero-config, easy integration with Pandas. |
| **OpenCV** | Image and video processing library. | Industry-standard computer vision operations with Python bindings. |
| **pandas & matplotlib** | Data manipulation and visualization libraries. | Powerful data analytics and comprehensive plotting capabilities. |
| **Python Standard Library** | Utilities for file I/O, regex, system processes, date/time handling. | Robust tooling for supporting application logic. |

-----

## Use Cases & Target Users

  - **Municipal Transportation Departments:** Traffic flow monitoring and violation detection.
  - **Parking Facility Operators:** Automated license plate-based entry/exit logging.
  - **Law Enforcement:** Quick violation checks and real-time surveillance analytics.
  - **Researchers & Academia:** Traffic pattern analysis & machine learning datasets collection.
  - **Smart City Solutions Providers:** Traffic management dashboards integrated with IoT devices.
  - **Educators & Students:** Open-source educational project for computer vision and data science.

-----

## Advanced Setup Instructions

### Prerequisites

  - Python 3.11+
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and added to system PATH
  - Conda environment recommended for dependency isolation

### Installation Steps

```bash
git clone https://github.com/KomatiBhavaniSankar/ANPR-ATCC-Infosys.git
cd ANPR-ATCC-Infosys


conda create -n cvproj python=3.11 -y
conda activate cvproj

pip install --upgrade pip
pip install -r requirements.txt

cd yolov10
pip install -e .
cd ..


# Ensure SQLite databases are created automatically, else run:
python sqldb.py
```

### Optional: GPU Acceleration

If CUDA-enabled GPU is available, install torch and ultralytics versions compatible with your CUDA version to accelerate YOLO models.

### Environment Variables

  - `TESSDATA_PREFIX` (optional): Path to Tesseract language data files if custom installed.

### Debugging Common Issues

  - Fix numpy version conflicts:
    ```bash
    pip uninstall numpy
    pip install numpy==1.26.4
    ```
  - Confirm tesseract CLI works via terminal by running:
    `tesseract --version`
  - Check camera permissions on your OS before using webcam inputs.

-----

## Detailed Usage

### License Plate Detector (LP)

  - Upload video/image or use webcam to detect vehicle plates in real-time.
  - OCR extracted plate text displayed, saved to SQLite DB and JSON snapshots every 20 seconds.
  - Visual overlays help easily identify bounding boxes and recognized characters.

### Automatic Traffic Classifier & Counter (ATCC)

  - Upload image/video or camera capture to detect various vehicle classes (cars, trucks, bikes).
  - View detailed analytics including vehicle counts, traffic levels (No/Low/Medium/High), and historic data logs.
  - Adjust confidence and IOU thresholds dynamically to tune detection sensitivity.

### Database Insights & Maintenance

  - View raw analysis tables with full timestamped historic records in-app.
  - Clear or reset databases with UI buttons.
  - Export DB files for offline analysis or import into third-party tools.

-----

## Contributing Guidelines

  - Fork and clone the repo.
  - Create a new feature or bugfix branch.
  - Follow Python style conventions (PEP8) and write descriptive commit messages.
  - Test your changes thoroughly with provided notebooks and media.
  - Document any new APIs, workflows, or configurations in README or docstrings.
  - Submit a pull request referencing related issues or features.

-----

## Testing & Validation

  - Example media files and Jupyter notebooks are provided under `notebook/` for functional validation.
  - Unit tests for database operations and core detection pipelines to be added.
  - Continuous integration support planned (GitHub Actions) for automatic test runs on pull requests.

-----

## Security & Privacy Considerations

  - All data stored locally—no external servers involved, keeping sensitive information secure by design.
  - For production deployments, consider encrypting SQLite DB files and securing access paths.
  - Webcam and file uploads handled within browser sandbox, with no persistent external uploads.
  - Future versions may integrate authentication layers to enable controlled multi-user access.

-----

## Future Work & Roadmap

  - **Multi-language OCR support:** Increase scope beyond English license plates with PaddleOCR and custom models.
  - **Integration with Vehicle Registration APIs:** Cross-check extracted plates with official databases for real-time alerts.
  - **Edge Device Deployment:** Lightweight models optimized for Jetson Nano, Raspberry Pi, or embedded cameras.
  - **Cloud Sync & Visualization:** Remote data dashboard on cloud platforms (AWS/GCP/Azure).
  - **Enhanced UI/UX:** Dark mode, user profiles, notifications, and mobile responsiveness.
  - **Expanded Vehicle Classification:** Include electric scooters, buses, emergency vehicles, and non-motorized entities.
  - **AI Model Improvements:** Experiment with Transformer-based detectors for superior accuracy.

-----

## References & Resources

  - [YOLOv10 GitHub](https://github.com/THU-MIG/yolov10)
  - [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - [Streamlit Official Site](https://streamlit.io/)
  - [SQLite Browser](https://sqlitebrowser.org/)
  - [Python OpenCV](https://opencv.org/)

-----

### License

This project is licensed under the [MIT License](https://www.google.com/search?q=./LICENSE) © 2025 Vidzai Digital.

-----

### 🌐 Project Vision

*ANPR-ATCC aims to revolutionize intelligent traffic monitoring by merging AI-powered vision, automation, and real-time analytics into one unified platform — contributing toward smarter, safer, and more efficient cities.*

-----
