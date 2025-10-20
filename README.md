
# 🚗 Automatic Number Plate Recognition (ANPR) & Traffic Classification (ATCC)

### 🔍 Powered by YOLOv10, Tesseract OCR & SQLite3

This project is a **Computer Vision-based Automatic Number Plate Recognition (ANPR)** and **Automatic Traffic Count & Classification (ATCC)** system.  
It detects vehicles, extracts license plate text using **Tesseract OCR**, and stores results automatically into an **SQL database** for analytics, tracking, and traffic insights.

---

## 🧠 Table of Contents

- [📘 Overview](#-overview)
- [🧠 Tech Stack](#-tech-stack)
- [⚙️ Installation Guide (Windows)](#️-installation-guide-windows)
- [🚀 Run the Project](#-run-the-project)
- [🗃️ View SQL Database](#️-view-sql-database)
- [🧩 Common Fixes](#-common-fixes)
- [📊 Output Example](#-output-example)
- [📁 Folder Structure](#-folder-structure)
- [🚦 ATCC (Traffic Count & Classification)](#-atcc-traffic-count--classification)
- [🧱 Future Enhancements](#-future-enhancements)
- [👨‍💻 Author](#-author)

---

## 📘 Overview

The system performs **real-time vehicle detection** and **license plate recognition** from live traffic cameras or pre-recorded video feeds.  
It is capable of **detecting multiple vehicles**, classifying them (Car, Bike, Truck, etc.), and performing **OCR-based license plate extraction** using **Tesseract OCR**.  

All recognized plate data, timestamps, and vehicle types are **automatically stored** in an **SQLite3 database** for further inspection or visualization.

### ✨ Key Features:
- 🚘 **Vehicle Detection & Classification** using **YOLOv10**
- 🔠 **License Plate Text Recognition** via **Tesseract OCR**
- 💾 **Data Storage** in **SQLite3 Database**
- 📊 **View, Query, and Export Results**
- 🎥 **Supports Real-Time Video Feed and Pre-Recorded Files**
- ⚙️ **Fully Offline, Cross-Platform, and Lightweight**

---

## 🧠 Tech Stack

| Category | Technology |
|-----------|-------------|
| **Language** | Python 3.11 |
| **Object Detection** | YOLOv10 (Ultralytics) |
| **Text Recognition (OCR)** | Tesseract OCR |
| **Database** | SQLite3 |
| **Libraries** | OpenCV, NumPy, Pandas, Matplotlib |
| **Environment** | Conda / Virtualenv |
| **Platform** | 🪟 Windows (Primary), Linux/macOS (Optional) |

---

## ⚙️ Installation Guide (Windows)

### 1️⃣ Clone YOLOv10 Repository
```bash
git clone https://github.com/THU-MIG/yolov10.git
````

### 2️⃣ Create & Activate Virtual Environment

Using Conda:

```bash
conda create -n cvproj python=3.11 -y
conda activate cvproj
```

Or using venv (PowerShell):

```powershell
python -m venv infosys
.\infosys\Scripts\Activate.ps1
```

---

### 3️⃣ Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Install YOLOv10 in Editable Mode

```bash
cd yolov10
pip install -e .
cd ..
```

---

### 5️⃣ Install Tesseract OCR (Windows)

#### 🔹 Step 1: Download & Install

Go to [Tesseract OCR (UB Mannheim)](https://github.com/UB-Mannheim/tesseract/wiki)
Download and install the Windows build.

During installation:

* ✅ Select **"Add to PATH"**
* ✅ Include **English language pack**
* ✅ Note installation path (usually `C:\Program Files\Tesseract-OCR\`)

#### 🔹 Step 2: Verify Installation

Run in Command Prompt:

```cmd
tesseract --version
```

If not recognized, manually set the path in Python:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## 🚀 Run the Project

### Start the SQLite Database

```bash
python sqldb.py
```

### Run the Main Detection Script

```bash
python main.py
```

This script:

* Opens your camera or video feed
* Detects vehicles
* Extracts number plate text
* Stores all results in an SQLite database

---

## 🗃️ View SQL Database

To visualize and query your records easily, use this online viewer:

🔗 [SQLite Viewer](https://inloop.github.io/sqlite-viewer/)

Simply upload your generated `.db` file (e.g., `anpr_data.db`) to inspect records.

---

## 🧩 Common Fixes

### 🔸 Numpy Version Conflict

If you see:

```
ValueError: numpy.dtype size changed
```

Run:

```bash
pip uninstall numpy
pip install numpy==1.26.4
```

---

### 🔸 Tesseract Not Found

If Tesseract isn’t recognized:

1. Add its path to Windows environment variables manually
   (e.g. `C:\Program Files\Tesseract-OCR\`)
2. Or define path in your script:

   ```python
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

---

## 📊 Output Example

| Vehicle ID | License Plate | Vehicle Type | Timestamp           |
| ---------- | ------------- | ------------ | ------------------- |
| 001        | AP29CE1234    | Car          | 2025-10-20 15:30:10 |
| 002        | TS07BM9876    | Truck        | 2025-10-20 15:31:05 |
| 003        | KA09LM4567    | Bike         | 2025-10-20 15:32:02 |

---

## 📁 Folder Structure

```
📁 License-Plate-Extraction-Save-Data-to-SQL-Database
├── main.py                  # Main script (ANPR + ATCC logic)
├── sqldb.py                 # SQLite3 database handler
├── yolov10/                 # YOLOv10 source repo
├── requirements.txt         # Dependencies
├── .gitignore
└── README.md
```

---

## 🚦 ATCC (Traffic Count & Classification)

The **Automatic Traffic Count and Classification (ATCC)** module extends the ANPR pipeline with vehicle analytics.

### 🚘 Features:

* Detects and counts **Cars, Trucks, Bikes, Buses**, etc.
* Tracks each detected object across frames using **object tracking**
* Classifies based on **YOLOv10 class labels**
* Maintains per-frame vehicle counts and categories

### ⚙️ Core Logic:

1. **Frame Capture** → Read each frame from video or live feed
2. **YOLOv10 Detection** → Get bounding boxes + labels
3. **Object Tracking (optional)** → Use ID matching between frames
4. **Counting Logic** → Increment counts when a vehicle crosses a defined line or ROI
5. **Data Logging** → Store vehicle type, ID, and timestamp in the SQLite database

---

## 🧱 Future Enhancements

* 🧾 Export results to **Excel/CSV** automatically
* 📶 Integrate **live dashboard (Streamlit)** for analytics
* ⚡ Add **GPU acceleration (CUDA)** for faster inference
* 🎯 Implement **vehicle speed estimation** using frame-rate and bounding box tracking
* ☁️ Connect to **cloud-based SQL servers** for large-scale deployments

---

## 👨‍💻 Author

**Bhavani Sankar Komati**
💻 *Computer Vision & AI Developer*



💼 LinkedIn: [https://linkedin.com/in/yourusername](https://www.linkedin.com/in/bhavani-sankar-komati/)

---

## 🏁 License

This project is released under the **MIT License** — free to use, modify, and distribute with attribution.

---

## 🧾 Citation

If you use this project in your research or development work, please cite:

> Bhavani Sankar Komati, “License Plate Extraction and Vehicle Classification using YOLOv10 and Tesseract OCR”, 2025.

---

## ⭐ Acknowledgements

* [Ultralytics YOLOv10](https://github.com/THU-MIG/yolov10)
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
* [OpenCV](https://opencv.org/)
* [SQLite](https://www.sqlite.org/)
