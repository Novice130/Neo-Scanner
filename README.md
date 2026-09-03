# Neo Scanner 📄

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Build Windows Installer](https://github.com/Novice130/Neo-Scanner/actions/workflows/build-windows.yml/badge.svg)](https://github.com/Novice130/Neo-Scanner/actions/workflows/build-windows.yml)

**Neo Scanner** is an AI-powered document scanner application that turns any computer with a connected camera or USB document camera into an intelligent scanning station. It features deep-learning boundary segmentation, handwriting OCR, curved notebook page dewarping, automatic page-turn capture, OneDrive watch-folder sync, and cross-continent remote control from your phone browser over Tailscale.

---

## Key Features

1. **Handwriting OCR & Searchable PDF Export**:
   - Two-stage local pipeline: **PaddleOCR** line layout detection + **Hugging Face TrOCR** (`microsoft/trocr-small-handwritten`) for handwriting recognition.
   - Optional **Google Gemini Vision LLM** fallback.
   - Generates searchable PDFs using **PyMuPDF** with invisible, selectable text layers (`render_mode=3`).
   - Runs in a background thread with real-time progress dialog.

2. **Page Dewarping Upgrade (Notebook Spine Flattening)**:
   - Deep-learning document boundary detection via **YOLOv8** (`yolov8n-seg.pt`).
   - Classical geometric correction using **cubic polynomial curve fitting** and **bicubic remapping (`cv2.remap`)** to flatten curved notebook pages near the spine.
   - Original OpenCV Hough/Canny contour detection preserved as an instant toggle.

3. **Student Session Tagging**:
   - Dedicated "Student Name / ID" field for school and grading workflows.
   - Automatically tags all capture entries and groups exported files into `{student_tag}_{date}.pdf` and `{student_tag}_{date}/` folders instead of flat directories.

4. **Auto-Capture on Page Turn**:
   - Computes low-latency frame differences on grayscale downscaled frames.
   - Detects page-turn motion and triggers capture once motion drops below a configurable threshold and settles for a settle window (0.5–1.0s).
   - Includes live visual motion status indicator (`Still`, `Page Turning`, `Settling...`, `Captured`) and safety cooldown.

5. **Auto-Export to Watched Folder (OneDrive Sync)**:
   - Configurable watched folder (defaults to local OneDrive sync directory `~/OneDrive/CamScan`).
   - One-click **"Finish & Export Session"** action that auto-exports the session PDF and clears the capture list, ready for the next student.

6. **Remote Control over Tailscale**:
   - Built-in **FastAPI** daemon server on port `8000`.
   - Low-latency **MJPEG live video stream** (`/api/feed`) and snapshot fallback (`/api/snapshot`).
   - Mobile-first web app accessible from phone browsers across continents over Tailscale:
     - Live video stream with tactile big **CAPTURE** button.
     - Student session input and live thumbnail strip.
     - Mode toggles (two-page, YOLO dewarp, auto-capture).
     - Remote session finalize action.

---

## Installation & Download

### Windows (Recommended)
Download the latest **`Neo_Scanner-Setup.exe`** from the [Releases page](https://github.com/Novice130/Neo-Scanner/releases). Double-click the installer and follow the wizard to install Neo Scanner with desktop and Start Menu shortcuts.

### Running from Source

#### Prerequisites
- Python 3.11
- Tkinter (`python-tk`)
- Webcam or USB document camera

#### Setup
```bash
# Clone repository
git clone https://github.com/Novice130/Neo-Scanner.git
cd Neo-Scanner

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
pip install paddleocr torch transformers ultralytics

# Launch Neo Scanner Desktop GUI
python camscan/main.py
```

---

## Remote Control Usage (Phone over Tailscale)

1. Connect the host PC running Neo Scanner to your [Tailscale](https://tailscale.com/) network.
2. In the Neo Scanner left sidebar, make sure **Remote Control** is enabled (it runs on port `8000`).
3. On your phone (connected to the same Tailscale network), open your browser and navigate to:
   ```
   http://<host-tailscale-ip>:8000
   ```
4. Enter the student's name/ID, watch the live camera feed, tap **CAPTURE** for each page, and tap **Finish Session** when done. The scanned document will automatically save to the host's watched OneDrive folder!

---

## Building the Windows Executable Locally

To compile the standalone Windows executable and installer yourself:

```cmd
# 1. Install PyInstaller and Inno Setup
pip install pyinstaller pyinstaller-hooks-contrib
choco install innosetup

# 2. Build with PyInstaller
pyinstaller neo_scanner.spec

# 3. Compile Windows Installer with Inno Setup
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```
The resulting installer will be located in `dist/Neo_Scanner-Setup-v1.0.0.exe`.

---

## Credits & Acknowledgments

Neo Scanner builds upon and integrates the remarkable work of several open-source projects:

- **Original Base Architecture**: [CamScan](https://github.com/suhren/camscan) by **Adam Suhren Gustafsson** ([@suhren](https://github.com/suhren)), providing the foundational OpenCV webcam document scanning loop and CustomTkinter structure.
- **Document Boundary Segmentation**: [YOLOv8](https://github.com/ultralytics/ultralytics) by **Ultralytics**.
- **Handwriting Text Line Layout**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) by **Baidu PaddlePaddle**.
- **Handwritten Text Recognition**: [TrOCR](https://huggingface.co/microsoft/trocr-small-handwritten) by **Microsoft Research** & **Hugging Face**.
- **Graphical User Interface**: [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) by **Tom Schimansky**.
- **Searchable PDF Generation**: [PyMuPDF](https://github.com/pymupdf/PyMuPDF) by **Artifex Software**.
- **Remote Web Server**: [FastAPI](https://fastapi.tiangolo.com/) by **Sebastián Ramírez** ([@tiangolo](https://github.com/tiangolo)) and [Uvicorn](https://www.uvicorn.org/).

---

## License

This project is licensed under the [MIT License](LICENSE.md).
Original CamScan copyright &copy; 2023 Adam Suhren Gustafsson.
Neo Scanner extensions and contributions &copy; 2026 Neo Scanner Contributors.
