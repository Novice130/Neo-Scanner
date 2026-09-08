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
   - Clean Perspective Crop (quadrilateral contour detection) and OpenCV contour detection.

3. **Hierarchical Dynamic Subfolder Exports**:
   - System Date auto-extracted from host clock (`{day} {short_month}`, e.g., `25 Aug`, `8 Sep`).
   - Dynamic Subject / Main Folder selector with add/remove (`Maths`, `English`, `Science`, etc.).
   - Student Roster selector with add/remove (`Ahmad`, `Ubaid`, etc.).
   - Structured target folder hierarchy: `{Watched Folder} / {Subject} / {System Date} / {Student Name} /`.
   - Automatic export of searchable PDF and high-res page images into the student's subfolder.

4. **Multi-User Profiles**:
   - Support for multiple teachers / users on the same computer with isolated persistent settings.
   - Switch profiles sequentially via header dropdown; instantly loads each user's subjects, students, camera choice, and OCR defaults.

5. **Zero-Trust Tailscale Remote Control & Phone Pairing**:
   - Startup prompt asking to launch remote session with "remember my choice".
   - Zero-trust subnet verification strictly limited to Tailscale (`100.64.0.0/10`) and localhost.
   - Rotating 6-digit PIN and bearer token pairing.
   - Pairing QR code dialog for instant phone connection.
   - Mobile web interface with live preview, tactile capture button, Subject & Student selectors, date badge, and remote session finalize.

6. **Auto-Capture on Page Turn**:
   - Computes low-latency frame differences on grayscale downscaled frames.
   - Detects page-turn motion and triggers capture once motion drops below threshold and settles.
   - Live visual motion indicator (`Still`, `Page Turning`, `Settling...`, `Captured`).

> 📖 **Detailed Feature Inventory & Verification Checklist**: See [**`FEATURES.md`**](FEATURES.md) for the full feature checklist, testing instructions, and cross-app integration guide.

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
