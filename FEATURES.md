# Neo Scanner — Enabled Features & Verification Checklist 📄

This document provides a comprehensive inventory of all features currently enabled in **Neo Scanner**, along with an interactive verification checklist for testing and steps for integrating these modules into other apps.

---

## 📋 Feature Verification Checklist

Use this checklist to test each feature on your machine. Mark each item as verified `[x]` or report issues if any step does not behave as expected.

### 1. Tailscale Remote Scanner Session
- [ ] **Startup Dialog**: When launching the app, a dialog asks: *"Start Tailscale Remote Scanner session for this workstation?"* with "Yes", "No", and "Remember my choice".
- [ ] **Tailscale IP Detection**: App detects your machine's `100.x.y.z` Tailscale CGNAT IP address automatically (or falls back to localhost if Tailscale CLI is offline).
- [ ] **Zero-Trust Security**:
  - [ ] Incoming connections are strictly limited to the Tailscale subnet (`100.64.0.0/10`) and localhost (`127.0.0.1`). Any external network IP is rejected with `403 Forbidden`.
  - [ ] Each session generates a unique bearer token and a rotating 6-digit PIN.
- [ ] **Pairing QR Code Dialog**: Clicking the **"Pairing QR"** button in the sidebar opens a window with:
  - Scannable QR code for phones.
  - Direct connection URL (`http://<tailscale-ip>:8000?token=...`).
  - 6-digit PIN display.
- [ ] **Mobile Web App (`http://<tailscale-ip>:8000`)**:
  - [ ] Opens cleanly on mobile browser (Safari, Chrome).
  - [ ] Live camera preview stream.
  - [ ] Prompts for PIN if opened without token in URL.
  - [ ] Big tactile **"Capture"** button captures current camera frame into the session.
  - [ ] Subject dropdown with `+ Add` button.
  - [ ] Student dropdown with `+ Add` button.
  - [ ] Live date badge showing system date (e.g. `25 Aug`, `8 Sep`).
  - [ ] Capture thumbnail gallery showing captured pages.
  - [ ] **"Finish & Export Session"** button exports the session PDF and clears the mobile gallery.

---

### 2. Multi-User Profiles & Persistent Settings
- [ ] **User Profile Dropdown**: Sidebar header features a profile selector (e.g., `Default`, `Teacher A`, `Teacher B`).
- [ ] **Add User ("+ New...")**: Clicking "+ New..." prompts for a user name and creates an independent profile.
- [ ] **Per-User Settings Isolation**:
  - [ ] Switching users switches their saved preferences (last selected Subject, last selected Student, Camera Index, Resolution, OCR engine, Post-processing filter, Watched Folder path).
  - [ ] Settings persist across app restarts in `~/Library/Application Support/NeoScanner/profiles.json` (macOS) or `%APPDATA%/NeoScanner/profiles.json` (Windows).
- [ ] **Sequential Usage**: Teachers/users can use the same scanning station one after another without overwriting each other's rosters or default options.

---

### 3. Dynamic Hierarchical Folder Structure
- [ ] **Target Directory Layout**:
  ```
  {Watched Folder} / {Subject} / {System Date} / {Student Name} /
  ```
  Example:
  ```
  ~/OneDrive/CamScan/Maths/8 Sep/Ahmad/
  ```
- [ ] **System Date Auto-Extraction**: Automatically extracts `{day} {short_month}` (e.g. `25 Aug`, `8 Sep`) from the system clock without manual date typing.
- [ ] **Subject / Main Folder Management**:
  - [ ] Subject dropdown preloaded with defaults: `Maths`, `English`, `Science`.
  - [ ] `+` button opens a dialog to add a new subject (persists in the active profile).
  - [ ] `-` button removes the currently selected custom subject.
- [ ] **Student Roster Management**:
  - [ ] Student dropdown with persistent student names.
  - [ ] `+` button opens a dialog to add a new student.
  - [ ] `-` button removes the selected student.
- [ ] **Live Destination Preview**: Sidebar displays live preview:
  `📁 Save: Maths / 8 Sep / Ahmad /` updating immediately whenever Subject, Date, or Student changes.
- [ ] **Session Export Outputs**:
  - [ ] Searchable PDF saved to: `{Watched Folder}/{Subject}/{Date}/{Student}/{student}_{date}.pdf`
  - [ ] High-res page images saved to: `{Watched Folder}/{Subject}/{Date}/{Student}/images/page_001.png`, etc.

---

### 4. Scanner Controls & Boundary Detection
- [ ] **Camera Controls**:
  - [ ] Camera source selection (built-in webcam, external USB document camera).
  - [ ] Resolution selector (e.g. 1920x1080, 1280x720).
  - [ ] Flip Horizontal / Flip Vertical toggles.
- [ ] **Boundary Detection**:
  - [ ] `Clean Perspective Crop (Recommended)`: Automatic 4-point quadrilateral detection and perspective warp.
  - [ ] `Classic Contour (OpenCV)`: OpenCV Canny/Hough boundary detection.
- [ ] **Dewarping**:
  - [ ] Spine dewarping toggle to flatten curved notebook pages using cubic curve fitting.
- [ ] **Auto-Capture on Page Turn**:
  - [ ] Motion detection sensing page turns with settling cooldown timer.

---

### 5. OCR & Post-Processing Filters
- [ ] **Post-Processing Options**:
  - [ ] `Magic Color (CamScanner)`: Enhances contrast, removes paper shadows, whitens background.
  - [ ] `None (Raw Natural)`: Preserves camera image as-is.
  - [ ] `Sharpen`: Kernel sharpening for fine text.
  - [ ] `Grayscale`: 8-bit black/white gradient.
  - [ ] `Black and White`: High-contrast binarized document scan.
- [ ] **OCR Engines**:
  - [ ] `PaddleOCR Fast (Books & Documents)`: Local printed text recognition.
  - [ ] `PaddleOCR + TrOCR (Handwriting)`: Two-stage line layout + TrOCR handwriting model.
  - [ ] `Vision LLM API`: Google Gemini Vision fallback.
  - [ ] `None (Instant PDF Export)`: Fast image-only PDF packaging without waiting for OCR.

---

## 🚀 How to Run and Test

### Running the Latest Code Directly
Since your system's `/Applications/Neo Scanner.app` may be an older build, run the application using the project virtual environment:

```bash
cd /Users/abdulhannan/Documents/Phet/Neo_Scanner
source .venv/bin/activate
python camscan/main.py
```

### Running the Automated Test Suite
All 34 automated unit tests can be executed with:

```bash
.venv/bin/pytest -v
```

### Rebuilding the Standalone Mac Application (`.app` / `.dmg`)
To update `/Applications/Neo Scanner.app` with all the latest features:

```bash
./build_mac_dmg.sh
```

---

## 🧩 Reusable Architecture for Your Other Apps

The modules created for Neo Scanner are decoupled and designed to be imported or copied directly into your other desktop/mobile/web applications:

| Module | File | Purpose & How to Reuse |
|---|---|---|
| **Multi-User Profiles** | `camscan/profiles.py` | Standalone `ProfileManager` and `UserProfile` dataclasses. Store per-user JSON configs in OS-standard directories via `platformdirs`. Call `mgr.get_active_profile()` and `mgr.save_profiles()`. |
| **Hierarchical Storage** | `camscan/session.py` | Functions `get_system_date_str(now)` and `build_session_export_dir(base_dir, subject, date_str, student)`. Automatically creates `{Subject}/{Date}/{Student}/` hierarchy in any Python app. |
| **Zero-Trust Tailscale** | `camscan/tailscale.py` | Subnet check `is_tailscale_or_local(ip)`, CLI detector `get_tailscale_ip()`, session token generator `generate_session_token()`, and QR code generator `generate_qr_code_image()`. |
| **Remote Web Daemon** | `camscan/remote.py` | FastAPI server with security middleware, live MJPEG stream, and REST endpoints for remote control. |

---

## 📁 Repository Directory Reference

```
Neo_Scanner/
├── camscan/
│   ├── app.py           # Main Tkinter desktop GUI, sidebar, layout
│   ├── auto_export.py   # Hierarchical PDF & image export logic
│   ├── capture.py       # Camera feed capture & frame buffering
│   ├── dewarp.py        # Boundary detection & curve dewarping
│   ├── export.py        # Searchable PDF assembly with PyMuPDF
│   ├── main.py          # Application entrypoint
│   ├── ocr.py           # PaddleOCR, TrOCR, and Vision LLM
│   ├── profiles.py      # [NEW] Multi-user profile management
│   ├── remote.py        # [NEW] FastAPI remote control & mobile web UI
│   ├── session.py       # [NEW] Dynamic path builder & session state
│   ├── tailscale.py     # [NEW] Zero-trust Tailscale security & QR pairing
│   └── widgets.py       # Tkinter UI dialogs (QR Dialog, Startup Dialog)
├── tests/
│   ├── test_profiles.py # Tests for profile switching & persistence
│   ├── test_remote.py   # Tests for remote endpoints & auth
│   ├── test_session.py  # Tests for hierarchical folder generation
│   └── test_tailscale.py# Tests for subnet verification & tokens
├── FEATURES.md          # [NEW] This document
├── ISSUES.md            # Known issues & roadmap
└── README.md            # General project overview
```
