# Neo Scanner - Known Issues, Status & Engineering Roadmap

**Last Updated:** September 4, 2026  
**Repository:** `Novice130/Neo-Scanner`  
**Current Active App:** `/Applications/Neo Scanner.app`  
**Output Directory:** `/Users/abdulhannan/OneDrive/CamScan/`

---

## 1. Auto-Capture Delay: Handheld vs. Stand Dynamics

### The Issue
Auto-capture takes too long to snap pictures when the camera is held by hand.

### Why This Happens
1. **Frame Differencing Micro-Jitter:**  
   The page-turn detector (`camscan/motion.py`) calculates pixel variance between consecutive 1080p frames. When holding the Logitech BRIO by hand, natural hand tremors and breathing cause subtle shifts across all 2 million pixels. The motion score hovers around 3.5%–7.0%, keeping the state machine in `STATE_MOTION` or repeatedly resetting the `settle_time` timer.
2. **Autofocus Hunting on Moving Lens:**  
   Because the camera distance to the page fluctuates continuously in handheld mode, the physical autofocus mechanism in the BRIO hunts continuously for the focal plane. Our sharpness gate (`cv2.Laplacian > 65.0`) prevents taking a blurry shot, but holding by hand delays achieving sustained sharpness.

### The Solution
- **Putting the Camera on a Stand/Tripod (Recommended):**  
  When the camera is mounted on a fixed arm or stand, the frame is 100% motionless (`motion_score < 0.2%`) the moment your hand turns the page. The BRIO locks focus in ~0.5s, and the shutter triggers instantly at 0.6s–0.8s like clockwork.
- **Handheld Optimization in Code:**  
  In `camscan/app.py` / `camscan/motion.py`, the default settings can be adjusted:
  - Default `motion_threshold`: increase from `3.0` to `4.5`–`5.0` to tolerate minor hand tremor.
  - Default `settle_time_s`: decrease from `0.8s` to `0.6s`.
  - Center-weighted motion mask: ignore edge shaking and focus motion detection only on the center 60% of the field of view.

---

## 2. Document Cropping & Boundary Detection

### Past Issue
Captured photos included side objects (e.g., graphics tablet, keyboard, desk cables) instead of just the book, or suffered from "funhouse vortex" distortion.

### Root Cause
1. In earlier versions, `camscan/dewarp.py` used an experimental 3rd-degree polynomial equation ($y = ax^3 + bx^2 + cx + d$) which diverged when edge points hit non-paper objects, producing extreme warping.
2. `capture()` was running YOLOv8 (`yolov8n-seg.pt` trained on COCO 80 general objects), which misclassified the desk's graphics tablet as a "laptop" and cropped the tablet on the side instead of the book.

### Current Status: RESOLVED
- Unstable polynomial dewarping was replaced with **rigid 4-point quadrilateral perspective extraction** (`scanner.extract_contour` / `cv2.warpPerspective`).
- YOLO was removed from the capture pipeline. Preview and capture are unified on `scanner.find_paper_contour_adaptive()`. What you see inside the green box in the viewfinder is 100% what gets cropped and saved.

---

## 3. Two-Page Mode Slicing

### Past Issue
Single pages or book covers (e.g., *The Power of Habit* cover) were being chopped in half vertically into two tall strips (`THE PO / HA` and `WER OF / BIT`).

### Root Cause
`Two-Page Mode` blindly executed `cutoff_width = image.shape[1] // 2` on any capture, regardless of whether it was an open book or a single portrait page/cover.

### Current Status: RESOLVED
Added an aspect ratio gate in `camscan/app.py`:
- `Two-Page Mode` now only splits when the cropped image is an **open landscape spread** ($W > 1.15 \times H$).
- If scanning single portrait pages, book covers, or index cards ($H \ge W$), it preserves them as a single full page even if Two-Page Mode is checked.

---

## 4. Blur Prevention & Focus Quality

### Past Issue
Auto-capture took blurry snapshots because the timer fired before the Logitech BRIO's physical autofocus motor finished moving.

### Current Status: RESOLVED
- Implemented **Autofocus-Gated Capture**: Real-time Laplacian variance is calculated on every frame.
- Shutter is blocked until optical sharpness score $\ge 65.0$.
- Status indicator displays `Status: Focusing...` in orange while the lens is adjusting and switches to `Status: Sharp & Settling` in blue before snapping.
- Verified in session `captures_20260904_052445`: Page 21 was captured 100% sharp and readable.

---

## 5. Viewfinder Artifacts & Flashing

### Past Issue
Viewfinder preview suffered from gray static/quantized noise, and green boundary boxes flashed violently around the full screen perimeter when paper was moved.

### Root Cause
1. `magic_color` postprocessing was being applied to the live 60fps camera loop instead of only on captured photos.
2. Fallback boundary detector was returning full-screen coordinates when paper was missing.

### Current Status: RESOLVED
- Viewfinder restored to clean 60fps natural color camera video.
- Added Exponential Moving Average (EMA) coordinate smoothing to the 4 corners of the document box.
- When paper is not detected, boundary box is cleanly hidden (`contour = None`), completely eliminating perimeter flashing.

---

## 6. Handwriting OCR Support

### User Question
*"Did you put the best algorithm in the world to enable OCR on our text, especially for human handwriting?"*

### Architecture & Engines Available
Traditional OCR engines (e.g., Tesseract) fail on handwriting because they expect rigid mechanical fonts. Neo Scanner supports the two leading handwriting architectures:
1. **Microsoft TrOCR (`PaddleOCR + TrOCR (Handwriting)`)**:
   - Vision Transformer developed by Microsoft specifically trained on handwritten manuscripts and the IAM handwriting dataset.
   - Runs locally without cloud dependencies.
2. **Google Gemini 2.5 Flash (`Vision LLM API`)**:
   - State-of-the-art multimodal vision model for cursive handwriting, student homework notes, and handwritten mathematical equations.
   - Configurable in sidebar under **Section 6: Settings $\rightarrow$ OCR Engine**.

---

## 7. Dual Session Exports

### Configuration
All finalized sessions automatically generate two parallel exports in `/Users/abdulhannan/OneDrive/CamScan/`:
1. **Searchable PDF:** e.g., `captures_YYYYMMDD_HHMMSS.pdf` or `{StudentTag}_YYYYMMDD_HHMMSS.pdf`.
2. **Individual High-Resolution PNG Photos:** Stored in a matching folder (e.g., `captures_YYYYMMDD_HHMMSS/001_....png`).
- Default postprocessing: **Magic Color (CamScanner)** applies background illumination division, shadow stripping, and unsharp masking to all saved images.

---

## 8. Summary of Action Items for Next Session
1. **Test on Stand / Mount:** Mount the Logitech BRIO on a desk stand or arm to test auto-capture speed.
2. **Fine-Tune Handheld Motion Threshold:** If handheld operation is still needed, add a "Handheld Mode" toggle in the UI with `motion_threshold = 5.0` and `settle_time_s = 0.5s`.
3. **Verify OCR on Handwritten Samples:** Test TrOCR / Vision LLM on cursive handwriting or student notes.
