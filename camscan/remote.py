"""
Remote Control Server for CamScan.
Provides a FastAPI web application allowing full camera monitoring, capture control,
student session tagging, and session finalization over Tailscale or local network.
"""

import asyncio
import io
import json
import logging
import os
import socket
import threading
import time
import typing as t

import cv2
import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    FastAPI = None

logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    """Attempt to find local or Tailscale IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class RemoteBridge:
    """Abstract interface between remote API server and scanner/GUI."""

    def get_frame(self) -> t.Optional[np.ndarray]:
        raise NotImplementedError

    def capture(self) -> dict:
        raise NotImplementedError

    def get_status(self) -> dict:
        raise NotImplementedError

    def set_student_tag(self, tag: str) -> str:
        raise NotImplementedError

    def set_settings(self, settings: dict) -> dict:
        raise NotImplementedError

    def finalize_session(self) -> dict:
        raise NotImplementedError

    def delete_capture(self, index: int) -> bool:
        raise NotImplementedError

    def get_thumbnail(self, index: int) -> t.Optional[np.ndarray]:
        raise NotImplementedError


class AppBridge(RemoteBridge):
    """
    Bridge connecting FastAPI endpoints thread-safely to a running CamScanApp instance.
    """

    def __init__(self, app):
        self.app = app

    def get_frame(self) -> t.Optional[np.ndarray]:
        try:
            if hasattr(self.app, "_latest_preview_frame") and self.app._latest_preview_frame is not None:
                return self.app._latest_preview_frame
            if hasattr(self.app, "camera"):
                return self.app.camera.capture()
        except Exception:
            pass
        return None

    def capture(self) -> dict:
        evt = threading.Event()
        result = {}

        def _do_capture():
            try:
                self.app.capture_image()
                count = len(self.app.entries)
                last_name = self.app.entries[-1].name if self.app.entries else ""
                result["success"] = True
                result["count"] = count
                result["name"] = last_name
            except Exception as e:
                result["success"] = False
                result["error"] = str(e)
            finally:
                evt.set()

        self.app.after(0, _do_capture)
        evt.wait(timeout=10.0)
        return result

    def get_status(self) -> dict:
        tag = self.app.var_student_tag.get()
        count = len(self.app.entries)
        captures = [
            {"name": e.name, "index": i}
            for i, e in enumerate(self.app.entries, start=1)
        ]
        return {
            "student_tag": tag,
            "capture_count": count,
            "captures": captures,
            "two_page_mode": bool(self.app.var_two_page_mode.get()),
            "boundary_detector": self.app.var_boundary_detector.get(),
            "auto_capture": bool(self.app.var_auto_capture.get()),
            "free_capture_mode": bool(self.app.var_free_capture_mode.get()),
            "watched_folder": self.app.var_watched_folder.get(),
        }

    def set_student_tag(self, tag: str) -> str:
        self.app.after(0, lambda: self.app.var_student_tag.set(tag))
        return tag

    def set_settings(self, settings: dict) -> dict:
        def _apply():
            if "two_page_mode" in settings:
                self.app.var_two_page_mode.set(1 if settings["two_page_mode"] else 0)
            if "boundary_detector" in settings:
                self.app.var_boundary_detector.set(settings["boundary_detector"])
            if "auto_capture" in settings:
                self.app.var_auto_capture.set(1 if settings["auto_capture"] else 0)
            if "free_capture_mode" in settings:
                self.app.var_free_capture_mode.set(
                    1 if settings["free_capture_mode"] else 0
                )

        self.app.after(0, _apply)
        return settings

    def finalize_session(self) -> dict:
        evt = threading.Event()
        res = {"success": True}

        def _do_finalize():
            try:
                self.app.finalize_session()
            except Exception as e:
                res["success"] = False
                res["error"] = str(e)
            finally:
                evt.set()

        self.app.after(0, _do_finalize)
        evt.wait(timeout=3.0)
        return res

    def delete_capture(self, index: int) -> bool:
        evt = threading.Event()
        success = [False]

        def _do_del():
            if 1 <= index <= len(self.app.entries):
                entry = self.app.entries[index - 1]
                entry.frame.destroy()
                self.app.entries.pop(index - 1)
                self.app.renumber_entries()
                success[0] = True
            evt.set()

        self.app.after(0, _do_del)
        evt.wait(timeout=3.0)
        return success[0]

    def get_thumbnail(self, index: int) -> t.Optional[np.ndarray]:
        if 1 <= index <= len(self.app.entries):
            img = self.app.entries[index - 1].current_image
            h, w = img.shape[:2]
            scale = min(180 / max(1, h), 140 / max(1, w))
            return cv2.resize(
                img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return None


# Embedded Mobile-First Responsive Web UI
MOBILE_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>CamScan Remote Control</title>
  <style>
    :root {
      --bg: #121212;
      --card-bg: #1e1e1e;
      --accent: #2196f3;
      --accent-hover: #1976d2;
      --success: #4caf50;
      --warning: #ff9800;
      --danger: #f44336;
      --text: #f5f5f5;
      --text-dim: #a0a0a0;
      --border: #333;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding-bottom: 90px; }
    header {
      background: #181818; border-bottom: 1px solid var(--border);
      padding: 12px 16px; display: flex; justify-content: space-between; align-items: center;
      position: sticky; top: 0; z-index: 100;
    }
    .brand { font-size: 1.1rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
    .badge {
      font-size: 0.75rem; background: #2a2a2a; color: var(--success);
      padding: 3px 8px; border-radius: 12px; font-weight: 600; display: inline-flex; align-items: center; gap: 4px;
    }
    .badge-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--success); }
    .container { max-width: 600px; margin: 0 auto; padding: 12px; }
    
    /* Viewport */
    .viewport-card {
      background: #000; border-radius: 12px; overflow: hidden;
      box-shadow: 0 4px 16px rgba(0,0,0,0.5); position: relative; margin-bottom: 14px;
      aspect-ratio: 4/3; display: flex; justify-content: center; align-items: center;
    }
    .viewport-card img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .stream-overlay {
      position: absolute; top: 10px; right: 10px; display: flex; gap: 6px;
    }
    .pill-btn {
      background: rgba(0,0,0,0.65); color: #fff; border: 1px solid rgba(255,255,255,0.2);
      border-radius: 20px; font-size: 0.75rem; padding: 4px 10px; cursor: pointer;
    }
    
    /* Session Tag Card */
    .card {
      background: var(--card-bg); border: 1px solid var(--border);
      border-radius: 12px; padding: 14px; margin-bottom: 14px;
    }
    .card-title { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-dim); margin-bottom: 8px; }
    .input-row { display: flex; gap: 8px; }
    input[type="text"] {
      flex: 1; background: #262626; border: 1px solid var(--border); border-radius: 8px;
      padding: 10px 12px; color: #fff; font-size: 0.95rem; outline: none;
    }
    input[type="text"]:focus { border-color: var(--accent); }
    button.btn {
      border: none; border-radius: 8px; font-weight: 600; font-size: 0.9rem;
      padding: 10px 16px; cursor: pointer; transition: background 0.2s, transform 0.1s;
    }
    button.btn:active { transform: scale(0.97); }
    .btn-primary { background: var(--accent); color: #fff; }
    .btn-primary:hover { background: var(--accent-hover); }
    .btn-success { background: var(--success); color: #fff; }
    .btn-danger { background: var(--danger); color: #fff; }
    .btn-secondary { background: #333; color: #eee; }
    
    /* Toggle Grid */
    .toggle-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
    .toggle-item {
      background: #262626; border: 1px solid var(--border); border-radius: 8px;
      padding: 10px; display: flex; justify-content: space-between; align-items: center;
      font-size: 0.85rem;
    }
    
    /* Captures Gallery */
    .captures-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .captures-strip {
      display: flex; gap: 10px; overflow-x: auto; padding-bottom: 6px;
    }
    .thumb-card {
      min-width: 90px; width: 90px; height: 110px; background: #000; border-radius: 8px;
      overflow: hidden; border: 1px solid var(--border); position: relative; flex-shrink: 0;
    }
    .thumb-card img { width: 100%; height: 100%; object-fit: cover; }
    .thumb-tag {
      position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.7);
      font-size: 0.65rem; padding: 2px 4px; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .thumb-del {
      position: absolute; top: 3px; right: 3px; background: rgba(244,67,54,0.85); color: #fff;
      border: none; border-radius: 50%; width: 20px; height: 20px; font-size: 11px; cursor: pointer;
    }

    /* Fixed Bottom Action Bar */
    .action-bar {
      position: fixed; bottom: 0; left: 0; right: 0; background: #1a1a1a;
      border-top: 1px solid var(--border); padding: 12px 16px; display: flex; gap: 12px;
      z-index: 100; max-width: 600px; margin: 0 auto;
    }
    .btn-capture {
      flex: 2; height: 52px; font-size: 1.1rem; border-radius: 26px;
      background: #e53935; color: #fff; border: 2px solid rgba(255,255,255,0.3);
      display: flex; justify-content: center; align-items: center; gap: 8px;
      box-shadow: 0 4px 12px rgba(229,57,53,0.4);
    }
    .btn-capture:active { background: #c62828; transform: scale(0.96); }
    .btn-finalize {
      flex: 1; height: 52px; font-size: 0.85rem; border-radius: 26px;
      background: #2e7d32; color: #fff; border: none;
    }

    /* Toast Notification */
    #toast {
      position: fixed; top: 60px; left: 50%; transform: translateX(-50%) translateY(-30px);
      background: #323232; color: #fff; padding: 10px 20px; border-radius: 20px;
      font-size: 0.85rem; opacity: 0; pointer-events: none; transition: all 0.25s ease;
      z-index: 200; border: 1px solid var(--accent); box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    #toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span>📄</span>
      <span>CamScan Remote</span>
    </div>
    <div class="badge" id="conn-badge">
      <span class="badge-dot"></span>
      <span id="conn-text">Tailscale Connected</span>
    </div>
  </header>

  <div id="toast">Notification</div>

  <div class="container">
    <!-- Camera Viewport -->
    <div class="viewport-card">
      <img id="camera-stream" src="/api/feed" alt="Camera Stream" />
      <div class="stream-overlay">
        <button class="pill-btn" id="stream-mode-btn" onclick="toggleStreamMode()">MJPEG Live</button>
        <button class="pill-btn" onclick="refreshFeed()">⟳</button>
      </div>
    </div>

    <!-- Student Session Card -->
    <div class="card">
      <div class="card-title">Student Session Tag</div>
      <div class="input-row">
        <input type="text" id="student-tag-input" placeholder="e.g. Alex_Smith_101" />
        <button class="btn btn-primary" onclick="saveStudentTag()">Set</button>
      </div>
    </div>

    <!-- Captures in Session -->
    <div class="card">
      <div class="captures-header">
        <div class="card-title" style="margin-bottom:0">Session Captures (<span id="cap-count">0</span>)</div>
      </div>
      <div class="captures-strip" id="captures-strip">
        <div style="font-size: 0.8rem; color: var(--text-dim); padding: 10px 0;">No pages captured yet in this session.</div>
      </div>
    </div>

    <!-- Mode Settings Card -->
    <div class="card">
      <div class="card-title">Scanner Modes</div>
      <div class="toggle-grid">
        <div class="toggle-item">
          <span>Two-Page Mode</span>
          <input type="checkbox" id="toggle-twopage" onchange="updateSettings()" />
        </div>
        <div class="toggle-item">
          <span>YOLO Dewarp</span>
          <input type="checkbox" id="toggle-yolo" onchange="updateSettings()" />
        </div>
        <div class="toggle-item">
          <span>Auto-Capture</span>
          <input type="checkbox" id="toggle-autocap" onchange="updateSettings()" />
        </div>
        <div class="toggle-item">
          <span>Free Capture</span>
          <input type="checkbox" id="toggle-freecap" onchange="updateSettings()" />
        </div>
      </div>
    </div>
  </div>

  <!-- Sticky Bottom Actions -->
  <div class="action-bar">
    <button class="btn btn-capture" id="btn-capture" onclick="triggerCapture()">
      <span>📷</span>
      <span>CAPTURE</span>
    </button>
    <button class="btn btn-finalize" onclick="finalizeSession()">
      <span>🏁 Finish Session</span>
    </button>
  </div>

  <script>
    let isMjpeg = true;
    let pollTimer = null;

    function showToast(msg, color = '#2196f3') {
      const toast = document.getElementById('toast');
      toast.innerText = msg;
      toast.style.borderColor = color;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 2500);
    }

    function toggleStreamMode() {
      const btn = document.getElementById('stream-mode-btn');
      const img = document.getElementById('camera-stream');
      if (isMjpeg) {
        // Switch to snapshot polling
        isMjpeg = false;
        btn.innerText = 'Polling (1s)';
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(() => {
          img.src = '/api/snapshot?t=' + Date.now();
        }, 1000);
      } else {
        // Switch to MJPEG
        isMjpeg = true;
        if (pollTimer) clearInterval(pollTimer);
        btn.innerText = 'MJPEG Live';
        img.src = '/api/feed';
      }
    }

    function refreshFeed() {
      const img = document.getElementById('camera-stream');
      if (isMjpeg) {
        img.src = '/api/feed?t=' + Date.now();
      } else {
        img.src = '/api/snapshot?t=' + Date.now();
      }
    }

    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();
        
        // Update tag if not actively typing
        const tagInput = document.getElementById('student-tag-input');
        if (document.activeElement !== tagInput && data.student_tag) {
          tagInput.value = data.student_tag;
        }

        document.getElementById('cap-count').innerText = data.capture_count;
        document.getElementById('toggle-twopage').checked = !!data.two_page_mode;
        document.getElementById('toggle-yolo').checked = (data.boundary_detector || '').toLowerCase().includes('yolo');
        document.getElementById('toggle-autocap').checked = !!data.auto_capture;
        document.getElementById('toggle-freecap').checked = !!data.free_capture_mode;

        // Render thumbnails
        const strip = document.getElementById('captures-strip');
        if (data.captures && data.captures.length > 0) {
          strip.innerHTML = data.captures.map(c => `
            <div class="thumb-card">
              <img src="/api/thumbnail/${c.index}?t=${Date.now()}" alt="p${c.index}">
              <div class="thumb-tag">#${c.index}</div>
              <button class="thumb-del" onclick="deleteCapture(${c.index})">&times;</button>
            </div>
          `).join('');
        } else {
          strip.innerHTML = '<div style="font-size: 0.8rem; color: var(--text-dim); padding: 10px 0;">No pages captured yet in this session.</div>';
        }
      } catch (e) {
        console.error('Status fetch error:', e);
      }
    }

    async function saveStudentTag() {
      const val = document.getElementById('student-tag-input').value.trim();
      try {
        const res = await fetch('/api/session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ student_tag: val })
        });
        if (res.ok) {
          showToast('Student session updated to: ' + (val || 'Untagged'), '#4caf50');
          fetchStatus();
        }
      } catch (e) {
        showToast('Failed to update session tag', '#f44336');
      }
    }

    async function triggerCapture() {
      const btn = document.getElementById('btn-capture');
      btn.disabled = true;
      btn.style.opacity = '0.6';
      try {
        const res = await fetch('/api/capture', { method: 'POST' });
        const data = await res.json();
        if (res.ok && data.success) {
          showToast('📸 Page captured! Total: ' + data.count, '#4caf50');
          fetchStatus();
        } else {
          showToast('Capture error: ' + (data.error || 'Failed'), '#f44336');
        }
      } catch (e) {
        showToast('Network error during capture', '#f44336');
      } finally {
        setTimeout(() => {
          btn.disabled = false;
          btn.style.opacity = '1';
        }, 500);
      }
    }

    async function finalizeSession() {
      if (!confirm('Finalize current student session and auto-export all pages?')) return;
      try {
        showToast('Exporting session to watched folder...', '#ff9800');
        const res = await fetch('/api/finalize', { method: 'POST' });
        const data = await res.json();
        if (res.ok && data.success) {
          showToast('✅ Session finalized & exported!', '#4caf50');
          document.getElementById('student-tag-input').value = '';
          fetchStatus();
        } else {
          showToast('Finalize failed: ' + (data.error || 'Error'), '#f44336');
        }
      } catch (e) {
        showToast('Error finalizing session', '#f44336');
      }
    }

    async function deleteCapture(idx) {
      if (!confirm('Delete page #' + idx + '?')) return;
      try {
        const res = await fetch('/api/captures/' + idx, { method: 'DELETE' });
        if (res.ok) {
          showToast('Page #' + idx + ' deleted', '#ff9800');
          fetchStatus();
        }
      } catch (e) {
        showToast('Failed to delete page', '#f44336');
      }
    }

    async function updateSettings() {
      const twopage = document.getElementById('toggle-twopage').checked;
      const yolo = document.getElementById('toggle-yolo').checked;
      const autocap = document.getElementById('toggle-autocap').checked;
      const freecap = document.getElementById('toggle-freecap').checked;
      try {
        await fetch('/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            two_page_mode: twopage,
            boundary_detector: yolo ? "YOLOv8 + Geometric Dewarp" : "Classic Contour (OpenCV)",
            auto_capture: autocap,
            free_capture_mode: freecap
          })
        });
        showToast('Settings saved', '#2196f3');
      } catch (e) {
        showToast('Failed to save settings', '#f44336');
      }
    }

    // Polling status loop
    setInterval(fetchStatus, 2500);
    fetchStatus();
  </script>
</body>
</html>
"""


def create_remote_app(bridge: RemoteBridge) -> "FastAPI":
    """
    Build and return configured FastAPI app wired to given RemoteBridge.
    """
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed. Please install fastapi and uvicorn.")

    app = FastAPI(title="CamScan Remote", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index_page():
        return HTMLResponse(content=MOBILE_UI_HTML)

    @app.get("/api/feed")
    async def video_feed():
        """
        MJPEG stream endpoint. Efficient, low-latency streaming over Tailscale.
        """
        async def frame_generator():
            while True:
                frame = bridge.get_frame()
                if frame is not None:
                    # Compress with moderate quality for responsive streaming across continents
                    ret, buffer = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 55]
                    )
                    if ret:
                        data = buffer.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                        )
                # Cap rate at ~12-15 FPS to preserve bandwidth over international Tailscale
                await asyncio.sleep(0.07)

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/snapshot")
    async def snapshot():
        """Single snapshot endpoint."""
        frame = bridge.get_frame()
        if frame is None:
            raise HTTPException(status_code=503, detail="Camera frame unavailable")
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            raise HTTPException(status_code=500, detail="Encoding error")
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    @app.get("/api/status")
    async def get_status():
        """Return scanner state and capture list."""
        return bridge.get_status()

    @app.get("/api/thumbnail/{index}")
    async def get_thumbnail(index: int):
        """Return thumbnail image for capture entry index."""
        thumb = bridge.get_thumbnail(index)
        if thumb is None:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        ret, buffer = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not ret:
            raise HTTPException(status_code=500, detail="Encoding error")
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    class SessionRequest(BaseModel):
        student_tag: str

    @app.post("/api/session")
    async def update_session(req: SessionRequest):
        """Update current student session tag."""
        clean = bridge.set_student_tag(req.student_tag)
        return {"success": True, "student_tag": clean}

    @app.post("/api/capture")
    async def capture_page():
        """Trigger capture on the host machine."""
        res = bridge.capture()
        return res

    @app.post("/api/finalize")
    async def finalize_session():
        """Finalize and auto-export session."""
        res = bridge.finalize_session()
        return res

    class SettingsRequest(BaseModel):
        two_page_mode: t.Optional[bool] = None
        boundary_detector: t.Optional[str] = None
        auto_capture: t.Optional[bool] = None
        free_capture_mode: t.Optional[bool] = None

    @app.post("/api/settings")
    async def update_settings(req: SettingsRequest):
        """Update scanner configuration."""
        data = {k: v for k, v in req.model_dump().items() if v is not None}
        updated = bridge.set_settings(data)
        return {"success": True, "settings": updated}

    @app.delete("/api/captures/{index}")
    async def delete_capture(index: int):
        """Delete specific capture by index."""
        ok = bridge.delete_capture(index)
        if not ok:
            raise HTTPException(status_code=404, detail="Capture index not found")
        return {"success": True}

    return app


class RemoteServerManager:
    """
    Manages starting and stopping Uvicorn server in a daemon background thread.
    """

    def __init__(self, bridge: RemoteBridge, host: str = "0.0.0.0", port: int = 8000):
        self.bridge = bridge
        self.host = host
        self.port = port
        self.app = create_remote_app(bridge)
        self._server = None
        self._thread = None
        self._running = False

    def start(self):
        """Start FastAPI/Uvicorn server in background thread."""
        if self._running:
            return

        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            loop="asyncio",
        )
        self._server = uvicorn.Server(config)

        def _run():
            self._running = True
            logger.info(f"Remote Server started on http://{self.host}:{self.port}")
            self._server.run()
            self._running = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background server."""
        if self._server:
            self._server.should_exit = True
            self._running = False

    def is_running(self) -> bool:
        return self._running

    def get_url(self) -> str:
        ip = get_local_ip()
        return f"http://{ip}:{self.port}"
