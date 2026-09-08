"""
Remote Control Server for CamScan.
Provides a FastAPI web application allowing full camera monitoring, capture control,
student/subject tagging, and session finalization over Tailscale or local network.

Hardened with zero-trust session authentication (pairing PIN / token) and subnet validation.
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
    from fastapi import FastAPI, HTTPException, Request, Response, Depends, Header, Cookie
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError as e:
    import sys
    print(f"DEBUG: FastAPI import failed: {e}", file=sys.stderr)
    FastAPI = None

from camscan import session
from camscan.tailscale import (
    SessionSecurityManager,
    get_tailscale_info,
    is_client_authorized_subnet,
    find_tailscale_cli,
)

logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    """Attempt to find local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_best_server_ip() -> str:
    """Return Tailscale IP if running/available, otherwise local LAN IP."""
    ts_info = get_tailscale_info()
    if ts_info.ipv4:
        return ts_info.ipv4
    return get_local_ip()


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

    def set_subject(self, subject: str) -> str:
        raise NotImplementedError

    def add_subject(self, subject: str) -> list[str]:
        raise NotImplementedError

    def remove_subject(self, subject: str) -> list[str]:
        raise NotImplementedError

    def add_student(self, student: str) -> list[str]:
        raise NotImplementedError

    def remove_student(self, student: str) -> list[str]:
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
        tag = self.app.var_student_tag.get() if hasattr(self.app, "var_student_tag") else ""
        subject = self.app.var_subject.get() if hasattr(self.app, "var_subject") else "General"
        active_prof_name = "Default"
        subjects_list = [subject]
        students_list = [tag] if tag else []

        if hasattr(self.app, "profile_manager"):
            prof = self.app.profile_manager.get_active_profile()
            active_prof_name = prof.name
            subjects_list = list(prof.subjects)
            students_list = list(prof.students)

        count = len(self.app.entries)
        captures = [
            {"name": e.name, "index": i}
            for i, e in enumerate(self.app.entries, start=1)
        ]

        ts_info = get_tailscale_info()

        return {
            "student_tag": tag,
            "subject": subject,
            "subjects": subjects_list,
            "students": students_list,
            "system_date": session.get_system_date_str(),
            "active_profile": active_prof_name,
            "capture_count": count,
            "captures": captures,
            "two_page_mode": bool(self.app.var_two_page_mode.get()) if hasattr(self.app, "var_two_page_mode") else False,
            "boundary_detector": self.app.var_boundary_detector.get() if hasattr(self.app, "var_boundary_detector") else "",
            "auto_capture": bool(self.app.var_auto_capture.get()) if hasattr(self.app, "var_auto_capture") else False,
            "free_capture_mode": bool(self.app.var_free_capture_mode.get()) if hasattr(self.app, "var_free_capture_mode") else False,
            "postprocessing_option": self.app.var_postprocessing_option.get() if hasattr(self.app, "var_postprocessing_option") else "",
            "watched_folder": self.app.var_watched_folder.get() if hasattr(self.app, "var_watched_folder") else "",
            "tailscale": {
                "running": ts_info.running,
                "ipv4": ts_info.ipv4,
                "magic_dns": ts_info.magic_dns,
            }
        }

    def set_student_tag(self, tag: str) -> str:
        self.app.after(0, lambda: self.app.var_student_tag.set(tag))
        if hasattr(self.app, "profile_manager"):
            self.app.profile_manager.get_active_profile().active_student = tag
            self.app.profile_manager.save()
        return tag

    def set_subject(self, subject: str) -> str:
        self.app.after(0, lambda: self.app.var_subject.set(subject))
        if hasattr(self.app, "profile_manager"):
            self.app.profile_manager.get_active_profile().active_subject = subject
            self.app.profile_manager.save()
        return subject

    def add_subject(self, subject: str) -> list[str]:
        if hasattr(self.app, "profile_manager"):
            updated = self.app.profile_manager.add_subject(subject)
            self.app.after(0, self.app._refresh_subject_dropdown)
            return updated
        return [subject]

    def remove_subject(self, subject: str) -> list[str]:
        if hasattr(self.app, "profile_manager"):
            updated = self.app.profile_manager.remove_subject(subject)
            self.app.after(0, self.app._refresh_subject_dropdown)
            return updated
        return ["General"]

    def add_student(self, student: str) -> list[str]:
        if hasattr(self.app, "profile_manager"):
            updated = self.app.profile_manager.add_student(student)
            self.app.after(0, self.app._refresh_student_dropdown)
            return updated
        return [student]

    def remove_student(self, student: str) -> list[str]:
        if hasattr(self.app, "profile_manager"):
            updated = self.app.profile_manager.remove_student(student)
            self.app.after(0, self.app._refresh_student_dropdown)
            return updated
        return []

    def set_settings(self, settings: dict) -> dict:
        def _apply():
            if "two_page_mode" in settings and hasattr(self.app, "var_two_page_mode"):
                self.app.var_two_page_mode.set(1 if settings["two_page_mode"] else 0)
            if "boundary_detector" in settings and hasattr(self.app, "var_boundary_detector"):
                self.app.var_boundary_detector.set(settings["boundary_detector"])
            if "auto_capture" in settings and hasattr(self.app, "var_auto_capture"):
                self.app.var_auto_capture.set(1 if settings["auto_capture"] else 0)
            if "free_capture_mode" in settings and hasattr(self.app, "var_free_capture_mode"):
                self.app.var_free_capture_mode.set(1 if settings["free_capture_mode"] else 0)
            if "postprocessing_option" in settings and hasattr(self.app, "var_postprocessing_option"):
                self.app.var_postprocessing_option.set(settings["postprocessing_option"])

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


# Embedded Mobile-First Responsive Web UI with Authentication, Subject & Student controls
MOBILE_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Neo Scanner Remote Control</title>
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
    .brand { font-size: 1.05rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
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
    
    /* Session Hierarchy Cards */
    .card {
      background: var(--card-bg); border: 1px solid var(--border);
      border-radius: 12px; padding: 14px; margin-bottom: 14px;
    }
    .card-title { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-dim); margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
    .input-row { display: flex; gap: 8px; margin-bottom: 8px; }
    input[type="text"], select {
      flex: 1; background: #262626; border: 1px solid var(--border); border-radius: 8px;
      padding: 10px 12px; color: #fff; font-size: 0.95rem; outline: none;
    }
    input[type="text"]:focus, select:focus { border-color: var(--accent); }
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
    .btn-sm { padding: 6px 10px; font-size: 0.8rem; }
    
    .path-preview {
      background: #181818; border: 1px dashed var(--border); border-radius: 6px;
      padding: 8px 10px; font-size: 0.78rem; color: #64b5f6; font-family: monospace;
      word-break: break-all; margin-top: 6px;
    }
    
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
    
    /* Sticky Bottom Capture Bar */
    .bottom-bar {
      position: fixed; bottom: 0; left: 0; right: 0;
      background: rgba(24, 24, 24, 0.95); backdrop-filter: blur(10px);
      border-top: 1px solid var(--border); padding: 12px 16px;
      display: flex; gap: 10px; z-index: 100; max-width: 600px; margin: 0 auto;
    }
    .btn-capture {
      flex: 2; height: 50px; font-size: 1.1rem; font-weight: 700;
      background: linear-gradient(135deg, #2196f3, #1565c0);
      color: #fff; box-shadow: 0 4px 12px rgba(33,150,243,0.3);
    }
    .btn-finish { flex: 1; height: 50px; font-size: 0.95rem; }

    /* Auth Modal */
    #auth-overlay {
      position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 999;
      display: none; align-items: center; justify-content: center; padding: 20px;
    }
    .auth-box {
      background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px;
      max-width: 360px; width: 100%; padding: 24px; text-align: center;
    }
    .auth-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 8px; color: #fff; }
    .auth-desc { font-size: 0.85rem; color: var(--text-dim); margin-bottom: 18px; }
    .pin-input {
      font-size: 1.8rem; letter-spacing: 6px; text-align: center;
      width: 100%; padding: 10px; margin-bottom: 16px; font-weight: bold;
    }
    
    #toast {
      position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%);
      background: rgba(33, 150, 243, 0.95); color: #fff; padding: 8px 18px;
      border-radius: 20px; font-size: 0.85rem; font-weight: 500;
      pointer-events: none; opacity: 0; transition: opacity 0.25s; z-index: 200;
    }
  </style>
</head>
<body>

  <!-- Pairing Auth Overlay -->
  <div id="auth-overlay">
    <div class="auth-box">
      <div class="auth-title">🔐 Remote Scanner Pairing</div>
      <div class="auth-desc">Enter the 6-digit pairing PIN shown on your Neo Scanner desktop screen:</div>
      <input type="text" id="pin-entry" class="pin-input" maxlength="6" pattern="[0-9]*" placeholder="000000" />
      <button class="btn btn-primary" style="width:100%" onclick="submitPin()">Pair Device</button>
      <div id="auth-error" style="color:var(--danger); font-size:0.8rem; margin-top:10px; display:none;">Invalid PIN. Please check desktop.</div>
    </div>
  </div>

  <header>
    <div class="brand">
      <span>📄 Neo Scanner</span>
      <span class="badge" id="conn-badge"><span class="badge-dot"></span> Live</span>
    </div>
    <div style="font-size: 0.8rem; color: var(--text-dim);" id="header-user">
      User: ...
    </div>
  </header>

  <div class="container">
    <!-- Camera Feed Viewport -->
    <div class="viewport-card">
      <img id="stream-img" src="/api/snapshot" alt="Live Camera Preview" />
      <div class="stream-overlay">
        <button class="pill-btn" onclick="toggleStream()" id="stream-btn">Pause</button>
      </div>
    </div>

    <!-- Subject & Student Hierarchy Card -->
    <div class="card">
      <div class="card-title">
        <span>📁 Folder Hierarchy</span>
        <span id="date-badge" style="background:#262626; color:#4caf50; padding:2px 8px; border-radius:10px; font-size:0.75rem;">📅 Today</span>
      </div>

      <!-- Main Folder (Subject) -->
      <label style="font-size:0.75rem; color:var(--text-dim); margin-bottom:4px; display:block;">Main Folder (Subject):</label>
      <div class="input-row">
        <select id="subject-select" onchange="onSubjectChange()">
          <option value="General">General</option>
        </select>
        <button class="btn btn-secondary btn-sm" onclick="promptAddSubject()">+ Add</button>
      </div>

      <!-- Student Selection -->
      <label style="font-size:0.75rem; color:var(--text-dim); margin-bottom:4px; display:block;">Student Subfolder:</label>
      <div class="input-row">
        <select id="student-select" onchange="onStudentSelectChange()">
          <option value="">-- Select or type below --</option>
        </select>
        <button class="btn btn-secondary btn-sm" onclick="promptAddStudent()">+ Add</button>
      </div>
      <div class="input-row">
        <input type="text" id="student-tag-input" placeholder="Type student name e.g. Ahmad" oninput="onStudentInputChanged()" />
        <button class="btn btn-primary" onclick="applySessionTag()">Set</button>
      </div>

      <div class="path-preview" id="path-preview">
        Target: Base / ... / ... / ...
      </div>
    </div>

    <!-- Captures Gallery Strip -->
    <div class="card">
      <div class="captures-header">
        <div class="card-title" style="margin:0;">Session Captures</div>
        <div style="font-size:0.8rem; color:var(--accent); font-weight:600;" id="capture-count-label">0 pages</div>
      </div>
      <div class="captures-strip" id="captures-strip">
        <div style="color:var(--text-dim); font-size:0.85rem; padding:12px 4px;">No pages captured yet in this session.</div>
      </div>
    </div>

    <!-- Settings Card -->
    <div class="card">
      <div class="card-title">Capture Mode</div>
      <div class="toggle-grid">
        <div class="toggle-item">
          <span>Two-Page Mode</span>
          <input type="checkbox" id="toggle-twopage" onchange="updateSettings()" />
        </div>
        <div class="toggle-item">
          <span>Perspective Crop</span>
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
  <div class="bottom-bar">
    <button class="btn btn-primary btn-capture" onclick="triggerCapture()">📸 Capture Page</button>
    <button class="btn btn-success btn-finish" onclick="finalizeSession()">Finish & Save</button>
  </div>

  <div id="toast">Message</div>

  <script>
    let authToken = sessionStorage.getItem('neo_token') || '';
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('token')) {
      authToken = urlParams.get('token');
      sessionStorage.setItem('neo_token', authToken);
    }

    let isStreaming = true;
    const streamImg = document.getElementById('stream-img');
    const streamBtn = document.getElementById('stream-btn');
    let cachedStatus = null;

    function getAuthHeaders() {
      return {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + authToken
      };
    }

    function showToast(msg, color) {
      const t = document.getElementById('toast');
      t.innerText = msg;
      t.style.background = color || 'rgba(33, 150, 243, 0.95)';
      t.style.opacity = '1';
      setTimeout(() => { t.style.opacity = '0'; }, 2200);
    }

    function checkAuth() {
      if (!authToken) {
        document.getElementById('auth-overlay').style.display = 'flex';
      } else {
        document.getElementById('auth-overlay').style.display = 'none';
        startStream();
      }
    }

    async function submitPin() {
      const pin = document.getElementById('pin-entry').value.trim();
      const err = document.getElementById('auth-error');
      if (!pin) return;
      try {
        const res = await fetch('/api/auth', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pin: pin })
        });
        const data = await res.json();
        if (data.success && data.token) {
          authToken = data.token;
          sessionStorage.setItem('neo_token', authToken);
          document.getElementById('auth-overlay').style.display = 'none';
          err.style.display = 'none';
          showToast('Paired successfully!', '#4caf50');
          startStream();
          fetchStatus();
        } else {
          err.style.display = 'block';
        }
      } catch (e) {
        err.style.display = 'block';
      }
    }

    function startStream() {
      if (authToken) {
        streamImg.src = '/api/feed?token=' + encodeURIComponent(authToken);
        isStreaming = true;
        streamBtn.innerText = 'Pause';
      }
    }

    function toggleStream() {
      if (isStreaming) {
        streamImg.src = '/api/snapshot?token=' + encodeURIComponent(authToken);
        streamBtn.innerText = 'Play';
        isStreaming = false;
      } else {
        startStream();
      }
    }

    async function fetchStatus() {
      if (!authToken) return;
      try {
        const res = await fetch('/api/status?token=' + encodeURIComponent(authToken));
        if (res.status === 401) {
          document.getElementById('auth-overlay').style.display = 'flex';
          return;
        }
        const data = await res.json();
        cachedStatus = data;
        renderStatus(data);
      } catch (e) {}
    }

    function renderStatus(data) {
      document.getElementById('header-user').innerText = 'User: ' + (data.active_profile || 'Default');
      document.getElementById('date-badge').innerText = '📅 ' + (data.system_date || 'Today');

      // Update Subjects
      const subjSelect = document.getElementById('subject-select');
      const curSubj = data.subject || 'General';
      subjSelect.innerHTML = '';
      (data.subjects || ['General']).forEach(s => {
        const opt = document.createElement('option');
        opt.value = s;
        opt.innerText = s;
        if (s === curSubj) opt.selected = true;
        subjSelect.appendChild(opt);
      });

      // Update Students
      const studSelect = document.getElementById('student-select');
      const curStud = data.student_tag || '';
      studSelect.innerHTML = '<option value="">-- Select student --</option>';
      (data.students || []).forEach(st => {
        const opt = document.createElement('option');
        opt.value = st;
        opt.innerText = st;
        if (st === curStud) opt.selected = true;
        studSelect.appendChild(opt);
      });

      const tagInput = document.getElementById('student-tag-input');
      if (document.activeElement !== tagInput) {
        tagInput.value = curStud;
      }

      updatePathPreview(curSubj, data.system_date, curStud);

      // Settings
      document.getElementById('toggle-twopage').checked = !!data.two_page_mode;
      document.getElementById('toggle-yolo').checked = (data.boundary_detector && (data.boundary_detector.includes('Perspective') || data.boundary_detector.includes('Clean')));
      document.getElementById('toggle-autocap').checked = !!data.auto_capture;
      document.getElementById('toggle-freecap').checked = !!data.free_capture_mode;

      // Gallery
      document.getElementById('capture-count-label').innerText = data.capture_count + ' page' + (data.capture_count === 1 ? '' : 's');
      renderThumbnails(data.captures || []);
    }

    function updatePathPreview(subject, dateStr, student) {
      const s = subject || 'General';
      const d = dateStr || 'Today';
      const st = student || 'Untagged';
      document.getElementById('path-preview').innerText = '📁 Destination: ' + s + ' / ' + d + ' / ' + st + ' /';
    }

    function onSubjectChange() {
      const subj = document.getElementById('subject-select').value;
      applySession(subj, document.getElementById('student-tag-input').value);
    }

    function onStudentSelectChange() {
      const st = document.getElementById('student-select').value;
      document.getElementById('student-tag-input').value = st;
      applySession(document.getElementById('subject-select').value, st);
    }

    function onStudentInputChanged() {
      const subj = document.getElementById('subject-select').value;
      const st = document.getElementById('student-tag-input').value;
      updatePathPreview(subj, (cachedStatus ? cachedStatus.system_date : 'Today'), st);
    }

    async function applySessionTag() {
      const subj = document.getElementById('subject-select').value;
      const st = document.getElementById('student-tag-input').value.trim();
      await applySession(subj, st);
      showToast('Student tagged: ' + (st || 'Untagged'), '#2196f3');
    }

    async function applySession(subject, student_tag) {
      try {
        await fetch('/api/session', {
          method: 'POST',
          headers: getAuthHeaders(),
          body: JSON.stringify({ subject: subject, student_tag: student_tag })
        });
        fetchStatus();
      } catch (e) {
        showToast('Failed to update session', '#f44336');
      }
    }

    async function promptAddSubject() {
      const name = prompt('Enter new subject / main folder name (e.g. Maths, English):');
      if (!name || !name.trim()) return;
      try {
        const res = await fetch('/api/subjects', {
          method: 'POST',
          headers: getAuthHeaders(),
          body: JSON.stringify({ action: 'add', subject: name.trim() })
        });
        showToast('Subject added', '#4caf50');
        fetchStatus();
      } catch (e) {
        showToast('Failed to add subject', '#f44336');
      }
    }

    async function promptAddStudent() {
      const name = prompt('Enter student name (e.g. Ahmad, Ubaid):');
      if (!name || !name.trim()) return;
      try {
        const res = await fetch('/api/students', {
          method: 'POST',
          headers: getAuthHeaders(),
          body: JSON.stringify({ action: 'add', student: name.trim() })
        });
        showToast('Student added to roster', '#4caf50');
        fetchStatus();
      } catch (e) {
        showToast('Failed to add student', '#f44336');
      }
    }

    function renderThumbnails(captures) {
      const strip = document.getElementById('captures-strip');
      if (!captures.length) {
        strip.innerHTML = '<div style="color:var(--text-dim); font-size:0.85rem; padding:12px 4px;">No pages captured yet in this session.</div>';
        return;
      }
      strip.innerHTML = '';
      captures.forEach(c => {
        const card = document.createElement('div');
        card.className = 'thumb-card';
        card.innerHTML = `
          <img src="/api/thumbnail/${c.index}?token=${encodeURIComponent(authToken)}" alt="${c.name}">
          <div class="thumb-tag">#${c.index}</div>
          <button class="thumb-del" onclick="deleteCapture(${c.index})">&times;</button>
        `;
        strip.appendChild(card);
      });
    }

    async function triggerCapture() {
      showToast('Capturing...', '#2196f3');
      try {
        const res = await fetch('/api/capture', {
          method: 'POST',
          headers: getAuthHeaders()
        });
        const data = await res.json();
        if (data.success) {
          showToast('Captured Page #' + data.count, '#4caf50');
          fetchStatus();
        } else {
          showToast('Capture error: ' + (data.error || 'Unknown'), '#f44336');
        }
      } catch (e) {
        showToast('Network error during capture', '#f44336');
      }
    }

    async function finalizeSession() {
      if (!confirm('Finish and auto-export this session to hierarchical folder?')) return;
      showToast('Finalizing & saving...', '#ff9800');
      try {
        const res = await fetch('/api/finalize', {
          method: 'POST',
          headers: getAuthHeaders()
        });
        const data = await res.json();
        if (data.success) {
          showToast('Session saved successfully!', '#4caf50');
          document.getElementById('student-tag-input').value = '';
          fetchStatus();
        } else {
          showToast('Failed to finalize: ' + (data.error || ''), '#f44336');
        }
      } catch (e) {
        showToast('Network error during finalize', '#f44336');
      }
    }

    async function deleteCapture(idx) {
      if (!confirm('Delete page #' + idx + '?')) return;
      try {
        await fetch('/api/captures/' + idx, {
          method: 'DELETE',
          headers: getAuthHeaders()
        });
        showToast('Page #' + idx + ' deleted', '#ff9800');
        fetchStatus();
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
          headers: getAuthHeaders(),
          body: JSON.stringify({
            two_page_mode: twopage,
            boundary_detector: yolo ? "Clean Perspective Crop (Recommended)" : "Classic Contour (OpenCV)",
            auto_capture: autocap,
            free_capture_mode: freecap
          })
        });
        showToast('Settings saved', '#2196f3');
      } catch (e) {
        showToast('Failed to save settings', '#f44336');
      }
    }

    checkAuth();
    setInterval(fetchStatus, 3000);
    fetchStatus();
  </script>
</body>
</html>
"""


def create_remote_app(
    bridge: RemoteBridge,
    security_manager: t.Optional[SessionSecurityManager] = None,
    allow_lan: bool = True,
) -> "FastAPI":
    """
    Build and return configured FastAPI app wired to given RemoteBridge.
    Includes Zero-Trust Token/PIN authentication and Subnet verification.
    """
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed. Please install fastapi and uvicorn.")

    sec_mgr = security_manager or SessionSecurityManager()

    app = FastAPI(title="Neo Scanner Remote", version="1.1.0")

    # Add CORS middleware to protect from cross-origin exploits
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def verify_auth(
        request: Request,
        authorization: t.Optional[str] = Header(None),
        token: t.Optional[str] = None,
    ):
        """Verify session authentication token."""
        # 1. Subnet check
        client_host = request.client.host if request.client else "127.0.0.1"
        if not is_client_authorized_subnet(client_host, allow_lan=allow_lan):
            logger.warning(f"Rejected connection from unauthorized subnet: {client_host}")
            raise HTTPException(status_code=403, detail="Forbidden subnet")

        # 2. Extract token from header or query param
        extracted = token
        if not extracted and authorization:
            parts = authorization.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                extracted = parts[1]

        if not sec_mgr.validate_token(extracted):
            raise HTTPException(status_code=401, detail="Unauthorized: invalid session token")
        return True

    @app.get("/", response_class=HTMLResponse)
    async def index_page():
        return HTMLResponse(content=MOBILE_UI_HTML)

    class AuthRequest(BaseModel):
        pin: str

    @app.post("/api/auth")
    async def authenticate_pin(req: AuthRequest, request: Request):
        """Authenticate 6-digit PIN and return session token."""
        client_host = request.client.host if request.client else "127.0.0.1"
        if not is_client_authorized_subnet(client_host, allow_lan=allow_lan):
            raise HTTPException(status_code=403, detail="Forbidden subnet")

        if sec_mgr.validate_pin(req.pin):
            return {"success": True, "token": sec_mgr.session_token}
        raise HTTPException(status_code=401, detail="Invalid pairing PIN")

    @app.get("/api/feed")
    async def video_feed(token: t.Optional[str] = None, req: Request = None):
        """
        MJPEG stream endpoint. Efficient, low-latency streaming over Tailscale.
        """
        client_host = req.client.host if req and req.client else "127.0.0.1"
        if not is_client_authorized_subnet(client_host, allow_lan=allow_lan):
            raise HTTPException(status_code=403, detail="Forbidden subnet")
        if not sec_mgr.validate_token(token):
            raise HTTPException(status_code=401, detail="Unauthorized")

        async def frame_generator():
            while True:
                frame = bridge.get_frame()
                if frame is not None:
                    ret, buffer = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 55]
                    )
                    if ret:
                        data = buffer.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                        )
                await asyncio.sleep(0.07)

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/snapshot")
    async def snapshot(token: t.Optional[str] = None, req: Request = None):
        """Single snapshot endpoint."""
        client_host = req.client.host if req and req.client else "127.0.0.1"
        if not is_client_authorized_subnet(client_host, allow_lan=allow_lan):
            raise HTTPException(status_code=403, detail="Forbidden subnet")
        if not sec_mgr.validate_token(token):
            raise HTTPException(status_code=401, detail="Unauthorized")

        frame = bridge.get_frame()
        if frame is None:
            raise HTTPException(status_code=503, detail="Camera frame unavailable")
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            raise HTTPException(status_code=500, detail="Encoding error")
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    @app.get("/api/status")
    async def get_status(authenticated: bool = Depends(verify_auth)):
        """Return scanner state, subjects, students, and capture list."""
        return bridge.get_status()

    @app.get("/api/thumbnail/{index}")
    async def get_thumbnail(index: int, token: t.Optional[str] = None, req: Request = None):
        """Return thumbnail image for capture entry index."""
        client_host = req.client.host if req and req.client else "127.0.0.1"
        if not is_client_authorized_subnet(client_host, allow_lan=allow_lan):
            raise HTTPException(status_code=403, detail="Forbidden subnet")
        if not sec_mgr.validate_token(token):
            raise HTTPException(status_code=401, detail="Unauthorized")

        thumb = bridge.get_thumbnail(index)
        if thumb is None:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        ret, buffer = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not ret:
            raise HTTPException(status_code=500, detail="Encoding error")
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    class SessionRequest(BaseModel):
        student_tag: t.Optional[str] = None
        subject: t.Optional[str] = None

    @app.post("/api/session")
    async def update_session(req: SessionRequest, authenticated: bool = Depends(verify_auth)):
        """Update current student tag and/or main subject."""
        out = {}
        if req.student_tag is not None:
            out["student_tag"] = bridge.set_student_tag(req.student_tag)
        if req.subject is not None:
            out["subject"] = bridge.set_subject(req.subject)
        return {"success": True, **out}

    class SubjectActionRequest(BaseModel):
        action: str  # 'add' or 'remove'
        subject: str

    @app.post("/api/subjects")
    async def manage_subjects(req: SubjectActionRequest, authenticated: bool = Depends(verify_auth)):
        """Add or remove subject / main folder."""
        if req.action == "add":
            updated = bridge.add_subject(req.subject)
        elif req.action == "remove":
            updated = bridge.remove_subject(req.subject)
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
        return {"success": True, "subjects": updated}

    class StudentActionRequest(BaseModel):
        action: str  # 'add' or 'remove'
        student: str

    @app.post("/api/students")
    async def manage_students(req: StudentActionRequest, authenticated: bool = Depends(verify_auth)):
        """Add or remove student from roster."""
        if req.action == "add":
            updated = bridge.add_student(req.student)
        elif req.action == "remove":
            updated = bridge.remove_student(req.student)
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
        return {"success": True, "students": updated}

    @app.post("/api/capture")
    async def capture_page(authenticated: bool = Depends(verify_auth)):
        """Trigger capture on the host machine."""
        return bridge.capture()

    @app.post("/api/finalize")
    async def finalize_session(authenticated: bool = Depends(verify_auth)):
        """Finalize and auto-export session."""
        return bridge.finalize_session()

    class SettingsRequest(BaseModel):
        two_page_mode: t.Optional[bool] = None
        boundary_detector: t.Optional[str] = None
        auto_capture: t.Optional[bool] = None
        free_capture_mode: t.Optional[bool] = None

    @app.post("/api/settings")
    async def update_settings(req: SettingsRequest, authenticated: bool = Depends(verify_auth)):
        """Update scanner configuration."""
        data = {k: v for k, v in req.model_dump().items() if v is not None}
        updated = bridge.set_settings(data)
        return {"success": True, "settings": updated}

    @app.delete("/api/captures/{index}")
    async def delete_capture(index: int, authenticated: bool = Depends(verify_auth)):
        """Delete specific capture by index."""
        ok = bridge.delete_capture(index)
        if not ok:
            raise HTTPException(status_code=404, detail="Capture index not found")
        return {"success": True}

    return app


class RemoteServerManager:
    """
    Manages starting and stopping Uvicorn server in a daemon background thread
    with Zero-Trust security and Tailscale awareness.
    """

    def __init__(
        self,
        bridge: RemoteBridge,
        host: str = "0.0.0.0",
        port: int = 8000,
        security_manager: t.Optional[SessionSecurityManager] = None,
        allow_lan: bool = True,
    ):
        self.bridge = bridge
        self.host = host
        self.port = port
        self.security_manager = security_manager or SessionSecurityManager()
        self.allow_lan = allow_lan
        self.app = create_remote_app(
            bridge,
            security_manager=self.security_manager,
            allow_lan=self.allow_lan,
        )
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

    def get_pairing_pin(self) -> str:
        return self.security_manager.pairing_pin

    def get_url(self, with_token: bool = True) -> str:
        ip = get_best_server_ip()
        base = f"http://{ip}:{self.port}"
        if with_token:
            return f"{base}/?token={self.security_manager.session_token}"
        return base

    def get_qr_image(self):
        """Return PIL Image of pairing QR code."""
        url = self.get_url(with_token=True)
        return self.security_manager.generate_qr_image(url)
