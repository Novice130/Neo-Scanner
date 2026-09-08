"""
Unit tests for camscan remote control API server with Zero-Trust authentication
and multi-user / subject / student hierarchy management.
"""

import numpy as np
import pytest
from starlette.testclient import TestClient

from camscan.remote import RemoteBridge, create_remote_app
from camscan.tailscale import SessionSecurityManager


class MockBridge(RemoteBridge):
    def __init__(self):
        self.student_tag = "Init_Tag"
        self.subject = "Maths"
        self.subjects = ["Maths", "English", "Science"]
        self.students = ["Ahmad", "Ubaid"]
        self.captures = []
        self.settings = {
            "two_page_mode": False,
            "boundary_detector": "Classic Contour (OpenCV)",
            "auto_capture": False,
            "free_capture_mode": False,
        }

    def get_frame(self):
        return np.full((120, 160, 3), 128, dtype=np.uint8)

    def capture(self):
        idx = len(self.captures) + 1
        name = f"{self.student_tag}_page_{idx}"
        self.captures.append({"name": name, "index": idx})
        return {"success": True, "count": len(self.captures), "name": name}

    def get_status(self):
        return {
            "student_tag": self.student_tag,
            "subject": self.subject,
            "subjects": self.subjects,
            "students": self.students,
            "system_date": "25 Aug",
            "active_profile": "Default",
            "capture_count": len(self.captures),
            "captures": self.captures,
            **self.settings,
        }

    def set_student_tag(self, tag: str):
        self.student_tag = tag
        return self.student_tag

    def set_subject(self, subject: str):
        self.subject = subject
        return self.subject

    def add_subject(self, subject: str):
        if subject not in self.subjects:
            self.subjects.append(subject)
        self.subject = subject
        return self.subjects

    def remove_subject(self, subject: str):
        if subject in self.subjects:
            self.subjects.remove(subject)
        return self.subjects

    def add_student(self, student: str):
        if student not in self.students:
            self.students.append(student)
        self.student_tag = student
        return self.students

    def remove_student(self, student: str):
        if student in self.students:
            self.students.remove(student)
        return self.students

    def set_settings(self, settings: dict):
        self.settings.update(settings)
        return self.settings

    def finalize_session(self):
        count = len(self.captures)
        self.captures = []
        return {"success": True, "exported_count": count, "pdf": "/fake/path.pdf"}

    def delete_capture(self, index: int):
        for i, c in enumerate(self.captures):
            if c["index"] == index:
                self.captures.pop(i)
                return True
        return False

    def get_thumbnail(self, index: int):
        return np.full((60, 80, 3), 200, dtype=np.uint8)


def test_remote_app_security_and_endpoints():
    bridge = MockBridge()
    sec_mgr = SessionSecurityManager()
    app = create_remote_app(bridge, security_manager=sec_mgr, allow_lan=True)
    client = TestClient(app)

    # 1. UI Root (HTML) is accessible
    res = client.get("/")
    assert res.status_code == 200
    assert "Neo Scanner Remote" in res.text

    # 2. Unauthenticated calls to /api/status or /api/capture fail with 401
    res = client.get("/api/status")
    assert res.status_code == 401

    res = client.post("/api/capture")
    assert res.status_code == 401

    # 3. Authenticate with PIN via /api/auth
    bad_auth = client.post("/api/auth", json={"pin": "000000"})
    assert bad_auth.status_code == 401

    good_auth = client.post("/api/auth", json={"pin": sec_mgr.pairing_pin})
    assert good_auth.status_code == 200
    auth_data = good_auth.json()
    assert auth_data["success"] is True
    token = auth_data["token"]
    assert token == sec_mgr.session_token

    headers = {"Authorization": f"Bearer {token}"}

    # 4. Authenticated Status
    res = client.get("/api/status", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["student_tag"] == "Init_Tag"
    assert data["subject"] == "Maths"
    assert "Maths" in data["subjects"]
    assert "Ahmad" in data["students"]
    assert data["system_date"] == "25 Aug"

    # 5. Snapshot & Feed with token query param
    res = client.get(f"/api/snapshot?token={token}")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"

    # 6. Manage Subjects (Add & Remove)
    res = client.post("/api/subjects", json={"action": "add", "subject": "History"}, headers=headers)
    assert res.status_code == 200
    assert "History" in res.json()["subjects"]
    assert bridge.subject == "History"

    res = client.post("/api/subjects", json={"action": "remove", "subject": "History"}, headers=headers)
    assert res.status_code == 200
    assert "History" not in res.json()["subjects"]

    # 7. Manage Students (Add & Remove)
    res = client.post("/api/students", json={"action": "add", "student": "Hamza"}, headers=headers)
    assert res.status_code == 200
    assert "Hamza" in res.json()["students"]
    assert bridge.student_tag == "Hamza"

    res = client.post("/api/students", json={"action": "remove", "student": "Hamza"}, headers=headers)
    assert res.status_code == 200
    assert "Hamza" not in res.json()["students"]

    # 8. Set Session (Subject & Student)
    res = client.post("/api/session", json={"subject": "Science", "student_tag": "Ubaid"}, headers=headers)
    assert res.status_code == 200
    assert bridge.subject == "Science"
    assert bridge.student_tag == "Ubaid"

    # 9. Capture, Thumbnail, Delete, Finalize
    res = client.post("/api/capture", headers=headers)
    assert res.status_code == 200
    cap_data = res.json()
    assert cap_data["success"] is True
    assert cap_data["count"] == 1

    res = client.get(f"/api/thumbnail/1?token={token}")
    assert res.status_code == 200

    res = client.delete("/api/captures/1", headers=headers)
    assert res.status_code == 200
    assert len(bridge.captures) == 0

    client.post("/api/capture", headers=headers)
    res = client.post("/api/finalize", headers=headers)
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert len(bridge.captures) == 0
