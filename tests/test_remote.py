"""
Unit tests for camscan remote control API server.
"""

import numpy as np
import pytest
from starlette.testclient import TestClient

from camscan.remote import RemoteBridge, create_remote_app


class MockBridge(RemoteBridge):
    def __init__(self):
        self.student_tag = "Init_Tag"
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
            "capture_count": len(self.captures),
            "captures": self.captures,
            **self.settings,
        }

    def set_student_tag(self, tag: str):
        self.student_tag = tag
        return self.student_tag

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


def test_remote_app_endpoints():
    bridge = MockBridge()
    app = create_remote_app(bridge)
    client = TestClient(app)

    # 1. UI Root
    res = client.get("/")
    assert res.status_code == 200
    assert "Neo Scanner Remote" in res.text

    # 2. Status
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["student_tag"] == "Init_Tag"
    assert data["capture_count"] == 0

    # 3. Snapshot
    res = client.get("/api/snapshot")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"

    # 4. Set Session
    res = client.post("/api/session", json={"student_tag": "Student_42"})
    assert res.status_code == 200
    assert bridge.student_tag == "Student_42"

    # 5. Capture
    res = client.post("/api/capture")
    assert res.status_code == 200
    cap_data = res.json()
    assert cap_data["success"] is True
    assert cap_data["count"] == 1

    # 6. Thumbnail
    res = client.get("/api/thumbnail/1")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"

    # 7. Update Settings
    res = client.post("/api/settings", json={"two_page_mode": True})
    assert res.status_code == 200
    assert bridge.settings["two_page_mode"] is True

    # 8. Delete Capture
    res = client.delete("/api/captures/1")
    assert res.status_code == 200
    assert len(bridge.captures) == 0

    # 9. Finalize
    client.post("/api/capture")
    res = client.post("/api/finalize")
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert len(bridge.captures) == 0
