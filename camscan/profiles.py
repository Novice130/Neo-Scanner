"""
Multi-user Profile and Settings Management for Neo Scanner.
Provides isolated, persistent settings for multiple users/teachers who use
the scanner sequentially.

Stores:
- Subjects (e.g. Maths, English, Science)
- Student roster for dropdowns (e.g. Ahmad, Ubaid)
- Hardware, camera, detector, auto-export, and Tailscale preferences.
"""

import json
import logging
import os
from pathlib import Path
import typing as t
from dataclasses import asdict, dataclass, field

try:
    import platformdirs

    CONFIG_DIR = Path(platformdirs.user_config_dir("NeoScanner", "Novice"))
except Exception:
    CONFIG_DIR = Path.home() / ".neo_scanner"

CONFIG_FILE = CONFIG_DIR / "profiles.json"
logger = logging.getLogger(__name__)

DEFAULT_SUBJECTS = ["Maths", "English", "Science"]
DEFAULT_STUDENTS = ["Ahmad", "Ubaid", "Hamza", "Zaid"]
DEFAULT_WATCHED_FOLDER = str(Path.home() / "OneDrive" / "CamScan")


@dataclass
class UserProfile:
    name: str = "Default"
    subjects: list[str] = field(default_factory=lambda: list(DEFAULT_SUBJECTS))
    active_subject: str = "Maths"
    students: list[str] = field(default_factory=lambda: list(DEFAULT_STUDENTS))
    active_student: str = ""
    watched_folder: str = DEFAULT_WATCHED_FOLDER
    remote_startup_action: str = "ask"  # 'ask', 'always', 'never'
    allow_lan_access: bool = True
    camera_index: int = 0
    camera_resolution: str = "Native"
    two_page_mode: bool = False
    free_capture_mode: bool = False
    boundary_detector: str = "Clean Perspective Crop (Recommended)"
    postprocessing_option: str = "Magic Color (CamScanner)"
    ocr_engine: str = "PaddleOCR Fast (Books & Documents)"
    auto_capture: bool = True
    motion_threshold: float = 3.0
    settle_time: float = 0.8
    ui_scaling: str = "100%"
    appearance_mode: str = "System"
    station_role: str = "host"  # 'host' (classroom scanner hub) or 'client' (teacher roaming controller)
    station_name: str = "Grade 6 Class"
    saved_hosts: list[dict] = field(
        default_factory=lambda: [
            {"name": "Grade 6 Class", "url": "http://127.0.0.1:8000", "pin": "", "token": ""},
            {"name": "Grade 7 Class", "url": "http://100.64.0.2:8000", "pin": "", "token": ""},
        ]
    )
    active_host_name: str = "Grade 6 Class"
    setup_completed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        # Filter only known fields
        known_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


class ProfileManager:
    """
    Manages loading, saving, creating, deleting, and switching user profiles.
    """

    def __init__(self, config_path: t.Optional[Path] = None):
        self.config_path = config_path or CONFIG_FILE
        self.profiles: dict[str, UserProfile] = {}
        self.active_profile_name: str = "Default"
        self.load()

    def load(self):
        """Load profiles from JSON file, initializing defaults if absent."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.active_profile_name = data.get("active_profile", "Default")
                raw_profiles = data.get("profiles", {})
                self.profiles = {}
                for name, p_data in raw_profiles.items():
                    self.profiles[name] = UserProfile.from_dict(p_data)
                if not self.profiles:
                    self._create_default_profile()
                elif self.active_profile_name not in self.profiles:
                    self.active_profile_name = next(iter(self.profiles.keys()))
                logger.info(
                    f"Loaded {len(self.profiles)} profile(s). Active: {self.active_profile_name}"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to read profiles from {self.config_path}: {e}")

        # Default fallback
        self._create_default_profile()
        self.save()

    def _create_default_profile(self):
        default_prof = UserProfile(name="Default")
        self.profiles = {"Default": default_prof}
        self.active_profile_name = "Default"

    def save(self):
        """Persist profiles to JSON file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "active_profile": self.active_profile_name,
                "profiles": {name: p.to_dict() for name, p in self.profiles.items()},
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved profiles to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save profiles to {self.config_path}: {e}")

    def get_profile_names(self) -> list[str]:
        return list(self.profiles.keys())

    def get_active_profile(self) -> UserProfile:
        if self.active_profile_name not in self.profiles:
            self._create_default_profile()
        return self.profiles[self.active_profile_name]

    def switch_profile(self, name: str) -> UserProfile:
        """Switch current active user profile."""
        if name in self.profiles:
            self.active_profile_name = name
            self.save()
        return self.get_active_profile()

    def add_profile(
        self, name: str, copy_from_active: bool = True
    ) -> UserProfile:
        """Create and switch to a new user profile."""
        name = name.strip()
        if not name:
            name = f"User_{len(self.profiles) + 1}"

        if name in self.profiles:
            # Already exists, just return
            return self.profiles[name]

        if copy_from_active:
            active = self.get_active_profile()
            new_prof = UserProfile.from_dict(active.to_dict())
            new_prof.name = name
        else:
            new_prof = UserProfile(name=name)

        self.profiles[name] = new_prof
        self.active_profile_name = name
        self.save()
        return new_prof

    def delete_profile(self, name: str) -> bool:
        """Delete user profile. Must keep at least one profile."""
        if len(self.profiles) <= 1 or name not in self.profiles:
            return False

        del self.profiles[name]
        if self.active_profile_name == name:
            self.active_profile_name = next(iter(self.profiles.keys()))
        self.save()
        return True

    def rename_profile(self, old_name: str, new_name: str) -> bool:
        """Rename an existing profile."""
        new_name = new_name.strip()
        if not new_name or old_name not in self.profiles or new_name in self.profiles:
            return False

        prof = self.profiles.pop(old_name)
        prof.name = new_name
        self.profiles[new_name] = prof
        if self.active_profile_name == old_name:
            self.active_profile_name = new_name
        self.save()
        return True

    # ---------------- Subject Helpers ----------------
    def add_subject(self, subject: str) -> list[str]:
        """Add main folder subject to active profile."""
        subject = subject.strip()
        if not subject:
            return self.get_active_profile().subjects
        prof = self.get_active_profile()
        if subject not in prof.subjects:
            prof.subjects.append(subject)
        prof.active_subject = subject
        self.save()
        return prof.subjects

    def remove_subject(self, subject: str) -> list[str]:
        """Remove subject from active profile."""
        prof = self.get_active_profile()
        if subject in prof.subjects:
            prof.subjects.remove(subject)
        if not prof.subjects:
            prof.subjects = ["General"]
        if prof.active_subject == subject:
            prof.active_subject = prof.subjects[0]
        self.save()
        return prof.subjects

    # ---------------- Student Helpers ----------------
    def add_student(self, student: str) -> list[str]:
        """Add student name to active profile roster."""
        student = student.strip()
        if not student:
            return self.get_active_profile().students
        prof = self.get_active_profile()
        if student not in prof.students:
            prof.students.append(student)
            prof.students.sort(key=lambda s: s.lower())
        prof.active_student = student
        self.save()
        return prof.students

    def remove_student(self, student: str) -> list[str]:
        """Remove student name from active profile roster."""
        prof = self.get_active_profile()
        if student in prof.students:
            prof.students.remove(student)
        if prof.active_student == student:
            prof.active_student = prof.students[0] if prof.students else ""
        self.save()
        return prof.students

    # ---------------- Station Role & Host Helpers ----------------
    def set_station_role(self, role: str) -> str:
        """Set active profile role: 'host' or 'client'."""
        prof = self.get_active_profile()
        prof.station_role = "client" if role.lower() == "client" else "host"
        self.save()
        return prof.station_role

    def set_station_name(self, name: str) -> str:
        """Set name identifying this station (e.g. 'Grade 6 Class')."""
        prof = self.get_active_profile()
        if name.strip():
            prof.station_name = name.strip()
            self.save()
        return prof.station_name

    def get_saved_hosts(self) -> list[dict]:
        """Return list of saved host stations for client connections."""
        prof = self.get_active_profile()
        return list(prof.saved_hosts)

    def add_saved_host(
        self, name: str, url: str, pin: str = "", token: str = ""
    ) -> list[dict]:
        """Add or update a host station for client connection."""
        name = name.strip()
        url = url.strip()
        if not name or not url:
            return self.get_saved_hosts()
        prof = self.get_active_profile()
        # Remove any existing entry with the same name
        prof.saved_hosts = [h for h in prof.saved_hosts if h.get("name") != name]
        prof.saved_hosts.append(
            {"name": name, "url": url, "pin": pin.strip(), "token": token.strip()}
        )
        prof.active_host_name = name
        self.save()
        return list(prof.saved_hosts)

    def remove_saved_host(self, name: str) -> list[dict]:
        """Remove a host station from saved hosts list."""
        prof = self.get_active_profile()
        prof.saved_hosts = [h for h in prof.saved_hosts if h.get("name") != name]
        if not prof.saved_hosts:
            prof.saved_hosts = [
                {
                    "name": "Default Station",
                    "url": "http://127.0.0.1:8000",
                    "pin": "",
                    "token": "",
                }
            ]
        if prof.active_host_name == name:
            prof.active_host_name = prof.saved_hosts[0]["name"]
        self.save()
        return list(prof.saved_hosts)

    def get_active_host(self) -> dict:
        """Get currently active host station configuration dictionary."""
        prof = self.get_active_profile()
        for h in prof.saved_hosts:
            if h.get("name") == prof.active_host_name:
                return dict(h)
        if prof.saved_hosts:
            prof.active_host_name = prof.saved_hosts[0].get("name", "Default")
            return dict(prof.saved_hosts[0])
        return {
            "name": "Default Station",
            "url": "http://127.0.0.1:8000",
            "pin": "",
            "token": "",
        }

    def set_active_host(self, name: str) -> dict:
        """Switch currently selected host station."""
        prof = self.get_active_profile()
        for h in prof.saved_hosts:
            if h.get("name") == name:
                prof.active_host_name = name
                self.save()
                return dict(h)
        return self.get_active_host()

