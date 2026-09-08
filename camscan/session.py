"""
Module for managing student session tagging and formatted directory/file naming.
"""

from datetime import datetime
import os
import re
import typing as t


def sanitize_tag(tag: str) -> str:
    """
    Sanitize a student name or ID for safe use in file and directory paths.
    Replaces spaces and invalid filename characters with underscores.
    """
    if not tag:
        return ""
    # Strip leading/trailing whitespace
    clean = tag.strip()
    # Replace spaces with underscores
    clean = re.sub(r"\s+", "_", clean)
    # Remove characters that are unsafe across Windows/macOS/Linux
    clean = re.sub(r'[^a-zA-Z0-9_\-.]', "", clean)
    return clean


def generate_session_filename(
    student_tag: str,
    ext: str = "pdf",
    timestamp: t.Optional[datetime] = None,
) -> str:
    """
    Generate output filename tagged with student ID/name and date.
    Example: 'JohnDoe_402_20260904_120530.pdf' or 'captures_20260904_120530.pdf'.
    """
    ts = timestamp or datetime.now()
    ts_str = ts.strftime(r"%Y%m%d_%H%M%S")
    clean = sanitize_tag(student_tag)

    ext_clean = ext.lstrip(".")
    if clean:
        return f"{clean}_{ts_str}.{ext_clean}"
    return f"captures_{ts_str}.{ext_clean}"


def generate_session_dirname(
    student_tag: str,
    timestamp: t.Optional[datetime] = None,
) -> str:
    """
    Generate output directory name tagged with student ID/name and date.
    Example: 'JohnDoe_402_20260904_120530' or 'captures_20260904_120530'.
    """
    ts = timestamp or datetime.now()
    ts_str = ts.strftime(r"%Y%m%d_%H%M%S")
    clean = sanitize_tag(student_tag)

    if clean:
        return f"{clean}_{ts_str}"
    return f"captures_{ts_str}"


def get_system_date_str(timestamp: t.Optional[datetime] = None) -> str:
    """
    Generate dynamic system date subfolder name.
    Example: '25 Aug' or '5 Sep' based on system clock.
    """
    ts = timestamp or datetime.now()
    return f"{ts.day} {ts.strftime('%b')}"


def build_session_export_dir(
    base_folder: str,
    subject: str = "General",
    date_str: t.Optional[str] = None,
    student_tag: str = "Untagged",
    timestamp: t.Optional[datetime] = None,
) -> str:
    """
    Build nested export directory path:
    {base_folder} / {Subject} / {Student} / {Date}/
    Example:
    '~/OneDrive/CamScan/Maths/Ahmad/25 Aug'
    """
    clean_base = os.path.expanduser(base_folder.strip()) if base_folder else "."
    clean_subject = sanitize_tag(subject) or "General"
    clean_date = date_str or get_system_date_str(timestamp)
    # Sanitize date characters while preserving spaces
    clean_date = re.sub(r'[^a-zA-Z0-9_\- ]', "", clean_date).strip() or "Today"
    clean_student = sanitize_tag(student_tag) or "Untagged"

    return os.path.join(clean_base, clean_subject, clean_student, clean_date)

