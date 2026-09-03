"""
Tests for student session tagging module.
"""

from datetime import datetime
from camscan.session import sanitize_tag, generate_session_filename, generate_session_dirname


def test_sanitize_tag():
    assert sanitize_tag("John Doe") == "John_Doe"
    assert sanitize_tag("Jane/Doe:123*") == "JaneDoe123"
    assert sanitize_tag("  Student-A_42  ") == "Student-A_42"
    assert sanitize_tag("") == ""


def test_generate_session_filename():
    fixed_time = datetime(2026, 9, 4, 14, 30, 0)
    # Tagged student
    filename = generate_session_filename("Alice Smith", ext="pdf", timestamp=fixed_time)
    assert filename == "Alice_Smith_20260904_143000.pdf"

    # Untagged
    filename_default = generate_session_filename("", ext="pdf", timestamp=fixed_time)
    assert filename_default == "captures_20260904_143000.pdf"


def test_generate_session_dirname():
    fixed_time = datetime(2026, 9, 4, 14, 30, 0)
    # Tagged student
    dirname = generate_session_dirname("Bob Jones", timestamp=fixed_time)
    assert dirname == "Bob_Jones_20260904_143000"

    # Untagged
    dirname_default = generate_session_dirname("", timestamp=fixed_time)
    assert dirname_default == "captures_20260904_143000"
