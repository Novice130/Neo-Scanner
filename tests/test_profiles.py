"""
Unit tests for multi-user profile management.
"""

from pathlib import Path
import pytest
from camscan.profiles import ProfileManager, UserProfile


def test_profile_manager_lifecycle(tmp_path: Path):
    cfg_file = tmp_path / "profiles.json"
    pm = ProfileManager(config_path=cfg_file)

    # 1. Default Profile
    assert pm.active_profile_name == "Default"
    assert "Default" in pm.get_profile_names()
    prof = pm.get_active_profile()
    assert "Maths" in prof.subjects
    assert "English" in prof.subjects
    assert "Ahmad" in prof.students

    # 2. Add New Profile
    prof2 = pm.add_profile("Teacher B", copy_from_active=True)
    assert pm.active_profile_name == "Teacher B"
    assert "Teacher B" in pm.get_profile_names()
    assert prof2.name == "Teacher B"

    # 3. Add & Remove Subjects
    subj_list = pm.add_subject("History")
    assert "History" in subj_list
    assert pm.get_active_profile().active_subject == "History"

    subj_list = pm.remove_subject("English")
    assert "English" not in subj_list
    assert "History" in subj_list

    # 4. Add & Remove Students
    stud_list = pm.add_student("Mustafa")
    assert "Mustafa" in stud_list
    assert pm.get_active_profile().active_student == "Mustafa"

    stud_list = pm.remove_student("Hamza")
    assert "Hamza" not in stud_list
    assert "Mustafa" in stud_list

    # 5. Switch Profile
    pm.switch_profile("Default")
    assert pm.active_profile_name == "Default"
    assert "English" in pm.get_active_profile().subjects
    assert "Mustafa" not in pm.get_active_profile().students

    # 6. Persistence across instances
    pm2 = ProfileManager(config_path=cfg_file)
    assert "Teacher B" in pm2.get_profile_names()
    assert pm2.active_profile_name == "Default"
    pm2.switch_profile("Teacher B")
    assert "History" in pm2.get_active_profile().subjects
    assert "Mustafa" in pm2.get_active_profile().students

    # 7. Delete Profile
    res = pm2.delete_profile("Teacher B")
    assert res is True
    assert "Teacher B" not in pm2.get_profile_names()
    assert pm2.active_profile_name == "Default"

    # Cannot delete last remaining profile
    res_last = pm2.delete_profile("Default")
    assert res_last is False
    assert len(pm2.get_profile_names()) == 1


def test_station_role_and_multi_host(tmp_path: Path):
    cfg_file = tmp_path / "profiles.json"
    pm = ProfileManager(config_path=cfg_file)

    # Default role is host
    assert pm.get_active_profile().station_role == "host"
    assert pm.get_active_profile().station_name == "Grade 6 Class"

    # 1. Switch to Client mode
    pm.set_station_role("client")
    assert pm.get_active_profile().station_role == "client"

    # 2. Manage multiple hosts (Grade 6 Class and Grade 7 Class)
    hosts = pm.get_saved_hosts()
    assert len(hosts) >= 2
    host_names = [h["name"] for h in hosts]
    assert "Grade 6 Class" in host_names
    assert "Grade 7 Class" in host_names

    # 3. Add Grade 8 Class host
    pm.add_saved_host("Grade 8 Class", "http://100.64.0.3:8000", pin="123456")
    updated_hosts = pm.get_saved_hosts()
    assert any(h["name"] == "Grade 8 Class" for h in updated_hosts)
    assert pm.get_active_host()["name"] == "Grade 8 Class"

    # 4. Switch active host back to Grade 6 Class
    active = pm.set_active_host("Grade 6 Class")
    assert active["name"] == "Grade 6 Class"
    assert pm.get_active_host()["name"] == "Grade 6 Class"

    # 5. Persistence
    pm2 = ProfileManager(config_path=cfg_file)
    assert pm2.get_active_profile().station_role == "client"
    assert pm2.get_active_host()["name"] == "Grade 6 Class"
    assert any(h["name"] == "Grade 8 Class" for h in pm2.get_saved_hosts())

    # 6. Remove a host
    pm2.remove_saved_host("Grade 8 Class")
    assert not any(h["name"] == "Grade 8 Class" for h in pm2.get_saved_hosts())

