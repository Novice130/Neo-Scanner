"""
Unit tests for Tailscale detection, zero-trust security manager, and subnet filters.
"""

from camscan.tailscale import (
    is_tailscale_ip,
    is_loopback_ip,
    is_private_lan_ip,
    is_client_authorized_subnet,
    SessionSecurityManager,
    get_tailscale_info,
)


def test_is_tailscale_ip():
    # CGNAT 100.64.0.0/10
    assert is_tailscale_ip("100.64.0.1") is True
    assert is_tailscale_ip("100.90.76.116") is True
    assert is_tailscale_ip("100.127.255.254") is True
    # Non-tailscale
    assert is_tailscale_ip("192.168.1.1") is False
    assert is_tailscale_ip("10.0.0.1") is False
    assert is_tailscale_ip("127.0.0.1") is False
    assert is_tailscale_ip("8.8.8.8") is False
    assert is_tailscale_ip("invalid-ip") is False


def test_is_loopback_ip():
    assert is_loopback_ip("127.0.0.1") is True
    assert is_loopback_ip("::1") is True
    assert is_loopback_ip("192.168.1.1") is False
    assert is_loopback_ip("100.90.76.116") is False


def test_is_private_lan_ip():
    assert is_private_lan_ip("192.168.1.10") is True
    assert is_private_lan_ip("10.0.0.5") is True
    assert is_private_lan_ip("172.16.0.1") is True
    # Tailscale should NOT be categorized as standard private LAN
    assert is_private_lan_ip("100.90.76.116") is False
    assert is_private_lan_ip("8.8.8.8") is False


def test_is_client_authorized_subnet():
    # Loopback and Tailscale are always allowed
    assert is_client_authorized_subnet("127.0.0.1", allow_lan=False) is True
    assert is_client_authorized_subnet("100.90.76.116", allow_lan=False) is True

    # LAN allowed only when allow_lan=True
    assert is_client_authorized_subnet("192.168.1.50", allow_lan=False) is False
    assert is_client_authorized_subnet("192.168.1.50", allow_lan=True) is True

    # Public Internet IPs always rejected
    assert is_client_authorized_subnet("142.250.190.46", allow_lan=True) is False


def test_session_security_manager():
    mgr = SessionSecurityManager()
    assert len(mgr.session_token) > 16
    assert len(mgr.pairing_pin) == 6
    assert mgr.pairing_pin.isdigit()

    # Validation
    assert mgr.validate_pin(mgr.pairing_pin) is True
    assert mgr.validate_pin("000000") is False
    assert mgr.validate_pin("") is False
    assert mgr.validate_pin(None) is False

    assert mgr.validate_token(mgr.session_token) is True
    assert mgr.validate_token("wrong-token") is False
    assert mgr.validate_token(None) is False

    # Regeneration
    old_token = mgr.session_token
    old_pin = mgr.pairing_pin
    mgr.regenerate()
    assert mgr.session_token != old_token
    assert mgr.validate_token(old_token) is False
    assert mgr.validate_token(mgr.session_token) is True

    # QR code generation
    img = mgr.generate_qr_image("http://100.90.76.116:8000/?token=test")
    assert img is not None
    assert img.size[0] > 100


def test_get_tailscale_info():
    info = get_tailscale_info()
    assert info is not None
    assert isinstance(info.installed, bool)
    assert isinstance(info.running, bool)
