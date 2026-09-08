"""
Tailscale integration and Zero-Trust Security Module for Neo Scanner.
Provides:
1. Tailscale discovery (CLI status, CGNAT 100.64.0.0/10 IP detection, MagicDNS).
2. Session Security Manager (pairing PIN, cryptographic token, zero-trust auth).
3. Subnet & IP filtering (Tailscale subnet, loopback, optional LAN).
4. Pairing QR code generator for instant mobile connection.
"""

import hmac
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import typing as t
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Tailscale IPv4 CGNAT subnet and IPv6 Unique Local Address space
TAILSCALE_IPV4_NETWORK = ipaddress.ip_network("100.64.0.0/10")
TAILSCALE_IPV6_NETWORK = ipaddress.ip_network("fd7a:115c:a1e0::/48")

# Windows flag to suppress console window creation
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


@dataclass
class TailscaleInfo:
    installed: bool = False
    running: bool = False
    ipv4: t.Optional[str] = None
    ipv6: t.Optional[str] = None
    magic_dns: t.Optional[str] = None
    backend_state: str = "Unknown"
    cli_path: t.Optional[str] = None
    error: t.Optional[str] = None


def find_tailscale_cli() -> t.Optional[str]:
    """Find tailscale executable across standard OS install locations."""
    # 1. Check system PATH
    found = shutil.which("tailscale")
    if found:
        return found

    # 2. Known standard paths
    candidates = []
    if sys.platform == "darwin":
        candidates = [
            "/usr/local/bin/tailscale",
            "/opt/homebrew/bin/tailscale",
            "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
            os.path.expanduser("~/Applications/Tailscale.app/Contents/MacOS/Tailscale"),
        ]
    elif sys.platform == "win32":
        prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        candidates = [
            os.path.join(prog_files, "Tailscale", "tailscale.exe"),
            os.path.join(prog_files_x86, "Tailscale", "tailscale.exe"),
            os.path.join(local_app_data, "Tailscale", "tailscale.exe"),
        ]
    else:  # Linux / Unix
        candidates = [
            "/usr/bin/tailscale",
            "/usr/local/bin/tailscale",
            "/snap/bin/tailscale",
        ]

    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return None


def get_tailscale_info() -> TailscaleInfo:
    """
    Query local Tailscale status and retrieve IPv4, MagicDNS, and state.
    """
    cli = find_tailscale_cli()
    if not cli:
        # Check if network interface has a 100.64.0.0/10 address even if CLI is not found
        cgnat_ip = find_cgnat_ip_from_socket()
        if cgnat_ip:
            return TailscaleInfo(
                installed=True,
                running=True,
                ipv4=cgnat_ip,
                backend_state="Running (Interface)",
            )
        return TailscaleInfo(installed=False, error="Tailscale CLI not found")

    try:
        kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": 3.0,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = _CREATE_NO_WINDOW

        res = subprocess.run([cli, "status", "--json"], **kwargs)
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            backend_state = data.get("BackendState", "Unknown")
            self_node = data.get("Self", {})
            ips = self_node.get("TailscaleIPs", [])
            dns = self_node.get("DNSName", "").rstrip(".")

            ipv4 = None
            ipv6 = None
            for ip_s in ips:
                try:
                    obj = ipaddress.ip_address(ip_s)
                    if obj.version == 4 and not ipv4:
                        ipv4 = ip_s
                    elif obj.version == 6 and not ipv6:
                        ipv6 = ip_s
                except ValueError:
                    pass

            is_running = backend_state.lower() == "running"
            return TailscaleInfo(
                installed=True,
                running=is_running,
                ipv4=ipv4,
                ipv6=ipv6,
                magic_dns=dns if dns else None,
                backend_state=backend_state,
                cli_path=cli,
            )
        else:
            return TailscaleInfo(
                installed=True,
                running=False,
                cli_path=cli,
                backend_state="Stopped",
                error=res.stderr.strip() or f"Return code {res.returncode}",
            )
    except Exception as e:
        logger.debug(f"Tailscale status query failed: {e}")
        # Fallback to interface check
        cgnat_ip = find_cgnat_ip_from_socket()
        if cgnat_ip:
            return TailscaleInfo(
                installed=True,
                running=True,
                ipv4=cgnat_ip,
                cli_path=cli,
                backend_state="Running (Interface Fallback)",
            )
        return TailscaleInfo(
            installed=True,
            running=False,
            cli_path=cli,
            error=str(e),
        )


def find_cgnat_ip_from_socket() -> t.Optional[str]:
    """Scan local interfaces for a Tailscale CGNAT (100.64.0.0/10) IP."""
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if is_tailscale_ip(ip):
                return ip
    except Exception:
        pass
    return None


def is_tailscale_ip(ip_str: str) -> bool:
    """Return True if ip_str belongs to Tailscale address space."""
    try:
        ip = ipaddress.ip_address(ip_str.strip())
        if ip.version == 4:
            return ip in TAILSCALE_IPV4_NETWORK
        elif ip.version == 6:
            return ip in TAILSCALE_IPV6_NETWORK
    except ValueError:
        pass
    return False


def is_loopback_ip(ip_str: str) -> bool:
    """Return True if ip_str is local loopback (127.0.0.1, ::1, localhost, testclient)."""
    if not ip_str:
        return False
    clean = ip_str.strip().lower()
    if clean in ("127.0.0.1", "::1", "localhost", "testclient"):
        return True
    try:
        ip = ipaddress.ip_address(clean)
        return ip.is_loopback
    except ValueError:
        pass
    return False


def is_private_lan_ip(ip_str: str) -> bool:
    """Return True if ip_str is standard private LAN (RFC1918)."""
    try:
        ip = ipaddress.ip_address(ip_str.strip())
        return ip.is_private and not is_tailscale_ip(ip_str)
    except ValueError:
        pass
    return False


def is_client_authorized_subnet(
    client_ip: str, allow_lan: bool = False
) -> bool:
    """
    Check if incoming connection is from an authorized subnet:
    Always allows Tailscale and Localhost. Only allows LAN if explicitly enabled.
    """
    if is_loopback_ip(client_ip) or is_tailscale_ip(client_ip):
        return True
    if allow_lan and is_private_lan_ip(client_ip):
        return True
    return False


class SessionSecurityManager:
    """
    Zero-Trust Session Authenticator:
    Generates cryptographically random pairing tokens and human-friendly PINs.
    Prevents unauthorized access from other nodes on the network/tailnet.
    """

    def __init__(self):
        self.session_token: str = secrets.token_urlsafe(24)
        # 6-digit numeric pairing PIN for manual typing on mobile
        self.pairing_pin: str = f"{secrets.randbelow(900000) + 100000}"
        self.created_at = os.times().elapsed

    def regenerate(self):
        """Regenerate token and PIN (e.g. on new session start)."""
        self.session_token = secrets.token_urlsafe(24)
        self.pairing_pin = f"{secrets.randbelow(900000) + 100000}"

    def validate_token(self, token: t.Optional[str]) -> bool:
        """Validate token constant-time."""
        if not token:
            return False
        return hmac.compare_digest(self.session_token, token.strip())

    def validate_pin(self, pin: t.Optional[str]) -> bool:
        """Validate 6-digit PIN constant-time."""
        if not pin:
            return False
        return hmac.compare_digest(self.pairing_pin, pin.strip())

    def generate_qr_image(self, url: str):
        """
        Generate PIL Image containing QR code for given URL.
        Requires qrcode package.
        """
        try:
            import qrcode

            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=8,
                border=3,
            )
            qr.add_data(url)
            qr.make(fit=True)
            return qr.make_image(fill_color="black", back_color="white")
        except ImportError:
            logger.warning("qrcode package is not installed.")
            return None
