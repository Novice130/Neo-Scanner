# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = []
datas += collect_data_files("customtkinter")

# Collect model weights if present
import os
if os.path.exists("yolov8n-seg.pt"):
    datas.append(("yolov8n-seg.pt", "."))

hiddenimports = [
    "PIL",
    "PIL._imagingtk",
    "PIL._tkinter_finder",
    "customtkinter",
    "cv2",
    "numpy",
    "pymupdf",
    "fitz",
    "fastapi",
    "uvicorn",
    "starlette",
    "camscan",
    "camscan.app",
    "camscan.camera",
    "camscan.scanner",
    "camscan.postprocessing",
    "camscan.widgets",
    "camscan.ocr",
    "camscan.dewarp",
    "camscan.session",
    "camscan.motion",
    "camscan.auto_export",
    "camscan.remote",
    "utils",
]

# Collect submodules for heavy libraries
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("fastapi")

a = Analysis(
    ["camscan/main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Neo_Scanner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Neo_Scanner",
)

import sys

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Neo Scanner.app",
        icon=None,
        bundle_identifier="com.novice130.neo-scanner",
        info_plist={
            "NSCameraUsageDescription": "Neo Scanner requires camera access to scan documents.",
            "NSHighResolutionCapable": "True",
            "CFBundleShortVersionString": "1.0.0",
        },
    )
