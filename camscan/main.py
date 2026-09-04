import multiprocessing
import os
import sys

# On Windows, unconditionally suppress visible command prompt windows for any child subprocesses
if sys.platform == "win32":
    import subprocess

    _orig_popen = subprocess.Popen
    _CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    def _silent_popen(*args, **kwargs):
        creationflags = kwargs.get("creationflags", 0)
        kwargs["creationflags"] = creationflags | _CREATE_NO_WINDOW
        return _orig_popen(*args, **kwargs)

    subprocess.Popen = _silent_popen

from camscan.app import CamScanApp


def main():
    # Fix for PyInstaller on Windows: prevent child processes (spawned by PyTorch / YOLO)
    # from continuously creating new console windows and re-executing main()
    multiprocessing.freeze_support()

    # If running as a frozen executable on Windows with no console, redirect stdout and stderr
    # to avoid NoneType errors or conhost window creation from third-party libraries
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")

    app = CamScanApp()
    app.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

