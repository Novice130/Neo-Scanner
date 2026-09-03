"""
Entry point for Neo Scanner.
"""

from camscan.app import CamScanApp


def main():
    app = CamScanApp()
    app.mainloop()


if __name__ == "__main__":
    main()
