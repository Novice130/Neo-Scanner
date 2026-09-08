"""
This is an application used for scanning documents using a camera connected to
your computer, like your webcam. This module specifically implements the GUI
part of the application, as well as the code used to handle, post process, and
export the captured images.
"""

from datetime import datetime
import functools
import logging
import os
import re
import threading
import time
import typing as t

import customtkinter as ctk
import cv2
import numpy as np
import PIL
import tkinter as tk
from tkinter import simpledialog, messagebox

from camscan import postprocessing, widgets
from camscan.camera import Camera
from camscan import (
    scanner,
    ocr,
    pdf_builder,
    dewarp,
    session,
    motion,
    auto_export,
    remote,
    tailscale,
    profiles,
)
from camscan import __app_display_name__, __version__
import utils

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

# Define the window title
WINDOW_TITLE = f"{__app_display_name__} {__version__}"

# Define the initial application window size
WINDOW_WIDTH = 1536
WINDOW_HEIGHT = 864

# Define the to wait before updating the camera feed (20ms)
CAMERA_FEED_WAIT_MS = 20

# Define constants related to the styling of widgets in the GUI
LEFT_MENU_PAD_X = 20
LEFT_MENU_PAD_Y = 5
RIGHT_MENU_PAD_X = 10
RIGHT_MENU_PAD_Y = 5
LEFT_MENU_PACK_KWARGS = dict(padx=LEFT_MENU_PAD_X, pady=LEFT_MENU_PAD_Y)
RIGHT_MENU_PACK_KWARGS = dict(padx=RIGHT_MENU_PAD_X, pady=RIGHT_MENU_PAD_Y)

# Keybind used to capture images with the cameras
CAPTURE_KEYBIND = "<space>"

# Specify supported file formats when exporting images as separate files.
# See the OpenCV documentation for more information on the supported file types:
# https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
EXPORT_SEPARATE_FILE_TYPES = [
    "png",
    "bmp",
    "dib",
    "jpeg",
    "jpg",
    "jpe",
    "jp2",
    "webp",
    "pbm",
    "pgm",
    "ppm",
    "pxm",
    "pnm",
    "sr",
    "ras",
    "tiff",
    "tif",
    "exr",
    "hdr",
    "pic",
]

# Specify supported file formats when exporting images as a single merged file
EXPORT_MERGED_FILE_TYPES = [
    "pdf",
]

# Supported OCR engines for searchable PDF export
OCR_OPTIONS = [
    "PaddleOCR Fast (Books & Documents)",
    "PaddleOCR + TrOCR (Handwriting)",
    "None (Instant PDF Export)",
    "Vision LLM API",
]

# Specify the supported postprocessing functions for the captured images
POSTPROCESSING_OPTIONS = {
    "Magic Color (CamScanner)": postprocessing.magic_color,
    "None (Raw Natural)": postprocessing.dummy,
    "Sharpen": postprocessing.sharpen,
    "Grayscale": postprocessing.grayscale,
    "Black and White": postprocessing.black_and_white,
}

# Define the list of pre-defined camera resolutions. In addition to these, the
# user can also enter custom resolutions manually.
RESOLUTIONS = [
    "3264x2448",
    "3264x1836",
    "2592x1944",
    "2048x1536",
    "1920x1080",
    "1600x1200",
    "1280x720",
    "1024x768",
    "800x600",
    "640x480",
]

# Supported boundary detection and dewarping algorithms
BOUNDARY_DETECTION_OPTIONS = [
    "Clean Perspective Crop (Recommended)",
    "Classic Contour (OpenCV)",
]

# Collection of tooltip strings shown for various widgets
TOOLTIPS = {
    # Left panel
    "camera_configuration": (
        "Open camera configuration for selecting camera and resolution"
    ),
    "camera_driver_settings": (
        "Open camera driver settings dialog (determined by the selected camera)"
    ),
    "boundary_detector": (
        "Select document boundary detector: YOLOv8 with cubic polynomial dewarping "
        "for curved notebook spines, or Classic OpenCV contour detection."
    ),
    "student_tag": (
        "Enter student name or ID to tag captures and group exported files/folders "
        "by student and date."
    ),
    "auto_capture": (
        "Automatically trigger capture when a page turn finishes and the document settles"
    ),
    "motion_threshold": (
        "Sensitivity threshold for detecting page turns (percentage of frame changed)"
    ),
    "settle_time": (
        "Window in seconds motion must stay below threshold before capture triggers"
    ),
    "postprocessing": "Set the postprocessing effect applied to the captured images",
    "system_appearance": "Set the user interface appearance of the application",
    "system_ui_scaling": "Set the user interface scale of the application",
    "free_capture_mode": (
        "Ignore the document detection algorithm and capture the entire image"
    ),
    "two_page_mode": "Split the captured image into equal left and right parts",
    "capture": (
        f"Capture an image and save to the captures pane (key bind {CAPTURE_KEYBIND})"
    ),
    "export_separate": "Export captures as separate files in a directory",
    "export_merged": "Export captures as a single merged file",
    "ocr_engine": "Choose OCR engine to create searchable handwriting text layer in exported PDF",
    "watched_folder": "Directory path to auto-export finalized student sessions (e.g. OneDrive sync folder)",
    "browse_watched_folder": "Browse and select destination watched folder",
    "finalize_session": "Finalize current student session and auto-export to watched folder",
    "remote_server": (
        "Enable remote control web server accessible from phone browser over Tailscale (port 8000)"
    ),
    # Right panel
    "select_all": "Select or deselect all captures",
    "delete": "Delete the selected captures",
    # Camera Configuration Window
    "camera_index": (
        "Select a camera by choosing its device index. Update this list with available"
        " devices using the camera identification button."
    ),
    "identify_cameras": (
        "Identify available cameras on the system and populate the camera index list"
    ),
    "camera_resolution": "Set the camera resolution from a preset list of resolutions",
    "custom_camera_resolution": (
        "Set a custom camera resolution using a string on the form <width>x<height>"
    ),
}


def opencv_to_pil_image(
    image: cv2.Mat,
    width: int = None,
    height: int = None,
) -> PIL.Image:
    """
    Given an OpenCV image, convert to to a PIL image. The function also supports
    resizing the image while keeping its original aspect ratio.
    :param image: The input OpenCV image
    :param width: Optional width to scale the image to
    :param width: Optional height to scale the image to
    :return: The image converted to a PIL image
    """
    return PIL.Image.fromarray(
        utils.resize_with_aspect_ratio(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            width=width,
            height=height,
        )
    )


def opencv_to_ctk_image(
    image: cv2.Mat,
    width: int = None,
    height: int = None,
) -> ctk.CTkImage:
    """
    Given an OpenCV image, convert to to a CTkImage. The function also supports
    resizing the image while keeping its original aspect ratio.
    :param image: The input OpenCV image
    :param width: Optional width to scale the image to
    :param width: Optional height to scale the image to
    :return: The image converted to a CTkImage
    """
    pil_image = opencv_to_pil_image(image=image, width=width, height=height)
    return ctk.CTkImage(
        pil_image,
        size=(pil_image.width, pil_image.height),
    )


class CaptureEntry:
    """
    Helper class for keeping track of the captured images. This class both
    contains the original OpenCV image, as well as the GUI element called an
    'Entry' which is comprised of several underlying widgets.
    :param image: The original OpenCV image capture from the camera
    :param name: A name given to the image which is displayed in the Entry
    :param index: The index number shown in the Entry
    :param master: The parent widget containing the Entry
    :param move_entry: A function used to move this entry up or down in the list
    """

    def __init__(
        self,
        image: cv2.Mat,
        name: str,
        index: int,
        master: ctk.CTkBaseClass,
        move_entry: t.Callable,
    ):
        self.var_selected = tk.IntVar(value=0)
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.name = name
        self.frame = ctk.CTkFrame(master=master)
        self.frame.grid(row=index, column=0, padx=5, pady=5, sticky="nsew")

        self.move_up_button = ctk.CTkButton(
            master=self.frame,
            text="🔼",
            width=24,
            height=24,
            font=ctk.CTkFont(size=24),
            fg_color="transparent",
            command=functools.partial(move_entry, self, -1),
        )
        self.selection_checkbox = ctk.CTkCheckBox(
            master=self.frame,
            text=None,
            checkbox_width=24,
            checkbox_height=24,
            width=24,
            height=24,
            variable=self.var_selected,
        )
        self.move_down_button = ctk.CTkButton(
            master=self.frame,
            text="🔽",
            width=24,
            height=24,
            font=ctk.CTkFont(size=24),
            fg_color="transparent",
            command=functools.partial(move_entry, self, 1),
        )
        self.image_widget = ctk.CTkButton(
            master=self.frame,
            fg_color="transparent",
            text=None,
            command=self.open_image_viewer_window,
        )
        self.index_label = ctk.CTkLabel(
            master=self.frame,
            text=str(index),
        )
        self.name_label = ctk.CTkLabel(
            master=self.frame,
            text=self.name,
        )
        self.index_label.grid(row=0, column=0, padx=5, pady=(5, 0), sticky="nsew")
        self.name_label.grid(row=0, column=1, padx=5, pady=(5, 0), sticky="nsew")
        self.move_up_button.grid(row=1, column=0, padx=5, pady=5)
        # The checkbox needs slightly more padding on the left to be aligned
        # with the up/down buttons in the same column
        self.selection_checkbox.grid(row=2, column=0, padx=(12, 5), pady=5)
        self.move_down_button.grid(row=3, column=0, padx=5, pady=5)
        self.image_widget.grid(
            row=1, column=1, rowspan=3, padx=5, pady=(0, 5), sticky="nsew"
        )

        self.set_current_image(image=image)

    def set_current_image(self, image: cv2.Mat):
        """
        Update the current displayed image of this Entry. This will not modify
        the original OpenCV image stored in this object. This will also update
        the smaller thumbnail picture used in the Entry.
        :param image: The new OpenCV image to set the current displayed image to
        """
        self.current_image = image.copy()
        thumbnail_image = opencv_to_ctk_image(image=image, width=230, height=400)
        self.image_widget.photo = thumbnail_image
        self.image_widget.configure(image=thumbnail_image)

    def open_image_viewer_window(self):
        """
        Open an image viewer window displaying the current image of this Entry.
        """
        window = ctk.CTkToplevel()
        window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        window.title(self.name)

        frame_widget = ctk.CTkFrame(master=window)
        image_widget = ctk.CTkLabel(master=frame_widget, text=None)

        # The current window size. Used to keep track of when it changes
        current_size = [0, 0]

        def _resize_image():
            """Resize the image to fill up the frame in the window"""
            max_width = frame_widget.winfo_width()
            max_height = frame_widget.winfo_height()

            # At startup, this area might be of size zero. If so, try later
            if not (max_width > 1 and max_height > 1):
                return

            # Convert the OpenCV image to a CTkImage to display in the widget
            new_image = opencv_to_ctk_image(
                image=self.current_image, width=max_width, height=max_height
            )
            image_widget.photo = new_image
            image_widget.configure(image=new_image)

        def _on_window_resize(event):
            # We need to make sure that the only widget that is allowed to
            # trigger the image resizing is the window itself. Otherwise, when
            # we update the image size, the image widget itself will generate
            # a 'Configure' event which will trigger this function again. That
            # will lead to an endless stream of events.
            if event.widget == window:
                return
            # We are only interested in updating the image size if the window
            # changes size. This event is also triggered when the window moves,
            # so we keep track of the current window size and compare to the new
            # one and update only if it changes.
            new_size = [event.width, event.height]
            if current_size == new_size:
                return
            # Modify values inplace to keep the reference intact
            current_size[:] = new_size[:]
            # Make sure that the containing frame widget has been updated
            # since it is that size which determines the maximum size of
            # the displayed image inside. In some cases, like when the user
            # expands the window to full-screen, this frame might not have
            # enough time to update before we attempt to update time image
            # within. Then it is not possible to fully expand the image to
            # the entire frame size. By manually calling the update here we
            # can ensure that the frame is the maximum size first.
            frame_widget.update()
            _resize_image()

        # Pack the widgets
        frame_widget.pack(fill=ctk.BOTH, expand=True)
        image_widget.pack()
        # Bind an event to when the window changes size to resize the image
        window.bind("<Configure>", _on_window_resize)
        # Make sure this window is on top of the main window
        window.lift()
        window.attributes("-topmost", True)
        # It can take some time for the window to set up its widgets and get its
        # proper size. Therefore, wait a little bit before displaying the image.
        window.after(ms=100, func=_resize_image)


class CamScanApp(ctk.CTk):
    """
    Application class for CamScan. This defines a CTk Window object containing
    the entire GUI of the application, as well as supporting code.

    Example usage:
        app = CameraScannerApp()
        app.mainloop()
    """

    def __init__(self):
        super().__init__()

        self.camera = Camera()
        self.entries = []

        # Multi-user profile management
        self.profile_manager = profiles.ProfileManager()
        self.active_profile = self.profile_manager.get_active_profile()
        self.var_active_profile = tk.StringVar(value=self.active_profile.name)
        self.var_subject = tk.StringVar(value=self.active_profile.active_subject)
        self.var_student_tag = tk.StringVar(value=self.active_profile.active_student)

        self.var_postprocessing_option = tk.StringVar(
            value=self.active_profile.postprocessing_option
            if self.active_profile.postprocessing_option in POSTPROCESSING_OPTIONS
            else list(POSTPROCESSING_OPTIONS.keys())[0]
        )
        self.var_two_page_mode = tk.IntVar(
            value=1 if self.active_profile.two_page_mode else 0
        )
        self.var_free_capture_mode = tk.IntVar(
            value=1 if self.active_profile.free_capture_mode else 0
        )
        self.var_select_all_captures = tk.IntVar(value=0)
        self.var_merged_captures_file_type = tk.StringVar(
            value=EXPORT_MERGED_FILE_TYPES[0]
        )
        self.var_separate_captures_file_type = tk.StringVar(
            value=EXPORT_SEPARATE_FILE_TYPES[0]
        )
        self.var_ocr_engine = tk.StringVar(
            value=self.active_profile.ocr_engine
            if self.active_profile.ocr_engine in OCR_OPTIONS
            else OCR_OPTIONS[0]
        )
        self.yolo_dewarp_engine = dewarp.YOLODewarpEngine()
        self.var_boundary_detector = tk.StringVar(
            value=self.active_profile.boundary_detector
            if self.active_profile.boundary_detector in BOUNDARY_DETECTION_OPTIONS
            else BOUNDARY_DETECTION_OPTIONS[0]
        )
        self.page_turn_detector = motion.PageTurnDetector(
            motion_threshold=self.active_profile.motion_threshold,
            settle_time_s=self.active_profile.settle_time,
            cooldown_s=2.0,
        )
        self._session_active: bool = False
        self.var_auto_capture = tk.IntVar(
            value=1 if getattr(self.active_profile, "auto_capture", True) else 0
        )
        self.var_motion_threshold = tk.StringVar(
            value=str(self.active_profile.motion_threshold)
        )
        self.var_settle_time = tk.StringVar(value=str(self.active_profile.settle_time))
        self.auto_exporter = auto_export.AutoExporter(
            watched_folder=self.active_profile.watched_folder
        )
        self.var_watched_folder = tk.StringVar(
            value=self.auto_exporter.watched_folder
        )

        # Remote Control Server (Tailscale / Phone access with Zero-Trust Security)
        self._latest_preview_frame = None
        self.remote_bridge = remote.AppBridge(self)
        self.remote_security_manager = tailscale.SessionSecurityManager()
        self.remote_server = remote.RemoteServerManager(
            self.remote_bridge,
            host="0.0.0.0",
            port=8000,
            security_manager=self.remote_security_manager,
            allow_lan=self.active_profile.allow_lan_access,
        )
        init_remote = 1 if self.active_profile.remote_startup_action == "always" else 0
        self.var_remote_server = tk.IntVar(value=init_remote)

        # Station Role and Host-Client configuration
        self.var_station_role = tk.StringVar(
            value=getattr(self.active_profile, "station_role", "host")
        )
        self.var_station_name = tk.StringVar(
            value=getattr(self.active_profile, "station_name", "Grade 6 Class")
        )
        self.var_active_host = tk.StringVar(
            value=getattr(self.active_profile, "active_host_name", "Grade 6 Class")
        )
        self.var_client_status = tk.StringVar(value="Ready")
        self.host_client = None

        # configure window
        self.title(WINDOW_TITLE)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        # Configure the grid layout
        self.grid_columnconfigure((0, 2), weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Configure the left sidebar (Scrollable so all controls fit and scroll smoothly)
        self.left_sidebar_frame = ctk.CTkScrollableFrame(self, width=280, corner_radius=0)

        # Add a label to the top of the sidebar
        self.left_sidebar_title_label = ctk.CTkLabel(
            self.left_sidebar_frame,
            text="Neo Scanner 📄",
            font=ctk.CTkFont(size=20, weight="bold"),
        )

        # Add dropdown for directly selecting active camera device by name
        self.camera_selection_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Active Camera:", anchor="w"
        )
        camera_names = (
            [f"{name} ({idx})" for idx, name in getattr(self.camera, "available_devices", [])]
            if getattr(self.camera, "available_devices", None)
            else ["Default Camera (0)"]
        )
        current_cam_display = f"{self.camera.device_name} ({self.camera.index})"
        self.var_camera_device = tk.StringVar(
            value=current_cam_display if current_cam_display in camera_names else camera_names[0]
        )
        self.camera_selection_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=camera_names,
            command=self.change_camera_event,
            variable=self.var_camera_device,
        )

        # Add button for resolution / advanced camera settings
        self.camera_settings_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Resolution & Advanced:", anchor="w"
        )
        self.configure_camera_button = ctk.CTkButton(
            self.left_sidebar_frame,
            text="Resolution Settings",
            command=self.configure_camera_event,
        )
        self.camera_settings_button = ctk.CTkButton(
            self.left_sidebar_frame,
            text="Camera Controls & Focus",
            command=lambda: self.camera.show_settings(self),
        )

        # Add a menu for the color settings
        self.postprocessing_menu_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Postprocessing:", anchor="w"
        )
        self.postprocessing_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=list(POSTPROCESSING_OPTIONS.keys()),
            command=self.change_postprocessing_event,
            variable=self.var_postprocessing_option,
        )

        # Add a menu for the application UI appearance
        self.appearance_mode_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Appearance Mode:", anchor="w"
        )
        self.appearance_mode_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=["System", "Dark", "Light"],
            command=change_ui_appearance_event,
        )
        self.appearance_mode_option_menu.set("System")

        # Add a menu for the application UI scaling
        self.scaling_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="UI Scaling:", anchor="w"
        )
        self.scaling_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=["80%", "90%", "100%", "110%", "120%"],
            command=change_ui_scaling_event,
        )
        self.scaling_option_menu.set("100%")

        # Station Role banner and Setup trigger
        self.station_banner_frame = ctk.CTkFrame(
            self.left_sidebar_frame, fg_color=("#e0e0e0", "#2b2b2b"), corner_radius=8
        )
        self.station_role_label = ctk.CTkLabel(
            self.station_banner_frame,
            text=f"🖥️ Host: {self.var_station_name.get()}",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.station_setup_btn = ctk.CTkButton(
            self.station_banner_frame,
            text="⚙️ Setup",
            width=65,
            height=24,
            command=self.prompt_setup_station_dialog,
        )
        self.station_role_label.pack(side=ctk.LEFT, padx=8, pady=4)
        self.station_setup_btn.pack(side=ctk.RIGHT, padx=6, pady=4)
        self.station_banner_frame.pack(fill=ctk.X, pady=(4, 6))

        # Classroom Host Selection (for Client Mode)
        self.client_host_frame = ctk.CTkFrame(self.left_sidebar_frame, fg_color="transparent")
        self.client_host_label = ctk.CTkLabel(
            self.client_host_frame,
            text="🏫 Host Station:",
            anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.client_host_subframe = ctk.CTkFrame(self.client_host_frame, fg_color="transparent")
        self.host_option_menu = ctk.CTkOptionMenu(
            self.client_host_subframe,
            values=[h.get("name", "Station") for h in self.profile_manager.get_saved_hosts()],
            variable=self.var_active_host,
            command=self.on_host_selected,
            width=135,
        )
        self.add_host_btn = ctk.CTkButton(
            self.client_host_subframe, text="+", width=28, command=self.prompt_add_host
        )
        self.del_host_btn = ctk.CTkButton(
            self.client_host_subframe, text="-", width=28, command=self.prompt_remove_host
        )
        self.client_status_label = ctk.CTkLabel(
            self.client_host_frame, text="⚪ Ready", font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.remote_capture_btn = ctk.CTkButton(
            self.client_host_frame,
            text="📸 Remote Capture",
            fg_color="#2e7d32",
            hover_color="#1b5e20",
            command=self.trigger_remote_host_capture,
        )
        self.remote_finalize_btn = ctk.CTkButton(
            self.client_host_frame,
            text="📁 Finish & Export Session",
            fg_color="#1565c0",
            hover_color="#0d47a1",
            command=self.finalize_remote_host_session,
        )

        self.client_host_label.pack(fill=ctk.X, pady=(2, 2))
        self.host_option_menu.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(0, 4))
        self.add_host_btn.pack(side=ctk.LEFT, padx=(0, 2))
        self.del_host_btn.pack(side=ctk.LEFT)
        self.client_host_subframe.pack(fill=ctk.X, pady=(0, 4))
        self.client_status_label.pack(fill=ctk.X, pady=(0, 4))
        self.remote_capture_btn.pack(fill=ctk.X, pady=(2, 2))
        self.remote_finalize_btn.pack(fill=ctk.X, pady=(2, 6))

        # Multi-user profile selector widgets
        self.user_profile_frame = ctk.CTkFrame(self.left_sidebar_frame, fg_color="transparent")
        self.user_profile_label = ctk.CTkLabel(
            self.user_profile_frame, text="👤 User:", font=ctk.CTkFont(size=12, weight="bold")
        )
        self.user_profile_option_menu = ctk.CTkOptionMenu(
            self.user_profile_frame,
            values=self.profile_manager.get_profile_names(),
            variable=self.var_active_profile,
            command=self.switch_user_profile,
            width=135,
        )
        self.add_profile_button = ctk.CTkButton(
            self.user_profile_frame,
            text="+",
            width=28,
            command=self.prompt_add_profile,
        )
        self.del_profile_button = ctk.CTkButton(
            self.user_profile_frame,
            text="🗑️",
            width=28,
            fg_color="#c62828",
            hover_color="#b71c1c",
            command=self.prompt_delete_profile,
        )
        self.user_profile_label.pack(side=ctk.LEFT, padx=(0, 4))
        self.user_profile_option_menu.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(0, 4))
        self.add_profile_button.pack(side=ctk.LEFT, padx=(0, 2))
        self.del_profile_button.pack(side=ctk.LEFT)

        # Main Folder (Subject)
        self.subject_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Main Folder (Subject):", anchor="w"
        )
        self.subject_frame = ctk.CTkFrame(self.left_sidebar_frame, fg_color="transparent")
        self.subject_combobox = ctk.CTkComboBox(
            self.subject_frame,
            values=self.active_profile.subjects,
            variable=self.var_subject,
            command=self.on_subject_selected,
        )
        self.add_subject_button = ctk.CTkButton(
            self.subject_frame, text="+", width=28, command=self.prompt_add_subject
        )
        self.del_subject_button = ctk.CTkButton(
            self.subject_frame, text="-", width=28, command=self.prompt_remove_subject
        )
        self.subject_combobox.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(0, 4))
        self.add_subject_button.pack(side=ctk.LEFT, padx=(0, 2))
        self.del_subject_button.pack(side=ctk.LEFT)

        # Dynamic System Date Subfolder
        self.system_date_label = ctk.CTkLabel(
            self.left_sidebar_frame,
            text=f"📅 Date Subfolder: {session.get_system_date_str()} (Auto)",
            font=ctk.CTkFont(size=11),
            text_color="#4caf50",
            anchor="w",
        )

        # Student Subfolder dropdown + text entry
        self.student_tag_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Student Subfolder:", anchor="w"
        )
        self.student_frame = ctk.CTkFrame(self.left_sidebar_frame, fg_color="transparent")
        self.student_combobox = ctk.CTkComboBox(
            self.student_frame,
            values=self.active_profile.students,
            variable=self.var_student_tag,
            command=self.on_student_selected,
        )
        self.add_student_button = ctk.CTkButton(
            self.student_frame, text="+", width=28, command=self.prompt_add_student
        )
        self.del_student_button = ctk.CTkButton(
            self.student_frame, text="-", width=28, command=self.prompt_remove_student
        )
        self.student_combobox.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(0, 4))
        self.add_student_button.pack(side=ctk.LEFT, padx=(0, 2))
        self.del_student_button.pack(side=ctk.LEFT)

        # Live export destination preview
        self.path_preview_label = ctk.CTkLabel(
            self.left_sidebar_frame,
            text=self._get_path_preview_text(),
            font=ctk.CTkFont(size=10),
            text_color="#64b5f6",
            wraplength=240,
            justify="left",
            anchor="w",
        )

        # Traces for reactive path preview
        self.var_subject.trace_add("write", self._update_path_preview)
        self.var_student_tag.trace_add("write", self._update_path_preview)
        self.var_watched_folder.trace_add("write", self._update_path_preview)

        # Remote server controls
        self.remote_server_check_box = ctk.CTkCheckBox(
            self.left_sidebar_frame,
            text="Remote Session",
            variable=self.var_remote_server,
            command=self.toggle_remote_server,
        )
        self.remote_url_label = ctk.CTkLabel(
            self.left_sidebar_frame,
            text=self._get_remote_status_text(),
            font=ctk.CTkFont(size=10),
            text_color="#2196f3",
            wraplength=240,
            justify="left",
        )
        self.remote_pin_label = ctk.CTkLabel(
            self.left_sidebar_frame,
            text=f"🔑 PIN: {self.remote_server.get_pairing_pin()}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#ffb74d",
            anchor="w",
        )
        self.remote_btn_frame = ctk.CTkFrame(self.left_sidebar_frame, fg_color="transparent")
        self.btn_show_qr = ctk.CTkButton(
            self.remote_btn_frame,
            text="📱 QR Code",
            height=28,
            font=ctk.CTkFont(size=11),
            command=self.show_pairing_qr_dialog,
        )
        self.btn_copy_link = ctk.CTkButton(
            self.remote_btn_frame,
            text="📋 Copy Link",
            height=28,
            font=ctk.CTkFont(size=11),
            command=self.copy_remote_link,
        )
        self.btn_show_qr.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(0, 4))
        self.btn_copy_link.pack(side=ctk.LEFT, fill=ctk.X, expand=True)

        # Add boundary detector selection
        self.boundary_detector_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Boundary Detector:", anchor="w"
        )
        self.boundary_detector_option_menu = ctk.CTkOptionMenu(
            self.left_sidebar_frame,
            values=BOUNDARY_DETECTION_OPTIONS,
            variable=self.var_boundary_detector,
        )

        # Add a button for capturing the screen
        self.capture_image_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Capture Image", anchor="w"
        )
        self.two_page_setting_check_box = ctk.CTkCheckBox(
            self.left_sidebar_frame,
            text="Two-page Mode",
            variable=self.var_two_page_mode,
        )
        self.free_capture_setting_check_box = ctk.CTkCheckBox(
            self.left_sidebar_frame,
            text="Free Capture Mode",
            variable=self.var_free_capture_mode,
        )
        # Auto-capture on page turn controls
        self.auto_capture_check_box = ctk.CTkCheckBox(
            self.left_sidebar_frame,
            text="Auto-capture on Turn",
            variable=self.var_auto_capture,
        )
        self.motion_settings_frame = ctk.CTkFrame(
            self.left_sidebar_frame, fg_color="transparent"
        )
        self.motion_threshold_label = ctk.CTkLabel(
            self.motion_settings_frame, text="Thresh%:", font=ctk.CTkFont(size=11)
        )
        self.motion_threshold_entry = ctk.CTkEntry(
            self.motion_settings_frame,
            width=45,
            height=24,
            font=ctk.CTkFont(size=11),
            textvariable=self.var_motion_threshold,
        )
        self.settle_time_label = ctk.CTkLabel(
            self.motion_settings_frame, text="Settle(s):", font=ctk.CTkFont(size=11)
        )
        self.settle_time_entry = ctk.CTkEntry(
            self.motion_settings_frame,
            width=45,
            height=24,
            font=ctk.CTkFont(size=11),
            textvariable=self.var_settle_time,
        )
        self.motion_threshold_label.pack(side=ctk.LEFT, padx=(0, 2))
        self.motion_threshold_entry.pack(side=ctk.LEFT, padx=(0, 6))
        self.settle_time_label.pack(side=ctk.LEFT, padx=(0, 2))
        self.settle_time_entry.pack(side=ctk.LEFT)

        self.motion_status_label = ctk.CTkLabel(
            self.left_sidebar_frame,
            text="Auto-capture: Off",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray",
        )
        self.session_buttons_frame = ctk.CTkFrame(
            self.left_sidebar_frame, fg_color="transparent"
        )
        self.session_action_button = ctk.CTkButton(
            self.session_buttons_frame,
            text="▶️ Start Capture",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=42,
            fg_color="#2e7d32",
            hover_color="#1b5e20",
            command=self.toggle_capture_session,
        )
        self.stop_capture_button = ctk.CTkButton(
            self.session_buttons_frame,
            text="⏹️ Stop",
            font=ctk.CTkFont(size=13, weight="bold"),
            width=70,
            height=42,
            fg_color="#c62828",
            hover_color="#b71c1c",
            command=self.stop_capture,
        )
        self.capture_image_button = ctk.CTkButton(
            self.left_sidebar_frame,
            text="Capture",
            command=self.capture_image,
        )

        # Add a menu for exporting separate captures
        self.export_separate_captures_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Export Separate Files", anchor="w"
        )
        self.export_separate_captures_option_menu = ctk.CTkComboBox(
            master=self.left_sidebar_frame,
            values=sorted(EXPORT_SEPARATE_FILE_TYPES),
            variable=self.var_separate_captures_file_type,
            state="readonly",
        )
        self.export_separate_captures_button = ctk.CTkButton(
            master=self.left_sidebar_frame,
            text="Export separate files",
            command=self.export_separate_captures,
        )

        # Add a menu for exporting merged captures
        self.export_merged_captures_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Export Merged Files", anchor="w"
        )
        self.export_merged_captures_option_menu = ctk.CTkComboBox(
            master=self.left_sidebar_frame,
            values=sorted(EXPORT_MERGED_FILE_TYPES),
            variable=self.var_merged_captures_file_type,
            state="readonly",
        )
        self.ocr_engine_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="OCR Engine (PDF):", anchor="w"
        )
        self.ocr_engine_option_menu = ctk.CTkOptionMenu(
            master=self.left_sidebar_frame,
            values=OCR_OPTIONS,
            variable=self.var_ocr_engine,
        )
        self.export_merged_captures_button = ctk.CTkButton(
            master=self.left_sidebar_frame,
            text="Export merged file",
            command=self.export_merged_captures,
        )

        # Add a section for auto-exporting to watched folder
        self.watched_folder_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="Watched Folder (OneDrive):", anchor="w"
        )
        self.watched_folder_frame = ctk.CTkFrame(
            self.left_sidebar_frame, fg_color="transparent"
        )
        self.watched_folder_entry = ctk.CTkEntry(
            self.watched_folder_frame,
            textvariable=self.var_watched_folder,
            font=ctk.CTkFont(size=11),
        )
        self.browse_watched_folder_button = ctk.CTkButton(
            self.watched_folder_frame,
            text="📁",
            width=28,
            command=self.browse_watched_folder,
        )
        self.watched_folder_entry.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(0, 4))
        self.browse_watched_folder_button.pack(side=ctk.LEFT)

        self.finalize_session_button = ctk.CTkButton(
            master=self.left_sidebar_frame,
            text="Finish & Export Session",
            fg_color="#2e7d32",
            hover_color="#1b5e20",
            command=self.finalize_session,
        )

        # Organize left menu items logically into clean sections
        self.left_sidebar_title_label.pack(padx=LEFT_MENU_PAD_X, pady=(15, 6))
        self.user_profile_frame.pack(padx=LEFT_MENU_PAD_X, pady=(0, 10), fill="x")

        # --- SECTION 1: 📸 SCAN & CAPTURE ---
        self.sec_scan_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="📸 Scan & Capture", font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
        )
        self.sec_scan_label.pack(padx=LEFT_MENU_PAD_X, pady=(10, 4), fill="x")
        self.session_buttons_frame.pack(padx=LEFT_MENU_PAD_X, pady=(4, 4), fill="x")
        self.session_action_button.pack(side=ctk.LEFT, fill=ctk.X, expand=True)
        self.capture_image_button.configure(
            text="📸 Manual Snap [Space]",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=32,
            fg_color="#1976D2",
            hover_color="#1565C0",
        )
        self.capture_image_button.pack(padx=LEFT_MENU_PAD_X, pady=(2, 8), fill="x")

        self.camera_selection_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.camera_selection_option_menu.pack(**LEFT_MENU_PACK_KWARGS)

        self.subject_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.subject_frame.pack(**LEFT_MENU_PACK_KWARGS)
        self.system_date_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.student_tag_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.student_frame.pack(**LEFT_MENU_PACK_KWARGS)
        self.path_preview_label.pack(**LEFT_MENU_PACK_KWARGS)

        self.two_page_setting_check_box.pack(**LEFT_MENU_PACK_KWARGS)
        self.free_capture_setting_check_box.pack(**LEFT_MENU_PACK_KWARGS)
        self.boundary_detector_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.boundary_detector_option_menu.pack(**LEFT_MENU_PACK_KWARGS)

        # --- SECTION 2: ⚡ AUTO-CAPTURE ON PAGE TURN ---
        self.sec_auto_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="⚡ Auto-Capture on Turn", font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
        )
        self.sec_auto_label.pack(padx=LEFT_MENU_PAD_X, pady=(15, 4), fill="x")
        self.auto_capture_check_box.pack(**LEFT_MENU_PACK_KWARGS)
        self.motion_settings_frame.pack(**LEFT_MENU_PACK_KWARGS)
        self.motion_status_label.pack(**LEFT_MENU_PACK_KWARGS)

        # --- SECTION 3: 🚀 FINALIZE & EXPORT ---
        self.sec_export_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="🚀 Finalize & Export", font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
        )
        self.sec_export_label.pack(padx=LEFT_MENU_PAD_X, pady=(15, 4), fill="x")
        self.finalize_session_button.configure(
            text="Finish & Export Session",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            fg_color="#2e7d32",
            hover_color="#1b5e20",
        )
        self.finalize_session_button.pack(padx=LEFT_MENU_PAD_X, pady=(4, 8), fill="x")

        self.watched_folder_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.watched_folder_frame.pack(**LEFT_MENU_PACK_KWARGS)
        self.ocr_engine_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.ocr_engine_option_menu.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_merged_captures_button.pack(**LEFT_MENU_PACK_KWARGS)
        self.export_separate_captures_button.pack(**LEFT_MENU_PACK_KWARGS)

        # --- SECTION 4: 📱 REMOTE CONTROL (TAILSCALE) ---
        self.sec_remote_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="📱 Remote Control (Tailscale)", font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
        )
        self.sec_remote_label.pack(padx=LEFT_MENU_PAD_X, pady=(15, 4), fill="x")
        self.remote_server_check_box.pack(**LEFT_MENU_PACK_KWARGS)
        self.remote_url_label.pack(padx=LEFT_MENU_PAD_X, pady=(2, 2), fill="x")
        self.remote_pin_label.pack(padx=LEFT_MENU_PAD_X, pady=(2, 4), fill="x")
        self.remote_btn_frame.pack(**LEFT_MENU_PACK_KWARGS)

        # --- SECTION 5: ⚙️ CAMERA & HARDWARE ---
        self.sec_settings_label = ctk.CTkLabel(
            self.left_sidebar_frame, text="⚙️ Camera & Appearance", font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
        )
        self.sec_settings_label.pack(padx=LEFT_MENU_PAD_X, pady=(15, 4), fill="x")
        self.configure_camera_button.pack(**LEFT_MENU_PACK_KWARGS)
        self.camera_settings_button.pack(**LEFT_MENU_PACK_KWARGS)
        self.postprocessing_menu_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.postprocessing_option_menu.pack(**LEFT_MENU_PACK_KWARGS)
        self.appearance_mode_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.appearance_mode_option_menu.pack(**LEFT_MENU_PACK_KWARGS)
        self.scaling_label.pack(**LEFT_MENU_PACK_KWARGS)
        self.scaling_option_menu.pack(padx=LEFT_MENU_PAD_X, pady=(0, 20))

        # Configure the central widget showing the camera feed
        self.camera_image_widget = ctk.CTkLabel(self, text=None, padx=0, pady=0)
        self.camera_image_label = ctk.CTkLabel(
            self,
            text="📷 Waiting for camera feed...\n\nIf using a Mac, ensure camera permissions are enabled:\nSystem Settings > Privacy & Security > Camera",
            font=ctk.CTkFont(size=16),
            padx=20,
            pady=20,
        )

        # Configure the right sidebar
        self.right_sidebar_frame = ctk.CTkFrame(self, corner_radius=0)
        self.right_sidebar_frame.grid_rowconfigure((0, 1), weight=0)
        self.right_sidebar_frame.grid_rowconfigure(2, weight=1)

        # Add a label to the top of the sidebar
        self.right_sidebar_title_label = ctk.CTkLabel(
            self.right_sidebar_frame,
            text="Captures",
            font=ctk.CTkFont(size=20, weight="bold"),
        )

        # Create scrollable frame for the captures
        self.scrollable_frame = ctk.CTkScrollableFrame(
            master=self.right_sidebar_frame,
            width=320,
        )
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        # Add widgets for selcting all captures and deleting
        self.select_all_captures_check_box = ctk.CTkCheckBox(
            self.right_sidebar_frame,
            text="Select All",
            command=self.select_all_entries,
            variable=self.var_select_all_captures,
        )

        self.delete_captures_button = ctk.CTkButton(
            master=self.right_sidebar_frame,
            text="🗑",
            width=24,
            height=24,
            font=ctk.CTkFont(size=24),
            fg_color="transparent",
            command=self.delete_selected_entries,
        )

        # Organize right menu items
        self.right_sidebar_title_label.grid(
            row=0, column=0, columnspan=2, padx=LEFT_MENU_PAD_X, pady=20
        )
        self.select_all_captures_check_box.grid(
            row=1, column=0, **RIGHT_MENU_PACK_KWARGS
        )
        self.delete_captures_button.grid(row=1, column=1, **RIGHT_MENU_PACK_KWARGS)
        self.scrollable_frame.grid(
            row=2, column=0, columnspan=2, sticky="nsew", **RIGHT_MENU_PACK_KWARGS
        )

        # Organize main frames
        self.left_sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.camera_image_label.grid(row=0, column=1, sticky="nsew")
        self.camera_image_widget.grid(row=0, column=1, sticky="nsew")
        self.camera_image_label.lift()
        self.right_sidebar_frame.grid(row=0, column=2, rowspan=4, sticky="nsew")

        # Tooltips
        # Left menu
        widgets.Tooltip(
            widget=self.configure_camera_button,
            text=TOOLTIPS["camera_configuration"],
        )
        widgets.Tooltip(
            widget=self.camera_settings_button,
            text=TOOLTIPS["camera_driver_settings"],
        )
        widgets.Tooltip(
            widget=self.postprocessing_option_menu,
            text=TOOLTIPS["postprocessing"],
        )
        widgets.Tooltip(
            widget=self.appearance_mode_option_menu,
            text=TOOLTIPS["system_appearance"],
        )
        widgets.Tooltip(
            widget=self.scaling_option_menu,
            text=TOOLTIPS["system_ui_scaling"],
        )
        widgets.Tooltip(
            widget=self.remote_server_check_box,
            text=TOOLTIPS["remote_server"],
        )
        widgets.Tooltip(
            widget=self.boundary_detector_option_menu,
            text=TOOLTIPS["boundary_detector"],
        )
        widgets.Tooltip(
            widget=self.student_combobox,
            text=TOOLTIPS["student_tag"],
        )
        widgets.Tooltip(
            widget=self.free_capture_setting_check_box,
            text=TOOLTIPS["free_capture_mode"],
        )
        widgets.Tooltip(
            widget=self.two_page_setting_check_box,
            text=TOOLTIPS["two_page_mode"],
        )
        widgets.Tooltip(
            widget=self.auto_capture_check_box,
            text=TOOLTIPS["auto_capture"],
        )
        widgets.Tooltip(
            widget=self.motion_threshold_entry,
            text=TOOLTIPS["motion_threshold"],
        )
        widgets.Tooltip(
            widget=self.settle_time_entry,
            text=TOOLTIPS["settle_time"],
        )
        widgets.Tooltip(
            widget=self.capture_image_button,
            text=TOOLTIPS["capture"],
        )
        widgets.Tooltip(
            widget=self.export_separate_captures_button,
            text=TOOLTIPS["export_separate"],
        )
        widgets.Tooltip(
            widget=self.ocr_engine_option_menu,
            text=TOOLTIPS["ocr_engine"],
        )
        widgets.Tooltip(
            widget=self.export_merged_captures_button,
            text=TOOLTIPS["export_merged"],
        )
        widgets.Tooltip(
            widget=self.watched_folder_entry,
            text=TOOLTIPS["watched_folder"],
        )
        widgets.Tooltip(
            widget=self.browse_watched_folder_button,
            text=TOOLTIPS["browse_watched_folder"],
        )
        widgets.Tooltip(
            widget=self.finalize_session_button,
            text=TOOLTIPS["finalize_session"],
        )
        # Right menu
        widgets.Tooltip(
            widget=self.select_all_captures_check_box,
            text=TOOLTIPS["select_all"],
        )
        widgets.Tooltip(
            widget=self.delete_captures_button,
            text=TOOLTIPS["delete"],
        )

        # Hotkeys
        self.bind(sequence=CAPTURE_KEYBIND, func=lambda _: self.capture_image())

        # Clean shutdown protocol
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize station role UI
        self._update_station_ui()

        # Check if initial station setup or remote prompt should be shown
        if not getattr(self.active_profile, "setup_completed", False):
            self.after(500, self.prompt_setup_station_dialog)
        elif self.active_profile.remote_startup_action == "ask":
            self.after(600, self.prompt_remote_session_dialog)
        elif self.active_profile.remote_startup_action == "always":
            try:
                self.remote_server.start()
            except Exception as e:
                logging.warning(f"Could not start remote server: {e}")

        self.show_frame()

    def capture(self) -> tuple[cv2.Mat, cv2.Mat, np.ndarray]:
        """
        Capture an image from the camera and run the document detection
        algorithm on the resulting image.
        """
        img_capture = self.camera.capture()
        if img_capture is None:
            now = time.time()
            if not hasattr(self, "_last_reinit_attempt") or (now - self._last_reinit_attempt > 2.5):
                self._last_reinit_attempt = now
                self.camera.initialize()
                img_capture = self.camera.capture()

        if img_capture is not None:
            # Use the robust document & book detector matching the live viewfinder
            scan_result = scanner.main(img_capture)
            if scan_result.warped is not None and scan_result.contour is not None:
                return (
                    img_capture,
                    scan_result.warped,
                    scan_result.contour,
                )
            return (img_capture, img_capture, None)

        return (None, None, None)

    def show_frame(self):
        """
        This function is continuously called to show the camera feed in the
        central widget of the application.
        """
        # Get the current width and height of the camera widget area
        max_width = self.camera_image_widget.winfo_width()
        max_height = self.camera_image_widget.winfo_height()

        if max_width <= 10 or max_height <= 10:
            win_w = self.winfo_width()
            win_h = self.winfo_height()
            max_width = max(max_width, win_w - 620, 640)
            max_height = max(max_height, win_h - 40, 480)

        # Grab raw camera frame directly (instantaneous)
        raw_image = self.camera.capture()
        if raw_image is None:
            now = time.time()
            if not hasattr(self, "_last_reinit_attempt") or (now - self._last_reinit_attempt > 2.5):
                self._last_reinit_attempt = now
                self.camera.initialize()
                raw_image = self.camera.capture()

        if raw_image is not None:
            # Ultra-fast lightweight boundary detection on downscaled thumbnail every 3 frames (~10fps)
            if not hasattr(self, "_frame_counter"):
                self._frame_counter = 0
                self._cached_contour = None
                self._smooth_contour = None
                self._contour_missing_frames = 0
            self._frame_counter += 1

            if self._frame_counter % 3 == 0 or self._cached_contour is None:
                th_h, th_w = 180, 240
                h_orig, w_orig = raw_image.shape[:2]
                small = cv2.resize(raw_image, (th_w, th_h), interpolation=cv2.INTER_NEAREST)
                small_box = scanner.find_paper_contour_adaptive(small)

                # Check if small_box is a plausible document (not full screen edge >90%)
                is_valid_doc = False
                if small_box is not None:
                    area = cv2.contourArea(small_box)
                    total_area = th_h * th_w
                    if 0.08 * total_area < area < 0.88 * total_area:
                        is_valid_doc = True

                if is_valid_doc:
                    scale_x = w_orig / float(th_w)
                    scale_y = h_orig / float(th_h)
                    new_contour = (small_box * [scale_x, scale_y]).astype(np.float32)
                    if self._smooth_contour is None:
                        self._smooth_contour = new_contour
                    else:
                        # EMA temporal smoothing eliminates vibration/jitter
                        self._smooth_contour = 0.65 * self._smooth_contour + 0.35 * new_contour
                    self._cached_contour = self._smooth_contour.astype(np.int32)
                    self._contour_missing_frames = 0
                else:
                    self._contour_missing_frames += 1
                    # If paper is lost for >5 cycles (~0.5s), gracefully clear so no flashing border
                    if self._contour_missing_frames > 5:
                        self._cached_contour = None
                        self._smooth_contour = None

            contour = self._cached_contour

            # Calculate real-time image sharpness (Laplacian variance on central region)
            ch, cw = raw_image.shape[:2]
            sample_gray = cv2.cvtColor(
                raw_image[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4],
                cv2.COLOR_BGR2GRAY,
            )
            focus_score = cv2.Laplacian(sample_gray, cv2.CV_64F).var()

            # Auto-capture on page turn processing if enabled
            if self.var_auto_capture.get():
                try:
                    self.page_turn_detector.motion_threshold = float(
                        self.var_motion_threshold.get()
                    )
                except ValueError:
                    pass
                try:
                    self.page_turn_detector.settle_time_s = float(
                        self.var_settle_time.get()
                    )
                except ValueError:
                    pass

                has_doc = (contour is not None or self.var_free_capture_mode.get())
                should_trigger, motion_score, motion_state = (
                    self.page_turn_detector.process_frame(
                        raw_image,
                        contour=contour,
                        has_document=has_doc,
                    )
                )

                if not self._session_active:
                    self.motion_status_label.configure(
                        text="Auto-capture: Ready (Click Start Capture)",
                        text_color="#2196F3",
                    )
                elif motion_state == motion.PageTurnDetector.STATE_HAND_DETECTED:
                    self.motion_status_label.configure(
                        text="Status: ✋ Holding book (Hand detected)",
                        text_color="#E91E63",
                    )
                elif motion_state == motion.PageTurnDetector.STATE_IDLE:
                    self.motion_status_label.configure(
                        text=f"Status: Waiting for page turn ({motion_score:.1f}%)",
                        text_color="#4CAF50",
                    )
                elif motion_state == motion.PageTurnDetector.STATE_MOTION:
                    self.motion_status_label.configure(
                        text=f"Status: Turning page... ({motion_score:.1f}%)",
                        text_color="#FF9800",
                    )
                elif motion_state == motion.PageTurnDetector.STATE_SETTLING:
                    self.motion_status_label.configure(
                        text=f"Status: Page settled — capturing... ({focus_score:.0f})",
                        text_color="#2196F3",
                    )
                elif motion_state == motion.PageTurnDetector.STATE_COOLDOWN:
                    self.motion_status_label.configure(
                        text=f"Status: Captured! ({len(self.entries)} pages)",
                        text_color="#4CAF50",
                    )

                # Snap when settled without hands during active session, with minimum basic sharpness
                if (
                    should_trigger
                    and self._session_active
                    and not self.page_turn_detector.hand_detected
                    and focus_score >= 10.0
                    and not getattr(self, "_is_capturing", False)
                ):
                    self._is_capturing = True
                    try:
                        self.capture_image()
                    finally:
                        self._is_capturing = False

            else:
                self.motion_status_label.configure(
                    text=f"Auto-capture: Off (Focus: {focus_score:.0f})",
                    text_color="gray",
                )

            # Live preview always displays clean, natural color 60fps video
            image = raw_image.copy()

            # Draw boundary box only when a valid paper was detected
            if not self.var_free_capture_mode.get() and contour is not None:
                image = utils.draw_contour(image=image, contour=contour)
            self._latest_preview_frame = image.copy()
            # Convert the OpenCV image to a CTkImage to display in the widget
            image_width = image.shape[1]
            image_height = image.shape[0]
            # If the image is larger than the max widget size, resize it first
            if image_width > max_width or image_height > max_height:
                image = opencv_to_ctk_image(
                    image=image, width=max_width, height=max_height
                )
            else:
                image = opencv_to_ctk_image(image=image)
            # Update the camera image widget
            self.camera_image_widget.photo = image
            self.camera_image_widget.configure(image=image)
            # Ensure the camera image widget is top of the 'No Camera' widget
            self.camera_image_widget.lift()
        else:
            # If there was no image captured, lift the 'No Camera' widget on top
            self.camera_image_label.lift()

        # Run again after a delay
        self.after(ms=CAMERA_FEED_WAIT_MS, func=self.show_frame)

    def capture_image(self):
        """
        Capture an image using the camera.
        """
        full_image, warped_image, _ = self.capture()

        # If we are using Free Capture mode, use the full uncropped image
        if self.var_free_capture_mode.get():
            if full_image is not None:
                image = full_image
            else:
                tk.messagebox.showerror(
                    title="Error",
                    message="Could not capture an image from the Camera.",
                )
                return
        # Otherwise, use the warped cropped extracted image
        elif warped_image is not None:
            image = warped_image
        elif full_image is not None:
            image = full_image
        else:
            tk.messagebox.showerror(
                title="Error",
                message="Could not capture an image from the Camera.",
            )
            return

        # Ensure session is marked active when a capture is made
        if not self._session_active:
            self._session_active = True
            if hasattr(self, "session_action_button"):
                self.session_action_button.configure(
                    text="📁 Finish & Save",
                    fg_color="#1565C0",
                    hover_color="#0D47A1",
                )
            if hasattr(self, "stop_capture_button"):
                self.stop_capture_button.pack(side=ctk.LEFT, padx=(4, 0))


        # Give the capture a name using student tag and timestamp string
        timestamp_str = datetime.now().strftime(r"%Y%m%d_%H%M%S_%f")
        clean_tag = session.sanitize_tag(self.var_student_tag.get())
        base_name = f"{clean_tag}_{timestamp_str}" if clean_tag else timestamp_str

        # If we are using two-page mode, cut into left and right parts only if it's an open spread (W > 1.15 * H)
        img_h, img_w = image.shape[:2]
        if self.var_two_page_mode.get() and (img_w > 1.15 * img_h):
            cutoff_width = img_w // 2
            left_image = image[:, :cutoff_width]
            right_image = image[:, cutoff_width:]
            new_entries = [
                CaptureEntry(
                    master=self.scrollable_frame,
                    image=left_image,
                    name=f"{base_name}_1",
                    index=len(self.entries) + 1,
                    move_entry=self.move_entry,
                ),
                CaptureEntry(
                    master=self.scrollable_frame,
                    image=right_image,
                    name=f"{base_name}_2",
                    index=len(self.entries) + 2,
                    move_entry=self.move_entry,
                ),
            ]
        # Otherwise, take the entire image as an entry (single page, cover, or portrait orientation)
        else:
            new_entries = [
                CaptureEntry(
                    master=self.scrollable_frame,
                    image=image,
                    name=base_name,
                    index=len(self.entries) + 1,
                    move_entry=self.move_entry,
                )
            ]

        # If a postprocessing function is selected, apply it to the new images
        self.apply_postprocessing(entries=new_entries)
        self.entries += new_entries

        # Update button text with page count
        n_pages = len(self.entries)
        if hasattr(self, "finalize_session_button"):
            tag_name = self.var_student_tag.get().strip() or "Session"
            self.finalize_session_button.configure(
                text=f"Finish & Export ({tag_name}: {n_pages}p)"
            )

        # Update the scrollable frame with the entries and move it to the bottom
        self.scrollable_frame.update()
        self.scrollable_frame._parent_canvas.yview_moveto(1.0)


    def move_entry(self, entry: CaptureEntry, distance: int):
        """
        Move an entry in the capture list either up or down by some distance.
        :param entry: The CaptureEntry to move
        :param distance: The move distance (-1 to move up, or +1 to move down)
        """
        # Find the current index 'i' of the entry and the destination index 'j'
        i = self.entries.index(entry)
        j = i + distance

        # If the destination index is out of range, skip the operation
        if j < 0 or j >= len(self.entries):
            return

        # Get the current grid rows of the entries. This is not really needed
        # since the indices i and j should be the same as the grid row
        i_grid_row = self.entries[i].frame.grid_info()["row"]
        j_grid_row = self.entries[j].frame.grid_info()["row"]

        # Switch grid positions
        logging.debug(f"Switching entries in rows {i_grid_row} and {j_grid_row}")
        self.entries[i].frame.grid(row=j_grid_row)
        self.entries[j].frame.grid(row=i_grid_row)

        # Switch index labels
        self.entries[i].index_label.configure(text=str(j + 1))
        self.entries[j].index_label.configure(text=str(i + 1))

        # Switch the locations of the entries in the list
        self.entries[i], self.entries[j] = self.entries[j], self.entries[i]

    def select_all_entries(self):
        """
        Select or deselect all current capture entries.
        """
        # Depending on the state of the checkbox, select or deselect all entries
        select = self.var_select_all_captures.get()
        for entry in self.entries:
            if select:
                entry.selection_checkbox.select()
            else:
                entry.selection_checkbox.deselect()

    def delete_selected_entries(self):
        """
        Delete all the currently selected capture entries.
        """
        # Select the entries based on the state of their checkbox variable
        entries_to_delete = [e for e in self.entries if e.var_selected.get()]
        logging.debug(f"Removing {len(entries_to_delete)} entries")

        # For each such entry, destroy its frame and remove from the list
        for entry in entries_to_delete:
            entry.frame.destroy()
            self.entries.remove(entry)

        # After deletion, update the grid positions of the remaining entries
        for i, entry in enumerate(self.entries):
            entry.frame.grid(row=i)

        # There is some peculiar behavior of the scrollbar in the scrollable
        # frame when all entries are deleted at once. If there are enough
        # entries (around 5+) to make the scrollbar active, and it is scrolled
        # all the way to the bottom, it will not correctly update its allowed
        # range of scrolling when the entries are deleted. Instead, it will
        # still be scrolled all the way to the bottom, with the scrollable frame
        # being completely empty. After testing, it seems that one (hacky)
        # solution to this is to do the following:
        # - Add back a widget in the grid (a dummy frame in this solution)
        # - Move the scroll all the way back up to the top (yview_moveto)
        # - Call the update function on the scrollable frame
        # - Delete the dummy frame after it is no longer needed.
        # By adding this dummy widget, it seems to make the update of the
        # scrollable frame also update the scrollbar to the correct range.
        # Without it, this does not work!
        if len(self.entries) == 0:
            dummy_frame = ctk.CTkFrame(master=self.scrollable_frame)
            dummy_frame.grid(row=0, column=0)
            self.scrollable_frame._parent_canvas.yview_moveto(0.0)
            self.scrollable_frame.update()
            dummy_frame.destroy()

        # Uncheck the checkbox for selecting all entries
        self.select_all_captures_check_box.deselect()

    def export_merged_captures(self):
        """
        Export all the current captures as a single merged file.
        """
        # Get the currently select file type to export as
        file_type = self.var_merged_captures_file_type.get()

        n = len(self.entries)

        # If there are no captures, show a message box and return
        if n == 0:
            tk.messagebox.showerror(
                title="Error",
                message="There are no captures to export",
            )
            return

        # Create the name of the output file tagged with student session and date
        initialfile = session.generate_session_filename(
            student_tag=self.var_student_tag.get(),
            ext=file_type,
        )

        # Bring up a dialog asking for the output file path
        file_path = tk.filedialog.asksaveasfilename(
            initialfile=initialfile,
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")],
        )

        # If no output file was chosen (e.g. dialog cancelled), return
        if not file_path:
            return

        raw_images = [entry.current_image.copy() for entry in self.entries]
        ocr_mode = self.var_ocr_engine.get()
        engine = ocr.get_ocr_engine(ocr_mode)

        if file_type.lower() == "pdf":
            if engine is None:
                pdf_builder.create_searchable_pdf(
                    images=raw_images,
                    ocr_results=None,
                    output_path=file_path,
                )
                tk.messagebox.showinfo(
                    title="Export Successful",
                    message=f"{n} captures exported as {file_type} to {file_path}",
                )
            else:
                self._export_pdf_with_ocr(
                    images=raw_images,
                    engine=engine,
                    output_path=file_path,
                    total_pages=n,
                )
        else:
            # Fallback for non-PDF merged formats
            images = [opencv_to_pil_image(entry.current_image) for entry in self.entries]
            first_image = images[0]
            remaining_images = images[1:]
            first_image.save(
                file_path,
                save_all=True,
                append_images=remaining_images,
            )
            tk.messagebox.showinfo(
                title="Export Successful",
                message=f"{n} captures exported as {file_type} to {file_path}",
            )

    def _export_pdf_with_ocr(
        self,
        images: list[np.ndarray],
        engine: ocr.BaseOCREngine,
        output_path: str,
        total_pages: int,
    ):
        """
        Run OCR on images and export searchable PDF in a background thread to keep UI responsive.
        """
        import threading

        progress_dialog = ctk.CTkToplevel(self)
        progress_dialog.title("Exporting Searchable PDF")
        progress_dialog.geometry("460x200")
        progress_dialog.resizable(False, False)
        progress_dialog.attributes("-topmost", True)
        progress_dialog.grab_set()

        status_label = ctk.CTkLabel(
            progress_dialog,
            text=f"Starting handwriting OCR on {total_pages} page(s)...",
            font=ctk.CTkFont(size=14, weight="bold"),
            wraplength=420,
        )
        status_label.pack(padx=20, pady=(25, 10))

        detail_label = ctk.CTkLabel(
            progress_dialog,
            text="Initializing OCR engine...",
            font=ctk.CTkFont(size=12),
            wraplength=420,
        )
        detail_label.pack(padx=20, pady=(0, 15))

        progressbar = ctk.CTkProgressBar(progress_dialog, width=400)
        progressbar.pack(padx=20, pady=10)
        progressbar.set(0)

        def _worker():
            ocr_results = []
            try:
                for idx, img in enumerate(images):
                    page_num = idx + 1
                    frac = idx / total_pages
                    self.after(
                        0,
                        lambda p=page_num, f=frac: (
                            status_label.configure(
                                text=f"Processing Page {p}/{total_pages}..."
                            ),
                            progressbar.set(f),
                        ),
                    )

                    def _progress_cb(msg: str, p=page_num):
                        self.after(
                            0,
                            lambda m=msg, p=page_num: detail_label.configure(
                                text=f"Page {p}/{total_pages}: {m}"
                            ),
                        )

                    lines = engine.recognize(img, progress_callback=_progress_cb)
                    ocr_results.append(lines)

                self.after(
                    0,
                    lambda: (
                        status_label.configure(text="Generating Searchable PDF..."),
                        detail_label.configure(text="Embedding text layers..."),
                        progressbar.set(0.95),
                    ),
                )

                pdf_builder.create_searchable_pdf(
                    images=images,
                    ocr_results=ocr_results,
                    output_path=output_path,
                )

                def _on_success():
                    progress_dialog.destroy()
                    tk.messagebox.showinfo(
                        title="Export Successful",
                        message=(
                            f"{total_pages} captures exported as searchable PDF "
                            f"with handwriting OCR to {output_path}"
                        ),
                    )

                self.after(0, _on_success)

            except Exception as exc:
                logging.exception("Failed during OCR export")

                def _on_error(err=str(exc)):
                    progress_dialog.destroy()
                    tk.messagebox.showerror(
                        title="OCR Export Error",
                        message=f"OCR or PDF export failed: {err}",
                    )

                self.after(0, _on_error)

        threading.Thread(target=_worker, daemon=True).start()

    def export_separate_captures(self):
        """
        Export all the current captures as separate files in a directory.
        """
        # Get the currently select file type to export as
        file_type = self.var_separate_captures_file_type.get()

        n = len(self.entries)

        # If there are no captures, show a message box and return
        if n == 0:
            tk.messagebox.showerror(
                title="Error",
                message="There are no captures to export",
            )
            return

        # Bring up a dialog asking for the output directory path
        file_dialog_dir = tk.filedialog.askdirectory()

        # If no output directory was chosen (e.g. dialog cancelled), return
        if not file_dialog_dir:
            return

        # Create the name of the output directory tagged with student session and date
        session_dirname = session.generate_session_dirname(
            student_tag=self.var_student_tag.get()
        )
        output_dir = f"{file_dialog_dir}/{session_dirname}"
        os.makedirs(output_dir, exist_ok=True)

        # For each capture, write the image to the output directory
        for i, entry in enumerate(self.entries, start=1):
            cv2.imwrite(
                filename=f"{output_dir}/{i}_{entry.name}.{file_type}",
                img=entry.current_image,
            )

        # Show a message box indicating to the user that the export succeeded
        tk.messagebox.showinfo(
            title="Export Successful",
            message=f"{n} captures exported as {file_type} to {output_dir}",
        )

    def browse_watched_folder(self):
        """Browse and select the watched OneDrive folder."""
        folder = tk.filedialog.askdirectory(initialdir=self.var_watched_folder.get())
        if folder:
            self.var_watched_folder.set(folder)
            self.auto_exporter.set_watched_folder(folder)

    def toggle_capture_session(self):
        """Toggle active student capture session: Start Capture <-> Finish Capture."""
        if not self._session_active:
            # Start capture session
            student_tag = self.var_student_tag.get().strip()
            if not student_tag:
                prof = self.profile_manager.get_active_profile()
                if prof.students:
                    student_tag = prof.students[0]
                    self.var_student_tag.set(student_tag)
                else:
                    student_tag = "Untagged"
                    self.var_student_tag.set(student_tag)

            self._session_active = True
            if hasattr(self, "session_action_button"):
                self.session_action_button.configure(
                    text="📁 Finish & Save",
                    fg_color="#1565C0",
                    hover_color="#0D47A1",
                )
            if hasattr(self, "stop_capture_button"):
                self.stop_capture_button.pack(side=ctk.LEFT, padx=(4, 0))
            if hasattr(self, "finalize_session_button"):
                self.finalize_session_button.configure(
                    text=f"Finish & Export ({student_tag})",
                )
            self.page_turn_detector.arm_new_session(auto_trigger_initial=True)
            self._update_path_preview()
        else:
            # Finish capture session
            self.finalize_session()

    def stop_capture(self):
        """Stop capturing and disarm auto-capture without deleting captured pages."""
        self._session_active = False
        self.page_turn_detector.reset()
        if hasattr(self, "session_action_button"):
            self.session_action_button.configure(
                text="▶️ Resume Capture",
                fg_color="#2e7d32",
                hover_color="#1b5e20",
            )
        if hasattr(self, "stop_capture_button"):
            self.stop_capture_button.pack_forget()
        if hasattr(self, "motion_status_label"):
            self.motion_status_label.configure(
                text=f"Auto-capture stopped ({len(self.entries)} pages ready to finish)",
                text_color="#FF9800",
            )

    def finalize_session(self):
        """
        Finalize the current student capture session and auto-export all pages
        to the watched OneDrive folder, then prepare for the next student.
        """
        # Immediately halt capture and disarm motion detector so no more frames snap while finishing!
        self._session_active = False
        self.page_turn_detector.reset()
        if hasattr(self, "session_action_button"):
            self.session_action_button.configure(
                text="▶️ Start Capture",
                fg_color="#2e7d32",
                hover_color="#1b5e20",
            )
        if hasattr(self, "stop_capture_button"):
            self.stop_capture_button.pack_forget()

        n = len(self.entries)
        if n == 0:
            tk.messagebox.showwarning(
                title="No Captures",
                message="No pages have been captured yet for this session.",
            )
            return


        watched_dir = self.var_watched_folder.get()
        self.auto_exporter.set_watched_folder(watched_dir)

        subject = self.var_subject.get() or "General"
        date_str = session.get_system_date_str()
        student_tag = self.var_student_tag.get()
        images = [entry.current_image.copy() for entry in self.entries]
        ocr_mode = self.var_ocr_engine.get()
        engine = ocr.get_ocr_engine(ocr_mode)

        # Auto-add student to active roster if non-empty
        if student_tag and student_tag.strip():
            self.profile_manager.add_student(student_tag.strip())
            self._refresh_student_dropdown()

        progress_dialog = ctk.CTkToplevel(self)
        progress_dialog.title("Finalizing Session")
        progress_dialog.geometry("460x190")
        progress_dialog.resizable(False, False)
        progress_dialog.attributes("-topmost", True)
        progress_dialog.grab_set()

        tag_display = student_tag if student_tag else "Untagged"
        status_label = ctk.CTkLabel(
            progress_dialog,
            text=f"Auto-exporting {n} page(s) for '{tag_display}' to {subject}/{tag_display}/{date_str}...",
            font=ctk.CTkFont(size=14, weight="bold"),
            wraplength=420,
        )
        status_label.pack(padx=20, pady=(25, 10))

        progressbar = ctk.CTkProgressBar(progress_dialog, width=390)
        progressbar.pack(padx=20, pady=10)
        progressbar.set(0.2)

        def _worker():
            try:
                def _cb(msg):
                    self.after(0, lambda m=msg: status_label.configure(text=m))

                results = self.auto_exporter.export_session(
                    images=images,
                    student_tag=student_tag,
                    subject=subject,
                    date_str=date_str,
                    ocr_engine=engine,
                    progress_callback=_cb,
                )

                def _on_done():
                    progress_dialog.destroy()
                    self.delete_all_entries()
                    self._session_active = False
                    if hasattr(self, "session_action_button"):
                        self.session_action_button.configure(
                            text="▶️ Start Capture",
                            fg_color="#2e7d32",
                            hover_color="#1b5e20",
                        )
                    if hasattr(self, "finalize_session_button"):
                        self.finalize_session_button.configure(
                            text="Finish & Export Session",
                        )
                    self.page_turn_detector.reset()
                    self.var_student_tag.set("")
                    self._update_path_preview()
                    pdf_path = results.get("pdf", "")
                    tk.messagebox.showinfo(
                        title="Session Finalized",
                        message=(
                            f"Session successfully finalized!\n\n"
                            f"Exported to:\n{pdf_path}\n\n"
                            f"Ready for next student session."
                        ),
                    )

                self.after(0, _on_done)
            except Exception as e:
                logging.exception("Auto-export failed")
                def _on_err(err=str(e)):
                    progress_dialog.destroy()
                    tk.messagebox.showerror(
                        title="Auto-Export Error",
                        message=f"Failed to auto-export session: {err}",
                    )
                self.after(0, _on_err)

        threading.Thread(target=_worker, daemon=True).start()

    def delete_all_entries(self):
        """Clear all entries when a session finalizes."""
        for entry in list(self.entries):
            entry.frame.destroy()
        self.entries.clear()
        if hasattr(self, "select_all_captures_check_box"):
            self.select_all_captures_check_box.deselect()
        # Reset scrollbar position
        dummy_frame = ctk.CTkFrame(master=self.scrollable_frame)
        dummy_frame.grid(row=0, column=0)
        self.scrollable_frame._parent_canvas.yview_moveto(0.0)
        self.scrollable_frame.update()
        dummy_frame.destroy()

    def _get_path_preview_text(self) -> str:
        subj = self.var_subject.get() or "General"
        dt = session.get_system_date_str()
        stud = self.var_student_tag.get() or "Untagged"
        return f"📁 Save: {subj} / {stud} / {dt} /"


    def _update_path_preview(self, *args):
        if hasattr(self, "path_preview_label"):
            self.path_preview_label.configure(text=self._get_path_preview_text())

    def _get_remote_status_text(self) -> str:
        if not self.var_remote_server.get():
            return "Server: Stopped"
        ts_info = tailscale.get_tailscale_info()
        if ts_info.running and ts_info.ipv4:
            return f"🟢 Tailscale: {ts_info.ipv4}:8000"
        elif ts_info.ipv4:
            return f"🟠 Tailscale (Idle): {ts_info.ipv4}:8000"
        return f"🌐 Local: {remote.get_local_ip()}:8000"

    def _refresh_subject_dropdown(self):
        prof = self.profile_manager.get_active_profile()
        self.subject_combobox.configure(values=prof.subjects)
        if prof.active_subject in prof.subjects:
            self.var_subject.set(prof.active_subject)
        elif prof.subjects:
            self.var_subject.set(prof.subjects[0])

    def _refresh_student_dropdown(self):
        prof = self.profile_manager.get_active_profile()
        self.student_combobox.configure(values=prof.students)
        if prof.active_student:
            self.var_student_tag.set(prof.active_student)

    def on_subject_selected(self, selected: str):
        prof = self.profile_manager.get_active_profile()
        prof.active_subject = selected
        self.profile_manager.save()
        self._update_path_preview()

    def prompt_add_subject(self):
        name = simpledialog.askstring(
            "Add Subject",
            "Enter main folder / subject name (e.g. Maths, Science):",
            parent=self,
        )
        if name and name.strip():
            self.profile_manager.add_subject(name.strip())
            self._refresh_subject_dropdown()
            self._update_path_preview()

    def prompt_remove_subject(self):
        cur = self.var_subject.get()
        if not cur:
            return
        if messagebox.askyesno(
            "Remove Subject", f"Remove '{cur}' from subject list?", parent=self
        ):
            self.profile_manager.remove_subject(cur)
            self._refresh_subject_dropdown()
            self._update_path_preview()

    def on_student_selected(self, selected: str):
        prof = self.profile_manager.get_active_profile()
        prof.active_student = selected
        self.profile_manager.save()
        self._update_path_preview()

    def prompt_add_student(self):
        name = self.var_student_tag.get().strip()
        if not name:
            name = simpledialog.askstring(
                "Add Student",
                "Enter student name (e.g. Ahmad, Ubaid):",
                parent=self,
            )
        if name and name.strip():
            self.profile_manager.add_student(name.strip())
            self._refresh_student_dropdown()
            self._update_path_preview()
            messagebox.showinfo(
                "Student Added",
                f"Student '{name.strip()}' added to roster.",
                parent=self,
            )

    def prompt_remove_student(self):
        cur = self.var_student_tag.get().strip()
        if not cur:
            return
        if messagebox.askyesno(
            "Remove Student", f"Remove '{cur}' from student list?", parent=self
        ):
            self.profile_manager.remove_student(cur)
            self.var_student_tag.set("")
            self._refresh_student_dropdown()
            self._update_path_preview()

    def switch_user_profile(self, new_profile_name: str):
        self._save_current_profile_settings()
        prof = self.profile_manager.switch_profile(new_profile_name)
        self.active_profile = prof
        self.var_active_profile.set(prof.name)
        self.var_subject.set(prof.active_subject)
        self.var_student_tag.set(prof.active_student)
        self._refresh_subject_dropdown()
        self._refresh_student_dropdown()
        self.var_two_page_mode.set(1 if prof.two_page_mode else 0)
        self.var_free_capture_mode.set(1 if prof.free_capture_mode else 0)
        self.var_boundary_detector.set(prof.boundary_detector)
        self.var_postprocessing_option.set(prof.postprocessing_option)
        self.var_ocr_engine.set(prof.ocr_engine)
        self.var_auto_capture.set(1 if getattr(prof, "auto_capture", True) else 0)
        self.var_motion_threshold.set(str(prof.motion_threshold))
        self.var_settle_time.set(str(prof.settle_time))
        self.var_watched_folder.set(prof.watched_folder)
        self.auto_exporter.set_watched_folder(prof.watched_folder)
        self.var_station_role.set(getattr(prof, "station_role", "host"))
        self.var_station_name.set(getattr(prof, "station_name", "Grade 6 Class"))
        self.var_active_host.set(getattr(prof, "active_host_name", "Grade 6 Class"))
        self._refresh_host_dropdown()
        self._update_station_ui()
        self._update_path_preview()

    def prompt_add_profile(self):
        name = simpledialog.askstring(
            "New User Profile",
            "Enter name for new user profile (e.g. Teacher B):",
            parent=self,
        )
        if name and name.strip():
            new_name = name.strip()
            self._save_current_profile_settings()
            self.profile_manager.add_profile(new_name, copy_from_active=True)
            self.user_profile_option_menu.configure(
                values=self.profile_manager.get_profile_names()
            )
            self.switch_user_profile(new_name)

    def prompt_delete_profile(self):
        cur = self.var_active_profile.get()
        if len(self.profile_manager.get_profile_names()) <= 1:
            messagebox.showwarning(
                "Cannot Delete",
                "At least one profile must be retained.",
                parent=self,
            )
            return
        if messagebox.askyesno(
            "Delete Profile",
            f"Delete profile '{cur}' and all its saved settings?",
            parent=self,
        ):
            self.profile_manager.delete_profile(cur)
            self.user_profile_option_menu.configure(
                values=self.profile_manager.get_profile_names()
            )
            new_active = self.profile_manager.active_profile_name
            self.switch_user_profile(new_active)

    def _save_current_profile_settings(self):
        prof = self.profile_manager.get_active_profile()
        prof.active_subject = self.var_subject.get()
        prof.active_student = self.var_student_tag.get()
        prof.two_page_mode = bool(self.var_two_page_mode.get())
        prof.free_capture_mode = bool(self.var_free_capture_mode.get())
        prof.boundary_detector = self.var_boundary_detector.get()
        prof.postprocessing_option = self.var_postprocessing_option.get()
        prof.ocr_engine = self.var_ocr_engine.get()
        prof.auto_capture = bool(self.var_auto_capture.get())
        try:
            prof.motion_threshold = float(self.var_motion_threshold.get())
        except ValueError:
            pass
        try:
            prof.settle_time = float(self.var_settle_time.get())
        except ValueError:
            pass
        prof.watched_folder = self.var_watched_folder.get()
        prof.station_role = self.var_station_role.get()
        prof.station_name = self.var_station_name.get()
        prof.active_host_name = self.var_active_host.get()
        self.profile_manager.save()

    def _init_host_client(self):
        active_host = self.profile_manager.get_active_host()
        url = active_host.get("url", "http://127.0.0.1:8000")
        pin = active_host.get("pin", "")
        token = active_host.get("token", "")
        self.host_client = remote.HostClient(base_url=url, token=token, pin=pin)

    def _update_station_ui(self):
        role = self.var_station_role.get()
        st_name = self.var_station_name.get()
        if hasattr(self, "station_role_label"):
            if role == "client":
                self.station_role_label.configure(text=f"💻 Client: {st_name}")
            else:
                self.station_role_label.configure(text=f"🖥️ Host: {st_name}")

        if hasattr(self, "client_host_frame"):
            if role == "client":
                self.client_host_frame.pack(
                    fill=ctk.X, pady=(2, 6), before=self.user_profile_frame
                )
                self._connect_to_active_host()
            else:
                self.client_host_frame.pack_forget()

    def prompt_setup_station_dialog(self):
        def _on_setup_done(role: str, station_name: str):
            self.profile_manager.set_station_role(role)
            self.profile_manager.set_station_name(station_name)
            self.active_profile.setup_completed = True
            self.profile_manager.save()
            self.var_station_role.set(role)
            self.var_station_name.set(station_name)
            self._update_station_ui()
            if role == "host":
                if not self.var_remote_server.get():
                    self.var_remote_server.set(1)
                    self.toggle_remote_server()
            else:
                self._connect_to_active_host()

        widgets.SetupStationDialog(
            self,
            current_role=self.var_station_role.get(),
            current_station_name=self.var_station_name.get(),
            callback=_on_setup_done,
        )

    def prompt_add_host(self):
        def _on_add(name: str, url: str, pin: str):
            self.profile_manager.add_saved_host(name, url, pin=pin)
            self.var_active_host.set(name)
            self._refresh_host_dropdown()
            self._connect_to_active_host()

        widgets.AddHostDialog(self, callback=_on_add)

    def prompt_remove_host(self):
        cur = self.var_active_host.get()
        if len(self.profile_manager.get_saved_hosts()) <= 1:
            messagebox.showwarning(
                "Cannot Delete", "At least one host station must be saved.", parent=self
            )
            return
        if messagebox.askyesno(
            "Remove Host", f"Remove host station '{cur}'?", parent=self
        ):
            self.profile_manager.remove_saved_host(cur)
            self.var_active_host.set(
                self.profile_manager.get_active_host().get("name", "Default")
            )
            self._refresh_host_dropdown()
            self._connect_to_active_host()

    def on_host_selected(self, host_name: str):
        self.profile_manager.set_active_host(host_name)
        self._connect_to_active_host()

    def _refresh_host_dropdown(self):
        if hasattr(self, "host_option_menu"):
            names = [
                h.get("name", "Station")
                for h in self.profile_manager.get_saved_hosts()
            ]
            self.host_option_menu.configure(values=names)

    def _connect_to_active_host(self):
        self._init_host_client()

        def _check():
            res = self.host_client.get_status()
            if res.get("offline"):
                self.after(
                    0,
                    lambda: self._update_host_connection_status(
                        False, "Offline / Unreachable"
                    ),
                )
            elif res.get("error") and res.get("status_code") == 401:
                pin = self.profile_manager.get_active_host().get("pin", "")
                if pin:
                    auth_res = self.host_client.authenticate(pin)
                    if auth_res.get("token"):
                        self.after(
                            0,
                            lambda: self._update_host_connection_status(
                                True, f"Connected ({self.host_client.base_url})"
                            ),
                        )
                        return
                self.after(
                    0,
                    lambda: self._update_host_connection_status(
                        False, "PIN Required"
                    ),
                )
            elif res.get("subject"):
                self.after(
                    0,
                    lambda: self._update_host_connection_status(
                        True, f"Connected ({self.host_client.base_url})"
                    ),
                )
            else:
                self.after(
                    0,
                    lambda: self._update_host_connection_status(
                        True, f"Connected ({self.host_client.base_url})"
                    ),
                )

        threading.Thread(target=_check, daemon=True).start()

    def _update_host_connection_status(self, connected: bool, msg: str):
        if hasattr(self, "client_status_label"):
            if connected:
                self.client_status_label.configure(
                    text=f"🟢 {msg}", text_color="#4caf50"
                )
            else:
                self.client_status_label.configure(
                    text=f"🔴 {msg}", text_color="#f44336"
                )

    def trigger_remote_host_capture(self):
        if not self.host_client:
            return

        def _do():
            res = self.host_client.trigger_capture()
            if res.get("success"):
                count = res.get("count", 0)
                self.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Remote Capture",
                        f"Captured successfully on host!\nTotal captures: {count}",
                        parent=self,
                    ),
                )
            else:
                err = res.get("error", "Failed")
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        "Capture Error",
                        f"Could not capture on host station:\n{err}",
                        parent=self,
                    ),
                )

        threading.Thread(target=_do, daemon=True).start()

    def finalize_remote_host_session(self):
        if not self.host_client:
            return

        def _do():
            res = self.host_client.finalize_session()
            if res.get("success"):
                pdf = res.get("pdf_path", "PDF compiled")
                self.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Session Exported",
                        f"Session exported successfully on Host station!\nOutput: {pdf}",
                        parent=self,
                    ),
                )
            else:
                err = res.get("error", "Export failed")
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        "Export Error",
                        f"Could not finalize session on host station:\n{err}",
                        parent=self,
                    ),
                )

        threading.Thread(target=_do, daemon=True).start()


    def prompt_remote_session_dialog(self):
        prof = self.profile_manager.get_active_profile()
        if prof.remote_startup_action == "always":
            if not self.var_remote_server.get():
                self.var_remote_server.set(1)
                self.toggle_remote_server()
            return
        elif prof.remote_startup_action == "never":
            return

        def _on_decision(enable: bool, remember: bool):
            if remember:
                prof.remote_startup_action = "always" if enable else "never"
                self.profile_manager.save()
            if enable:
                self.var_remote_server.set(1)
                self.toggle_remote_server()
                self.show_pairing_qr_dialog()
            else:
                self.var_remote_server.set(0)
                self.toggle_remote_server()

        widgets.StartupRemoteDialog(self, prof.name, _on_decision)

    def show_pairing_qr_dialog(self):
        qr_img = self.remote_server.get_qr_image()
        url = self.remote_server.get_url(with_token=True)
        pin = self.remote_server.get_pairing_pin()
        widgets.QRCodeDialog(self, qr_img, url, pin)

    def copy_remote_link(self):
        url = self.remote_server.get_url(with_token=True)
        self.clipboard_clear()
        self.clipboard_append(url)
        if hasattr(self, "btn_copy_link"):
            self.btn_copy_link.configure(text="✅ Copied!")
            self.after(1500, lambda: self.btn_copy_link.configure(text="📋 Copy Link"))

    def toggle_remote_server(self):
        """Toggle remote control server on/off."""
        if self.var_remote_server.get():
            try:
                self.remote_server.start()
                self.remote_url_label.configure(
                    text=self._get_remote_status_text(),
                    text_color="#2196f3",
                )
                self.remote_pin_label.configure(
                    text=f"🔑 PIN: {self.remote_server.get_pairing_pin()}",
                )
            except Exception as e:
                logging.warning(f"Could not start remote server: {e}")
                self.remote_url_label.configure(
                    text="Error starting",
                    text_color="#f44336",
                )
        else:
            self.remote_server.stop()
            self.remote_url_label.configure(
                text="Server: Stopped",
                text_color="gray",
            )

    def on_close(self):
        """Clean shutdown of remote server, saving settings and application exit."""
        try:
            self._save_current_profile_settings()
        except Exception:
            pass
        try:
            self.remote_server.stop()
        except Exception:
            pass
        self.destroy()

    def change_camera_event(self, selected_text: str):
        """Switch active camera when chosen from dropdown."""
        import re
        match = re.search(r"\((\d+)\)$", selected_text)
        if match:
            idx = int(match.group(1))
            self.camera.set_index(idx)

    def change_postprocessing_event(self, *args):
        """
        Handle the event when the chose postprocessing function changes.
        When it does, apply it to all current capture entries.
        """
        self.apply_postprocessing(entries=self.entries)

    def apply_postprocessing(self, entries: list[CaptureEntry]):
        """
        Apply currently chosen postprocessing function to given capture entries.
        :param entries: The capture entries to apply the postprocessing to
        """
        postprocessing_option = self.var_postprocessing_option.get()
        postprocessing_function = POSTPROCESSING_OPTIONS[postprocessing_option]
        for entry in entries:
            new_image = postprocessing_function(entry.original_image)
            entry.set_current_image(image=new_image)

    def configure_camera_event(self):
        """
        Handle the event for configuring the camera. This is done by opening a
        separate window with the available configuration.
        """

        def _set_camera_index(index: int):
            """Callback for changing the camera device index"""
            self.camera.set_index(index=int(index))

        def _update_available_camera_indices():
            """Callback for updating the available camera device indices"""
            camera_indices = self.camera.get_available_device_indices()
            camera_index_combobox.configure(values=list(map(str, camera_indices)))
            if camera_indices:
                camera_index_combobox.set(value=str(camera_indices[0]))
                _set_camera_index(camera_indices[0])

        def _set_camera_resolution(resolution_string: str):
            """Set the camera resolution from a resolution string"""
            regex = re.compile(r"^(\d+)x(\d+)$")
            matches = regex.findall(resolution_string)
            if matches:
                resolution = (int(matches[0][0]), int(matches[0][1]))
                self.camera.set_resolution(resolution=resolution)
            else:
                tk.messagebox.showerror(
                    title="Error",
                    message=(
                        "The resolution string must be on the form '<width>x<height>'"
                    ),
                )

        # Create a new top-level window for the camera configuration
        window = ctk.CTkToplevel()
        window.resizable(width=False, height=False)
        window.title("Camera Configuration")

        # Define the variables
        possible_camera_indices = list(map(str, range(10)))
        current_resolution_string = "x".join(map(str, self.camera.resolution))
        var_camera_index = tk.StringVar(value=possible_camera_indices[0])
        var_camera_resolution = tk.StringVar(value=current_resolution_string)
        var_custom_camera_resolution = tk.StringVar(value=current_resolution_string)

        # Define the widgets
        camera_index_label = ctk.CTkLabel(
            master=window,
            text="Select Camera Index:",
        )
        camera_index_combobox = ctk.CTkOptionMenu(
            master=window,
            values=possible_camera_indices,
            command=_set_camera_index,
            state="readonly",
            variable=var_camera_index,
        )
        find_camera_indices_button = ctk.CTkButton(
            master=window,
            text="Identify Cameras",
            command=_update_available_camera_indices,
        )
        camera_resolution_label = ctk.CTkLabel(
            master=window,
            text="Camera Resolution:",
        )
        camera_resolution_combobox = ctk.CTkOptionMenu(
            master=window,
            values=RESOLUTIONS,
            command=_set_camera_resolution,
            variable=var_camera_resolution,
        )
        custom_camera_resolution_label = ctk.CTkLabel(
            master=window,
            text="Custom Camera Resolution:",
        )
        custom_camera_resolution_entry = ctk.CTkEntry(
            master=window, textvariable=var_custom_camera_resolution
        )

        custom_camera_resolution_button = ctk.CTkButton(
            master=window,
            text="Set Custom Resolution",
            command=functools.partial(
                _set_camera_resolution, var_custom_camera_resolution.get()
            ),
        )

        # Pack the widgets
        pack_kwargs = dict(padx=10, pady=5)
        camera_index_label.pack(padx=10, pady=(20, 5))
        find_camera_indices_button.pack(**pack_kwargs)
        camera_index_combobox.pack(**pack_kwargs)
        camera_resolution_label.pack(**pack_kwargs)
        camera_resolution_combobox.pack(**pack_kwargs)
        custom_camera_resolution_label.pack(**pack_kwargs)
        custom_camera_resolution_entry.pack(**pack_kwargs)
        custom_camera_resolution_button.pack(padx=10, pady=(5, 20))

        # Add tooltips
        widgets.Tooltip(
            widget=camera_index_combobox,
            text=TOOLTIPS["camera_index"],
        )
        widgets.Tooltip(
            widget=find_camera_indices_button,
            text=TOOLTIPS["identify_cameras"],
        )
        widgets.Tooltip(
            widget=camera_resolution_combobox,
            text=TOOLTIPS["camera_resolution"],
        )
        widgets.Tooltip(
            widget=custom_camera_resolution_button,
            text=TOOLTIPS["custom_camera_resolution"],
        )

        # Make sure this window is on top of the main window
        # We could simply just set topmost to True and leave it at that, but
        # that will prevent the Tooltips from working properly. We can instead
        # set it to topmost temporarily, use grab_set to set focus, and then
        # set topmost back to False. This brings the window to the front.
        # From the documentation it seems that using .lift(aboveThis=self) would
        # work, but I was not able to make that work.
        window.attributes("-topmost", True)
        window.grab_set()
        window.attributes("-topmost", False)


def change_ui_appearance_event(new_appearance_mode: str):
    """
    Handle the event to update the application appearance.
    :param new_appearance_mode: The appearance mode (System, Dark, Light)
    """
    ctk.set_appearance_mode(new_appearance_mode)


def change_ui_scaling_event(new_scaling: str):
    """
    Handle the event to update the application UI scale.
    :param new_scaling: The new scaling string on the form XX%
    """
    new_scaling_float = int(new_scaling.replace("%", "")) / 100
    ctk.set_widget_scaling(new_scaling_float)


if __name__ == "__main__":
    app = CamScanApp()
    app.mainloop()
