"""
This module provides an abstracted Camera class wrapping OpenCV video capture.
"""

import logging
import platform

import cv2

# Depending on the platform, there might be the need to change the API backend
# preference. For Windows specifically we use DirectShow. Read more here:
# https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
SYSTEM = platform.system()
if SYSTEM == "Windows":
    API_PREFERENCE = cv2.CAP_DSHOW
elif SYSTEM == "Darwin":
    API_PREFERENCE = cv2.CAP_AVFOUNDATION
else:
    API_PREFERENCE = cv2.CAP_ANY


def enable_camera_autofocus(device_name: str = None):
    """
    On macOS, explicitly commands connected cameras (like Logitech BRIO)
    to enable Continuous Auto Focus and Continuous Auto Exposure via AVFoundation.
    """
    if SYSTEM != "Darwin":
        return
    try:
        import ctypes, ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.restype = ctypes.c_void_p

        ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/AVFoundation.framework/AVFoundation"
        )
        avf = ctypes.CDLL(
            "/System/Library/Frameworks/AVFoundation.framework/AVFoundation"
        )
        ptr = ctypes.c_void_p.in_dll(avf, "AVMediaTypeVideo")
        ns_video = ctypes.cast(
            ctypes.byref(ptr), ctypes.POINTER(ctypes.c_void_p)
        ).contents.value

        AVCaptureDevice = objc.objc_getClass(b"AVCaptureDevice")
        devicesWithMediaType = objc.sel_registerName(b"devicesWithMediaType:")
        count_sel = objc.sel_registerName(b"count")
        objectAtIndex_sel = objc.sel_registerName(b"objectAtIndex:")
        localizedName_sel = objc.sel_registerName(b"localizedName")
        UTF8String_sel = objc.sel_registerName(b"UTF8String")
        isFocusModeSupported_sel = objc.sel_registerName(b"isFocusModeSupported:")
        setFocusMode_sel = objc.sel_registerName(b"setFocusMode:")
        isExposureModeSupported_sel = objc.sel_registerName(b"isExposureModeSupported:")
        setExposureMode_sel = objc.sel_registerName(b"setExposureMode:")
        lockForConfiguration_sel = objc.sel_registerName(b"lockForConfiguration:")
        unlockForConfiguration_sel = objc.sel_registerName(b"unlockForConfiguration")

        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        dev_list = objc.objc_msgSend(AVCaptureDevice, devicesWithMediaType, ns_video)
        objc.objc_msgSend.restype = ctypes.c_size_t
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        count = objc.objc_msgSend(dev_list, count_sel)

        for i in range(count):
            objc.objc_msgSend.restype = ctypes.c_void_p
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            dev = objc.objc_msgSend(dev_list, objectAtIndex_sel, i)
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            ns = objc.objc_msgSend(dev, localizedName_sel)
            objc.objc_msgSend.restype = ctypes.c_char_p
            name = objc.objc_msgSend(ns, UTF8String_sel).decode("utf-8")

            if device_name is None or device_name.lower() in name.lower() or "brio" in name.lower():
                err = ctypes.c_void_p(0)
                objc.objc_msgSend.restype = ctypes.c_bool
                objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                locked = objc.objc_msgSend(dev, lockForConfiguration_sel, ctypes.byref(err))
                if locked:
                    # 2 = AVCaptureFocusModeContinuousAutoFocus
                    objc.objc_msgSend.restype = ctypes.c_bool
                    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
                    if objc.objc_msgSend(dev, isFocusModeSupported_sel, 2):
                        objc.objc_msgSend.restype = None
                        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
                        objc.objc_msgSend(dev, setFocusMode_sel, 2)

                    # 2 = AVCaptureExposureModeContinuousAutoExposure
                    objc.objc_msgSend.restype = ctypes.c_bool
                    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
                    if objc.objc_msgSend(dev, isExposureModeSupported_sel, 2):
                        objc.objc_msgSend.restype = None
                        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
                        objc.objc_msgSend(dev, setExposureMode_sel, 2)

                    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                    objc.objc_msgSend(dev, unlockForConfiguration_sel)
                    logging.info(f"Successfully activated autofocus & auto-exposure on {name}")
    except Exception as e:
        logging.warning(f"Failed to configure autofocus on macOS: {e}")


def get_available_devices() -> list[tuple[int, str]]:
    """
    Returns a list of (index, display_name) for connected cameras.
    On macOS, physical cameras (Logitech BRIO, FaceTime) are mapped first to match
    OpenCV's indexing, with virtual camera drivers (OBS) mapped last.
    """
    devices = []
    if SYSTEM == "Darwin":
        try:
            import ctypes, ctypes.util

            objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
            objc.objc_getClass.restype = ctypes.c_void_p
            objc.sel_registerName.restype = ctypes.c_void_p
            objc.objc_msgSend.restype = ctypes.c_void_p
            ctypes.cdll.LoadLibrary(
                "/System/Library/Frameworks/AVFoundation.framework/AVFoundation"
            )
            avf = ctypes.CDLL(
                "/System/Library/Frameworks/AVFoundation.framework/AVFoundation"
            )
            ptr = ctypes.c_void_p.in_dll(avf, "AVMediaTypeVideo")
            ns_video = ctypes.cast(
                ctypes.byref(ptr), ctypes.POINTER(ctypes.c_void_p)
            ).contents.value

            AVCaptureDevice = objc.objc_getClass(b"AVCaptureDevice")
            devicesWithMediaType = objc.sel_registerName(b"devicesWithMediaType:")
            count_sel = objc.sel_registerName(b"count")
            objectAtIndex_sel = objc.sel_registerName(b"objectAtIndex:")
            localizedName_sel = objc.sel_registerName(b"localizedName")
            UTF8String_sel = objc.sel_registerName(b"UTF8String")

            objc.objc_msgSend.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            dev_list = objc.objc_msgSend(
                AVCaptureDevice, devicesWithMediaType, ns_video
            )

            count_sel = objc.sel_registerName(b"count")
            objc.objc_msgSend.restype = ctypes.c_size_t
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            count = objc.objc_msgSend(dev_list, count_sel)

            raw_names = []
            for i in range(count):
                objc.objc_msgSend.restype = ctypes.c_void_p
                objc.objc_msgSend.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                ]
                dev = objc.objc_msgSend(dev_list, objectAtIndex_sel, i)

                objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                name_ns = objc.objc_msgSend(dev, localizedName_sel)

                objc.objc_msgSend.restype = ctypes.c_char_p
                name_bytes = objc.objc_msgSend(name_ns, UTF8String_sel)
                name_str = (
                    name_bytes.decode("utf-8") if name_bytes else f"Camera {i}"
                )
                raw_names.append(name_str)

            # In OpenCV on macOS, external hardware cameras (e.g. Logitech BRIO)
            # are indexed first (0), followed by built-in FaceTime (1), and virtual extensions (OBS) last
            external_hw = [n for n in raw_names if any(k in n.lower() for k in ["brio", "logitech", "usb", "uvc", "external"])]
            builtin_hw = [n for n in raw_names if any(k in n.lower() for k in ["facetime", "macbook", "apple", "internal"]) and n not in external_hw]
            other_hw = [n for n in raw_names if n not in external_hw and n not in builtin_hw and not any(v in n.lower() for v in ["virtual", "obs", "snap"])]
            virtual_cams = [n for n in raw_names if any(v in n.lower() for v in ["virtual", "obs", "snap"])]

            ordered = external_hw + builtin_hw + other_hw + virtual_cams
            for i, name in enumerate(ordered):
                devices.append((i, name))
        except Exception as e:
            logging.warning(f"Error enumerating AVFoundation devices: {e}")

    if not devices:
        for index in range(5):
            cap = (
                cv2.VideoCapture(index, API_PREFERENCE)
                if API_PREFERENCE is not None
                else cv2.VideoCapture(index)
            )
            if cap.isOpened():
                devices.append((index, f"Camera {index}"))
            cap.release()

    return devices


class Camera:
    """
    A class representing a video camera. It mainly wraps and abstracts an OpenCV
    VideoCapture object, and makes tasks like changing resolution and input
    devices a bit easier.
    """

    def __init__(
        self,
        index: int = None,
        resolution: tuple[int, int] = (1920, 1080),
        target_fps: int = 30,
    ):
        devs = get_available_devices()
        self.available_devices = devs

        if index is None:
            # Auto-select the best camera: prefer physical USB/document/FaceTime cameras over virtual (OBS)
            selected_idx = 0
            if devs:
                # First, look for external webcams / document cameras
                for idx, name in devs:
                    nl = name.lower()
                    if any(k in nl for k in ["brio", "logitech", "document", "usb"]):
                        selected_idx = idx
                        break
                else:
                    # Next, look for built-in FaceTime / physical camera
                    for idx, name in devs:
                        nl = name.lower()
                        if "virtual" not in nl and "obs" not in nl:
                            selected_idx = idx
                            break
                    else:
                        selected_idx = devs[0][0]
            self.index = selected_idx
        else:
            self.index = index

        self.device_name = "Camera"
        for idx, name in devs:
            if idx == self.index:
                self.device_name = name
                break

        self.resolution = resolution
        self.target_fps = target_fps
        self._video_capture = None
        self.initialize()

    def initialize(self):
        """
        Initialize the camera by opening a video capture feed using settings
        like resolution and framerate specified in this instance.
        """
        if self._video_capture is not None:
            try:
                self._video_capture.release()
            except Exception:
                pass

        if API_PREFERENCE is not None:
            self._video_capture = cv2.VideoCapture(self.index, API_PREFERENCE)
        else:
            self._video_capture = cv2.VideoCapture(self.index)

        if not self._video_capture.isOpened():
            self._video_capture = cv2.VideoCapture(self.index)

        if self._video_capture.isOpened():
            self._video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._video_capture.set(cv2.CAP_PROP_FPS, self.target_fps)
            logging.info(f"Camera {self.index} ({self.device_name}) successfully opened.")
            enable_camera_autofocus(self.device_name)
        else:
            logging.error(f"Cannot open camera {self.index}")

    def set_index(self, index: int):
        """
        Set the OpenCV device index of the camera.
        """
        self.index = index
        for idx, name in self.available_devices:
            if idx == self.index:
                self.device_name = name
                break
        self.initialize()

    def set_resolution(self, resolution: tuple[int, int]):
        """
        Set the capture resolution of the camera.
        :param resolution: A tuple of the resolution on the form (width, height)
        """
        self.resolution = resolution
        self.initialize()

    def show_settings(self, master=None):
        """
        Bring up real interactive camera controls: Auto-Focus trigger,
        exposure controls, and driver settings dialog.
        """
        if SYSTEM == "Windows":
            self._video_capture.set(cv2.CAP_PROP_SETTINGS, 1)
            return

        import customtkinter as ctk

        dialog = ctk.CTkToplevel(master)
        dialog.title(f"Camera Controls - {self.device_name}")
        dialog.geometry("380x280")
        dialog.attributes("-topmost", True)

        title = ctk.CTkLabel(
            dialog,
            text=f"⚙️ {self.device_name}",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        title.pack(pady=(15, 10))

        status_lbl = ctk.CTkLabel(dialog, text="Autofocus: Ready", text_color="#4CAF50")

        def _refocus():
            enable_camera_autofocus(self.device_name)
            status_lbl.configure(
                text="✓ Continuous Autofocus Activated!", text_color="#4CAF50"
            )

        focus_btn = ctk.CTkButton(
            dialog,
            text="🔍 Trigger Continuous Autofocus Now",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            fg_color="#1976D2",
            hover_color="#1565C0",
            command=_refocus,
        )
        focus_btn.pack(padx=20, pady=10, fill="x")

        status_lbl.pack(pady=4)

        info_lbl = ctk.CTkLabel(
            dialog,
            text=(
                "Controls Logitech BRIO continuous focus & auto-exposure\n"
                "directly via macOS hardware layer."
            ),
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        info_lbl.pack(padx=20, pady=(5, 15))

        close_btn = ctk.CTkButton(
            dialog,
            text="Done",
            command=dialog.destroy,
            width=100,
        )
        close_btn.pack(pady=5)

    def capture(self) -> cv2.Mat:
        """
        Capture an image from the video stream and extract documents from it.
        :return: An OpenCV image of the captured frame
        """
        is_frame_read_correctly, img_capture = self._video_capture.read()

        if not is_frame_read_correctly:
            return None

        return img_capture

    def get_available_device_indices(self) -> list[int]:
        """
        Identify cameras available to OpenCV by naively attempting to initiate
        video capture on a range of device indices and saving the ones that
        successfully open.
        :return: A list of device indices where the video capture was successful
        """
        found_camera_indices = []
        for index in range(10):
            dummy_capture = cv2.VideoCapture(
                index=index,
                apiPreference=API_PREFERENCE,
            )
            if dummy_capture.isOpened():
                found_camera_indices.append(index)
            dummy_capture.release()

        # Ensure the camera is still properly initiated after opening captures
        self.initialize()

        return found_camera_indices
