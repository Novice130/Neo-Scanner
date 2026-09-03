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


def get_available_devices() -> list[tuple[int, str]]:
    """
    Returns a list of (index, display_name) for connected cameras.
    On macOS, queries AVFoundation via native runtime ctypes to get exact camera names.
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

            AVCaptureDevice = objc.objc_getClass(b"AVCaptureDevice")
            devicesWithMediaType = objc.sel_registerName(b"devicesWithMediaType:")
            NSConstantString = objc.objc_getClass(b"NSString")
            stringWithUTF8String = objc.sel_registerName(b"stringWithUTF8String:")

            objc.objc_msgSend.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_char_p,
            ]
            video_type = objc.objc_msgSend(
                NSConstantString, stringWithUTF8String, b"vide"
            )

            objc.objc_msgSend.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            dev_list = objc.objc_msgSend(
                AVCaptureDevice, devicesWithMediaType, video_type
            )

            count_sel = objc.sel_registerName(b"count")
            objectAtIndex_sel = objc.sel_registerName(b"objectAtIndex:")
            localizedName_sel = objc.sel_registerName(b"localizedName")
            UTF8String_sel = objc.sel_registerName(b"UTF8String")

            objc.objc_msgSend.restype = ctypes.c_size_t
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            count = objc.objc_msgSend(dev_list, count_sel)

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
                devices.append((i, name_str))
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
            logging.info(f"Camera {self.index} successfully opened.")
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

    def show_settings(self):
        """
        Bring up the settings of the camera as a dialog window.
        """
        self._video_capture.set(cv2.CAP_PROP_SETTINGS, 1)

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
