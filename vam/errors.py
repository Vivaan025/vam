"""Exceptions raised by vam (experiment 013). Catch `VamError` for all of them.

VideoOpenError  the container could not be opened (missing, truncated, not a video)
DecodeError     the container opened but no frame could be decoded for the request
ScanError       the keyframe scan produced nothing usable (no keyframes, demux failure)
"""


class VamError(Exception):
    """Base class for every error vam raises on purpose."""


class VideoOpenError(VamError, IOError):
    def __init__(self, path: str, detail: str = ""):
        self.path = path
        super().__init__(f"could not open video: {path}" + (f" ({detail})" if detail else ""))


class DecodeError(VamError, ValueError):
    def __init__(self, path: str, detail: str = ""):
        self.path = path
        super().__init__(f"could not decode frames from {path}" + (f" ({detail})" if detail else ""))


class ScanError(VamError, ValueError):
    def __init__(self, path: str, detail: str = ""):
        self.path = path
        super().__init__(f"keyframe scan failed for {path}" + (f" ({detail})" if detail else ""))
