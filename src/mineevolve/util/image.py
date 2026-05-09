"""POV / image utilities."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np


def encode_pov_to_base64(pov: Any, format: str = "PNG") -> str:
    """Encode a HxWx3 uint8 numpy array as a base64 PNG string."""

    if not isinstance(pov, np.ndarray):
        return ""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return ""
    img = Image.fromarray(pov.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("ascii")
