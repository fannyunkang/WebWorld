"""Screenshot loading helpers for Qwen-VL style multimodal messages."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScreenshotPayload:
    path: str
    mime_type: str
    base64_data: str


def encode_screenshot(path: str | Path) -> ScreenshotPayload:
    """Encode a screenshot as a transport-safe payload for VLM clients."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Screenshot not found: {image_path}")

    suffix = image_path.suffix.lower()
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")

    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return ScreenshotPayload(
        path=str(image_path),
        mime_type=mime_type,
        base64_data=data,
    )
