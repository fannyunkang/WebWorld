"""Annotate screenshots with element ids for visual grounding."""

from __future__ import annotations

from pathlib import Path

from mm_webagent.browser.element_registry import ElementRegistry


def annotate_screenshot(
    screenshot_path: str | Path,
    registry: ElementRegistry,
    output_path: str | Path | None = None,
) -> str:
    """Draw element boxes and bid labels on a screenshot.

    Pillow is imported lazily so text-only workflows do not require it.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Install pillow to annotate screenshots.") from exc

    source = Path(screenshot_path)
    target = Path(output_path) if output_path else source.with_name(f"{source.stem}_annotated{source.suffix}")
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for info in registry.values():
        if not info.bbox:
            continue
        x1 = info.bbox.x
        y1 = info.bbox.y
        x2 = x1 + info.bbox.width
        y2 = y1 + info.bbox.height
        label = str(info.bid)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=2)
        text_box = draw.textbbox((x1, y1), label, font=font)
        pad = 3
        draw.rectangle(
            (text_box[0] - pad, text_box[1] - pad, text_box[2] + pad, text_box[3] + pad),
            fill=(255, 64, 64),
        )
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)

    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return str(target)
