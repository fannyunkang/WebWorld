"""Collect multimodal browser observations from a Playwright page."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mm_webagent.browser.element_registry import BoundingBox, ElementInfo, ElementRegistry


INTERACTIVE_SELECTOR = "a,button,input,textarea,select,[role],[tabindex]"


@dataclass
class BrowserObservation:
    screenshot: str
    page_state: str
    url: str
    viewport: dict[str, int | None]
    scroll: dict[str, float]
    element_registry: ElementRegistry = field(default_factory=dict)


class PlaywrightObserver:
    """Capture screenshots and a compact element tree for Qwen-VL policies."""

    def __init__(self, output_dir: str | Path = "runs/browser_observations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def observe(self, page: Any, step_index: int = 0) -> BrowserObservation:
        registry = self._register_interactive_elements(page)
        screenshot_path = self.output_dir / f"step_{step_index:04d}.png"
        page.screenshot(path=str(screenshot_path), full_page=False)
        viewport = page.viewport_size or {"width": None, "height": None}
        scroll = page.evaluate("() => ({x: window.scrollX, y: window.scrollY})")
        return BrowserObservation(
            screenshot=str(screenshot_path),
            page_state=self._page_state(page, registry),
            url=page.url,
            viewport=viewport,
            scroll=scroll,
            element_registry=registry,
        )

    def _register_interactive_elements(self, page: Any) -> ElementRegistry:
        handles = page.query_selector_all(INTERACTIVE_SELECTOR)
        registry: ElementRegistry = {}
        bid = 1
        for handle in handles:
            try:
                if not handle.is_visible():
                    continue
                bbox = handle.bounding_box()
                if not bbox:
                    continue
                current_bid = str(bid)
                handle.evaluate("(el, bid) => el.setAttribute('data-mm-bid', bid)", current_bid)
                info = handle.evaluate(
                    """el => ({
                        role: el.getAttribute('role') || el.tagName.toLowerCase(),
                        name: el.getAttribute('aria-label') || el.innerText || el.value || el.getAttribute('placeholder') || '',
                        type: el.getAttribute('type') || '',
                        href: el.getAttribute('href') || ''
                    })"""
                )
                registry[current_bid] = ElementInfo(
                    bid=current_bid,
                    role=str(info.get("role") or "element"),
                    name=" ".join(str(info.get("name") or "").split()),
                    selector=f'[data-mm-bid="{current_bid}"]',
                    bbox=BoundingBox(
                        x=float(bbox["x"]),
                        y=float(bbox["y"]),
                        width=float(bbox["width"]),
                        height=float(bbox["height"]),
                    ),
                    attributes={"type": info.get("type", ""), "href": info.get("href", "")},
                )
                bid += 1
            except Exception:
                continue
        return registry

    def _page_state(self, page: Any, registry: ElementRegistry) -> str:
        title = page.title()
        lines = [f"RootWebArea {title!r}, url={page.url!r}"]
        for info in registry.values():
            attrs = []
            if info.attributes.get("type"):
                attrs.append(f"type={info.attributes['type']!r}")
            if info.bbox:
                attrs.append(f"bbox=({info.bbox.x:.0f},{info.bbox.y:.0f},{info.bbox.width:.0f},{info.bbox.height:.0f})")
            suffix = ", " + ", ".join(attrs) if attrs else ""
            lines.append(f"[{info.bid}] {info.role} {info.name!r}, clickable, visible{suffix}")
        return "\n".join(lines)
