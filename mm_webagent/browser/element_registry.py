"""Element registry shared by browser observers and action executors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


@dataclass(frozen=True)
class ElementInfo:
    bid: str
    role: str
    name: str
    selector: str | None = None
    bbox: BoundingBox | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


ElementRegistry = dict[str, ElementInfo]
