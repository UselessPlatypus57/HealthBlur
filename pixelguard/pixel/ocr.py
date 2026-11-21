"""OCR engines for burned-in text."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np


@dataclass
class TextBox:
    x0: int
    y0: int
    x1: int
    y1: int
    text: str
    confidence: float
    frame: int = 0
    source: str = ""  # render pass that found it

    @property
    def area(self) -> int:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)

    @property
    def height(self) -> int:
        return max(1, self.y1 - self.y0)

    def iou(self, o: "TextBox") -> float:
        ix = max(0, min(self.x1, o.x1) - max(self.x0, o.x0))
        iy = max(0, min(self.y1, o.y1) - max(self.y0, o.y0))
        inter = ix * iy
        union = self.area + o.area - inter
        return inter / union if union else 0.0

    def containment(self, o: "TextBox") -> float:
        """Share of the smaller box that lies inside the other box."""
        ix = max(0, min(self.x1, o.x1) - max(self.x0, o.x0))
        iy = max(0, min(self.y1, o.y1) - max(self.y0, o.y0))
        smaller = min(self.area, o.area)
        return ix * iy / smaller if smaller else 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class OCREngine(Protocol):
    name: str

    def detect(self, image_u8: np.ndarray) -> list[TextBox]: ...


class EasyOCREngine:
    """EasyOCR (CRAFT detector + CRNN recogniser), CPU by default.

    Model weights (~100 MB) are downloaded to ~/.EasyOCR on first use.
    """

    name = "easyocr"
    _lock = threading.RLock()

    def __init__(self, languages=("en",), gpu: bool = False):
        self.languages = list(languages)
        self.gpu = gpu
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            with self._lock:
                if self._reader is None:
                    import easyocr

                    self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        return self._reader

    def detect(self, image_u8: np.ndarray) -> list[TextBox]:
        h, w = image_u8.shape[:2]
        mag = 2.0 if max(h, w) <= 640 else 1.0
        with self._lock:  # EasyOCR readers are not thread-safe
            results = self.reader.readtext(
                image_u8,
                detail=1,
                paragraph=False,
                min_size=6,
                text_threshold=0.6,
                low_text=0.3,
                link_threshold=0.4,
                mag_ratio=mag,
                canvas_size=max(2560, int(max(h, w) * mag)),
            )
        boxes = []
        for pts, text, conf in results:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            boxes.append(
                TextBox(
                    int(max(0, min(xs))), int(max(0, min(ys))), int(min(w, max(xs) + 1)), int(min(h, max(ys) + 1)),
                    str(text), float(conf),
                )
            )
        return boxes


class StaticOCREngine:
    """Returns pre-set boxes. Used in tests and for replaying ground truth."""

    name = "static"

    def __init__(self, boxes_by_call: list[list[TextBox]] | None = None, default: list[TextBox] | None = None):
        self.queue = list(boxes_by_call or [])
        self.default = default or []

    def detect(self, image_u8: np.ndarray) -> list[TextBox]:
        if self.queue:
            return [TextBox(**b.to_dict()) for b in self.queue.pop(0)]
        return [TextBox(**b.to_dict()) for b in self.default]


def merge_boxes(boxes: list[TextBox], iou_threshold: float = 0.4) -> list[TextBox]:
    """De-duplicate detections from several render passes / frames."""
    kept: list[TextBox] = []
    for b in sorted(boxes, key=lambda b: -b.confidence):
        dup = next((k for k in kept if k.iou(b) >= iou_threshold or k.containment(b) >= 0.8), None)
        if dup is None:
            kept.append(b)
        else:
            # A clearly wider detection of the same line ("Fitzgerald, Rowan"
            # vs "Fitzgerald,") carries more context, so keep its reading unless
            # it is far less certain than the narrower one (garbled re-read).
            wider = (b.x1 - b.x0) > 1.15 * (dup.x1 - dup.x0)
            if wider and len(b.text.strip()) > len(dup.text.strip()) and b.confidence >= 0.5 * dup.confidence:
                dup.text = b.text
            dup.x0, dup.y0 = min(dup.x0, b.x0), min(dup.y0, b.y0)
            dup.x1, dup.y1 = max(dup.x1, b.x1), max(dup.y1, b.y1)
            dup.confidence = max(dup.confidence, b.confidence)
    if len(kept) < len(boxes):  # a grown box may now contain others: repeat until stable
        return merge_boxes(kept, iou_threshold)
    return kept


_ENGINES: dict[tuple, OCREngine] = {}


def get_engine(name: str = "easyocr", languages=("en",)) -> OCREngine:
    key = (name, tuple(languages))
    if key not in _ENGINES:
        if name == "easyocr":
            _ENGINES[key] = EasyOCREngine(languages)
        else:
            raise ValueError(f"unknown OCR engine {name!r}")
    return _ENGINES[key]
