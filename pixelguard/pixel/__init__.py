"""Pixel stage: find burned-in text with OCR and redact identifying text."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from pydicom.dataset import Dataset

from ..config import Config
from ..phi import KnownIdentifiers
from .ocr import OCREngine, StaticOCREngine, TextBox, get_engine, merge_boxes
from .policy import TextDecision, decide
from .redact import apply_redaction, build_mask, integrity_ok, padded
from .render import PixelInfo, UnsupportedPixelData, frame_to_u8, load_pixels, preview_u8, store_pixels


@dataclass
class ScanResult:
    decisions: list[TextDecision] = field(default_factory=list)
    frames_scanned: list[int] = field(default_factory=list)
    n_frames: int = 1
    seconds: float = 0.0

    @property
    def redact(self) -> list[TextDecision]:
        return [d for d in self.decisions if d.decision == "redact"]


@dataclass
class PixelResult:
    performed: bool = False
    skipped_reason: str | None = None
    scan: ScanResult | None = None
    redacted_boxes: list[list[int]] = field(default_factory=list)
    redacted_fraction: float = 0.0
    integrity_ok: bool | None = None
    redact_seconds: float = 0.0
    shape: tuple | None = None
    mask: np.ndarray | None = None
    before_preview: np.ndarray | None = None
    after_preview: np.ndarray | None = None

    def to_dict(self, previews: bool) -> dict:
        s = self.scan
        return {
            "performed": self.performed,
            "skipped_reason": self.skipped_reason,
            "frames": s.n_frames if s else None,
            "frames_scanned": s.frames_scanned if s else [],
            "ocr_seconds": round(s.seconds, 3) if s else 0.0,
            "redact_seconds": round(self.redact_seconds, 3),
            "text_regions": [d.to_dict(previews) for d in s.decisions] if s else [],
            "redacted_regions": len(self.redacted_boxes),
            "redacted_boxes_padded": self.redacted_boxes,
            "redacted_pixel_fraction": round(self.redacted_fraction, 6),
            "integrity_outside_mask": self.integrity_ok,
            "shape": list(self.shape) if self.shape else None,
        }


def frames_to_scan(n: int, cfg: Config) -> list[int]:
    if cfg.ocr_all_frames or n <= cfg.ocr_max_frames:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, cfg.ocr_max_frames)})


def scan_array(arr: np.ndarray, ds: Dataset, known: KnownIdentifiers | None, cfg: Config, engine: OCREngine) -> ScanResult:
    t0 = time.perf_counter()
    res = ScanResult(n_frames=arr.shape[0])
    boxes: list[TextBox] = []
    for fi in frames_to_scan(arr.shape[0], cfg):
        res.frames_scanned.append(fi)
        frame = arr[fi]
        passes = ("minmax",) if frame.ndim == 3 else cfg.ocr_passes
        rendered: list[np.ndarray] = []
        for mode in passes:
            u8 = frame_to_u8(frame, ds, mode)
            if u8 is None or any(np.array_equal(u8, r) for r in rendered):
                continue
            rendered.append(u8)
            for b in engine.detect(u8):
                b.frame, b.source = fi, mode
                boxes.append(b)
    merged = merge_boxes(boxes)
    res.decisions = decide(merged, known, cfg, arr.shape[1:3])
    res.seconds = time.perf_counter() - t0
    return res


def clean_pixels(ds: Dataset, known: KnownIdentifiers, cfg: Config, engine: OCREngine, *, keep_previews: bool = False) -> PixelResult:
    """Detect and redact burned-in text in ``ds`` (modified in place)."""
    pr = PixelResult()
    try:
        arr, info = load_pixels(ds)
    except UnsupportedPixelData as exc:
        pr.skipped_reason = str(exc)
        return pr
    pr.shape = arr.shape
    pr.scan = scan_array(arr, ds, known, cfg, engine)
    if keep_previews:
        pr.before_preview = preview_u8(arr, ds)

    t0 = time.perf_counter()
    hw = arr.shape[1:3]
    to_redact = [d.box for d in pr.scan.redact]
    pr.mask = build_mask(hw, to_redact, cfg)
    boxes = [padded(b, hw, cfg) for b in to_redact]
    pr.redacted_boxes = [list(b) for b in boxes]
    new = apply_redaction(arr, pr.mask, boxes, cfg.redact_method, info.photometric)
    pr.integrity_ok = integrity_ok(arr, new, pr.mask)
    pr.redacted_fraction = float(pr.mask.mean())
    store_pixels(ds, new, info)
    pr.redact_seconds = time.perf_counter() - t0
    pr.performed = True
    if keep_previews:
        pr.after_preview = preview_u8(new, ds)
    return pr


__all__ = [
    "OCREngine", "PixelInfo", "PixelResult", "ScanResult", "StaticOCREngine", "TextBox", "TextDecision",
    "UnsupportedPixelData", "clean_pixels", "get_engine", "load_pixels", "preview_u8", "scan_array",
]
