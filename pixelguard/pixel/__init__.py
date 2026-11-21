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
    projections: bool = False  # temporal max/min projections were scanned

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
    original: np.ndarray | None = None  # for the integrity check after remediation
    remediated_regions: int = 0

    def to_dict(self, previews: bool) -> dict:
        s = self.scan
        return {
            "performed": self.performed,
            "skipped_reason": self.skipped_reason,
            "frames": s.n_frames if s else None,
            "frames_scanned": s.frames_scanned if s else [],
            "temporal_projections": s.projections if s else False,
            "ocr_seconds": round(s.seconds, 3) if s else 0.0,
            "redact_seconds": round(self.redact_seconds, 3),
            "text_regions": [d.to_dict(previews) for d in s.decisions] if s else [],
            "redacted_regions": len(self.redacted_boxes),
            "redacted_boxes_padded": self.redacted_boxes,
            "redacted_pixel_fraction": round(self.redacted_fraction, 6),
            "integrity_outside_mask": self.integrity_ok,
            "remediated_regions": self.remediated_regions,
            "shape": list(self.shape) if self.shape else None,
        }


def frames_to_scan(n: int, cfg: Config) -> list[int]:
    if cfg.ocr_all_frames or n <= cfg.ocr_max_frames:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, cfg.ocr_max_frames)})


PROJECTION_MAX, PROJECTION_MIN = -1, -2  # pseudo frame indices in reports


def border_bands(u8: np.ndarray, cfg: Config):
    """Top and bottom bands of the image (where burned-in text lives),
    contrast-stretched and enlarged so that small, faint text is legible to
    the recogniser. Yields (band_u8, y_offset, scale)."""
    from PIL import Image

    g = u8 if u8.ndim == 2 else np.asarray(Image.fromarray(u8).convert("L"))
    h, w = g.shape
    bh = min(h // 2, max(40, int(h * cfg.border_band_fraction)))
    for y0 in sorted({0, h - bh}):
        band = g[y0 : y0 + bh].astype(np.float32)
        lo, hi = np.percentile(band, 1), np.percentile(band, 99.7)
        if hi - lo < 4:
            continue
        band = ((band - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
        s = cfg.border_zoom_scale
        big = np.asarray(Image.fromarray(band).resize((w * s, bh * s), Image.BICUBIC))
        yield big, y0, s


def _views(arr: np.ndarray, ds: Dataset, cfg: Config, res: ScanResult):
    """Everything OCR should look at: (frame label, pass name, image, y offset, scale)."""
    frames = frames_to_scan(arr.shape[0], cfg)
    res.frames_scanned = list(frames)
    planes: list[tuple[int, np.ndarray]] = [(fi, arr[fi]) for fi in frames]
    if cfg.ocr_projections and arr.shape[0] > len(frames):
        # Temporal projections contain every frame's brightest / darkest pixel,
        # so text shown in any frame is visible in at least one of them.
        planes.append((PROJECTION_MAX, arr.max(axis=0)))
        if arr.ndim == 3:
            planes.append((PROJECTION_MIN, arr.min(axis=0)))
        res.projections = True
    for idx, (fi, frame) in enumerate(planes):
        passes = ("minmax",) if frame.ndim == 3 else cfg.ocr_passes
        rendered: list[np.ndarray] = []
        for mode in passes:
            u8 = frame_to_u8(frame, ds, mode)
            if u8 is None or any(np.array_equal(u8, r) for r in rendered):
                continue
            rendered.append(u8)
            yield fi, mode, u8, 0, 1
        # Zoomed border bands: first frame and projections are enough, since
        # burned-in text is static.
        if cfg.border_zoom and rendered and (idx == 0 or fi < 0):
            for band, y0, s in border_bands(rendered[0], cfg):
                yield fi, "border-zoom", band, y0, s


def scan_array(arr: np.ndarray, ds: Dataset, known: KnownIdentifiers | None, cfg: Config, engine: OCREngine) -> ScanResult:
    t0 = time.perf_counter()
    res = ScanResult(n_frames=arr.shape[0])
    boxes: list[TextBox] = []
    h, w = arr.shape[1:3]
    for fi, mode, img, y0, s in _views(arr, ds, cfg, res):
        for b in engine.detect(img):
            if s != 1:
                b.x0, b.x1 = int(b.x0 / s), min(w, int(np.ceil(b.x1 / s)))
                b.y0, b.y1 = int(b.y0 / s) + y0, min(h, int(np.ceil(b.y1 / s)) + y0)
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
    pr.original = arr
    pr.scan = scan_array(arr, ds, known, cfg, engine)
    if keep_previews:
        pr.before_preview = preview_u8(arr, ds)

    t0 = time.perf_counter()
    hw = arr.shape[1:3]
    to_redact = [d.box for d in pr.scan.redact]
    keep = [d.box for d in pr.scan.decisions if d.decision == "keep"]
    pr.mask = build_mask(hw, to_redact, cfg, keep)
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


def remediate(ds: Dataset, pr: PixelResult, boxes: list[list[int]], cfg: Config, *, keep_previews: bool = False) -> int:
    """Redact regions the audit still found readable in the written file ``ds``."""
    arr, info = load_pixels(ds)
    hw = arr.shape[1:3]
    tbs = [TextBox(int(b[0]), int(b[1]), int(b[2]), int(b[3]), "", 1.0) for b in boxes]
    add = build_mask(hw, tbs, cfg)
    pad = [padded(b, hw, cfg) for b in tbs]
    new = apply_redaction(arr, add, pad, cfg.redact_method, info.photometric)
    store_pixels(ds, new, info)
    pr.mask = add if pr.mask is None else (pr.mask | add)
    pr.redacted_boxes += [list(b) for b in pad]
    pr.redacted_fraction = float(pr.mask.mean())
    if pr.original is not None and pr.original.shape == new.shape:
        pr.integrity_ok = integrity_ok(pr.original, new, pr.mask)
    pr.remediated_regions += len(tbs)
    if keep_previews:
        pr.after_preview = preview_u8(new, ds)
    return len(tbs)


__all__ = [
    "OCREngine", "PixelInfo", "PixelResult", "ScanResult", "StaticOCREngine", "TextBox", "TextDecision",
    "UnsupportedPixelData", "clean_pixels", "get_engine", "load_pixels", "preview_u8", "remediate", "scan_array",
]
