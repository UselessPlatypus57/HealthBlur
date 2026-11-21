"""Apply redactions to pixel arrays in their native dtype."""

from __future__ import annotations

import numpy as np

from ..config import Config, RedactMethod
from .ocr import TextBox


def padded(box: TextBox, shape_hw: tuple[int, int], cfg: Config) -> tuple[int, int, int, int]:
    h, w = shape_hw
    pad = cfg.pad_px + int(round(cfg.pad_ratio * box.height))
    return max(0, box.x0 - pad), max(0, box.y0 - pad), min(w, box.x1 + pad), min(h, box.y1 + pad)


def build_mask(shape_hw: tuple[int, int], boxes: list[TextBox], cfg: Config) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=bool)
    for b in boxes:
        x0, y0, x1, y1 = padded(b, shape_hw, cfg)
        mask[y0:y1, x0:x1] = True
    return mask


def fill_value(arr: np.ndarray, photometric: str):
    if arr.ndim == 4:  # color
        return 0
    if photometric == "MONOCHROME1":
        return arr.max()  # displays as black
    return arr.min()


def apply_redaction(arr: np.ndarray, mask: np.ndarray, boxes: list[tuple[int, int, int, int]], method: RedactMethod, photometric: str) -> np.ndarray:
    """``arr`` is (frames, rows, cols[, 3]); ``mask`` is (rows, cols). Every
    frame is redacted because burned-in text is normally static."""
    out = arr.copy()
    if not mask.any():
        return out
    if method == RedactMethod.FILL:
        out[:, mask] = fill_value(arr, photometric)
        return out
    for x0, y0, x1, y1 in boxes:
        if x1 <= x0 or y1 <= y0:
            continue
        for f in range(out.shape[0]):
            region = out[f, y0:y1, x0:x1].astype(np.float64)
            if method == RedactMethod.BLUR:
                r = max(4, (y1 - y0) // 2)
                for _ in range(3):
                    region = _box_blur(region, r)
            else:  # PIXELATE
                region = _pixelate(region, max(6, (y1 - y0) // 2))
            info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
            if info is not None:
                region = np.clip(np.rint(region), info.min, info.max)
            out[f, y0:y1, x0:x1] = region.astype(arr.dtype)
    return out


def integrity_ok(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> bool:
    """True if every pixel outside the redaction mask is bit-identical."""
    if before.shape != after.shape:
        return False
    keep = ~mask
    return bool(np.array_equal(before[:, keep], after[:, keep]))


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    for axis in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (r, r)
        p = np.pad(a, pad, mode="edge")
        c = np.cumsum(p, axis=axis)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        n = a.shape[axis]
        a = (np.take(c, range(2 * r + 1, 2 * r + 1 + n), axis=axis) - np.take(c, range(0, n), axis=axis)) / (2 * r + 1)
    return a


def _pixelate(a: np.ndarray, block: int) -> np.ndarray:
    out = a.copy()
    h, w = a.shape[:2]
    for y in range(0, h, block):
        for x in range(0, w, block):
            blk = a[y : y + block, x : x + block]
            out[y : y + block, x : x + block] = blk.mean(axis=(0, 1))
    return out
