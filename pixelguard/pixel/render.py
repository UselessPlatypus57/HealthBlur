"""Decode DICOM pixel data and render 8-bit views for OCR and previews."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydicom.dataset import Dataset


class UnsupportedPixelData(Exception):
    pass


@dataclass
class PixelInfo:
    n_frames: int
    samples: int  # 1 = grayscale, 3 = color
    photometric: str
    was_compressed: bool
    converted_to_rgb: bool


def load_pixels(ds: Dataset) -> tuple[np.ndarray, PixelInfo]:
    """Return pixel data as (frames, rows, cols[, 3]) and update ``ds`` so
    that writing the array back as uncompressed little endian is valid."""
    if "PixelData" not in ds:
        raise UnsupportedPixelData("no Pixel Data element")
    if int(ds.get("BitsAllocated", 16)) == 1:
        raise UnsupportedPixelData("1-bit pixel data is not supported")

    tsyntax = ds.file_meta.TransferSyntaxUID if "TransferSyntaxUID" in getattr(ds, "file_meta", {}) else None
    was_compressed = bool(tsyntax is not None and tsyntax.is_compressed)
    converted = False
    if was_compressed:
        try:
            ds.decompress(as_rgb=True)
        except Exception as exc:  # missing codec, corrupt stream
            raise UnsupportedPixelData(f"cannot decompress {tsyntax.name}: {exc}") from exc
        converted = str(ds.get("PhotometricInterpretation", "")) == "RGB" and int(ds.get("SamplesPerPixel", 1)) == 3

    photometric = str(ds.get("PhotometricInterpretation", "MONOCHROME2"))
    try:
        ds.pixel_array_options(as_rgb=False)
    except Exception:
        pass
    arr = ds.pixel_array
    samples = int(ds.get("SamplesPerPixel", 1))
    n_frames = int(ds.get("NumberOfFrames", 1) or 1)

    if samples == 3 and photometric.startswith("YBR"):
        from pydicom.pixels import convert_color_space

        arr = convert_color_space(arr, photometric, "RGB")
        ds.PhotometricInterpretation = "RGB"
        photometric = "RGB"
        converted = True
    if samples == 3:
        ds.PlanarConfiguration = 0  # numpy array is interleaved (rows, cols, 3)
    elif photometric not in ("MONOCHROME1", "MONOCHROME2"):
        if photometric == "PALETTE COLOR":
            raise UnsupportedPixelData("palette color images are not supported")

    if n_frames == 1:
        arr = arr[np.newaxis, ...]
    return np.array(arr, copy=True), PixelInfo(n_frames, samples, photometric, was_compressed, converted)


def store_pixels(ds: Dataset, arr: np.ndarray, info: PixelInfo) -> None:
    out = arr[0] if info.n_frames == 1 else arr
    ds.PixelData = np.ascontiguousarray(out).tobytes()
    if info.was_compressed or info.converted_to_rgb:
        from pydicom.uid import ExplicitVRLittleEndian

        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds["PixelData"].is_undefined_length = False


def frame_to_u8(frame: np.ndarray, ds: Dataset, mode: str = "minmax") -> np.ndarray | None:
    """8-bit rendering of one frame. ``mode``: minmax | voi | equalize."""
    if frame.ndim == 3:  # color
        f = frame.astype(np.float32)
        if frame.dtype != np.uint8:
            lo, hi = float(f.min()), float(f.max())
            f = (f - lo) / max(hi - lo, 1.0) * 255.0
        return f.clip(0, 255).astype(np.uint8)

    f = frame.astype(np.float32)
    if mode == "voi":
        if "WindowCenter" not in ds or "WindowWidth" not in ds:
            return None
        try:
            from pydicom.pixels import apply_modality_lut, apply_voi_lut

            f = apply_voi_lut(apply_modality_lut(frame, ds), ds, index=0).astype(np.float32)
        except Exception:
            return None
    if mode == "equalize":
        hist, bins = np.histogram(f.ravel(), bins=1024)
        cdf = hist.cumsum().astype(np.float64)
        cdf /= max(cdf[-1], 1)
        f = np.interp(f.ravel(), bins[:-1], cdf).reshape(f.shape) * 255.0
    else:
        lo, hi = float(f.min()), float(f.max())
        f = (f - lo) / max(hi - lo, 1e-6) * 255.0
    u8 = f.clip(0, 255).astype(np.uint8)
    if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
        u8 = 255 - u8
    return u8


def preview_u8(arr: np.ndarray, ds: Dataset, frame: int = 0) -> np.ndarray:
    fr = arr[min(frame, arr.shape[0] - 1)]
    out = frame_to_u8(fr, ds, "voi") if fr.ndim == 2 else None
    if out is None:
        out = frame_to_u8(fr, ds, "minmax")
    return out
