import numpy as np

from pixelguard.config import Config, RedactMethod
from pixelguard.pixel.ocr import TextBox
from pixelguard.pixel.redact import apply_redaction, build_mask, integrity_ok, padded


def _case(arr, photometric, method=RedactMethod.FILL):
    cfg = Config()
    box = TextBox(10, 10, 40, 20, "x", 1.0)
    hw = arr.shape[1:3]
    mask = build_mask(hw, [box], cfg)
    out = apply_redaction(arr, mask, [padded(box, hw, cfg)], method, photometric)
    return out, mask


def test_fill_monochrome2_uses_minimum_and_preserves_dtype():
    rng = np.random.default_rng(0)
    arr = rng.integers(100, 4000, (1, 64, 64), dtype=np.uint16)
    out, mask = _case(arr, "MONOCHROME2")
    assert out.dtype == np.uint16
    assert (out[0][mask] == arr.min()).all()
    assert integrity_ok(arr, out, mask)


def test_fill_monochrome1_uses_maximum():
    arr = np.random.default_rng(1).integers(0, 4095, (1, 64, 64), dtype=np.uint16)
    out, mask = _case(arr, "MONOCHROME1")
    assert (out[0][mask] == arr.max()).all()


def test_fill_rgb_multiframe_all_frames():
    arr = np.random.default_rng(2).integers(0, 255, (4, 64, 64, 3), dtype=np.uint8)
    out, mask = _case(arr, "RGB")
    assert (out[:, mask] == 0).all()
    assert integrity_ok(arr, out, mask)


def test_blur_and_pixelate_touch_only_mask():
    arr = np.random.default_rng(3).integers(0, 255, (2, 64, 64), dtype=np.uint8)
    for method in (RedactMethod.BLUR, RedactMethod.PIXELATE):
        out, mask = _case(arr, "MONOCHROME2", method)
        assert integrity_ok(arr, out, mask)
        assert not np.array_equal(out[:, mask], arr[:, mask])
        assert out.dtype == arr.dtype


def test_signed_pixels():
    arr = np.random.default_rng(4).integers(-1024, 3000, (1, 32, 64), dtype=np.int16)
    out, mask = _case(arr, "MONOCHROME2")
    assert (out[0][mask] == arr.min()).all()


def test_padding_does_not_clip_kept_clinical_text():
    cfg = Config()
    phi = TextBox(10, 10, 60, 20, "DOE^JANE", 1.0)
    kept = TextBox(10, 22, 60, 30, "R", 1.0)  # just below, inside phi's padding
    m = build_mask((64, 64), [phi], cfg, [kept])
    assert m[10:20, 10:60].all()  # PHI fully covered
    assert not m[22:30, 10:60].any()  # clinical label untouched
    m2 = build_mask((64, 64), [phi], cfg)
    assert m2[22:24, 10:60].any()  # without the carve-out padding would clip it
