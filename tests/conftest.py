import json
import sys
from pathlib import Path

import numpy as np
import pydicom
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixelguard.pixel.ocr import TextBox  # noqa: E402
from pixelguard.pixel.render import frame_to_u8, load_pixels  # noqa: E402
from pixelguard.synth import generate  # noqa: E402


def _fingerprint(u8: np.ndarray) -> bytes:
    h, w = u8.shape[:2]
    return u8[h // 2 - 16 : h // 2 + 16, w // 2 - 16 : w // 2 + 16].tobytes()


class GroundTruthOCR:
    """Fake OCR that 'reads' the synthetic ground-truth text wherever it is
    still visible in the image. Lets pipeline tests run without EasyOCR.

    Files are recognised by a fingerprint of the image centre (text is only
    burned into the corners, and redaction never touches the centre)."""

    name = "ground-truth"

    def __init__(self, truth_dir: Path):
        self.by_print: dict[bytes, list[dict]] = {}
        for tf in Path(truth_dir).glob("*.truth.json"):
            t = json.loads(tf.read_text())
            ds = pydicom.dcmread(str(Path(truth_dir) / t["file"]))
            arr, _ = load_pixels(ds)
            for f in range(arr.shape[0]):
                for mode in ("minmax", "voi", "equalize"):
                    u8 = frame_to_u8(arr[f], ds, mode)
                    if u8 is not None:
                        self.by_print[_fingerprint(u8)] = t["pixel_regions"]
        self.calls = 0

    def detect(self, img: np.ndarray) -> list[TextBox]:
        self.calls += 1
        regions = self.by_print.get(_fingerprint(img), [])
        out = []
        for r in regions:
            x0, y0, x1, y1 = r["bbox"]
            patch = img[y0:y1, x0:x1].astype(np.float32)
            if patch.size and patch.std() > 1.0:
                out.append(TextBox(x0, y0, x1, y1, r["text"], 0.95))
        return out


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    generate(d, n=12, seed=5, naive_fraction=0.25, hard_fraction=0.0, compress_fraction=0.2)
    return d


@pytest.fixture(scope="session")
def gt_engine(synth_dir):
    return GroundTruthOCR(synth_dir)


@pytest.fixture(scope="session")
def truths(synth_dir):
    return {t["file"]: t for t in (json.loads(p.read_text()) for p in synth_dir.glob("*.truth.json"))}
