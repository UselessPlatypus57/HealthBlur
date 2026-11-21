"""Synthetic DICOM generator with exact ground truth.

Creates CT, MR, DX, US (single and multi-frame) and secondary-capture images
with fictional PHI placed both in the header and burned into the pixels,
alongside clinically useful annotations (L/R markers, technique, scale).
A share of the files mimics a dataset that a metadata-only tool already
"anonymized": the header says ANONYMOUS but the pixels still show the name.

Every file has a ``.truth.json`` sidecar listing each burned-in text region
(box, text, PHI or clinical) and every PHI value injected into the header.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    CTImageStorage, DigitalXRayImageStorageForPresentation, ExplicitVRLittleEndian, MRImageStorage,
    RLELossless, SecondaryCaptureImageStorage, UltrasoundImageStorage, UltrasoundMultiFrameImageStorage,
    generate_uid,
)

from .names import CITIES, FIRST, INSTITUTIONS, LAST, STREETS

CLINICAL_TEXT = {
    "CT": ["W:400 L:40", "Se:2 Im:15", "120 kVp", "5 mm", "AX", "R", "L"],
    "MR": ["AX T2 FLAIR", "Se:4 Im:12", "R", "L", "TR 9000", "3 mm"],
    "DX": ["PORTABLE", "AP", "R", "L", "SUPINE", "ERECT"],
    "US": ["TGC", "GAIN 50%", "FR 30Hz", "3.5MHz", "DR 60", "5 cm"],
    "SC": ["R", "L", "x1.5", "5 cm"],
}
FONT_CANDIDATES = ["arial.ttf", "consola.ttf", "cour.ttf", "verdana.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"]


@dataclass
class Patient:
    first: str
    last: str
    mrn: str
    dob: date
    sex: str
    address: str
    phone: str
    other_id: str


def _fonts():
    found = []
    for name in FONT_CANDIDATES:
        try:
            ImageFont.truetype(name, 12)
            found.append(name)
        except OSError:
            pass
    return found


def _font(fonts, size, rng):
    if fonts:
        return ImageFont.truetype(rng.choice(fonts), size)
    return ImageFont.load_default(size=size)


def make_patient(rng: random.Random) -> Patient:
    first, last = rng.choice(FIRST), rng.choice(LAST)
    dob = date(1935, 1, 1) + timedelta(days=rng.randrange(0, 365 * 70))
    return Patient(
        first=first, last=last,
        mrn=str(rng.randrange(10_000_000, 99_999_999)),
        dob=dob,
        sex=rng.choice("MF"),
        address=f"{rng.randrange(10, 9999)} {rng.choice(STREETS)}^{rng.choice(CITIES)}",
        phone=f"({rng.randrange(200, 999)}) {rng.randrange(200, 999)}-{rng.randrange(1000, 9999)}",
        other_id=f"H{rng.randrange(100000, 999999)}",
    )


# ------------------------------------------------------------------ images
def _phantom(modality: str, rng: np.random.Generator, frames: int):
    if modality == "CT":
        h = w = 512
        yy, xx = np.mgrid[:h, :w]
        img = np.zeros((h, w), np.float32)
        body = ((xx - 256) / 220) ** 2 + ((yy - 256) / 170) ** 2 < 1
        img[body] = 1064
        for cx in (180, 332):
            img[((xx - cx) / 60) ** 2 + ((yy - 240) / 90) ** 2 < 1] = 224
        img[((xx - 256) / 22) ** 2 + ((yy - 380) / 22) ** 2 < 1] = 1724
        img += rng.normal(0, 18, img.shape)
        return img.clip(0, 4095).astype(np.uint16)[None], {"bits": 12, "max": 4095}
    if modality == "MR":
        h = w = 256
        yy, xx = np.mgrid[:h, :w]
        img = np.zeros((h, w), np.float32)
        img[((xx - 128) / 95) ** 2 + ((yy - 128) / 115) ** 2 < 1] = 900
        img[((xx - 128) / 80) ** 2 + ((yy - 128) / 100) ** 2 < 1] = 650
        img[((xx - 110) / 12) ** 2 + ((yy - 120) / 25) ** 2 < 1] = 1300
        img[((xx - 146) / 12) ** 2 + ((yy - 120) / 25) ** 2 < 1] = 1300
        img += rng.normal(0, 25, img.shape)
        return img.clip(0, 4095).astype(np.uint16)[None], {"bits": 12, "max": 4095}
    if modality == "DX":
        h, w = 768, 640
        yy, xx = np.mgrid[:h, :w]
        img = 2600 - 900 * np.exp(-(((xx - 200) / 90) ** 2 + ((yy - 380) / 220) ** 2)) - 900 * np.exp(-(((xx - 440) / 90) ** 2 + ((yy - 380) / 220) ** 2))
        img += 300 * np.sin(yy / 18.0) * np.exp(-(((xx - 320) / 260) ** 2))
        img += rng.normal(0, 30, img.shape)
        return img.clip(0, 4095).astype(np.uint16)[None], {"bits": 12, "max": 4095}
    if modality == "US":
        h, w = 480, 640
        yy, xx = np.mgrid[:h, :w]
        ang = np.arctan2(xx - 320, yy + 40)
        r = np.hypot(xx - 320, yy + 40)
        sector = (np.abs(ang) < 0.6) & (r > 80) & (r < 470)
        out = []
        for _ in range(frames):
            sp = rng.rayleigh(40, (h, w)) * sector
            sp += 60 * sector * np.exp(-(((xx - 320) / 60) ** 2 + ((yy - 260) / 30) ** 2))
            out.append(sp.clip(0, 230).astype(np.uint8))
        return np.stack(out), {"bits": 8, "max": 255}
    # SC: RGB screen capture
    h, w = 480, 640
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.linspace(20, 60, w, dtype=np.uint8)[None, :]
    img[..., 2] = np.linspace(40, 90, h, dtype=np.uint8)[:, None]
    yy, xx = np.mgrid[:h, :w]
    blob = ((xx - 320) / 150) ** 2 + ((yy - 240) / 120) ** 2 < 1
    img[blob] = (150, 120, 110)
    img = (img.astype(np.int16) + rng.integers(-6, 7, img.shape)).clip(0, 255).astype(np.uint8)
    return img[None], {"bits": 8, "max": 255}


def _layout_lines(modality, patient: Patient, study_dt, institution, accession, physician, rng: random.Random, hard: bool):
    """Return [(text, is_phi, category)] for top-left, top-right, bottom-left, bottom-right blocks."""
    name_fmt = rng.choice(["caret", "label", "comma", "plain"])
    name = {
        "caret": f"{patient.last.upper()}^{patient.first.upper()}",
        "label": f"Name: {patient.first} {patient.last}",
        "comma": f"{patient.last}, {patient.first}",
        "plain": f"{patient.first} {patient.last}",
    }[name_fmt]
    mrn = rng.choice([f"MRN: {patient.mrn}", f"ID {patient.mrn}", patient.mrn])
    dob = rng.choice([f"DOB: {patient.dob:%m/%d/%Y}", f"DOB {patient.dob:%Y-%m-%d}", f"{patient.dob:%d-%b-%Y}"])
    when = rng.choice([f"{study_dt:%Y-%m-%d} 14:{rng.randrange(10, 59)}", f"{study_dt:%d/%m/%Y}", f"{study_dt:%b %d, %Y}"])
    phi_pool = [
        (name, True, "NAME"), (mrn, True, "ID"), (dob, True, "DATE"), (when, True, "DATE"),
        (institution, True, "INSTITUTION"), (f"Acc# {accession}", True, "ID"), (f"Dr. {physician}", True, "NAME"),
    ]
    # Name and MRN always present; others sampled.
    phi = phi_pool[:2] + rng.sample(phi_pool[2:], k=rng.randint(1, 3))
    clin = [(t, False, "CLINICAL") for t in rng.sample(CLINICAL_TEXT[modality], k=min(3, len(CLINICAL_TEXT[modality])))]
    return phi, clin


def _burn(arr: np.ndarray, lines_by_corner, fonts, rng: random.Random, value, hard: bool, color: tuple | None):
    """Draw text into every frame; return ground-truth regions."""
    h, w = arr.shape[1:3]
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    regions = []
    base = max(11, min(h, w) // 30)
    for corner, lines in lines_by_corner.items():
        size = rng.choice([base - 2, base, base + 2]) if not hard else rng.choice([9, 10, base])
        font = _font(fonts, max(8, size), rng)
        margin = 6
        line_h = int(size * 1.35)
        for i, (text, is_phi, cat) in enumerate(lines):
            tb = draw.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            x = margin if corner.endswith("left") else w - margin - tw
            if corner.startswith("top"):
                y = margin + i * line_h
            else:
                y = h - margin - (len(lines) - i) * line_h
            draw.text((x - tb[0], y - tb[1]), text, fill=255, font=font)
            bb = draw.textbbox((x - tb[0], y - tb[1]), text, font=font)
            regions.append({"bbox": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])], "text": text, "is_phi": is_phi, "category": cat, "font_px": size})
    alpha = np.asarray(canvas, np.float32) / 255.0
    out = arr.astype(np.float32)
    if out.ndim == 4:
        col = np.array(color or (255, 255, 0), np.float32)
        out = out * (1 - alpha[None, ..., None]) + col * alpha[None, ..., None]
    else:
        out = out * (1 - alpha[None]) + float(value) * alpha[None]
    return out.round().astype(arr.dtype), regions


# ---------------------------------------------------------------- datasets
def _base_dataset(modality, sop_class, arr, meta_bits, rng: random.Random, seed_tag):
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = sop_class
    sop_uid = generate_uid(entropy_srcs=[seed_tag, "sop"])
    fm.MediaStorageSOPInstanceUID = sop_uid
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = fm
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    ds.Modality = modality
    frames, rows, cols = arr.shape[:3]
    ds.Rows, ds.Columns = rows, cols
    if arr.ndim == 4:
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
    else:
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8 if arr.dtype == np.uint8 else 16
    ds.BitsStored = meta_bits["bits"]
    ds.HighBit = meta_bits["bits"] - 1
    ds.PixelRepresentation = 0
    if frames > 1:
        ds.NumberOfFrames = frames
        ds.FrameIncrementPointer = 0x00181063
        ds.FrameTime = 33.3
    ds.ImageType = ["ORIGINAL", "PRIMARY"]
    return ds


def generate(out_dir: str | Path, n: int = 40, seed: int = 7, naive_fraction: float = 0.25, hard_fraction: float = 0.2, compress_fraction: float = 0.15) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    fonts = _fonts()
    patients = [make_patient(rng) for _ in range(max(1, n // 2))]
    plan = ["CT", "MR", "DX", "US", "US-CINE", "SC"]
    manifest = []
    for i in range(n):
        kind = plan[i % len(plan)]
        modality = "US" if kind.startswith("US") else kind
        p = patients[i % len(patients)]
        naive = rng.random() < naive_fraction
        hard = rng.random() < hard_fraction
        study_dt = date(2019, 1, 1) + timedelta(days=rng.randrange(0, 365 * 6))
        institution = rng.choice(INSTITUTIONS)
        accession = f"A{rng.randrange(1_000_000, 9_999_999)}"
        physician = rng.choice(LAST)
        tag = f"{seed}-{i}"

        frames = 6 if kind == "US-CINE" else 1
        arr, bits = _phantom(modality, nrng, frames)
        phi_lines, clin_lines = _layout_lines(modality, p, study_dt, institution, accession, physician, rng, hard)
        corners = {"top-left": phi_lines[:3], "top-right": phi_lines[3:], "bottom-left": [clin_lines[0]], "bottom-right": clin_lines[1:]}
        corners = {k: v for k, v in corners.items() if v}
        text_value = bits["max"] if not hard else int(bits["max"] * rng.choice([0.45, 0.6]))
        monochrome1 = modality == "DX" and rng.random() < 0.4
        if monochrome1:
            # MONOCHROME1 displays low values as white: invert anatomy, burn text near 0
            arr = (bits["max"] - arr).astype(arr.dtype)
            text_value = 0 if not hard else int(bits["max"] * 0.4)
        arr, regions = _burn(arr, corners, fonts, rng, text_value, hard, color=rng.choice([(255, 255, 0), (255, 255, 255), (0, 255, 0)]))

        sop = {"CT": CTImageStorage, "MR": MRImageStorage, "DX": DigitalXRayImageStorageForPresentation,
               "SC": SecondaryCaptureImageStorage}.get(modality, UltrasoundMultiFrameImageStorage if frames > 1 else UltrasoundImageStorage)
        ds = _base_dataset(modality, sop, arr, bits, rng, tag)
        if monochrome1:
            ds.PhotometricInterpretation = "MONOCHROME1"
        meta_phi: list[dict] = []

        def put(keyword, value, is_phi=True):
            setattr(ds, keyword, value)
            if is_phi and not naive:
                meta_phi.append({"keyword": keyword, "value": str(value)})

        if naive:
            ds.PatientName = "ANONYMOUS"
            ds.PatientID = f"ANON{i:04d}"
            ds.PatientIdentityRemoved = "YES"
            ds.BurnedInAnnotation = "NO"  # the naive tool did not check
        else:
            put("PatientName", f"{p.last}^{p.first}")
            put("PatientID", p.mrn)
            put("PatientBirthDate", p.dob.strftime("%Y%m%d"))
            put("PatientAddress", p.address)
            put("PatientTelephoneNumbers", p.phone)
            put("OtherPatientIDs", p.other_id)
            put("AccessionNumber", accession)
            put("ReferringPhysicianName", f"{physician}^Sam")
            put("InstitutionName", institution)
            put("InstitutionAddress", f"{rng.randrange(1, 999)} {rng.choice(STREETS)}, {rng.choice(CITIES)}")
            put("StationName", f"{modality}{rng.randrange(1, 9)}-{rng.choice(CITIES).upper()[:5]}")
            put("OperatorsName", f"{rng.choice(LAST)}^{rng.choice(FIRST)}")
            put("StudyDate", study_dt.strftime("%Y%m%d"))
            put("SeriesDate", study_dt.strftime("%Y%m%d"))
            put("DeviceSerialNumber", f"SN{rng.randrange(100000, 999999)}")
            item = Dataset()
            item.PatientID = p.other_id + "X"
            item.IssuerOfPatientID = "LEGACY_RIS"
            ds.OtherPatientIDsSequence = Sequence([item])
            meta_phi.append({"keyword": "OtherPatientIDsSequence[0].PatientID", "value": item.PatientID})
            if rng.random() < 0.5:
                put("ImageComments", f"Pt {p.first} {p.last} tolerated exam well")
            if modality in ("US", "SC"):
                ds.BurnedInAnnotation = "YES"
            # private tag carrying the name, as some vendors do
            block = ds.private_block(0x0019, "ACME_RIS_1", create=True)
            block.add_new(0x10, "LO", f"{p.last.upper()} {p.first.upper()}")
            meta_phi.append({"keyword": "(0019,1010) private", "value": f"{p.last.upper()} {p.first.upper()}"})
        ds.PatientSex = p.sex
        age = (study_dt - p.dob).days // 365
        ds.PatientAge = f"{age:03d}Y"
        ds.StudyTime = "142233"
        series_desc = {"CT": "CT CHEST W CONTRAST", "MR": "AX T2 FLAIR", "DX": "CHEST AP PORTABLE", "US": "ABDOMEN", "SC": "SCREEN CAPTURE"}[modality]
        name_in_desc = (not naive) and rng.random() < 0.25
        if name_in_desc:
            series_desc += f" {p.last.upper()}"
            meta_phi.append({"keyword": "SeriesDescription", "value": p.last.upper()})
        ds.SeriesDescription = series_desc
        ds.StudyDescription = series_desc.split()[0] + " STUDY"
        ds.BodyPartExamined = {"CT": "CHEST", "MR": "BRAIN", "DX": "CHEST", "US": "ABDOMEN", "SC": "ABDOMEN"}[modality]
        ds.Manufacturer = "SyntheticCo"
        ds.StudyInstanceUID = generate_uid(entropy_srcs=[str(seed), p.mrn, str(study_dt), "study"])
        ds.SeriesInstanceUID = generate_uid(entropy_srcs=[tag, "series"])
        ds.FrameOfReferenceUID = generate_uid(entropy_srcs=[tag, "for"])
        ds.StudyID = str(rng.randrange(1000, 9999))
        ds.SeriesNumber = 1
        ds.InstanceNumber = i + 1
        if modality in ("CT", "MR", "DX"):
            ds.PixelSpacing = [0.7, 0.7]
            ds.WindowCenter, ds.WindowWidth = ({"CT": (40, 400), "MR": (700, 1400), "DX": (2048, 4096)})[modality]
        if modality == "CT":
            ds.RescaleIntercept, ds.RescaleSlope, ds.RescaleType = -1024, 1, "HU"
            ds.KVP = 120
            ds.SliceThickness = 5
            ds.ImagePositionPatient = [-180, -180, 10.0 * i]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        if modality == "MR":
            ds.SliceThickness = 3
        overlay = (not naive) and modality in ("CT", "MR") and rng.random() < 0.35
        if overlay:
            from pydicom.pixels import pack_bits

            ov = np.zeros((ds.Rows, ds.Columns), np.uint8)
            ov_img = Image.new("1", (ds.Columns, ds.Rows), 0)
            ImageDraw.Draw(ov_img).text((10, ds.Rows - 30), p.mrn, fill=1)
            ov = np.asarray(ov_img, np.uint8)
            ds.add_new(0x60000010, "US", ds.Rows)
            ds.add_new(0x60000011, "US", ds.Columns)
            ds.add_new(0x60000040, "CS", "G")
            ds.add_new(0x60000050, "SS", [1, 1])
            ds.add_new(0x60000100, "US", 1)
            ds.add_new(0x60000102, "US", 0)
            ds.add_new(0x60003000, "OW", pack_bits(ov))
            meta_phi.append({"keyword": "OverlayData", "value": "__overlay__"})

        ds.PixelData = (arr[0] if frames == 1 else arr).tobytes()
        compressed = rng.random() < compress_fraction and arr.ndim == 3
        if compressed:
            try:
                ds.compress(RLELossless, encoding_plugin="pydicom")
            except Exception:
                compressed = False

        fname = f"synth_{i:04d}_{kind}.dcm"
        ds.save_as(str(out_dir / fname), enforce_file_format=True)
        truth = {
            "file": fname,
            "modality": modality,
            "kind": kind,
            "naive_anonymized": naive,
            "hard_case": hard,
            "monochrome1": monochrome1,
            "compressed": compressed,
            "frames": frames,
            "shape": [ds.Rows, ds.Columns],
            "patient_key": p.mrn,
            "pixel_regions": regions,
            "metadata_phi": meta_phi,
            "known_phi_strings": [f"{p.first} {p.last}", f"{p.last}^{p.first}", p.mrn, p.dob.strftime("%Y%m%d")],
        }
        (out_dir / (fname + ".truth.json")).write_text(json.dumps(truth, indent=2))
        manifest.append(truth)
    (out_dir / "manifest.json").write_text(json.dumps({"seed": seed, "n": n, "files": [m["file"] for m in manifest]}, indent=2))
    return manifest
