"""Run configuration."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import asdict, dataclass, field
from enum import Enum


class DateMode(str, Enum):
    REMOVE = "remove"  # DICOM PS3.15 Basic Profile: dates emptied or removed
    SHIFT = "shift"  # consistent per-patient offset (keeps intervals)
    KEEP = "keep"  # keep original dates (only for data-use agreements that allow it)


class RedactMethod(str, Enum):
    FILL = "fill"  # irreversible solid fill (default)
    BLUR = "blur"  # weaker: flagged by the audit
    PIXELATE = "pixelate"  # weaker: flagged by the audit


class TextPolicy(str, Enum):
    # Redact every piece of detected text except recognised clinical annotations.
    CONSERVATIVE = "conservative"
    # Redact only text that matches a PHI rule or a known identifier.
    TARGETED = "targeted"


@dataclass
class Config:
    date_mode: DateMode = DateMode.REMOVE
    redact_method: RedactMethod = RedactMethod.FILL
    text_policy: TextPolicy = TextPolicy.CONSERVATIVE

    # Secret used for pseudonyms, UID remapping, date offsets and report hashes.
    # Supply the same key across runs to keep pseudonyms stable; otherwise a
    # random key is generated per run and links across runs are impossible.
    secret_key: bytes = field(default_factory=lambda: secrets.token_bytes(32), repr=False)

    # Metadata options (DICOM PS3.15 Annex E option names in comments)
    retain_patient_characteristics: bool = True  # 113108: sex, age, size, weight
    clean_descriptors: bool = True  # 113105: keep descriptions if no PHI found
    keep_private_tags: bool = False
    remove_overlays: bool = True
    remove_curves: bool = True
    pseudonym_prefix: str = "ANON"

    # Pixel options
    clean_pixels: bool = True  # False = metadata-only (baseline / comparison)
    ocr_languages: tuple[str, ...] = ("en",)
    ocr_passes: tuple[str, ...] = ("minmax", "voi")
    ocr_min_confidence: float = 0.25  # below this, unmatched text is only flagged
    ocr_max_frames: int = 3  # frames sampled from multi-frame objects
    ocr_all_frames: bool = False
    pad_px: int = 3
    pad_ratio: float = 0.2  # extra padding relative to box height

    # Verification
    verify: bool = True
    verify_pixels: bool = True
    quarantine_failures: bool = True

    # Reporting
    report_previews: bool = False  # masked text previews (e.g. "D**^J***") in reports

    def hmac_hex(self, value: str, n: int = 12) -> str:
        return hmac.new(self.secret_key, value.encode("utf-8", "replace"), hashlib.sha256).hexdigest()[:n]

    def hmac_int(self, value: str) -> int:
        return int.from_bytes(hmac.new(self.secret_key, value.encode("utf-8", "replace"), hashlib.sha256).digest(), "big")

    def public_dict(self) -> dict:
        """Configuration without the secret, safe to place in reports."""
        d = asdict(self)
        d.pop("secret_key", None)
        for k, v in list(d.items()):
            if isinstance(v, Enum):
                d[k] = v.value
            elif isinstance(v, tuple):
                d[k] = list(v)
        d["secret_key_fingerprint"] = hashlib.sha256(self.secret_key).hexdigest()[:8]
        return d

    @staticmethod
    def key_from_text(text: str) -> bytes:
        return hashlib.sha256(text.encode("utf-8")).digest()
