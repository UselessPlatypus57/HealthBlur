"""Decide which detected text boxes to redact."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import Config, TextPolicy
from ..phi import KnownIdentifiers, detect, is_clinical_text, mask
from .ocr import TextBox


@dataclass
class TextDecision:
    box: TextBox
    decision: str  # redact | keep | review
    categories: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    def to_dict(self, previews: bool) -> dict:
        b = self.box
        d = {
            "bbox": [b.x0, b.y0, b.x1, b.y1],
            "frame": b.frame,
            "confidence": round(b.confidence, 3),
            "decision": self.decision,
            "categories": self.categories,
            "rules": self.rules,
            "text_length": len(b.text),
        }
        if self.decision == "keep":
            d["text"] = b.text  # clinical annotation, not PHI
        elif previews:
            d["preview"] = mask(b.text)
        return d


def group_lines(boxes: list[TextBox]) -> list[list[TextBox]]:
    lines: list[list[TextBox]] = []
    for b in sorted(boxes, key=lambda b: ((b.y0 + b.y1) / 2, b.x0)):
        cy = (b.y0 + b.y1) / 2
        target = None
        for line in lines:
            ref = line[-1]
            rcy = (ref.y0 + ref.y1) / 2
            h = max(b.height, ref.height)
            similar = max(b.height, ref.height) <= 1.8 * min(b.height, ref.height)
            if similar and abs(cy - rcy) <= 0.6 * h and ref.frame == b.frame:
                gap = min(max(b.x0 - m.x1, m.x0 - b.x1, 0) for m in line)  # 0 when overlapping
                if gap <= 3 * h:
                    target = line
                    break
        if target is None:
            lines.append([b])
        else:
            target.append(b)
    for line in lines:
        line.sort(key=lambda b: b.x0)
    return lines


def normalize_ocr_text(text: str) -> str:
    """Undo common OCR spacing artefacts: "08 , 2021" -> "08, 2021"."""
    return re.sub(r"\s+([,.:;/])", r"\1", text).strip()


def decide(boxes: list[TextBox], known: KnownIdentifiers | None, cfg: Config, image_hw: tuple[int, int] | None = None) -> list[TextDecision]:
    """Classify each OCR box as redact / keep / review.

    Order of evidence: (1) the box itself matches a PHI rule or a known
    identifier; (2) the box is a clinical annotation; (3) the box shares a
    line with PHI (e.g. "Name:" label or a split name); (4) policy fallback.
    Clinical words directly next to PHI on the same line are redacted too
    ("Harbor *View* Clinic", "*Dr.* Smith").
    """
    out: list[TextDecision] = []
    max_h = 0.15 * image_hw[0] if image_hw else None
    for line in group_lines(boxes):
        line_text = normalize_ocr_text(" ".join(b.text for b in line))
        line_f = detect(line_text, known) if len(line) > 1 else []
        line_strong = [f for f in line_f if f.strength == "strong"]
        decisions: list[TextDecision] = []
        for b in line:
            text = normalize_ocr_text(b.text)
            bf = detect(text, known)
            strong = [f for f in bf if f.strength == "strong"]
            weak = [f for f in bf + line_f if f.strength == "weak"]
            safe = not bf and is_clinical_text(text)
            confident = b.confidence >= cfg.ocr_min_confidence
            implausible = max_h is not None and b.height > max_h  # anatomy mistaken for text
            if strong:
                decisions.append(_dec(b, "redact", strong))
            elif implausible:
                decisions.append(TextDecision(b, "review", ["UNRECOGNIZED_TEXT"], ["implausible-text-size"]))
            elif safe:
                decisions.append(TextDecision(b, "keep", [], ["clinical-annotation"]))
            elif line_strong:
                decisions.append(_dec(b, "redact", line_strong, prefix="line-context:"))
            elif weak and confident:
                decisions.append(_dec(b, "redact", weak))
            elif cfg.text_policy == TextPolicy.CONSERVATIVE and confident:
                decisions.append(TextDecision(b, "redact", ["UNRECOGNIZED_TEXT"], ["policy:conservative-unrecognized-text"]))
            else:
                decisions.append(TextDecision(b, "review", ["UNRECOGNIZED_TEXT"], ["low-confidence-or-targeted-policy"]))
        # Clinical-looking words touching redacted PHI are part of the identifier.
        changed = True
        while changed:
            changed = False
            for d in decisions:
                if d.decision != "keep":
                    continue
                for o in decisions:
                    if o.decision == "redact" and o.categories != ["UNRECOGNIZED_TEXT"] and _adjacent(d.box, o.box):
                        d.decision, d.categories, d.rules = "redact", list(o.categories), ["adjacent-to-phi"]
                        changed = True
                        break
        out.extend(decisions)
    return out


def _adjacent(a: TextBox, b: TextBox) -> bool:
    h = max(a.height, b.height)
    gap = max(b.x0 - a.x1, a.x0 - b.x1, 0)
    return gap <= 1.2 * h


def _dec(b: TextBox, decision: str, findings, prefix: str = "") -> TextDecision:
    cats = sorted({f.category for f in findings})
    rules = sorted({prefix + f.rule for f in findings})
    return TextDecision(b, decision, cats, rules)
