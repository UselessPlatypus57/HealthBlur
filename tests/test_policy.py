from pixelguard.config import Config, TextPolicy
from pixelguard.phi import KnownIdentifiers
from pixelguard.pixel.ocr import TextBox, merge_boxes
from pixelguard.pixel.policy import decide, group_lines


def B(x0, y0, x1, y1, text, conf=0.95):
    return TextBox(x0, y0, x1, y1, text, conf)


def _decisions(boxes, policy=TextPolicy.TARGETED, known=None):
    return {d.box.text: d for d in decide(boxes, known or KnownIdentifiers(), Config(text_policy=policy), (256, 256))}


def test_split_name_is_grouped_and_redacted_without_known_identifiers():
    # OCR splits "Fitzgerald, Rowan" into two boxes; the header says ANONYMOUS.
    d = _decisions([B(88, 4, 129, 19, "Rowan", 1.0), B(4, 3, 85, 22, "Fitzgerald,")])
    assert d["Rowan"].decision == "redact" and "NAME" in d["Rowan"].categories
    assert d["Fitzgerald,"].decision == "redact"


def test_overlapping_boxes_form_one_line():
    lines = group_lines([B(88, 4, 129, 19, "Rowan"), B(4, 3, 129, 22, "Fitzgerald, Rowan")])
    assert len(lines) == 1


def test_merge_keeps_longer_reading_and_absorbs_contained_boxes():
    merged = merge_boxes([B(88, 4, 129, 19, "Rowan", 1.0), B(4, 3, 85, 22, "Fitzgerald,", 0.95), B(4, 3, 129, 22, "Fitzgerald, Rowan", 0.81)])
    assert len(merged) == 1 and merged[0].text == "Fitzgerald, Rowan" and (merged[0].x0, merged[0].x1) == (4, 129)


def test_label_makes_rest_of_line_phi():
    d = _decisions([B(4, 4, 50, 18, "Name:"), B(56, 4, 140, 18, "Quinn Ellsworth"), B(230, 4, 240, 18, "R")])
    assert d["Name:"].decision == "redact"
    assert d["Quinn Ellsworth"].decision == "redact"
    assert d["R"].decision == "keep"  # far-away orientation marker survives


def test_clinical_word_next_to_phi_is_redacted():
    d = _decisions([B(4, 4, 60, 18, "Harbor"), B(64, 4, 100, 18, "View"), B(104, 4, 150, 18, "Clinic")])
    assert all(d[t].decision == "redact" for t in ("Harbor", "View", "Clinic"))


def test_conservative_vs_targeted_on_unknown_text():
    boxes = [B(4, 4, 80, 18, "Qwxz Plorb")]
    assert _decisions(boxes, TextPolicy.CONSERVATIVE)["Qwxz Plorb"].decision == "redact"
    assert _decisions([B(4, 4, 80, 18, "zzqv")], TextPolicy.TARGETED)["zzqv"].decision == "review"


def test_clinical_annotations_kept_and_huge_boxes_not_redacted():
    d = _decisions([B(4, 230, 70, 246, "Se.4 Im:12"), B(170, 230, 250, 246, "AX T2 FLAIR"), B(40, 40, 200, 200, "0O", 0.7)], TextPolicy.CONSERVATIVE)
    assert d["Se.4 Im:12"].decision == "keep"
    assert d["AX T2 FLAIR"].decision == "keep"
    assert d["0O"].decision == "review"
