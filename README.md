# PixelGuard (HealthBlur)

**Stop medical images from leaking patient identities.**

DICOM files can identify a patient in two places: the **header** (name, MRN,
birth date, accession number, private vendor tags…) and the **pixels** (text
burned into the image by ultrasound machines, screen captures, scanned
films, overlay planes). Most anonymization tools only clean the header.
PixelGuard cleans both, records every change, and then **audits its own
output** to catch anything it missed.

> **PixelGuard does not guarantee de-identification.** Automated detection
> can miss identifiers (unusual fonts, very low contrast, handwriting, text
> that appears only in some frames, faces or jewelry in the anatomy, private
> tags with unknown meaning). Treat every result as needing review by a
> qualified person before data leaves your organisation.

## The problem in one run

```powershell
pixelguard demo
```

It generates six files that a header-only tool has already "anonymized"
(`PatientName = ANONYMOUS`, `PatientIdentityRemoved = YES`). PixelGuard's
audit then re-reads the pixels and shows something like this:

```
Step 2: metadata-only clean-up + audit -> 6/6 files still show identifying text in the pixels:
     file 0 (CT): DATE, ID, LABEL, NAME in the top-left, top-right corner(s)
     file 3 (US): ID, INSTITUTION, LABEL, NAME in the top-left, top-right corner(s)
     ...
Step 3: full PixelGuard pipeline -> audit PASS 5, WARN 1, FAIL 0; 34 regions redacted.
```

## How it works

```
read ─► harvest identifiers ─► pixel stage ─► metadata stage ─► write ─► audit ─► PASS / WARN / FAIL (quarantine)
        (from the header,      OCR (EasyOCR)   PS3.15 Basic     uncompressed  re-read file,
         before anything       → PHI rules     Profile rules,   Explicit VR   re-OCR pixels,
         is removed)           → redact        UID remap,       Little Endian walk every element
                                               pseudonyms
```

**1. Harvest known identifiers.** Before changing anything, PixelGuard collects
the patient's name tokens, IDs, accession numbers, institution and dates in a
dozen formats from the header. Most burned-in PHI repeats these values, so the
pixel stage looks for those exact strings and tolerates OCR slips
(`275O5O51` → `27505051`, `THORNBURV` → `THORNBURY`).

**2. Pixel stage.**
- Every frame is rendered to 8 bits in two ways (full range and the stored
  VOI window), and each rendering is read with EasyOCR. Multi-frame objects
  are sampled (first, middle and last frame by default; `--all-frames` reads
  every frame).
- Each text box is classified using:
  - known-identifier matches;
  - generic PHI patterns: `LAST^FIRST`, `Name:`/`DOB:`/`MRN:` labels, dates,
    ID numbers, phone numbers, SSNs, ages over 89, institution names;
  - line context: a box on the same line as PHI is also PHI;
  - a vocabulary of clinical annotations (`L`, `R`, `AP`, `W:400 L:40`,
    `Se:2 Im:15`, `120 kVp`, `5 cm`, `TGC`…), which are **kept**.
- Two policies:
  - `conservative` (default): redact all text except recognised clinical
    annotations.
  - `targeted`: redact only text that matches a PHI rule.
- Redaction works in the native pixel type. A solid fill is the default
  (display-black, so MONOCHROME1 is handled). `blur` and `pixelate` are
  available, but the audit flags them as weaker.
- **Integrity check:** pixels outside the redacted boxes must be
  bit-identical to the original, and the audit fails if they aren't.

**3. Metadata stage.** Follows the DICOM PS3.15 Annex E Basic Profile:
- Remove, empty, pseudonymize or remap-UID per attribute.
- Catch-all rules cover every person-name (PN) value, all private tags,
  overlay planes and curves.
- Descriptors (series and study descriptions, protocol name) are kept unless
  PHI is detected in them.
- Nested sequences are processed recursively.

Other behaviour:
- Pseudonyms, new UIDs and date offsets come from a keyed HMAC. The same
  `--key` gives the same output across runs, so patients stay linkable
  across a dataset, and nothing can be reversed without the key.
- Dates: `remove` (default) empties or removes them. `shift` moves each
  patient by one consistent offset, so intervals are preserved. `keep` leaves
  them unchanged.
- Ages over 89 become `090Y`.
- The output carries `PatientIdentityRemoved`, `DeidentificationMethod` and
  `DeidentificationMethodCodeSequence` (113100, 113101, and so on).

**4. Post-anonymization audit.** The audit is independent of the cleaning
step. It re-reads the **written file**, walks every element, including
sequences and file meta, and re-runs OCR on the output pixels. Checks:

| check | fails when |
|---|---|
| `metadata.known_identifiers` | any harvested identifier value is still in any element |
| `metadata.private_tags` / `metadata.overlays` | private attributes or overlay planes remain |
| `metadata.uids` | any original instance UID remains |
| `pixel.residual_text` | re-OCR still reads text matching a PHI rule (FAIL) or unrecognised text (WARN) |
| `pixel.integrity` | pixels outside redaction boxes changed |
| `risk.*` (WARN) | the header declares burned-in annotation but no text was found; high-risk modality (US, SC, OT…) with no text found; frames were sampled; weak redaction method |

`FAIL` files are moved to `_quarantine/`. A file that raises an error is
never written, so the pipeline fails closed. `WARN` means the file passed
but has a review flag.

**5. Audit trail.** `audit_report.json` and `audit_report.html` record:
- every tag transformation (path, action, rule, new value);
- every text region (box, decision, category, rule, OCR confidence);
- every check result and per-stage timings.

The reports **never contain original values**. A removed value is
identified only by a keyed HMAC and its length. Output paths are built from
pseudonyms and new UIDs. Input file names, which often contain patient
names, are written only to an optional `--linkage` CSV.

## Install

PixelGuard needs Python 3.10 or newer. It was developed on Python 3.14 with
CPU-only PyTorch.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,codecs]"
```

EasyOCR downloads about 100 MB of model weights to `~/.EasyOCR` on first
use. The `codecs` extra adds JPEG and JPEG 2000 decoding for compressed
inputs. RLE and uncompressed data work without it.

## Usage

```powershell
# De-identify a folder (writes output files + audit_report.html/json)
pixelguard deidentify path\to\dicoms -o clean_out

# Options
pixelguard deidentify in -o out --policy targeted --dates shift --key-file secret.txt --workers 3

# Read-only scan: what PHI is where?
pixelguard scan path\to\dicoms --json findings.json

# Audit an existing "anonymized" dataset without modifying it
pixelguard audit path\to\already_anonymized --report-dir audit_out

# Local web dashboard (http://127.0.0.1:8765)
pixelguard serve
```

The dashboard can:
- take files by drag and drop, or run the synthetic demo;
- show the original and the de-identified image side by side, with boxes for
  redacted, kept and flagged text, and dashed boxes for anything the audit
  still reads;
- list the audit checks and the metadata changes;
- download the clean files (quarantined files are excluded).

It binds to localhost only, and uploads are stored under neutral names.

## Measuring it

```powershell
pixelguard benchmark --n 30 --workers 3     # writes benchmark_out/benchmark.md + per-variant reports
```

The benchmark generates synthetic CT, MR, DX (including MONOCHROME1), US,
multi-frame US and RGB screen captures. Each file has fake PHI in the header,
in private tags, in nested sequences, in overlays and in descriptions, plus
PHI burned into the pixels next to clinical annotations. A share of the files
are header-only "anonymized", and some are hard cases (9–10 px fonts, low
contrast). Each file has a `.truth.json` with exact boxes.

Metrics:

| metric | definition |
|---|---|
| pixel PHI recall | PHI text regions ≥ 90 % covered by redaction |
| pixel false-positive rate | clinical annotations ≥ 50 % covered by redaction |
| collateral pixel share | redacted pixels outside any ground-truth text box |
| metadata PHI recall | injected header values absent from every output element |
| clinical retention | Modality, spacing, window, KVP, orientation… preserved |
| audit pass rate | PASS (strict) and PASS + WARN |
| speed | seconds per file and files per minute |

### Results (synthetic, seed 7, 30 files, CPU, 3 workers)

Test set:
- 118 burned-in PHI regions, 90 clinical annotations and 373 header PHI
  values, across 30 files;
- 7 of the 30 files are header-only "anonymized";
- 6 are hard cases (9–10 px fonts at 45–60 % contrast).

| variant | pixel PHI recall (normal / hard) | pixel FP rate | metadata PHI recall | clinical attribute retention | audit PASS / PASS+WARN / FAIL | throughput |
|---|---|---|---|---|---|---|
| metadata-only (baseline) | 0 % / 0 % | – | 100 % | 100 % | 0 / 0 / **30** | 16.6 files/min |
| targeted | 98.9 % / 77.4 % | 0.0 % | 100 % | 100 % | 23 / 30 / 0 | 9.0 files/min |
| **conservative** (default) | **100 % / 77.4 %** | 4.4 % | 100 % | 100 % | 25 / 30 / 0 | 9.0 files/min |

How to read this:
- **The metadata-only baseline leaks in every file.** Each one still shows
  PHI in its pixels, and the audit catches all 30 (FAIL, quarantined).
- **Conservative** redacted every PHI region in the normal cases. Its
  4.4 % false-positive rate comes from clinical labels that OCR misread
  badly enough that they no longer matched the clinical vocabulary.
- **Targeted** keeps every clinical label, but misses PHI that matches no
  rule, for example a bare "First Last" name read only in part.
- **Hard cases are the weak spot.** Tiny, low-contrast text is only partly
  found.
- **Important:** those hard-case misses still passed the audit, because the
  audit re-reads pixels with the same OCR engine. Text the OCR cannot read
  is invisible to both steps. The audit pass rate is therefore an *upper
  bound* on safety, not proof of it. Keep human review in the loop,
  especially for ultrasound and screen captures.
- Throughput is on a 12-thread CPU with 3 workers and no GPU. Each file is
  read by OCR twice for cleaning and twice for auditing (two renderings each
  time). `--no-verify` roughly halves the time, but is not recommended.

These numbers come from synthetic data with known ground truth. They show
how the configurations compare. They are **not** evidence of performance on
your scanners and fonts, so validate on a representative, labelled sample
of your own data before relying on PixelGuard.

### External / public data

To check a public de-identification benchmark (for example TCIA's
Pseudo-PHI DICOM collection), or your own labelled sample, write the known
PHI strings to a CSV with the columns `file,value` (`file` = `*` means every
file) and run:

```powershell
pixelguard evaluate-external path\to\data -o ext_out --known-csv known_phi.csv
```

This reports metadata recall and the audit pass rate. Region-level pixel
recall needs box annotations, so for external data rely on
`pixel.residual_text` and manual review.

## Tests

```powershell
pytest              # 70 fast tests (fake OCR replays ground truth)
pytest -m ocr       # end-to-end with the real EasyOCR model
```

## Known limitations

- OCR finds printed text. It does not find handwriting, text rotated more
  than about 30°, or text that is very small or very low contrast (see the
  hard-case recall above).
- Faces in head CT/MR volume renderings, and identifiers inside the anatomy
  (jewelry, tattoos, engraved implants), are not detected.
- Structured Reports and encapsulated PDFs are skipped and flagged, not
  cleaned.
- Private tags are removed rather than selectively kept. Keeping them
  (`--keep-private`) puts their contents outside PixelGuard's checks.
- The PS3.15 rule table is a curated subset backed by catch-alls. It is not
  a certified implementation of the profile.
