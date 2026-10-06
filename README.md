# INEC Automated Document Verification System

A prototype triage tool for Continuous Voter Registration (CVR) document
processing, built to reduce the manual effort ICT staff spend inspecting
identity documents, checking photo quality, and cross-checking for
duplicate records.

---

## 1. Problem Statement

During Continuous Voter Registration, ICT staff manually:
- Read and transcribe applicant details off scanned ID/registration forms
- Visually inspect uploaded photos for quality
- Manually search existing records for possible duplicate registrations

This is slow, repetitive, and inconsistent across different reviewers.
This project automates the *first-pass triage* of that work — it does not
replace human judgement, but pre-sorts applications into "looks fine" and
"needs a human look" categories, so reviewers spend their time where it
matters.

---

## 2. System Architecture

```
Uploaded document image
        │
        ├──► Module 1: OCR Text Extraction (Tesseract)
        │        → structured fields: surname, first name, DOB, LGA, state, VIN
        │
        ├──► Module 2: Document Image Quality Check (OpenCV)
        │        → Laplacian variance → sharp / blurry classification
        │
        ├──► Module 2B: Applicant Photo Check (OpenCV Haar Cascade)
        │        → is a face present? is that face region clear?
        │
        ├──► Module 3: Duplicate Detection (Fuzzy Matching + SQLite)
        │        → name similarity + DOB agreement against existing records
        │
        ▼
Module 4: Streamlit Dashboard
        → Approved / Flagged for Review, with reasons
```

Each module is independently testable (see `modules/*.py`, each has a
`__main__` block that runs real test cases against sample data).

---

## 3. Module-by-Module Design Rationale

### Module 1 — OCR (`ocr_module.py`)
- Preprocesses the image (grayscale + Otsu thresholding) before running
  Tesseract, which meaningfully improves OCR accuracy on scanned documents.
- Uses rule-based line parsing (not ML) to extract fields — a deliberate
  choice: ID layouts are fixed-format, so regex/keyword matching is fast,
  fully explainable, and needs no training data.
- **Known limitation found during testing:** text from other parts of the
  document layout (e.g. a "PHOTO" placeholder label) can bleed into a
  name field if it shares a text line with it. Mitigated with a noise-word
  filter, but a layout-aware (bounding-box-per-field) OCR approach would
  be more robust — noted as future work.

### Module 2 — Document Quality (`blur_module.py`)
- Uses variance of the Laplacian (an edge-detection filter) as a
  focus/sharpness proxy — a well-established, training-free technique.
- Threshold (default 100) is tunable and should be calibrated against a
  real sample set from actual scanning hardware, not treated as universal.

### Module 2B — Applicant Photo Check (`face_module.py`)
- **Scope, by design:** this checks *presence and clarity* of a face in
  the applicant photo. It does **not** perform facial matching/recognition
  (e.g. comparing the ID photo to a live selfie) — that is a materially
  different and heavier task requiring a deep-learning model (e.g.
  DeepFace) and paired real face data to validate honestly. It was
  deliberately scoped out rather than included half-tested.
- Uses OpenCV's built-in Haar Cascade face detector — zero extra
  dependencies, fast, explainable.
- **Known limitation found during testing:** the default minimum face
  size for detection (60px) caused real, valid ID photo crops to be
  missed, because ID photo crops are often small relative to a full
  photograph. Lowered to 40px after testing against a realistic sample.

### Module 3 — Duplicate Detection (`dedup_module.py`)
- Fuzzy string matching (word-order-independent similarity scoring)
  instead of exact SQL matching, to catch typos and reordered names.
- **Deliberately requires DOB agreement in addition to name similarity**
  — name-only matching would flag huge numbers of unrelated applicants
  who share a common surname, which is a real concern given how
  concentrated some Nigerian surnames are in certain regions. This
  trade-off (stricter matching, fewer false positives) is a design
  decision worth explaining if asked.
- Implemented with Python's built-in `difflib` for offline development;
  functionally equivalent to `fuzzywuzzy.token_sort_ratio`, which can be
  swapped in directly (see docstring in the file).

### Module 5 — Fingerprint Comparison (`fingerprint_module.py`) — PROTOTYPE, separate tab
- Presented in its own dashboard tab, deliberately separate from the main
  document pipeline, because it is **not** part of the certified document
  check — it's a standalone demonstration.
- Uses OpenCV's ORB (Oriented FAST and Rotated BRIEF) to detect visual
  keypoints in two fingerprint images, then matches their descriptors with
  a brute-force matcher and Lowe's ratio test to filter weak matches.
- **Honest scope statement, shown directly in the UI:** this is image-level
  feature matching, not certified biometric fingerprint verification. Real
  fingerprint systems (including BVAS) use minutiae-based matching —
  ridge endings and bifurcations extracted via specialised enhancement —
  via dedicated libraries (e.g. NIST NBIS, SourceAFIS). ORB was designed
  for general visual scenes, not fingerprint ridge patterns, which are
  highly repetitive and can produce false matches.
- **Tested on two datasets:**
  - A synthetic placeholder ridge-pattern (built purely to test the code
    path before real data was available) — this test *did* produce a
    false positive between two unrelated patterns, which is expected
    behavior for ORB on repetitive textures and is documented rather
    than hidden.
  - 4 real images from the SOCOFing dataset (same finger altered, index
    vs. middle finger same person, index finger vs. a different person,
    and a self-match sanity check) — all 4 produced the correct result.
    This is a small sample, not a statistically rigorous validation.
- **Dataset credit:** Shehu, Y. I., Ruiz-Garcia, A., Palade, V., & James, A.
  (2018). *Sokoto Coventry Fingerprint Dataset (SOCOFing)*. Used under its
  CC BY-NC-SA 4.0 licence for academic/non-commercial purposes. Available
  at https://www.kaggle.com/datasets/ruizgara/socofing

### Module 4 — Dashboard (`app.py`)
- Streamlit web dashboard; wires all modules together, shows a clear
  Approved / Flagged status with specific reasons, plus raw technical
  output available on demand (for transparency, not shown by default).
- Runs entirely locally — no internet dependency at runtime, which
  matters for reliability during a live demonstration.

---

## 4. Testing Summary

All modules were tested with real inputs (synthetic sample documents,
generated by `modules/generate_sample_id.py`) and real output — not
hand-written expected results.

| Test | Input | Result |
|---|---|---|
| OCR — clean document | `sample_id_clear.png` | All 6 fields extracted correctly |
| Blur — clear document | `sample_id_clear.png` | Variance 3107 → Sharp → Pass |
| Blur — degraded document | `sample_id_blurry.png` | Variance 1.4 → Severely Blurred → Flagged |
| Face — present & clear | `sample_id_with_face.png` | Detected, variance 908 → Clear |
| Face — present but blurry | `sample_id_blurry_face_only.png` | Detected, variance 8.3 → Blurry → Flagged |
| Face — absent | `sample_id_clear.png` (placeholder box) | Correctly detects no face |
| Dedup — exact match | OKAFOR OBINNA, matching DOB | Flagged (100% similarity) |
| Dedup — typo variant | OKAFFOR OBINNA, matching DOB | Flagged (96%+ similarity) |
| Dedup — reordered name | OBINNA OKAFOR, matching DOB | Flagged (word-order independent) |
| Dedup — same name, different DOB | OKAFOR OBINNA, different DOB | Correctly NOT flagged |
| Dedup — new applicant | Unrelated name/DOB | Correctly NOT flagged |

---

## 5. Limitations & Future Work

- OCR field parsing is layout-dependent; a more robust version would use
  bounding-box-constrained OCR per field rather than line-based parsing.
- No true facial recognition/matching (ID photo vs. live capture) — see
  Module 2B rationale above.
- Fingerprint comparison (Module 5) is an image-feature-matching
  prototype (ORB), not certified biometric verification — see Module 5
  rationale above. Production-grade matching would require dedicated
  fingerprint scanner hardware and minutiae-based matching libraries
  (e.g. NIST NBIS, SourceAFIS).
- Blur and face-clarity thresholds are defaults from general literature,
  not calibrated against real INEC scanning/camera hardware — this
  would be the first thing to tune with real deployment data.
- No authentication/access control layer — this is a prototype triage
  tool, not a production-hardened system.

---

## 6. How to Run

See `INSTALL_WINDOWS.md` for full setup instructions. Quick reference
once set up:

```
cd inec_verification
venv\Scripts\activate
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
streamlit run app.py
```
