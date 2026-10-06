"""
ocr_module.py
--------------
MODULE 1: Document Text Extraction (OCR)

What this does:
  1. Loads an uploaded ID image
  2. Preprocesses it (grayscale + threshold) so Tesseract reads it more accurately
  3. Runs Tesseract OCR to turn pixels -> raw text
  4. Parses the raw text into structured fields using simple rule-based matching

Why preprocessing matters:
  Tesseract is a pattern-matcher trained mostly on clean black-on-white text.
  Colour noise, shadows, and low contrast all reduce accuracy. Converting to
  grayscale and applying a binary threshold (pure black/white) removes a lot
  of that noise before the text even reaches the OCR engine.
"""
import cv2
import pytesseract
import re
import os

# ---------------------------------------------------------------------------
# WINDOWS-SPECIFIC SETUP
# ---------------------------------------------------------------------------
# On Linux/Mac, the `tesseract` command is usually already on PATH after
# installation. On WINDOWS, it is NOT automatically on PATH, so pytesseract
# can't find the engine unless you tell it exactly where the .exe lives.
#
# If you installed Tesseract with the default UB-Mannheim installer path,
# uncomment the line below (adjust the path if you installed it elsewhere):
#
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#
# You can also set it via an environment variable instead of hardcoding it:
_tess_cmd = os.environ.get("TESSERACT_CMD")
if _tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd


def preprocess_for_ocr(image_np):
    """
    Convert a raw image (as loaded by OpenCV, BGR format) into a cleaned-up
    black-and-white version optimised for OCR accuracy.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Otsu's method automatically picks the best threshold value to split
    # the image into pure black and pure white, rather than us guessing one.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_document_text(image_np):
    """
    Runs Tesseract OCR on a preprocessed image and returns the raw extracted text.
    """
    processed = preprocess_for_ocr(image_np)
    raw_text = pytesseract.image_to_string(processed)
    return raw_text


_KNOWN_NOISE_WORDS = {"PHOTO", "IMAGE", "PASSPORT", "SIGNATURE", "SIG"}
_LABEL_MAP = {
    "SURNAME": "surname",
    "FIRST NAME": "first_name",
    "OTHER NAMES": "other_names",
    "DATE OF BIRTH": "dob",
    "GENDER": "gender",
    "OCCUPATION": "occupation",
    "ADDRESS": "address",
    "VIN": "vin",
    "DELIM": "_delim",  # special-cased: splits into state/lga/ward below
}


def _clean_name_field(raw_value):
    """
    Defends against stray layout text (e.g. a 'PHOTO' placeholder label
    that shares a text line with a name field) bleeding into a name field.
    Keeps only alphabetic tokens that aren't known layout-noise words.
    """
    tokens = raw_value.strip().split()
    cleaned_tokens = [t for t in tokens if t.isalpha() and t not in _KNOWN_NOISE_WORDS]
    return " ".join(cleaned_tokens) if cleaned_tokens else raw_value.strip()


def parse_extracted_fields(raw_text):
    """
    Parses fields from a PVC-style layout where each field is a LABEL on
    its own line, followed by the VALUE on the next non-empty line (this
    mirrors the real INEC Permanent Voter's Card layout, which does not
    use inline "LABEL: value" formatting).

    Approach: scan lines in order; when a line matches a known label
    (allowing for OCR noise/whitespace), treat the next non-empty line as
    that field's value. This is more layout-realistic than a single-line
    keyword split, but still fully rule-based and explainable -- no ML.

    DELIM is a special case: on a real PVC it's a single combined field
    ("STATE / LGA / WARD"), so we split it into three separate values.
    """
    data = {
        "surname": None, "first_name": None, "other_names": None,
        "dob": None, "gender": None, "occupation": None, "address": None,
        "state": None, "lga": None, "ward": None, "vin": None,
    }

    lines = [line.strip() for line in raw_text.split("\n")]
    lines = [line for line in lines if line]  # drop blank lines

    i = 0
    while i < len(lines):
        clean = lines[i].upper()
        matched_field = None
        for label, field_key in _LABEL_MAP.items():
            if clean == label or clean.startswith(label):
                matched_field = field_key
                break

        if matched_field and i + 1 < len(lines):
            value_line = lines[i + 1].strip()

            if matched_field == "_delim":
                # Expect "STATE / LGA / WARD"
                parts = [p.strip() for p in value_line.split("/")]
                if len(parts) >= 1:
                    data["state"] = parts[0] if parts[0] else None
                if len(parts) >= 2:
                    data["lga"] = parts[1] if parts[1] else None
                if len(parts) >= 3:
                    data["ward"] = parts[2] if parts[2] else None
            elif matched_field in ("surname", "first_name", "other_names"):
                data[matched_field] = _clean_name_field(value_line)
            elif matched_field == "dob":
                match = re.search(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", value_line)
                data["dob"] = match.group(0) if match else value_line
            elif matched_field == "vin":
                match = re.search(r"[A-Z0-9]{15,25}", value_line.upper())
                data["vin"] = match.group(0) if match else value_line
            else:
                data[matched_field] = value_line

            i += 2  # skip past the value line we just consumed
        else:
            i += 1

    return data


if __name__ == "__main__":
    # Quick manual test against the sample image we generated
    import os

    img_path = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "sample_id_clear.png")
    image_np = cv2.imread(img_path)

    print("=" * 60)
    print("RAW OCR OUTPUT:")
    print("=" * 60)
    raw = extract_document_text(image_np)
    print(raw)

    print("=" * 60)
    print("PARSED STRUCTURED FIELDS:")
    print("=" * 60)
    fields = parse_extracted_fields(raw)
    for k, v in fields.items():
        print(f"  {k}: {v}")
