"""
generate_sample_id.py
----------------------
Creates synthetic mock Permanent Voter's Card (PVC) style images for
testing the OCR pipeline, styled to resemble the real INEC PVC layout:
a green header band, a VIN, a DELIM (State/LGA/Ward) line, split name
fields, DOB, gender, occupation, and address, with a photo box on the
right.

IMPORTANT: All names, VINs, and addresses generated here are entirely
FICTIONAL, invented for testing purposes. This does not copy or use any
real person's actual card data or photo.

This is ONLY for development/testing. In the real system, this image
would instead come from a file the applicant uploads.
"""
from PIL import Image, ImageDraw, ImageFont
import os

FIELD_LABEL_COLOR = (90, 90, 90)
FIELD_VALUE_COLOR = (20, 20, 20)


def _load_fonts():
    try:
        bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        reg = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        return {
            "title": ImageFont.truetype(bold, 15),
            "subtitle": ImageFont.truetype(bold, 12),
            "label": ImageFont.truetype(reg, 11),
            "value": ImageFont.truetype(bold, 15),
            "code": ImageFont.truetype(reg, 10),
        }
    except IOError:
        d = ImageFont.load_default()
        return {"title": d, "subtitle": d, "label": d, "value": d, "code": d}


def generate_mock_id(
    surname="OKAFOR",
    first_name="OBINNA",
    other_names="CHIDI",
    dob="1994-05-12",
    gender="MALE",
    occupation="TRADING",
    lga="AWKA SOUTH",
    state="ANAMBRA",
    ward="WARD 04",
    address="NO. 12 ENUGU ROAD, AWKA",
    filename="sample_id_clear.png",
    blur=False,
):
    fonts = _load_fonts()
    width, height = 900, 560
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # --- Header band (mirrors the real PVC's green header) ---
    draw.rectangle([(0, 0), (width, 66)], fill=(9, 82, 46))
    draw.text((24, 12), "FEDERAL REPUBLIC OF NIGERIA", fill="white", font=fonts["title"])
    draw.text((24, 34), "INDEPENDENT NATIONAL ELECTORAL COMMISSION", fill="white", font=fonts["subtitle"])

    vin = f"90F{abs(hash(surname + first_name + dob)) % 10**16:016d}"
    code = f"CODE{abs(hash(surname)) % 100:02d}-{abs(hash(first_name)) % 100:02d}-{abs(hash(dob)) % 1000:03d}"

    draw.text((24, 82), "CODE", fill=FIELD_LABEL_COLOR, font=fonts["label"])
    draw.text((24, 98), code, fill=FIELD_VALUE_COLOR, font=fonts["value"])

    draw.text((24, 130), "VIN", fill=FIELD_LABEL_COLOR, font=fonts["label"])
    draw.text((24, 146), vin, fill=FIELD_VALUE_COLOR, font=fonts["value"])

    draw.text((24, 178), "DELIM", fill=FIELD_LABEL_COLOR, font=fonts["label"])
    draw.text((24, 194), f"{state} / {lga} / {ward}", fill=FIELD_VALUE_COLOR, font=fonts["value"])

    rows = [
        ("SURNAME", surname),
        ("FIRST NAME", first_name),
        ("OTHER NAMES", other_names),
        ("DATE OF BIRTH", dob),
        ("GENDER", gender),
        ("OCCUPATION", occupation),
        ("ADDRESS", address),
    ]
    y = 228
    for label, value in rows:
        draw.text((24, y), label, fill=FIELD_LABEL_COLOR, font=fonts["label"])
        draw.text((24, y + 16), value, fill=FIELD_VALUE_COLOR, font=fonts["value"])
        y += 42

    # Photo box, top-right, PVC-style proportions
    draw.rectangle([(680, 86), (860, 300)], outline="black", width=2)
    draw.text((712, 185), "PHOTO", fill="gray", font=fonts["label"])

    if blur:
        import cv2
        import numpy as np
        arr = np.array(img)
        arr = cv2.GaussianBlur(arr, (25, 25), 0)
        img = Image.fromarray(arr)

    out_path = os.path.join(os.path.dirname(__file__), "..", "sample_docs", filename)
    img.save(out_path)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    generate_mock_id(filename="sample_id_clear.png", blur=False)
    generate_mock_id(filename="sample_id_blurry.png", blur=True)
    generate_mock_id(
        surname="UDOSEN", first_name="FAVOUR", other_names="MERCY", dob="2001-02-15",
        gender="FEMALE", occupation="STUDENT", lga="UYO", state="AKWA IBOM", ward="WARD 02",
        address="NO. 5 ABAK ROAD, UYO", filename="sample_id_2.png",
    )
    generate_mock_id(
        surname="BELLO", first_name="AMINA", other_names="ZARA", dob="1999-09-09",
        gender="FEMALE", occupation="TAILORING", lga="KUMBOTSO", state="KANO", ward="WARD 07",
        address="NO. 22 ZOO ROAD, KANO", filename="sample_id_new_applicant.png",
    )
