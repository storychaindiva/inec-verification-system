"""
blur_module.py
----------------
MODULE 2: Image Quality / Blur Detection (OpenCV)

Why this matters:
  A blurry or badly-lit ID photo/scan is unusable for manual review AND
  breaks the OCR/facial-match steps downstream. Better to catch it up front
  and ask the applicant to re-upload than let a bad file flow through the
  whole pipeline.

How it works — Variance of Laplacian:
  The Laplacian operator is an edge-detection filter: it highlights areas
  of rapid intensity change (i.e. edges/detail) in an image.
    - A SHARP image has lots of well-defined edges -> high pixel-value
      variance after applying the Laplacian.
    - A BLURRY image has soft transitions, few sharp edges -> low variance.

  So: variance of the Laplacian is a cheap, well-established proxy for
  "how in-focus is this image". No ML model, no training data needed —
  just a mathematical property of the pixel matrix. That's a strong,
  easily-defensible technique to explain in a viva.
"""
import cv2


def analyze_image_blur(image_np, blur_threshold=100.0):
    """
    Returns (variance, is_clear).

    blur_threshold=100.0 is the commonly cited default from the classic
    "detect blur" technique (Pech-Pacheco et al.). It's a reasonable
    starting point but SHOULD be tuned against your own real sample set —
    scanned documents behave differently from photographs, and a threshold
    that works for one INEC scanner/camera setup might not suit another.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    is_clear = variance >= blur_threshold
    return float(variance), is_clear


def classify_quality(variance):
    """
    Optional: turn the raw variance number into a human-readable tier.
    Purely for the dashboard's benefit — thresholds here are illustrative,
    not scientifically fixed; tune them once you test with real scans.
    """
    if variance >= 300:
        return "SHARP"
    elif variance >= 100:
        return "ACCEPTABLE"
    elif variance >= 40:
        return "LOW QUALITY"
    else:
        return "SEVERELY BLURRED"


if __name__ == "__main__":
    import os

    sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_docs")
    test_files = ["sample_id_clear.png", "sample_id_blurry.png", "sample_id_2.png"]

    print("=" * 60)
    print("BLUR ANALYSIS RESULTS")
    print("=" * 60)
    for fname in test_files:
        path = os.path.join(sample_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"{fname}: COULD NOT LOAD")
            continue
        variance, is_clear = analyze_image_blur(img)
        tier = classify_quality(variance)
        status = "PASS" if is_clear else "FLAGGED"
        print(f"{fname:25s} | variance={variance:9.2f} | {tier:16s} | {status}")
