"""
face_module.py
----------------
MODULE 2B: Facial Photo Presence & Clarity Check

SCOPE (deliberately limited): this module answers two questions only:
  1. Is there a detectable human face in the applicant's photo?
  2. Is that face region clear/in-focus (not just the whole image)?

It does NOT do facial matching/recognition (comparing one face to another,
e.g. ID photo vs a live selfie) — that's a materially different, heavier
task (would need DeepFace/face_recognition + real paired face data to
validate honestly). This module is an honest, tested subset: it catches
a blank/missing photo box or an unusably blurry face crop, which is a
real and common data-quality problem in registration photo uploads.

How it works — Haar Cascade:
  A Haar Cascade is a classical (pre-deep-learning) object detector, built
  into OpenCV, trained to recognise the general pattern of a frontal human
  face (eyes/nose/mouth contrast regions). It's fast, has zero extra
  dependencies beyond opencv-python (the cascade file ships inside the
  package), and is a reasonable, explainable choice for "is a face present"
  -- as opposed to needing a full deep-learning model for that sub-task.
"""
import cv2
import os

_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
_face_detector = cv2.CascadeClassifier(_CASCADE_PATH)


def detect_faces(image_np, min_size=40):
    """
    Runs face detection on the image. Returns a list of face bounding boxes
    (x, y, w, h). An empty list means no face was detected.

    min_size=40 (pixels) is deliberately smaller than OpenCV's common
    default of 60-80px. Real ID document photo crops are often small
    (a passport-style photo scanned as part of a larger form page), so a
    higher minimum size threshold would cause false "no face detected"
    results on perfectly valid, if modest-resolution, applicant photos.
    This was found by testing against a realistically-sized photo crop,
    not assumed upfront -- worth mentioning if asked in a defense.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = _face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size),
    )
    return list(faces)


def check_face_presence_and_clarity(image_np, blur_threshold=80.0):
    """
    Combines face detection with a blur check ON THE FACE REGION SPECIFICALLY
    (not the whole document) -- a photo can have a sharp document but a
    blurry/low-res face crop, or vice versa.

    Returns a dict describing the result.
    """
    faces = detect_faces(image_np)

    if len(faces) == 0:
        return {
            "face_detected": False,
            "face_count": 0,
            "face_variance": None,
            "face_is_clear": False,
            "note": "No face detected in the uploaded image.",
        }

    # If multiple faces are detected (e.g. noise in the document layout),
    # use the LARGEST detected region -- most likely the actual ID photo.
    largest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest
    face_crop = image_np[y:y + h, x:x + w]

    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    is_clear = variance >= blur_threshold

    return {
        "face_detected": True,
        "face_count": len(faces),
        "face_variance": float(variance),
        "face_is_clear": is_clear,
        "face_box": (int(x), int(y), int(w), int(h)),
        "note": "Face detected and clear." if is_clear else "Face detected but low quality/blurry.",
    }


if __name__ == "__main__":
    import os as _os

    sample_dir = _os.path.join(_os.path.dirname(__file__), "..", "sample_docs")

    print("=" * 70)
    print("FACE DETECTION TEST CASES")
    print("=" * 70)

    # Test 1: real face photo (bundled scikit-image sample, saved earlier)
    ref_path = _os.path.join(sample_dir, "test_face_reference.png")
    if _os.path.exists(ref_path):
        img = cv2.imread(ref_path)
        result = check_face_presence_and_clarity(img)
        print("\nTest 1: Real face photo (reference image)")
        print(f"  {result}")

        # Test 2: deliberately blurred version of the same face
        blurred = cv2.GaussianBlur(img, (35, 35), 0)
        result_blur = check_face_presence_and_clarity(blurred)
        print("\nTest 2: Same face, deliberately blurred")
        print(f"  {result_blur}")
    else:
        print("Reference face image not found -- run generate step first.")

    # Test 3: our mock ID card (no real face, just a placeholder box)
    id_path = _os.path.join(sample_dir, "sample_id_clear.png")
    if _os.path.exists(id_path):
        img = cv2.imread(id_path)
        result_id = check_face_presence_and_clarity(img)
        print("\nTest 3: Mock ID card (placeholder 'PHOTO' box, no real face)")
        print(f"  {result_id}")
