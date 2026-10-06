"""
fingerprint_module.py
------------------------
MODULE 5: Image-Based Fingerprint Feature Matching (PROTOTYPE)

SCOPE AND HONEST LIMITS -- read this before presenting it as more than it is:

This module compares two fingerprint IMAGES using ORB (Oriented FAST and
Rotated BRIEF) -- a general-purpose keypoint detector/descriptor built into
OpenCV. It is NOT real biometric fingerprint verification. Real systems
(e.g. what BVAS hardware uses) extract MINUTIAE -- ridge endings and
bifurcations, found via ridge-thinning and specialised enhancement -- and
match those structured point-sets. That requires dedicated libraries
(e.g. NIST NBIS, SourceAFIS) and purpose-built algorithms, not a
general-purpose keypoint matcher.

ORB was never designed for fingerprints. Fingerprint ridge patterns are
highly repetitive, so ORB keypoints in one ridge valley can look very
similar to keypoints in a different, unrelated ridge valley -- this is a
real and expected source of false matches, and the testing in this file's
__main__ block is designed to surface that honestly rather than hide it.

This module is presented as: "a working image-feature-matching prototype,
demonstrating the matching CONCEPT" -- not as a fingerprint verification
system suitable for production or for BVAS integration.
"""
import cv2
import numpy as np

# Lowe's ratio test threshold -- a standard technique (Lowe, 2004) for
# filtering out weak/ambiguous keypoint matches before counting them.
RATIO_TEST_THRESHOLD = 0.75
MATCH_COUNT_THRESHOLD = 15  # tunable; see __main__ test block for real numbers


def _load_grayscale(path_or_array):
    if isinstance(path_or_array, str):
        img = cv2.imread(path_or_array, cv2.IMREAD_GRAYSCALE)
    else:
        img = path_or_array
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def compare_fingerprints(image_a, image_b):
    """
    Compares two fingerprint images using ORB keypoint detection + a
    brute-force descriptor matcher with Lowe's ratio test.

    Returns a dict with the raw keypoint/match counts and a verdict --
    the verdict is explicitly a "prototype match", not a certified
    biometric result.
    """
    img_a = _load_grayscale(image_a)
    img_b = _load_grayscale(image_b)

    orb = cv2.ORB_create(nfeatures=500)
    kp_a, des_a = orb.detectAndCompute(img_a, None)
    kp_b, des_b = orb.detectAndCompute(img_b, None)

    if des_a is None or des_b is None or len(kp_a) == 0 or len(kp_b) == 0:
        return {
            "keypoints_a": len(kp_a) if kp_a else 0,
            "keypoints_b": len(kp_b) if kp_b else 0,
            "good_matches": 0,
            "is_match": False,
            "note": "Could not extract enough features from one or both images.",
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des_a, des_b, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_TEST_THRESHOLD * n.distance:
                good_matches.append(m)

    is_match = len(good_matches) >= MATCH_COUNT_THRESHOLD

    return {
        "keypoints_a": len(kp_a),
        "keypoints_b": len(kp_b),
        "good_matches": len(good_matches),
        "is_match": is_match,
        "note": (
            "Prototype match (image-feature level, not certified biometric verification)."
            if is_match else
            "No sufficient feature match found."
        ),
    }


if __name__ == "__main__":
    import os

    sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "fingerprints")

    print("=" * 70)
    print("FINGERPRINT MATCHING TEST -- PLACEHOLDER SYNTHETIC IMAGES")
    print("These are NOT real fingerprints. They only verify the code runs")
    print("correctly end-to-end. Real pass/fail numbers require real")
    print("fingerprint samples (e.g. SOCOFing) -- see conversation for that step.")
    print("=" * 70)

    a = os.path.join(sample_dir, "placeholder_same_A.png")
    a_altered = os.path.join(sample_dir, "placeholder_same_A_altered.png")
    b = os.path.join(sample_dir, "placeholder_different_B.png")

    if os.path.exists(a):
        print("\nTest 1: Same source, slightly shifted (simulates a re-scan)")
        print(" ", compare_fingerprints(a, a_altered))

        print("\nTest 2: Different source image entirely")
        print(" ", compare_fingerprints(a, b))

        print("\nTest 3: Identical image against itself (sanity check)")
        print(" ", compare_fingerprints(a, a))
    else:
        print("Placeholder images not found -- generate them first.")
