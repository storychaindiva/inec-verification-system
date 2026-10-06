"""
app.py
-------
INEC Automated Document Verification System — Main Dashboard

Wires together:
  Module 1 (ocr_module.py)    -> text extraction from uploaded ID
  Module 2 (blur_module.py)   -> image quality check
  Module 3 (dedup_module.py)  -> duplicate detection against voter DB

Run with:
    streamlit run app.py

Requires (install once):
    pip install streamlit opencv-python pytesseract Pillow
    (fuzzywuzzy/python-Levenshtein not required -- dedup_module uses
     Python's built-in difflib. See dedup_module.py docstring if you'd
     rather switch to fuzzywuzzy.)

Also requires the Tesseract OCR ENGINE itself to be installed on the
system (separate from the pytesseract Python package). See the
INSTALL_WINDOWS.md guide for exact steps.

STYLING NOTE:
  All custom styling uses the system font stack (Segoe UI / Helvetica /
  Arial) rather than a Google Font import. This is deliberate -- the app
  needs to run correctly with no internet connection at defense time, and
  a remote font request would silently fail (or hang) offline.
"""
import streamlit as st
import cv2
import numpy as np
import time
import sys
import os

# Make sibling modules importable regardless of where streamlit is launched from
sys.path.append(os.path.join(os.path.dirname(__file__)))

from modules.ocr_module import extract_document_text, parse_extracted_fields
from modules.blur_module import analyze_image_blur, classify_quality
from modules.dedup_module import check_database_duplicates, init_database
from modules.face_module import check_face_presence_and_clarity
from modules.fingerprint_module import compare_fingerprints

st.set_page_config(page_title="INEC CVR Verification System", layout="wide", page_icon=None)
init_database()

# ---------------------------------------------------------------------------
# DESIGN SYSTEM (custom CSS — overrides Streamlit's default look)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    :root {
        --ink: #1A1A1A;
        --ink-secondary: #5B6660;
        --surface: #FFFFFF;
        --panel: #F6F7F6;
        --border: #E2E5E2;
        --primary: #0B4D2C;
        --primary-dark: #073A20;
        --gold: #C9A227;
        --success: #1E7A34;
        --success-bg: #EAF4EC;
        --alert: #A3341E;
        --alert-bg: #FBEEEB;
    }

    /* Force light background everywhere, regardless of OS/browser dark mode */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--surface) !important;
    }
    html, body, [class*="css"] {
        font-family: "Segoe UI", -apple-system, "Helvetica Neue", Arial, sans-serif;
        color: var(--ink) !important;
    }
    p, span, label, div { color: var(--ink); }

    /* Kill default Streamlit top padding/branding bulk */
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1150px; }
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

    /* Header banner -- solid green, echoes the ID card design */
    .app-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        border-radius: 10px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 14px rgba(11, 77, 44, 0.18);
    }
    .app-header .wordmark {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: var(--gold);
        margin-bottom: 0.4rem;
    }
    .app-header h1 {
        font-size: 1.85rem;
        font-weight: 600;
        margin: 0;
        color: #FFFFFF !important;
        line-height: 1.3;
    }
    .app-header p {
        color: #DCE8DF !important;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        max-width: 640px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--panel) !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--ink-secondary) !important;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.5rem;
        margin-top: 0.5rem;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: var(--primary);
        border: none;
        font-weight: 500;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(11, 77, 44, 0.25);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--primary-dark);
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.55rem 1.1rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.75rem;
    }
    .status-approved {
        background-color: var(--success-bg);
        color: var(--success) !important;
        border: 1px solid var(--success);
    }
    .status-flagged {
        background-color: var(--alert-bg);
        color: var(--alert) !important;
        border: 1px solid var(--alert);
    }

    /* Stat cards */
    .stat-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .stat-card .label {
        font-size: 0.78rem;
        color: var(--ink-secondary) !important;
        margin-bottom: 0.3rem;
    }
    .stat-card .value {
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--ink) !important;
    }
    .stat-card .sub {
        font-size: 0.78rem;
        color: var(--ink-secondary) !important;
        margin-top: 0.2rem;
    }

    /* Issue list */
    .issue-item {
        padding: 0.6rem 0.9rem;
        background-color: var(--alert-bg);
        border-left: 3px solid var(--alert);
        border-radius: 4px;
        margin-bottom: 0.4rem;
        font-size: 0.92rem;
        color: var(--ink) !important;
    }

    .section-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--ink) !important;
        margin-top: 1.8rem;
        margin-bottom: 0.7rem;
    }

    /* Uploaded document image */
    [data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    /* Scope/limitation banner (used on the Fingerprint tab) */
    .scope-banner {
        background-color: #FBF6E8;
        border: 1px solid var(--gold);
        border-radius: 6px;
        padding: 0.8rem 1.1rem;
        font-size: 0.88rem;
        color: var(--ink);
        margin-bottom: 1.2rem;
    }
    .scope-banner b { color: #8A6A10; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="wordmark">INEC — CONTINUOUS VOTER REGISTRATION</div>
    <h1>Automated Document Verification System</h1>
    <p>Checks document clarity, extracts applicant details, and flags possible duplicate records for ICT review.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TOP-LEVEL TABS
# ---------------------------------------------------------------------------
tab_doc, tab_finger = st.tabs(["Document Verification", "Fingerprint Comparison (Prototype)"])

# ---------------------------------------------------------------------------
# SIDEBAR: UPLOAD + SETTINGS (applies to the Document Verification tab)
# ---------------------------------------------------------------------------
st.sidebar.caption("Settings below apply to the Document Verification tab.")
st.sidebar.markdown("### Upload document")
uploaded_doc = st.sidebar.file_uploader(
    "Applicant ID or registration form",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

st.sidebar.markdown("### Settings")
blur_threshold = st.sidebar.slider(
    "Blur sensitivity threshold", min_value=20, max_value=500, value=100, step=10,
    help="Images with a Laplacian variance below this value are flagged as low quality.",
)
run_button = st.sidebar.button("Run verification", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 1: DOCUMENT VERIFICATION
# ---------------------------------------------------------------------------
with tab_doc:
    if uploaded_doc is None:
        st.info("Upload an ID document image from the left panel to begin.")
    else:
        file_bytes = np.asarray(bytearray(uploaded_doc.read()), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, 1)

        col_preview, col_results = st.columns([1, 2])

        with col_preview:
            st.markdown('<div class="section-label">Uploaded document</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), use_container_width=True)

        if run_button:
            with col_results:
                start = time.time()

                with st.spinner("Analyzing image quality..."):
                    variance, is_clear = analyze_image_blur(image_np, blur_threshold=blur_threshold)
                    quality_tier = classify_quality(variance)

                with st.spinner("Checking applicant photo..."):
                    face_result = check_face_presence_and_clarity(image_np)

                with st.spinner("Extracting document text..."):
                    raw_text = extract_document_text(image_np)
                    fields = parse_extracted_fields(raw_text)

                with st.spinner("Checking for duplicate records..."):
                    dup_result = {"is_duplicate": False, "score": 0, "matched_name": None}
                    if fields.get("surname") and fields.get("first_name"):
                        dup_result = check_database_duplicates(
                            fields["surname"], fields["first_name"], fields.get("dob")
                        )

                elapsed = time.time() - start

                issues = []
                if not is_clear:
                    issues.append("Image quality is below the acceptable threshold.")
                if not face_result["face_detected"]:
                    issues.append("No applicant photo detected in the document.")
                elif not face_result["face_is_clear"]:
                    issues.append("Applicant photo was detected but is low quality/blurry.")
                if dup_result["is_duplicate"]:
                    issues.append(f"Possible duplicate of existing record: {dup_result['matched_name']}.")
                if not fields.get("surname") or not fields.get("first_name"):
                    issues.append("Could not confidently extract the applicant's name fields.")

                approved = len(issues) == 0

                st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)

                if approved:
                    st.markdown(
                        '<span class="status-badge status-approved">Approved</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span class="status-badge status-flagged">Flagged for review</span>',
                        unsafe_allow_html=True,
                    )
                    for issue in issues:
                        st.markdown(f'<div class="issue-item">{issue}</div>', unsafe_allow_html=True)

                st.caption(f"Processed in {elapsed:.2f} seconds")

                st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="label">Image quality</div>
                        <div class="value">{quality_tier.title()}</div>
                        <div class="sub">Laplacian variance: {variance:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    if face_result["face_detected"]:
                        face_value = "Clear" if face_result["face_is_clear"] else "Blurry"
                        face_sub = f"Variance: {face_result['face_variance']:.1f}"
                    else:
                        face_value = "Not found"
                        face_sub = "No photo detected"
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="label">Applicant photo</div>
                        <div class="value">{face_value}</div>
                        <div class="sub">{face_sub}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="label">Duplicate match score</div>
                        <div class="value">{dup_result['score']}%</div>
                        <div class="sub">Threshold for review: 85%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    extracted_count = sum(1 for v in fields.values() if v)
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="label">Fields extracted</div>
                        <div class="value">{extracted_count} of {len(fields)}</div>
                        <div class="sub">Surname, first name, DOB, LGA, state, VIN</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="section-label">Extracted details</div>', unsafe_allow_html=True)
                field_labels = {
                    "surname": "Surname", "first_name": "First name", "dob": "Date of birth",
                    "lga": "LGA", "state": "State of origin", "vin": "VIN",
                }
                detail_cols = st.columns(2)
                for i, (key, label) in enumerate(field_labels.items()):
                    with detail_cols[i % 2]:
                        st.markdown(f"**{label}:** {fields.get(key) or '—'}")

                with st.expander("Technical details (raw OCR output, duplicate-check data)"):
                    t1, t2, t3 = st.tabs(["Raw OCR text", "Duplicate check data", "Face check data"])
                    with t1:
                        st.text(raw_text)
                    with t2:
                        st.json(dup_result)
                    with t3:
                        st.json(face_result)
        else:
            with col_results:
                st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
                st.caption("Click Run verification in the left panel to process this document.")

# ---------------------------------------------------------------------------
# TAB 2: FINGERPRINT COMPARISON (PROTOTYPE)
# ---------------------------------------------------------------------------
with tab_finger:
    st.markdown("""
    <div class="scope-banner">
        <b>Prototype scope:</b> this compares two fingerprint IMAGES using general-purpose
        visual feature matching (ORB). It does not perform certified biometric fingerprint
        verification, which requires minutiae-based matching and dedicated hardware/libraries.
        Treat results here as a demonstration of the matching concept, not a production
        verification result.
    </div>
    """, unsafe_allow_html=True)

    fp_col1, fp_col2 = st.columns(2)
    with fp_col1:
        st.markdown('<div class="section-label">Fingerprint A</div>', unsafe_allow_html=True)
        fp_file_a = st.file_uploader(
            "Fingerprint A", type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed", key="fp_a",
        )
        if fp_file_a is not None:
            st.image(fp_file_a, width=220)

    with fp_col2:
        st.markdown('<div class="section-label">Fingerprint B</div>', unsafe_allow_html=True)
        fp_file_b = st.file_uploader(
            "Fingerprint B", type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed", key="fp_b",
        )
        if fp_file_b is not None:
            st.image(fp_file_b, width=220)

    fp_compare_button = st.button("Compare fingerprints", type="primary")

    if fp_compare_button:
        if fp_file_a is None or fp_file_b is None:
            st.warning("Upload both Fingerprint A and Fingerprint B before comparing.")
        else:
            with st.spinner("Extracting and matching features..."):
                bytes_a = np.asarray(bytearray(fp_file_a.read()), dtype=np.uint8)
                bytes_b = np.asarray(bytearray(fp_file_b.read()), dtype=np.uint8)
                img_a = cv2.imdecode(bytes_a, cv2.IMREAD_GRAYSCALE)
                img_b = cv2.imdecode(bytes_b, cv2.IMREAD_GRAYSCALE)
                fp_result = compare_fingerprints(img_a, img_b)

            st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
            if fp_result["is_match"]:
                st.markdown(
                    '<span class="status-badge status-approved">Prototype match</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="status-badge status-flagged">No match found</span>',
                    unsafe_allow_html=True,
                )
            st.caption(fp_result["note"])

            fm1, fm2, fm3 = st.columns(3)
            with fm1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="label">Keypoints — Fingerprint A</div>
                    <div class="value">{fp_result['keypoints_a']}</div>
                </div>
                """, unsafe_allow_html=True)
            with fm2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="label">Keypoints — Fingerprint B</div>
                    <div class="value">{fp_result['keypoints_b']}</div>
                </div>
                """, unsafe_allow_html=True)
            with fm3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="label">Good feature matches</div>
                    <div class="value">{fp_result['good_matches']}</div>
                    <div class="sub">Match threshold: 15</div>
                </div>
                """, unsafe_allow_html=True)
