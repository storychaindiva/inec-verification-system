# Running the INEC Verification System on Windows

Follow these steps in order. Each one is verifiable — don't move to the next
step until the current one's check command works.

---

## Step 1: Install Python

1. Download Python 3.11 or 3.12 from https://www.python.org/downloads/
2. Run the installer. **Important:** on the first install screen, tick
   **"Add python.exe to PATH"** before clicking Install.
3. Verify it worked — open Command Prompt (`Win + R`, type `cmd`, Enter) and run:
   ```
   python --version
   ```
   You should see something like `Python 3.12.x`. If you get "not recognized",
   Python wasn't added to PATH — rerun the installer and check that box.

---

## Step 2: Install Tesseract OCR (the engine itself)

`pip install pytesseract` (later) only installs a Python *wrapper* — it does
NOT install the actual OCR engine. You need both.

1. Download the Windows installer from the UB Mannheim build (the standard
   Windows distribution of Tesseract):
   https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer. Note the install path shown — by default it's:
   ```
   C:\Program Files\Tesseract-OCR\
   ```
3. Verify it worked — open Command Prompt and run:
   ```
   "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
   ```
   You should see version info printed. If this works but plain `tesseract
   --version` doesn't, that's expected — Tesseract isn't on PATH by default
   on Windows, which is fine, because we handle that in code (see Step 5).

---

## Step 3: Copy the project folder

Copy the entire `inec_verification` folder onto your Windows PC — e.g. to
`C:\Users\<you>\Documents\inec_verification`.

---

## Step 4: Create a virtual environment and install dependencies

Open Command Prompt, navigate into the project folder, and run:

```
cd C:\Users\<you>\Documents\inec_verification
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

You'll know the virtual environment is active because your prompt line will
start with `(venv)`. You need to run `venv\Scripts\activate` every time you
open a new terminal to work on this project.

---

## Step 5: Point pytesseract to the Tesseract install

Since Tesseract isn't on Windows PATH by default, tell Python where it is.
**Easiest option** — set it for just this terminal session, every time before
running the app:

```
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

(`ocr_module.py` already checks for this environment variable — see the
top of that file.)

Alternative: edit `modules/ocr_module.py` directly and uncomment the line
that hardcodes the path, if you'd rather not set the env variable each time.

---

## Step 6: Run the dashboard

Still inside the activated virtual environment:

```
streamlit run app.py
```

This should automatically open `http://localhost:8501` in your browser. If
it doesn't open automatically, copy that URL into your browser manually.

You should see the INEC Verification dashboard. Upload one of the sample
images from `sample_docs/` (or a real one) and click **Run Verification
Pipeline** in the sidebar.

---

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `'streamlit' is not recognized` | Virtual environment isn't active — run `venv\Scripts\activate` first |
| `TesseractNotFoundError` | Step 5 wasn't done in this terminal session, or the path is wrong — re-check the exact install path |
| Blank/white page in browser | Wait a few seconds on first run (Streamlit is compiling); otherwise check the Command Prompt window for a Python traceback |
| `ModuleNotFoundError: No module named 'cv2'` | `pip install -r requirements.txt` didn't complete — rerun it and watch for errors |

---

## Before the defense: run this checklist

- [ ] App launches with `streamlit run app.py` with **no internet connection**
      (turn off wifi and test — this confirms nothing secretly depends on a
      network call, which matters if the venue's wifi is unreliable)
- [ ] Test with a clear sample image → confirm APPROVED status
- [ ] Test with a blurry sample image → confirm FLAGGED status
- [ ] Test with a duplicate name (matching one of the seeded DB records) →
      confirm FLAGGED status
- [ ] Your colleague can explain, in her own words, what each of the 3
      pipeline stages does and WHY that technique was chosen — not just
      that it works
