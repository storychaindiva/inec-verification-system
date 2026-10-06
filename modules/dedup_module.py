"""
dedup_module.py
------------------
MODULE 3: Duplicate Detection via Fuzzy String Matching + SQLite

Why fuzzy matching instead of exact SQL WHERE matching:
  SQL `WHERE surname = 'OKAFOR' AND first_name = 'OBINNA'` only catches
  EXACT string matches. It fails on:
    - Typos: "OKAFOR" vs "OKAFFOR"
    - Word order swaps: "OKAFOR OBINNA" vs "OBINNA OKAFOR"
    - Minor spelling variants: "MOHAMMED" vs "MUHAMMED"
  Fuzzy matching scores SIMILARITY (0-100) instead of exact equality, so
  these near-duplicates still get flagged for a human to check.

Match rule used here (deliberately conservative to avoid false positives):
  A record is flagged as a likely duplicate only if:
    - name_similarity_score >= NAME_THRESHOLD, AND
    - date_of_birth matches exactly (or is missing on one side)
  Name-only matching would flag huge numbers of unrelated people who simply
  share a common surname. Requiring DOB agreement too makes a flag mean
  something specific enough for a human reviewer to act on.

NOTE ON LIBRARIES:
  This file uses Python's built-in `difflib` (SequenceMatcher) so it runs
  with zero extra installs. On your Windows machine, once you `pip install
  fuzzywuzzy python-Levenshtein`, you can swap in fuzzywuzzy's
  `token_sort_ratio` — it does the same *job* (word-order-independent
  similarity scoring) with a C-optimised backend. Both approaches are shown
  below so you can defend either choice.
"""
import sqlite3
import os
from difflib import SequenceMatcher

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "voter_register.db")

NAME_MATCH_THRESHOLD = 85  # 0-100 scale; tune against real test data


def init_database():
    """Creates the voters table and seeds it with sample records, if empty."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS voters (
            vin TEXT PRIMARY KEY,
            surname TEXT,
            first_name TEXT,
            dob TEXT,
            state TEXT
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM voters")
    if cursor.fetchone()[0] == 0:
        sample_voters = [
            ("90F1234567890123456", "OKAFOR", "OBINNA", "1994-05-12", "ANAMBRA"),
            ("90F9876543210987654", "UDOSEN", "FAVOUR", "2001-02-15", "AKWA IBOM"),
            ("90F5554443332221110", "ADEBAYO", "OLUMIDE", "1988-11-23", "LAGOS"),
        ]
        cursor.executemany("INSERT INTO voters VALUES (?, ?, ?, ?, ?)", sample_voters)
        conn.commit()
    conn.close()


def token_sort_similarity(a: str, b: str) -> float:
    """
    Zero-dependency equivalent of fuzzywuzzy's token_sort_ratio:
      1. Split each string into words (tokens)
      2. Sort the tokens alphabetically (this is what makes word ORDER
         stop mattering -- "OKAFOR OBINNA" and "OBINNA OKAFOR" both sort
         to "OBINNA OKAFOR")
      3. Rejoin and compare similarity with SequenceMatcher

    Equivalent fuzzywuzzy call on your Windows machine:
        from fuzzywuzzy import fuzz
        score = fuzz.token_sort_ratio(a, b)
    """
    tokens_a = sorted(a.upper().split())
    tokens_b = sorted(b.upper().split())
    norm_a, norm_b = " ".join(tokens_a), " ".join(tokens_b)
    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    return round(ratio * 100, 1)


def check_database_duplicates(surname, first_name, dob=None):
    """
    Compares the incoming applicant's name against every record in the
    voter database and returns the closest match plus its score.

    Returns a dict: { is_duplicate, score, matched_vin, matched_name, matched_dob }
    """
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT surname, first_name, dob, vin FROM voters")
    records = cursor.fetchall()
    conn.close()

    incoming_full_name = f"{surname} {first_name}"

    best = {"score": 0, "matched_vin": None, "matched_name": None, "matched_dob": None}

    for db_surname, db_first, db_dob, db_vin in records:
        db_full_name = f"{db_surname} {db_first}"
        score = token_sort_similarity(incoming_full_name, db_full_name)

        if score > best["score"]:
            best = {
                "score": score,
                "matched_vin": db_vin,
                "matched_name": db_full_name,
                "matched_dob": db_dob,
            }

    dob_matches = (dob is not None and dob == best["matched_dob"])
    is_duplicate = best["score"] >= NAME_MATCH_THRESHOLD and dob_matches

    return {
        "is_duplicate": is_duplicate,
        "score": best["score"],
        "matched_vin": best["matched_vin"],
        "matched_name": best["matched_name"],
        "matched_dob": best["matched_dob"],
        "dob_provided": dob,
    }


if __name__ == "__main__":
    init_database()

    print("=" * 70)
    print("DUPLICATE DETECTION TEST CASES")
    print("=" * 70)

    test_cases = [
        # (surname, first_name, dob, description)
        ("OKAFOR", "OBINNA", "1994-05-12", "Exact match (should flag)"),
        ("OKAFFOR", "OBINNA", "1994-05-12", "Typo in surname (should still flag)"),
        ("OBINNA", "OKAFOR", "1994-05-12", "Swapped name order (should still flag)"),
        ("OKAFOR", "OBINNA", "1985-01-01", "Same name, DIFFERENT dob (should NOT flag)"),
        ("BELLO", "AMINA", "1999-09-09", "Completely new applicant (should NOT flag)"),
    ]

    for surname, first_name, dob, description in test_cases:
        result = check_database_duplicates(surname, first_name, dob)
        verdict = "🚩 FLAGGED AS DUPLICATE" if result["is_duplicate"] else "✅ TREATED AS NEW"
        print(f"\n{description}")
        print(f"  Input: {surname} {first_name} ({dob})")
        print(f"  Closest DB match: {result['matched_name']} ({result['matched_dob']}) "
              f"| score={result['score']}")
        print(f"  Verdict: {verdict}")
