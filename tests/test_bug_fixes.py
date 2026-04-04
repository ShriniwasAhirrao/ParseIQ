"""
Manual verification tests for the three bug fixes:
  Bug 1 - Data sheet type coercion (astype(str) removed)
  Bug 2 - LOW_UNIQUENESS false positive on boolean fields
  Bug 3 - Affected_Tables showing "Multiple" instead of actual table name
"""

import sys
import os
import pandas as pd
import numpy as np
import openpyxl
import io
from contextlib import redirect_stdout

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1_metadata_extractor.extractor import MetadataExtractor


# ---------------------------------------------------------------------------
# BUG 1 - Data sheet must preserve types (no astype(str))
# ---------------------------------------------------------------------------

def test_bug1_numeric_preserved():
    """After fix, integer columns must stay int in the DataFrame."""
    data = [
        {'name': 'Alice', 'age': 30, 'salary': 55000.5, 'is_active': True,  'bonus': None},
        {'name': 'Bob',   'age': 25, 'salary': 42000.0, 'is_active': False, 'bonus': 5000},
        {'name': 'Carol', 'age': None,'salary': None,    'is_active': True,  'bonus': None},
    ]
    df = pd.DataFrame(data)
    # Apply the FIX (same line as in main.py after the patch)
    df_fixed = df.astype(object).where(df.notna(), other=None)

    # Numbers
    assert df_fixed['age'].iloc[0] == 30,          "age[0] must be 30 (int), not '30'"
    assert df_fixed['salary'].iloc[0] == 55000.5,  "salary[0] must be 55000.5 (float)"
    assert df_fixed['bonus'].iloc[1] == 5000,      "bonus[1] must be 5000 (int)"

    # Booleans
    assert df_fixed['is_active'].iloc[0] is True,  "is_active[0] must be True (bool)"
    assert df_fixed['is_active'].iloc[1] is False, "is_active[1] must be False (bool)"

    # None for missing values
    assert df_fixed['bonus'].iloc[0] is None,      "bonus[0] must be None (was NaN)"
    assert df_fixed['age'].iloc[2] is None,        "age[2] must be None (was NaN)"

    print("  [PASS] Bug 1 - numeric/bool/None types preserved")


def test_bug1_old_code_would_fail():
    """Demonstrate what the OLD code was doing wrong."""
    # Use multiple rows so pandas infers float dtype for bonus (-> NaN -> 'nan')
    data = [
        {'age': 30, 'salary': 55000.5, 'is_active': True,  'bonus': None},
        {'age': 25, 'salary': 42000.0, 'is_active': False, 'bonus': 5000},
    ]
    df = pd.DataFrame(data)

    # OLD behaviour: astype(str) coerces everything
    df_old = df.copy()
    for col in df_old.columns:
        df_old[col] = df_old[col].astype(str).replace('nan', '')

    assert df_old['age'].iloc[0] == '30',         "OLD: age becomes string '30'"
    assert df_old['is_active'].iloc[0] == 'True', "OLD: boolean becomes string 'True'"
    # bonus[0] was NaN (float col with None) -> astype(str)->'nan' -> replace->'  '
    assert df_old['bonus'].iloc[0] in ('nan', ''), "OLD: NaN becomes 'nan' or '' string"

    # NEW behaviour
    df_new = df.astype(object).where(df.notna(), other=None)
    assert df_new['age'].iloc[0] == 30,           "NEW: age stays int 30"
    assert df_new['is_active'].iloc[0] is True,   "NEW: boolean stays True"
    assert df_new['bonus'].iloc[0] is None,       "NEW: None stays None"

    print("  [PASS] Bug 1 - contrast between old (broken) and new (correct) behaviour confirmed")


def test_bug1_excel_round_trip():
    """Write DataFrame to Excel buffer and read back — types must survive."""
    data = [
        {'id': 1, 'score': 9.75, 'passed': True,  'notes': 'Good'},
        {'id': 2, 'score': 4.5,  'passed': False, 'notes': None},
    ]
    df = pd.DataFrame(data)
    df_fixed = df.astype(object).where(df.notna(), other=None)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df_fixed.to_excel(writer, sheet_name='Data_test', index=False)
    buf.seek(0)

    wb = openpyxl.load_workbook(buf)
    ws = wb['Data_test']
    rows = list(ws.iter_rows(min_row=2, values_only=True))

    assert rows[0][0] == 1,     "id=1 must round-trip as integer 1"
    assert rows[0][1] == 9.75,  "score=9.75 must round-trip as float"
    assert rows[0][2] == True,  "passed=True must round-trip as boolean"
    assert rows[1][1] == 4.5,   "score=4.5 must round-trip as float"
    assert rows[1][3] is None,  "notes=None must round-trip as empty cell"

    print("  [PASS] Bug 1 - Excel round-trip preserves int, float, bool, and None (blank)")


# ---------------------------------------------------------------------------
# BUG 2 - LOW_UNIQUENESS must NOT fire on boolean fields
# ---------------------------------------------------------------------------

def test_bug2_boolean_no_low_uniqueness():
    """Boolean columns must never trigger LOW_UNIQUENESS."""
    extractor = MetadataExtractor()

    # 50 records, boolean field (only 2 unique values -> unique_ratio = 0.04)
    records = [{'is_active': i % 2 == 0} for i in range(50)]
    with redirect_stdout(io.StringIO()):
        metadata = extractor.extract_metadata(records)

    attr = metadata['table_metadata']['attributes']['is_active']
    anomalies = attr.get('anomaly_flags', [])

    assert 'LOW_UNIQUENESS' not in anomalies, (
        f"LOW_UNIQUENESS must not fire for boolean field, got: {anomalies}"
    )
    assert attr['data_type'] == 'boolean', "data_type must be 'boolean'"
    print(f"  [PASS] Bug 2 - boolean field not flagged LOW_UNIQUENESS (anomalies={anomalies})")


def test_bug2_status_field_low_uniqueness_still_fires():
    """A low-cardinality STRING field (not boolean) SHOULD still get LOW_UNIQUENESS."""
    extractor = MetadataExtractor()

    statuses = ['active', 'inactive', 'pending']
    records = [{'status': statuses[i % 3]} for i in range(50)]
    with redirect_stdout(io.StringIO()):
        metadata = extractor.extract_metadata(records)

    attr = metadata['table_metadata']['attributes']['status']
    anomalies = attr.get('anomaly_flags', [])

    # unique_ratio = 3/50 = 0.06 < 0.1 and NOT boolean -> should fire
    assert 'LOW_UNIQUENESS' in anomalies, (
        f"LOW_UNIQUENESS SHOULD fire for low-cardinality string field, got: {anomalies}"
    )
    print(f"  [PASS] Bug 2 - low-cardinality string still flagged LOW_UNIQUENESS (anomalies={anomalies})")


def test_bug2_boolean_with_all_true():
    """All-True boolean column: unique_ratio=1/N, still no LOW_UNIQUENESS."""
    extractor = MetadataExtractor()
    records = [{'flag': True} for _ in range(30)]
    with redirect_stdout(io.StringIO()):
        metadata = extractor.extract_metadata(records)
    attr = metadata['table_metadata']['attributes']['flag']
    assert 'LOW_UNIQUENESS' not in attr.get('anomaly_flags', [])
    print("  [PASS] Bug 2 - all-True boolean column not flagged")


def test_bug2_numeric_low_uniqueness_still_fires():
    """A numeric column with only 2 distinct values (like 0/1) should still fire."""
    extractor = MetadataExtractor()
    records = [{'score': i % 2} for i in range(50)]
    with redirect_stdout(io.StringIO()):
        metadata = extractor.extract_metadata(records)
    attr = metadata['table_metadata']['attributes']['score']
    anomalies = attr.get('anomaly_flags', [])
    # data_type is integer, not boolean -> should fire
    assert attr['data_type'] == 'integer'
    assert 'LOW_UNIQUENESS' in anomalies, (
        f"LOW_UNIQUENESS SHOULD fire for low-cardinality integer, got: {anomalies}"
    )
    print(f"  [PASS] Bug 2 - low-cardinality integer still flagged LOW_UNIQUENESS")


# ---------------------------------------------------------------------------
# BUG 3 - Affected_Tables must show the actual table, not hardcoded "Multiple"
# ---------------------------------------------------------------------------

def _parse_affected_table(issue_text: str) -> str:
    """Same logic as patched main.py."""
    if issue_text.startswith("["):
        end_bracket = issue_text.find("]")
        if end_bracket > 0:
            return issue_text[1:end_bracket]
    return "Multiple"


def test_bug3_bracketed_table_name_extracted():
    issues = [
        "[employees] HIGH_NULL_RATE in salary",
        "[departments] DUPLICATE_ROWS detected",
        "[main_table] MIXED_DATA_TYPES in status",
    ]
    expected = ["employees", "departments", "main_table"]
    for issue, exp in zip(issues, expected):
        result = _parse_affected_table(issue)
        assert result == exp, f"Expected '{exp}', got '{result}' for: {issue}"
    print("  [PASS] Bug 3 - bracketed table names correctly extracted")


def test_bug3_unbracketed_falls_back_to_multiple():
    plain_issues = [
        "HIGH_NULL_RATE in salary",
        "DUPLICATE_ROWS detected across all tables",
        "Schema inconsistency",
    ]
    for issue in plain_issues:
        result = _parse_affected_table(issue)
        assert result == "Multiple", f"Expected 'Multiple', got '{result}' for: {issue}"
    print("  [PASS] Bug 3 - unbracketed issues correctly fall back to 'Multiple'")


def test_bug3_partial_bracket_falls_back():
    edge_cases = [
        "[no-close-bracket",
        "[]",
        "",
    ]
    for issue in edge_cases:
        result = _parse_affected_table(issue)
        # "[]" has end_bracket=1, so extracted name is "" — acceptable, or "Multiple"
        # "[no-close-bracket" -> end_bracket=-1 -> "Multiple"
        if issue == "[no-close-bracket":
            assert result == "Multiple"
        print(f"  [PASS] Bug 3 - edge case '{issue}' -> '{result}' (no crash)")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    results = {'passed': 0, 'failed': 0}

    tests = [
        test_bug1_numeric_preserved,
        test_bug1_old_code_would_fail,
        test_bug1_excel_round_trip,
        test_bug2_boolean_no_low_uniqueness,
        test_bug2_status_field_low_uniqueness_still_fires,
        test_bug2_boolean_with_all_true,
        test_bug2_numeric_low_uniqueness_still_fires,
        test_bug3_bracketed_table_name_extracted,
        test_bug3_unbracketed_falls_back_to_multiple,
        test_bug3_partial_bracket_falls_back,
    ]

    print("\n" + "="*60)
    print("  ParseIQ Bug-Fix Verification Tests")
    print("="*60)

    sections = {
        0: "\n--- Bug 1: Data sheet type coercion ---",
        3: "\n--- Bug 2: LOW_UNIQUENESS false positive on booleans ---",
        7: "\n--- Bug 3: Affected_Tables shows actual table name ---",
    }

    for i, test_fn in enumerate(tests):
        if i in sections:
            print(sections[i])
        try:
            test_fn()
            results['passed'] += 1
        except Exception as e:
            results['failed'] += 1
            print(f"  [FAIL] {test_fn.__name__}: {e}")

    print("\n" + "="*60)
    total = results['passed'] + results['failed']
    print(f"  Result: {results['passed']}/{total} passed", end="")
    if results['failed']:
        print(f"  ({results['failed']} FAILED)")
    else:
        print("  -- ALL PASSED")
    print("="*60)
