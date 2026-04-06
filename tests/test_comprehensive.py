"""
Comprehensive test suite — covers all gaps identified in coverage analysis.

Covers:
  - FileLoader: CSV, XML, Excel, file size limit, get_file_info, _detect_encoding,
                all _flatten_nested_json branches (FK injection, collision, primitive
                arrays, embedded objects, null handling, path-qualified names)
  - MetadataExtractor: every anomaly flag, quality scoring, data profiling,
                       schema analysis, export_metadata_report, summary report,
                       _determine_data_type all branches, boolean/array/object analysis
  - main.py: _safe_sheet, convert_enriched_json_to_csv, _generate_summary
  - Config: create_prompt_template_path
"""

import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# FileLoader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFileLoaderCSV(unittest.TestCase):
    def setUp(self):
        from file_loader.loader import FileLoader
        self.loader = FileLoader()

    def _write_temp(self, content: str, suffix: str) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='w', encoding='utf-8')
        tmp.write(content)
        tmp.close()
        return tmp.name

    # ── CSV ──────────────────────────────────────────────────────────────────

    def test_load_csv_basic(self):
        path = self._write_temp("a,b,c\n1,2,3\n4,5,6\n", ".csv")
        try:
            result = self.loader.load_file(path)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]['a'], 1)
        finally:
            os.unlink(path)

    def test_load_csv_null_values_become_none(self):
        """NaN values from empty CSV cells must be converted to None (Bug 6 fix)."""
        path = self._write_temp("a,b\n1,\n2,3\n", ".csv")
        try:
            result = self.loader.load_file(path)
            self.assertIsNone(result[0]['b'])
            self.assertEqual(result[1]['b'], 3)
        finally:
            os.unlink(path)

    def test_load_csv_semicolon_delimiter(self):
        path = self._write_temp("a;b;c\n1;2;3\n4;5;6\n", ".csv")
        try:
            result = self.loader.load_file(path)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
        finally:
            os.unlink(path)

    # ── XML ──────────────────────────────────────────────────────────────────

    def test_load_xml_basic(self):
        xml = """<?xml version="1.0"?>
<root>
  <record><name>Alice</name><age>30</age></record>
  <record><name>Bob</name><age>25</age></record>
</root>"""
        path = self._write_temp(xml, ".xml")
        try:
            result = self.loader.load_file(path)
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
        finally:
            os.unlink(path)

    def test_load_xml_invalid_raises(self):
        path = self._write_temp("not xml at all <<<", ".xml")
        try:
            with self.assertRaises(ValueError):
                self.loader.load_file(path)
        finally:
            os.unlink(path)

    # ── Excel ─────────────────────────────────────────────────────────────────

    def test_load_excel_basic(self):
        path = tempfile.mktemp(suffix=".xlsx")
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        df.to_excel(path, index=False)
        try:
            result = self.loader.load_file(path)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 3)
            self.assertIn('col1', result[0])
        finally:
            os.unlink(path)

    # ── File size limit ───────────────────────────────────────────────────────

    def test_load_file_too_large_raises(self):
        with patch('parseiq.file_loader.loader.Path.exists', return_value=True):
            mock_stat = MagicMock()
            mock_stat.st_size = 200 * 1024 * 1024  # 200 MB
            with patch('parseiq.file_loader.loader.Path.stat', return_value=mock_stat):
                with self.assertRaises(ValueError) as ctx:
                    self.loader.load_file('big.json')
                self.assertIn('too large', str(ctx.exception).lower())

    # ── get_file_info ─────────────────────────────────────────────────────────

    def test_get_file_info_returns_correct_fields(self):
        path = self._write_temp('{"a": 1}', ".json")
        try:
            info = self.loader.get_file_info(path)
            self.assertIn('filename', info)
            self.assertIn('extension', info)
            self.assertIn('size_bytes', info)
            self.assertIn('size_mb', info)
            self.assertIn('is_supported', info)
            self.assertTrue(info['is_supported'])
            self.assertEqual(info['extension'], '.json')
        finally:
            os.unlink(path)

    def test_get_file_info_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.loader.get_file_info('no_such_file.json')

    # ── _detect_encoding ──────────────────────────────────────────────────────

    def test_detect_encoding_utf8(self):
        path = self._write_temp('hello world', ".txt")
        try:
            enc = self.loader._detect_encoding(Path(path))
            self.assertIsInstance(enc, str)
            self.assertGreater(len(enc), 0)
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# FileLoader — flattener edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestFlattenNestedJSON(unittest.TestCase):
    def setUp(self):
        from file_loader.loader import FileLoader
        self.loader = FileLoader()

    def flatten(self, data):
        return self.loader._flatten_nested_json(data)

    def test_flat_dict_becomes_main_table(self):
        result = self.flatten({"name": "Alice", "age": 30})
        self.assertIn("main_table", result)
        self.assertEqual(result["main_table"][0]["name"], "Alice")

    def test_list_of_dicts_becomes_main_table(self):
        result = self.flatten([{"a": 1}, {"a": 2}])
        self.assertIn("main_table", result)
        self.assertEqual(len(result["main_table"]), 2)

    def test_nested_array_of_dicts_becomes_child_table(self):
        data = {
            "employees": [
                {"id": 1, "name": "Alice", "projects": [{"proj_id": 10, "title": "X"}]}
            ]
        }
        result = self.flatten(data)
        self.assertIn("employees", result)
        self.assertIn("projects", result)

    def test_fk_injected_into_child_table(self):
        data = {
            "orders": [
                {"order_id": 99, "items": [{"item_id": 1, "qty": 2}]}
            ]
        }
        result = self.flatten(data)
        self.assertIn("items", result)
        child_rec = result["items"][0]
        self.assertIn("_ref_orders_id", child_rec)
        self.assertEqual(child_rec["_ref_orders_id"], 99)

    def test_embedded_shallow_dict_flattened_inline(self):
        data = {
            "users": [
                {"id": 1, "address": {"city": "NYC", "zip": "10001"}}
            ]
        }
        result = self.flatten(data)
        row = result["users"][0]
        self.assertIn("address__city", row)
        self.assertIn("address__zip", row)
        self.assertNotIn("address", row)

    def test_primitive_array_joined_as_string(self):
        data = {
            "products": [
                {"id": 1, "tags": ["python", "data", "ai"]}
            ]
        }
        result = self.flatten(data)
        row = result["products"][0]
        self.assertIn("tags", row)
        self.assertIn("python", row["tags"])
        self.assertIn("ai", row["tags"])

    def test_null_scalar_passes_through(self):
        data = {
            "users": [
                {"id": 1, "email": None}
            ]
        }
        result = self.flatten(data)
        self.assertIsNone(result["users"][0]["email"])

    def test_null_dict_field_skips_artifact_column(self):
        """If a field is dict in most records but null in one, no artifact column."""
        data = {
            "users": [
                {"id": 1, "address": {"city": "NYC"}},
                {"id": 2, "address": None},
            ]
        }
        result = self.flatten(data)
        for row in result["users"]:
            self.assertNotIn("address", row)  # no raw "address" artifact

    def test_same_name_tables_from_siblings_merged(self):
        """departments from division[0] and division[1] should merge into one table."""
        data = {
            "divisions": [
                {"div_id": 1, "departments": [{"dept_id": 10, "name": "Eng"}]},
                {"div_id": 2, "departments": [{"dept_id": 20, "name": "Sales"}]},
            ]
        }
        result = self.flatten(data)
        self.assertIn("departments", result)
        self.assertEqual(len(result["departments"]), 2)

    def test_empty_list_skipped(self):
        data = {
            "users": [{"id": 1, "tags": []}]
        }
        result = self.flatten(data)
        row = result["users"][0]
        self.assertIn("tags", row)
        self.assertEqual(row["tags"], "")

    def test_path_sentinel_keys_removed_from_output(self):
        data = {"items": [{"id": 1, "sub": [{"val": 2}]}]}
        result = self.flatten(data)
        for key in result:
            self.assertFalse(key.startswith("__path__"),
                             f"Sentinel key leaked: {key}")

    def test_get_record_id_exact_match(self):
        record = {"employees_id": 5, "name": "Bob"}
        result = self.loader._get_record_id(record, "employees")
        self.assertEqual(result, 5)

    def test_get_record_id_singular_form(self):
        record = {"employee_id": 7, "name": "Carol"}
        result = self.loader._get_record_id(record, "employees")
        self.assertEqual(result, 7)

    def test_get_record_id_generic_id(self):
        record = {"id": 42, "name": "Dave"}
        result = self.loader._get_record_id(record, "items")
        self.assertEqual(result, 42)

    def test_get_record_id_no_id_returns_none(self):
        record = {"name": "no id here"}
        result = self.loader._get_record_id(record, "things")
        self.assertIsNone(result)

    def test_get_record_id_ignores_ref_columns(self):
        """_ref_* FK columns should never be used as own ID."""
        record = {"_ref_parent_id": 99, "name": "orphan"}
        result = self.loader._get_record_id(record, "children")
        self.assertIsNone(result)


# ─────────────────────────────────────────────────────────────────────────────
# MetadataExtractor — _determine_data_type
# ─────────────────────────────────────────────────────────────────────────────

class TestDetermineDataType(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def _dtype(self, values):
        return self.extractor._determine_data_type(values)

    def test_empty_returns_unknown(self):
        self.assertEqual(self._dtype([]), "unknown")

    def test_integers(self):
        self.assertEqual(self._dtype([1, 2, 3]), "integer")

    def test_floats(self):
        self.assertEqual(self._dtype([1.1, 2.2]), "float")

    def test_booleans(self):
        self.assertEqual(self._dtype([True, False, True]), "boolean")

    def test_strings(self):
        self.assertEqual(self._dtype(["hello", "world"]), "string")

    def test_string_integer(self):
        self.assertEqual(self._dtype(["1", "2", "3"]), "integer")

    def test_string_float(self):
        self.assertEqual(self._dtype(["1.5", "2.5"]), "float")

    def test_lists(self):
        self.assertEqual(self._dtype([[1, 2], [3, 4]]), "array")

    def test_dicts(self):
        self.assertEqual(self._dtype([{"a": 1}, {"b": 2}]), "object")


# ─────────────────────────────────────────────────────────────────────────────
# MetadataExtractor — _detect_anomalies (all flags)
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectAnomalies(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def test_high_null_rate_flagged(self):
        attr = {"null_percentage": 45, "data_type": "string"}
        flags = self.extractor._detect_anomalies(attr, ["a"])
        self.assertIn("HIGH_NULL_RATE", flags)

    def test_no_flag_when_null_rate_low(self):
        attr = {"null_percentage": 5, "data_type": "string", "avg_length": 5}
        flags = self.extractor._detect_anomalies(attr, ["hello"])
        self.assertNotIn("HIGH_NULL_RATE", flags)

    def test_avg_length_too_short(self):
        attr = {"null_percentage": 0, "data_type": "string", "avg_length": 1}
        flags = self.extractor._detect_anomalies(attr, ["a", "b"])
        self.assertIn("AVG_LENGTH_TOO_SHORT", flags)

    def test_avg_length_too_long(self):
        attr = {"null_percentage": 0, "data_type": "string", "avg_length": 600}
        flags = self.extractor._detect_anomalies(attr, ["x" * 600])
        self.assertIn("AVG_LENGTH_TOO_LONG", flags)

    def test_negative_values_flagged(self):
        attr = {"null_percentage": 0, "data_type": "integer",
                "min_value": -5, "outliers": {"count": 0}}
        flags = self.extractor._detect_anomalies(attr, [-5, 1, 2])
        self.assertIn("NEGATIVE_VALUES_DETECTED", flags)

    def test_numeric_outliers_flagged(self):
        attr = {"null_percentage": 0, "data_type": "integer",
                "min_value": 1, "outliers": {"count": 3}}
        flags = self.extractor._detect_anomalies(attr, [1, 2, 3])
        self.assertIn("NUMERIC_OUTLIERS_DETECTED", flags)

    def test_low_uniqueness_flagged(self):
        # unique_ratio < 0.1 and len(values) > 10
        attr = {"null_percentage": 0, "data_type": "string",
                "avg_length": 5, "unique_ratio": 0.05}
        flags = self.extractor._detect_anomalies(attr, ["x"] * 15)
        self.assertIn("LOW_UNIQUENESS", flags)

    def test_low_uniqueness_not_flagged_small_set(self):
        # len(values) <= 10 → no flag
        attr = {"null_percentage": 0, "data_type": "string",
                "avg_length": 5, "unique_ratio": 0.05}
        flags = self.extractor._detect_anomalies(attr, ["x"] * 5)
        self.assertNotIn("LOW_UNIQUENESS", flags)

    def test_pattern_inconsistency_flagged(self):
        attr = {
            "null_percentage": 0, "data_type": "string", "avg_length": 10,
            "unique_ratio": 0.9,
            "pattern_analysis": {
                "regex_patterns": {
                    "email": {"percentage": 65},
                    "phone": {"percentage": 0},
                }
            }
        }
        flags = self.extractor._detect_anomalies(attr, ["a@b.com", "notanemail"])
        self.assertIn("PATTERN_INCONSISTENCY", flags)

    def test_future_date_flagged(self):
        future = (datetime.today() + timedelta(days=30)).strftime('%Y-%m-%d')
        attr = {"null_percentage": 0, "data_type": "string", "avg_length": 10,
                "unique_ratio": 1.0}
        flags = self.extractor._detect_anomalies(attr, [future])
        self.assertIn("FUTURE_DATE_DETECTED", flags)

    def test_past_date_not_flagged(self):
        past = "2020-01-01"
        attr = {"null_percentage": 0, "data_type": "string", "avg_length": 10,
                "unique_ratio": 1.0}
        flags = self.extractor._detect_anomalies(attr, [past])
        self.assertNotIn("FUTURE_DATE_DETECTED", flags)

    def test_mixed_data_types_flagged(self):
        attr = {"null_percentage": 0, "data_type": "string", "avg_length": 5,
                "unique_ratio": 1.0}
        flags = self.extractor._detect_anomalies(attr, [1, "hello", 2.5])
        self.assertIn("MIXED_DATA_TYPES", flags)

    def test_mixed_types_not_flagged_single_value(self):
        attr = {"null_percentage": 0, "data_type": "integer", "unique_ratio": 1.0}
        flags = self.extractor._detect_anomalies(attr, [42])
        self.assertNotIn("MIXED_DATA_TYPES", flags)


# ─────────────────────────────────────────────────────────────────────────────
# MetadataExtractor — quality scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityScoring(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def test_attribute_score_100_for_perfect_data(self):
        attr = {"null_percentage": 0, "anomaly_flags": []}
        score = self.extractor._calculate_attribute_quality_score(attr)
        self.assertEqual(score, 100.0)

    def test_attribute_score_penalizes_nulls(self):
        attr = {"null_percentage": 40, "anomaly_flags": []}
        score = self.extractor._calculate_attribute_quality_score(attr)
        self.assertLess(score, 100.0)
        self.assertGreaterEqual(score, 0.0)

    def test_attribute_score_penalizes_anomalies(self):
        attr = {"null_percentage": 0, "anomaly_flags": ["HIGH_NULL_RATE", "LOW_UNIQUENESS"]}
        score = self.extractor._calculate_attribute_quality_score(attr)
        self.assertLess(score, 100.0 - 15)  # at least 2 × 15 deduction

    def test_attribute_score_never_below_zero(self):
        attr = {"null_percentage": 100, "anomaly_flags": ["A", "B", "C", "D", "E", "F", "G"]}
        score = self.extractor._calculate_attribute_quality_score(attr)
        self.assertGreaterEqual(score, 0.0)

    def test_table_quality_score_empty_attrs_returns_zero(self):
        score = self.extractor._calculate_quality_score({})
        self.assertEqual(score, 0.0)

    def test_table_quality_score_averages_attributes(self):
        attrs = {
            "col1": {"quality_score": 80, "anomaly_flags": []},
            "col2": {"quality_score": 60, "anomaly_flags": []},
        }
        score = self.extractor._calculate_quality_score(attrs)
        # base = 70, no anomaly penalty
        self.assertAlmostEqual(score, 70.0, places=1)

    def test_table_quality_score_penalizes_anomalies(self):
        attrs = {
            "col1": {"quality_score": 90, "anomaly_flags": ["HIGH_NULL_RATE"]},
        }
        score = self.extractor._calculate_quality_score(attrs)
        # 90 - 3*1 = 87
        self.assertAlmostEqual(score, 87.0, places=1)


# ─────────────────────────────────────────────────────────────────────────────
# MetadataExtractor — _generate_anomaly_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateAnomalySummary(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def test_no_anomalies(self):
        attrs = {"col1": {"anomaly_flags": []}}
        summary = self.extractor._generate_anomaly_summary(attrs)
        self.assertEqual(summary["total_anomalies"], 0)
        self.assertEqual(summary["affected_attributes"], 0)

    def test_counts_anomalies_correctly(self):
        attrs = {
            "col1": {"anomaly_flags": ["HIGH_NULL_RATE", "LOW_UNIQUENESS"]},
            "col2": {"anomaly_flags": ["NEGATIVE_VALUES_DETECTED"]},
            "col3": {"anomaly_flags": []},
        }
        summary = self.extractor._generate_anomaly_summary(attrs)
        self.assertEqual(summary["total_anomalies"], 3)
        self.assertEqual(summary["affected_attributes"], 2)
        self.assertIn("HIGH_NULL_RATE", summary["anomaly_types"])
        self.assertEqual(summary["anomaly_types"]["HIGH_NULL_RATE"], 1)

    def test_anomaly_rate_calculated(self):
        attrs = {
            "col1": {"anomaly_flags": ["X"]},
            "col2": {"anomaly_flags": []},
        }
        summary = self.extractor._generate_anomaly_summary(attrs)
        self.assertEqual(summary["anomaly_rate"], 50.0)


# ─────────────────────────────────────────────────────────────────────────────
# MetadataExtractor — _identify_top_issues
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentifyTopIssues(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def test_high_null_rate_issue_reported(self):
        attrs = {
            "email": {
                "null_percentage": 60,
                "quality_score": 40,
                "anomaly_flags": ["HIGH_NULL_RATE"],
                "outliers": {}
            }
        }
        issues = self.extractor._identify_top_issues(attrs)
        self.assertTrue(any("email" in i for i in issues))

    def test_future_date_issue_reported(self):
        attrs = {
            "expiry": {
                "null_percentage": 0,
                "quality_score": 85,
                "anomaly_flags": ["FUTURE_DATE_DETECTED"],
                "outliers": {}
            }
        }
        issues = self.extractor._identify_top_issues(attrs)
        self.assertTrue(any("future date" in i.lower() for i in issues))

    def test_mixed_types_issue_reported(self):
        attrs = {
            "amount": {
                "null_percentage": 0,
                "quality_score": 85,
                "anomaly_flags": ["MIXED_DATA_TYPES"],
                "outliers": {}
            }
        }
        issues = self.extractor._identify_top_issues(attrs)
        self.assertTrue(any("mixed" in i.lower() for i in issues))

    def test_max_10_issues_returned(self):
        attrs = {
            f"col{i}": {
                "null_percentage": 55,
                "quality_score": 20,
                "anomaly_flags": ["HIGH_NULL_RATE"],
                "outliers": {}
            }
            for i in range(20)
        }
        issues = self.extractor._identify_top_issues(attrs)
        self.assertLessEqual(len(issues), 10)


# ─────────────────────────────────────────────────────────────────────────────
# MetadataExtractor — data profiling helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestDataProfilingHelpers(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def test_analyze_duplicates_none(self):
        records = [{"a": 1}, {"a": 2}, {"a": 3}]
        result = self.extractor._analyze_duplicates(records)
        self.assertEqual(result["total_duplicates"], 0)
        self.assertEqual(result["unique_records"], 3)

    def test_analyze_duplicates_detected(self):
        records = [{"a": 1}, {"a": 1}, {"a": 2}]
        result = self.extractor._analyze_duplicates(records)
        self.assertEqual(result["total_duplicates"], 1)
        self.assertEqual(result["unique_records"], 2)

    def test_analyze_duplicates_empty(self):
        result = self.extractor._analyze_duplicates([])
        self.assertEqual(result, {})

    def test_record_completeness_full(self):
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = self.extractor._analyze_record_completeness(records)
        self.assertEqual(result["avg_completeness"], 100.0)
        self.assertEqual(result["records_100_complete"], 2)

    def test_record_completeness_with_nulls(self):
        records = [{"a": 1, "b": None}, {"a": 2, "b": 3}]
        result = self.extractor._analyze_record_completeness(records)
        self.assertLess(result["avg_completeness"], 100.0)
        self.assertEqual(result["records_100_complete"], 1)

    def test_record_completeness_empty(self):
        result = self.extractor._analyze_record_completeness([])
        self.assertEqual(result, {})

    def test_analyze_schema_consistency_perfect(self):
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = self.extractor._analyze_schema(records)
        self.assertEqual(result["schema_consistency"], 100.0)
        self.assertEqual(result["unique_schemas"], 1)
        self.assertEqual(len(result["required_attributes"]), 2)

    def test_analyze_schema_optional_attributes(self):
        records = [{"a": 1, "b": 2}, {"a": 3}]
        result = self.extractor._analyze_schema(records)
        self.assertIn("b", result["optional_attributes"])

    def test_analyze_schema_empty_returns_empty(self):
        result = self.extractor._analyze_schema([])
        self.assertEqual(result, {})


# ─────────────────────────────────────────────────────────────────────────────
# MetadataExtractor — type-specific analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestTypeSpecificAnalysis(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def test_analyze_boolean_attribute(self):
        result = self.extractor._analyze_boolean_attribute([True, False, True, True])
        self.assertEqual(result["true_count"], 3)
        self.assertEqual(result["false_count"], 1)
        self.assertAlmostEqual(result["true_percentage"], 75.0)

    def test_analyze_boolean_empty(self):
        result = self.extractor._analyze_boolean_attribute([])
        self.assertEqual(result, {})

    def test_analyze_array_attribute(self):
        result = self.extractor._analyze_array_attribute([[1, 2], [3], []])
        self.assertEqual(result["min_length"], 0)
        self.assertEqual(result["max_length"], 2)
        self.assertEqual(result["empty_arrays"], 1)

    def test_analyze_array_empty(self):
        result = self.extractor._analyze_array_attribute([])
        self.assertEqual(result, {})

    def test_analyze_object_attribute(self):
        result = self.extractor._analyze_object_attribute([
            {"x": 1, "y": 2},
            {"x": 3, "z": 4}
        ])
        self.assertIn("unique_keys", result)
        self.assertEqual(result["unique_keys"], 3)  # x, y, z

    def test_analyze_object_empty(self):
        result = self.extractor._analyze_object_attribute([])
        self.assertEqual(result, {})

    def test_analyze_charset(self):
        result = self.extractor._analyze_charset(["Hello World 123!"])
        self.assertIn("alphabetic", result)
        self.assertIn("numeric", result)
        self.assertIn("whitespace", result)
        self.assertIn("special_chars", result)
        total = sum(result.values())
        self.assertAlmostEqual(total, 100.0, places=0)

    def test_analyze_charset_empty_strings(self):
        result = self.extractor._analyze_charset([""])
        # All zeros → return raw dict (no division by zero)
        self.assertIsInstance(result, dict)

    def test_analyze_string_attribute_basic(self):
        result = self.extractor._analyze_string_attribute(["hello", "world", "python"])
        self.assertEqual(result["min_length"], 5)
        self.assertEqual(result["max_length"], 6)
        self.assertIn("common_values", result)

    def test_analyze_numeric_attribute_basic(self):
        result = self.extractor._analyze_numeric_attribute([1, 2, 3, 4, 5])
        self.assertEqual(result["min_value"], 1.0)
        self.assertEqual(result["max_value"], 5.0)
        self.assertAlmostEqual(result["mean"], 3.0)
        self.assertIn("outliers", result)

    def test_analyze_numeric_attribute_empty(self):
        result = self.extractor._analyze_numeric_attribute([])
        self.assertEqual(result, {})


# ─────────────────────────────────────────────────────────────────────────────
# MetadataExtractor — _analyze_patterns
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzePatterns(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def test_email_pattern_detected(self):
        values = ["alice@example.com", "bob@domain.org", "carol@test.net"]
        result = self.extractor._analyze_patterns(values)
        email_pct = result["regex_patterns"]["email"]["percentage"]
        self.assertEqual(email_pct, 100.0)

    def test_no_pattern_match(self):
        values = ["hello", "world", "foo"]
        result = self.extractor._analyze_patterns(values)
        self.assertIn("dominant_pattern", result)

    def test_empty_values_returns_empty(self):
        result = self.extractor._analyze_patterns([])
        self.assertEqual(result, {})


# ─────────────────────────────────────────────────────────────────────────────
# MetadataExtractor — export_metadata_report & _generate_summary_report
# ─────────────────────────────────────────────────────────────────────────────

class TestExportMetadataReport(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def test_export_json_format(self):
        metadata = {"dataset_type": "single_table", "table_metadata": {}}
        output = self.extractor.export_metadata_report(metadata, "json")
        parsed = json.loads(output)
        self.assertEqual(parsed["dataset_type"], "single_table")

    def test_export_summary_format_single_table(self):
        metadata = {
            "extraction_timestamp": "2026-01-01T00:00:00",
            "dataset_type": "single_table",
            "table_metadata": {
                "data_quality_score": 85.0,
                "dataset_info": {"total_records": 100, "total_attributes": 5,
                                 "memory_usage_bytes": 1024},
                "top_issues": ["'email' has high null rate (35.0%)"],
                "anomaly_summary": {"total_anomalies": 1}
            },
            "summary": {"total_records": 100}
        }
        output = self.extractor.export_metadata_report(metadata, "summary")
        self.assertIn("DATA QUALITY METADATA REPORT", output)
        self.assertIn("85", output)

    def test_export_summary_format_multi_table(self):
        metadata = {
            "extraction_timestamp": "2026-01-01T00:00:00",
            "dataset_type": "multi_table",
            "overall_quality_score": 77.5,
            "summary": {
                "total_tables": 2,
                "total_records": 200,
                "table_names": ["users", "orders"],
                "table_record_counts": {"users": 100, "orders": 100}
            },
            "tables": {
                "users": {"data_quality_score": 90.0, "top_issues": []},
                "orders": {"data_quality_score": 65.0, "top_issues": []}
            },
            "global_anomaly_summary": {
                "total_anomalies": 3,
                "affected_tables": 1,
                "critical_issues": ["orders: 'amount' has 2 numeric outliers"]
            }
        }
        output = self.extractor.export_metadata_report(metadata, "summary")
        self.assertIn("DATASET OVERVIEW", output)
        self.assertIn("users", output)
        self.assertIn("orders", output)

    def test_export_unsupported_format_raises(self):
        with self.assertRaises(ValueError):
            self.extractor.export_metadata_report({}, "pdf")


# ─────────────────────────────────────────────────────────────────────────────
# MetadataExtractor — duplicate-row anomaly injected at table level
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicateRowAnomalyTableLevel(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def test_duplicate_rows_deduct_quality_and_flag_anomaly(self):
        records = [
            {"id": 1, "name": "Alice"},
            {"id": 1, "name": "Alice"},   # duplicate
            {"id": 2, "name": "Bob"},
        ]
        result = self.extractor._analyze_table_detailed(records, "test_table")
        self.assertIn("DUPLICATE_ROWS_DETECTED", result["anomaly_summary"]["anomaly_types"])
        self.assertLess(result["data_quality_score"], 100.0)
        self.assertTrue(any("duplicate" in issue.lower() for issue in result["top_issues"]))

    def test_no_duplicate_rows_no_flag(self):
        records = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        result = self.extractor._analyze_table_detailed(records, "clean_table")
        self.assertNotIn("DUPLICATE_ROWS_DETECTED", result["anomaly_summary"]["anomaly_types"])


# ─────────────────────────────────────────────────────────────────────────────
# MetadataExtractor — FK columns excluded from analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestFKColumnsExcluded(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def test_ref_columns_not_in_attributes(self):
        records = [
            {"_ref_orders_id": 99, "item_id": 1, "qty": 2},
            {"_ref_orders_id": 99, "item_id": 2, "qty": 5},
        ]
        result = self.extractor._analyze_table_detailed(records, "items")
        self.assertNotIn("_ref_orders_id", result["attributes"])
        self.assertIn("item_id", result["attributes"])
        self.assertIn("qty", result["attributes"])


# ─────────────────────────────────────────────────────────────────────────────
# MetadataExtractor — _analyze_cross_table_relationships (second definition)
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateAnomalySummary(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def test_aggregate_anomaly_summary_counts(self):
        tables = {
            "users": {
                "anomaly_summary": {"total_anomalies": 2,
                                    "anomaly_types": {"HIGH_NULL_RATE": 1, "MIXED_DATA_TYPES": 1}},
                "top_issues": ["'email' has high null rate"]
            },
            "orders": {
                "anomaly_summary": {"total_anomalies": 0, "anomaly_types": {}},
                "top_issues": []
            }
        }
        result = self.extractor._aggregate_anomaly_summary(tables)
        self.assertEqual(result["total_anomalies"], 2)
        self.assertEqual(result["affected_tables"], 1)
        self.assertIn("HIGH_NULL_RATE", result["anomaly_types"])
        self.assertGreater(len(result["critical_issues"]), 0)


# ─────────────────────────────────────────────────────────────────────────────
# MetadataExtractor — full end-to-end single table
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractMetadataEndToEnd(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def test_single_table_list_input(self):
        data = [
            {"id": 1, "name": "Alice", "email": "alice@example.com", "score": 95},
            {"id": 2, "name": "Bob",   "email": None,                "score": 80},
            {"id": 3, "name": "Carol", "email": "carol@test.com",    "score": 105},
        ]
        result = self.extractor.extract_metadata(data)
        self.assertEqual(result["dataset_type"], "single_table")
        self.assertIn("table_metadata", result)
        self.assertEqual(result["summary"]["total_records"], 3)
        self.assertIn("id", result["table_metadata"]["attributes"])
        self.assertIn("email", result["table_metadata"]["attributes"])

    def test_multi_table_dict_input(self):
        data = {
            "users":  [{"user_id": 1, "name": "Alice"}, {"user_id": 2, "name": "Bob"}],
            "orders": [{"order_id": 10, "user_id": 1, "total": 99.99}],
        }
        result = self.extractor.extract_metadata(data)
        self.assertEqual(result["dataset_type"], "multi_table")
        self.assertIn("users", result["tables"])
        self.assertIn("orders", result["tables"])
        self.assertEqual(result["summary"]["total_tables"], 2)
        self.assertEqual(result["summary"]["total_records"], 3)

    def test_single_dict_input(self):
        data = {"company": "Acme", "founded": 1985}
        result = self.extractor.extract_metadata(data)
        self.assertEqual(result["dataset_type"], "single_table")

    def test_unsupported_type_raises(self):
        with self.assertRaises((ValueError, AttributeError)):
            self.extractor.extract_metadata("just a string")


# ─────────────────────────────────────────────────────────────────────────────
# main.py — _safe_sheet
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeSheet(unittest.TestCase):
    """Test the _safe_sheet inner function via convert_enriched_json_to_csv output."""

    def test_safe_sheet_truncates_long_name(self):
        """Sheet name (prefix + table) must never exceed 31 chars."""
        from main import convert_enriched_json_to_csv
        import tempfile, json, os

        # Build minimal JSON files to drive the function
        input_data = {"a_very_long_table_name_that_exceeds_limits": [{"col": 1}]}
        raw_meta = {
            "dataset_type": "multi_table",
            "tables": {
                "a_very_long_table_name_that_exceeds_limits": {
                    "data_quality_score": 90,
                    "attributes": {"col": {"quality_score": 90, "anomaly_flags": [],
                                           "null_percentage": 0, "data_type": "integer"}},
                    "anomaly_summary": {"total_anomalies": 0, "anomaly_types": {}},
                    "top_issues": [],
                    "dataset_info": {"total_records": 1, "total_attributes": 1},
                    "data_profiling": {"duplicate_analysis": {}, "record_completeness": {}}
                }
            },
            "summary": {"total_tables": 1, "total_records": 1,
                        "table_names": ["a_very_long_table_name_that_exceeds_limits"],
                        "table_record_counts": {"a_very_long_table_name_that_exceeds_limits": 1}},
            "overall_quality_score": 90,
            "global_anomaly_summary": {"total_anomalies": 0, "affected_tables": 0,
                                       "anomaly_types": {}, "critical_issues": []}
        }
        enriched_meta = {"overall_assessment": {"quality_grade": "A", "overall_score": 90},
                         "table_assessments": {}}

        tmpdir = tempfile.mkdtemp()
        in_path  = os.path.join(tmpdir, "input.json")
        raw_path = os.path.join(tmpdir, "raw.json")
        enr_path = os.path.join(tmpdir, "enriched.json")
        out_dir  = os.path.join(tmpdir, "output")

        with open(in_path, "w")  as f: json.dump(input_data, f)
        with open(raw_path, "w") as f: json.dump(raw_meta, f)
        with open(enr_path, "w") as f: json.dump(enriched_meta, f)

        try:
            convert_enriched_json_to_csv(
                input_data_path=in_path,
                raw_path=raw_path,
                enriched_path=enr_path,
                output_dir=out_dir,
                create_excel=True,
                csv_per_table=False
            )
            # Verify Excel file was created
            excel_files = [f for f in os.listdir(out_dir) if f.endswith('.xlsx')]
            self.assertGreater(len(excel_files), 0, "No Excel file created")

            # Verify all sheet names are ≤ 31 chars
            import openpyxl
            wb = openpyxl.load_workbook(os.path.join(out_dir, excel_files[0]))
            for sheet_name in wb.sheetnames:
                self.assertLessEqual(len(sheet_name), 31,
                    f"Sheet name too long: '{sheet_name}' ({len(sheet_name)} chars)")
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# main.py — _create_fallback_enrichment
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateFallbackEnrichment(unittest.TestCase):
    @patch('main.Config.get_api_key', return_value='dummy')
    def setUp(self, _):
        from main import MetadataEnrichmentAgent
        self.agent = MetadataEnrichmentAgent(debug=False)

    def test_fallback_has_overall_assessment(self):
        fb = self.agent._create_fallback_enrichment({"summary": {"total_records": 5}})
        self.assertIn("overall_assessment", fb)

    def test_fallback_has_quality_grade(self):
        fb = self.agent._create_fallback_enrichment({"summary": {}})
        self.assertIn("quality_grade", fb["overall_assessment"])

    def test_fallback_does_not_raise_on_empty_input(self):
        try:
            self.agent._create_fallback_enrichment({})
        except Exception as e:
            self.fail(f"_create_fallback_enrichment raised on empty input: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Config — create_prompt_template_path
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigPromptTemplatePath(unittest.TestCase):
    def test_returns_expected_path(self):
        from config import Config
        path = Config.create_prompt_template_path()
        self.assertIn("prompt_template.txt", path)
        self.assertIn("step2_llm_enricher", path)


# ─────────────────────────────────────────────────────────────────────────────
# StatisticalUtils — edge cases not covered
# ─────────────────────────────────────────────────────────────────────────────

class TestStatisticalUtilsEdgeCases(unittest.TestCase):
    def setUp(self):
        from step1_metadata_extractor.utils import StatisticalUtils
        self.utils = StatisticalUtils()

    def test_calculate_percentiles_custom(self):
        result = self.utils.calculate_percentiles([10, 20, 30, 40, 50], [50])
        self.assertAlmostEqual(result["50th"], 30.0)

    def test_analyze_string_lengths_single_char(self):
        result = self.utils.analyze_string_lengths(["a", "bb", "ccc"])
        self.assertEqual(result["single_char_strings"], 1)

    def test_detect_distribution_type_insufficient(self):
        result = self.utils.detect_distribution_type([1, 2])
        self.assertEqual(result["likely_distribution"], "insufficient_data")

    def test_calculate_correlation_matrix_single_col(self):
        result = self.utils.calculate_correlation_matrix({"col1": [1, 2, 3]})
        self.assertIn("message", result)

    def test_detect_time_series_patterns_insufficient(self):
        result = self.utils.detect_time_series_patterns([1, 2, 3])
        self.assertIn("message", result)


# ─────────────────────────────────────────────────────────────────────────────
# Integration — flattener + extractor pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestFlattenThenExtract(unittest.TestCase):
    """End-to-end: flatten a nested JSON then run metadata extraction on it."""

    def setUp(self):
        from file_loader.loader import FileLoader
        from step1_metadata_extractor.extractor import MetadataExtractor
        self.loader = FileLoader()
        self.extractor = MetadataExtractor()

    def test_nested_json_fully_analyzed(self):
        raw = {
            "company": [
                {
                    "company_id": 1,
                    "name": "Acme",
                    "departments": [
                        {"dept_id": 10, "dept_name": "Engineering",
                         "employees": [
                             {"emp_id": 100, "name": "Alice", "salary": 90000},
                             {"emp_id": 101, "name": "Bob",   "salary": None},
                         ]},
                    ]
                }
            ]
        }
        tables = self.loader._flatten_nested_json(raw)
        # Must produce company, departments, employees
        self.assertIn("company", tables)
        self.assertIn("departments", tables)
        self.assertIn("employees", tables)

        # Run extractor on the flattened output
        metadata = self.extractor.extract_metadata(tables)
        self.assertEqual(metadata["dataset_type"], "multi_table")
        self.assertIn("employees", metadata["tables"])

        # Null salary in employees should be detected
        emp_attrs = metadata["tables"]["employees"]["attributes"]
        self.assertIn("salary", emp_attrs)
        self.assertGreater(emp_attrs["salary"]["null_percentage"], 0)

    def test_fk_columns_not_in_extractor_output(self):
        raw = {
            "orders": [
                {"order_id": 1, "items": [{"item_id": 10, "qty": 2}]}
            ]
        }
        tables = self.loader._flatten_nested_json(raw)
        metadata = self.extractor.extract_metadata(tables)
        items_attrs = metadata["tables"]["items"]["attributes"]
        fk_cols = [k for k in items_attrs if k.startswith("_ref_")]
        self.assertEqual(fk_cols, [], f"FK columns leaked into analysis: {fk_cols}")


if __name__ == "__main__":
    unittest.main()
