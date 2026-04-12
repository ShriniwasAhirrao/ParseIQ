"""
ParseIQ — Live JSON Dataset Tests
==================================
Tests the full pipeline with REAL data structures ranging from
trivially simple to deeply complex nested JSON.

Each test writes a JSON file, runs Pipeline.run(llm=False), and inspects
the PipelineResult + generated output files for correctness.

Categories:
  1. Flat / simple JSON
  2. Multi-table sibling arrays
  3. Nested objects (1–2 levels)
  4. Deeply nested (3+ levels)
  5. Mixed / polymorphic records
  6. Real-world-shaped data (e-commerce, IoT, GitHub-like, healthcare)
  7. Edge cases (nulls, empty arrays, unicode, huge records)
  8. Stress: wide tables, many tables, large row counts
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import shutil
import string
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from parseiq.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(data: Any, name: str = "data.json", tmp: str = None) -> "PipelineResult":
    """Write JSON to temp, run pipeline, return result."""
    if tmp is None:
        tmp = tempfile.mkdtemp()
    json_path = os.path.join(tmp, name)
    out_dir = os.path.join(tmp, "out")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = Pipeline.from_file(json_path, output_dir=out_dir).run(llm=False)
    return result, out_dir, tmp, buf.getvalue()


def _cleanup(tmp: str):
    shutil.rmtree(tmp, ignore_errors=True)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 1. FLAT / SIMPLE JSON
# ===========================================================================

class TestFlatJSON(unittest.TestCase):
    """Plain list-of-dicts — the simplest possible JSON input."""

    def test_01_minimal_two_rows(self):
        """Absolute minimum: 2 records, 2 columns."""
        data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(len(result.tables), 1)
            self.assertGreaterEqual(result.overall_quality_score, 0)
            self.assertIn("summary", result.raw_metadata)
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 2)
        finally:
            _cleanup(tmp)

    def test_02_ten_rows_mixed_types(self):
        """10 rows with int, float, string, boolean, null."""
        data = [
            {"id": i, "value": i * 1.5, "label": f"item_{i}",
             "active": i % 2 == 0, "notes": None if i % 3 == 0 else "ok"}
            for i in range(1, 11)
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 10)
            # Quality score should exist for the table
            self.assertTrue(len(result.quality_scores) >= 1)
        finally:
            _cleanup(tmp)

    def test_03_all_strings(self):
        """Dataset where every field is a string."""
        data = [
            {"first": "John", "last": "Doe", "email": "john@example.com", "city": "NY"},
            {"first": "Jane", "last": "Smith", "email": "jane@example.com", "city": "LA"},
            {"first": "Bob", "last": "Brown", "email": "bob@example.com", "city": "SF"},
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 3)
        finally:
            _cleanup(tmp)

    def test_04_all_numeric(self):
        """Pure numeric dataset — sensor readings style."""
        random.seed(42)
        data = [
            {"sensor_id": i, "temp": round(random.gauss(22, 3), 2),
             "humidity": round(random.uniform(30, 90), 1),
             "pressure": round(random.gauss(1013, 5), 1)}
            for i in range(50)
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 50)
            # Check raw_metadata.json was created
            raw_path = os.path.join(out, "raw_metadata.json")
            self.assertTrue(os.path.exists(raw_path))
            raw = _read_json(raw_path)
            # Should have table metadata with statistical info
            tables_meta = raw["tables"]
            self.assertTrue(len(tables_meta) >= 1)
        finally:
            _cleanup(tmp)

    def test_05_single_record(self):
        """Just one record — edge but valid."""
        data = [{"x": 100, "y": 200, "z": 300}]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 1)
        finally:
            _cleanup(tmp)

    def test_06_100_rows_realistic_users(self):
        """100-row user table with realistic variety."""
        random.seed(99)
        domains = ["gmail.com", "yahoo.com", "outlook.com", "company.io"]
        cities = ["New York", "London", "Tokyo", "Berlin", "Mumbai", "Sydney"]
        data = []
        for i in range(100):
            data.append({
                "user_id": i + 1,
                "username": f"user_{i+1:04d}",
                "email": f"user{i+1}@{random.choice(domains)}",
                "age": random.randint(18, 80),
                "salary": round(random.uniform(30000, 150000), 2),
                "city": random.choice(cities),
                "is_active": random.random() > 0.2,
                "signup_date": (datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1500))).strftime("%Y-%m-%d"),
                "score": round(random.gauss(50, 15), 1),
                "notes": None if random.random() < 0.15 else "standard",
            })
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 100)
            # Should detect some quality patterns
            self.assertGreater(result.overall_quality_score, 0)
            # Check output files exist
            for f in result.output_files:
                self.assertTrue(os.path.exists(f), f"Missing: {f}")
        finally:
            _cleanup(tmp)


# ===========================================================================
# 2. MULTI-TABLE SIBLING ARRAYS
# ===========================================================================

class TestMultiTableSiblings(unittest.TestCase):
    """Root dict with multiple top-level arrays → multiple tables."""

    def test_01_two_tables(self):
        data = {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@test.com"},
                {"id": 2, "name": "Bob", "email": "bob@test.com"},
            ],
            "orders": [
                {"order_id": 101, "user_id": 1, "total": 49.99},
                {"order_id": 102, "user_id": 2, "total": 129.50},
                {"order_id": 103, "user_id": 1, "total": 9.99},
            ],
        }
        result, out, tmp, log = _run(data)
        try:
            self.assertGreaterEqual(len(result.tables), 2)
            self.assertIn("users", result.tables)
            self.assertIn("orders", result.tables)
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 5)
        finally:
            _cleanup(tmp)

    def test_02_five_tables(self):
        """Five sibling tables of different sizes."""
        data = {
            "customers": [{"id": i, "name": f"Cust{i}"} for i in range(20)],
            "products": [{"sku": f"SKU-{i}", "price": i * 9.99} for i in range(15)],
            "orders": [{"oid": i, "cust_id": i % 20, "amount": i * 5} for i in range(50)],
            "reviews": [{"rid": i, "sku": f"SKU-{i%15}", "stars": (i % 5) + 1} for i in range(30)],
            "returns": [{"ret_id": i, "oid": i * 3, "reason": "defective"} for i in range(10)],
        }
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(len(result.tables), 5)
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 125)
            # Each table should have a quality score
            self.assertEqual(len(result.quality_scores), 5)
        finally:
            _cleanup(tmp)

    def test_03_tables_with_different_schemas(self):
        """Tables with completely different column sets."""
        data = {
            "employees": [
                {"emp_id": 1, "first_name": "Alice", "dept": "Engineering", "salary": 95000},
                {"emp_id": 2, "first_name": "Bob", "dept": "Marketing", "salary": 72000},
            ],
            "sensors": [
                {"device": "temp-01", "reading": 22.5, "timestamp": "2024-01-15T10:30:00"},
                {"device": "temp-02", "reading": 23.1, "timestamp": "2024-01-15T10:31:00"},
            ],
            "logs": [
                {"level": "ERROR", "message": "Connection timeout", "code": 504},
                {"level": "WARN", "message": "Slow query", "code": 200},
                {"level": "INFO", "message": "Startup complete", "code": 200},
            ],
        }
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(len(result.tables), 3)
            for tname in ["employees", "sensors", "logs"]:
                self.assertIn(tname, result.tables)
        finally:
            _cleanup(tmp)


# ===========================================================================
# 3. NESTED OBJECTS (1–2 levels)
# ===========================================================================

class TestNestedObjects(unittest.TestCase):
    """Records containing embedded objects that get flattened with __ separator."""

    def test_01_single_level_nested_object(self):
        """address: {city, zip} → address__city, address__zip"""
        data = [
            {"id": 1, "name": "Alice", "address": {"city": "NYC", "zip": "10001"}},
            {"id": 2, "name": "Bob", "address": {"city": "LA", "zip": "90001"}},
            {"id": 3, "name": "Carol", "address": {"city": "Chicago", "zip": "60601"}},
        ]
        result, out, tmp, log = _run(data)
        try:
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            # Find the main table metadata
            tables_meta = raw["tables"]
            tname = list(tables_meta.keys())[0]
            attrs = tables_meta[tname]["table_metadata"]["attributes"]
            # Should have flattened columns
            col_names = list(attrs.keys())
            self.assertIn("address__city", col_names)
            self.assertIn("address__zip", col_names)
        finally:
            _cleanup(tmp)

    def test_02_two_level_nested(self):
        """company.address.geo → should flatten or recurse."""
        data = [
            {
                "id": 1,
                "name": "Alice",
                "company": {
                    "name": "Acme",
                    "address": {"street": "123 Main", "city": "NYC"},
                },
            },
            {
                "id": 2,
                "name": "Bob",
                "company": {
                    "name": "Globe",
                    "address": {"street": "456 Oak", "city": "LA"},
                },
            },
        ]
        result, out, tmp, log = _run(data)
        try:
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            tables_meta = raw["tables"]
            # Should have extracted the company address fields somewhere
            tname = list(tables_meta.keys())[0]
            attrs = tables_meta[tname]["table_metadata"]["attributes"]
            col_names = list(attrs.keys())
            # Expect flattened like company__name, company__address__city
            has_company_cols = any("company" in c for c in col_names)
            self.assertTrue(has_company_cols, f"Expected company columns in {col_names}")
        finally:
            _cleanup(tmp)

    def test_03_nested_with_null_objects(self):
        """Some records have the nested object, some have null."""
        data = [
            {"id": 1, "profile": {"bio": "Developer", "website": "alice.dev"}},
            {"id": 2, "profile": None},
            {"id": 3, "profile": {"bio": "Designer", "website": "carol.io"}},
            {"id": 4, "profile": None},
            {"id": 5, "profile": {"bio": "Manager", "website": None}},
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 5)
            # Pipeline should handle nulls gracefully
            self.assertGreater(result.overall_quality_score, 0)
        finally:
            _cleanup(tmp)

    def test_04_nested_with_arrays_and_objects(self):
        """Mix of nested objects and primitive arrays."""
        data = [
            {
                "id": 1,
                "name": "Alice",
                "tags": ["python", "data"],
                "meta": {"joined": "2023-01", "level": "senior"},
            },
            {
                "id": 2,
                "name": "Bob",
                "tags": ["java", "backend", "cloud"],
                "meta": {"joined": "2024-06", "level": "junior"},
            },
        ]
        result, out, tmp, log = _run(data)
        try:
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            tname = list(raw["tables"].keys())[0]
            attrs = raw["tables"][tname]["table_metadata"]["attributes"]
            col_names = list(attrs.keys())
            # tags should be joined as comma-separated string
            self.assertIn("tags", col_names)
            # meta should be flattened
            self.assertIn("meta__joined", col_names)
        finally:
            _cleanup(tmp)


# ===========================================================================
# 4. DEEPLY NESTED JSON (3+ levels) — child tables with FK injection
# ===========================================================================

class TestDeeplyNested(unittest.TestCase):
    """JSON with arrays-of-objects nested inside arrays-of-objects → child tables."""

    def test_01_parent_child_two_levels(self):
        """departments → employees (1 level of nesting)."""
        data = {
            "departments": [
                {
                    "dept_id": 1, "name": "Engineering",
                    "employees": [
                        {"emp_id": 101, "name": "Alice", "role": "Lead"},
                        {"emp_id": 102, "name": "Bob", "role": "Dev"},
                    ],
                },
                {
                    "dept_id": 2, "name": "Marketing",
                    "employees": [
                        {"emp_id": 201, "name": "Carol", "role": "Manager"},
                    ],
                },
            ]
        }
        result, out, tmp, log = _run(data)
        try:
            # Should have departments table + employees child table
            self.assertGreaterEqual(len(result.tables), 2)
            self.assertIn("departments", result.tables)
            self.assertIn("employees", result.tables)
            # FK columns (_ref_*) are injected into raw data by the flattener
            # but the metadata extractor filters them from attributes (by design).
            # Verify the child table has the expected non-FK columns instead.
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            emp_attrs = list(raw["tables"]["employees"]["table_metadata"]["attributes"].keys())
            self.assertIn("emp_id", emp_attrs)
            self.assertIn("name", emp_attrs)
        finally:
            _cleanup(tmp)

    def test_02_three_levels_deep(self):
        """company → divisions → teams → members."""
        data = {
            "company": [
                {
                    "id": 1, "name": "MegaCorp",
                    "divisions": [
                        {
                            "div_id": 10, "name": "Tech",
                            "teams": [
                                {
                                    "team_id": 100, "name": "Backend",
                                    "members": [
                                        {"member_id": 1001, "name": "Alice", "level": "L5"},
                                        {"member_id": 1002, "name": "Bob", "level": "L4"},
                                    ]
                                },
                                {
                                    "team_id": 101, "name": "Frontend",
                                    "members": [
                                        {"member_id": 1003, "name": "Carol", "level": "L4"},
                                    ]
                                },
                            ],
                        },
                        {
                            "div_id": 11, "name": "Sales",
                            "teams": [
                                {
                                    "team_id": 200, "name": "Enterprise",
                                    "members": [
                                        {"member_id": 2001, "name": "Dave", "level": "L6"},
                                    ]
                                },
                            ],
                        },
                    ],
                },
            ]
        }
        result, out, tmp, log = _run(data)
        try:
            # Should create 4 tables: company, divisions, teams, members
            self.assertGreaterEqual(len(result.tables), 3)
            self.assertIn("company", result.tables)
            self.assertIn("divisions", result.tables)
            self.assertIn("teams", result.tables)
            self.assertIn("members", result.tables)
            # FK columns are injected into raw data but filtered from metadata
            # attributes by the extractor. Verify child tables have their own columns.
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            div_attrs = list(raw["tables"]["divisions"]["table_metadata"]["attributes"].keys())
            team_attrs = list(raw["tables"]["teams"]["table_metadata"]["attributes"].keys())
            member_attrs = list(raw["tables"]["members"]["table_metadata"]["attributes"].keys())
            self.assertIn("div_id", div_attrs)
            self.assertIn("team_id", team_attrs)
            self.assertIn("member_id", member_attrs)
        finally:
            _cleanup(tmp)

    def test_03_same_name_children_from_different_parents(self):
        """Two parent types both have 'items' children — should merge or disambiguate."""
        data = {
            "orders": [
                {
                    "order_id": 1,
                    "items": [
                        {"sku": "A1", "qty": 2, "price": 10.0},
                        {"sku": "B2", "qty": 1, "price": 25.0},
                    ],
                },
                {
                    "order_id": 2,
                    "items": [
                        {"sku": "C3", "qty": 5, "price": 5.0},
                    ],
                },
            ],
            "invoices": [
                {
                    "invoice_id": 1001,
                    "items": [
                        {"line": 1, "description": "Service A", "amount": 500},
                        {"line": 2, "description": "Service B", "amount": 300},
                    ],
                },
            ],
        }
        result, out, tmp, log = _run(data)
        try:
            # Should have orders, invoices, and items tables
            # items might be merged or disambiguated
            self.assertGreaterEqual(len(result.tables), 3)
            self.assertIn("orders", result.tables)
            self.assertIn("invoices", result.tables)
        finally:
            _cleanup(tmp)

    def test_04_four_levels_school_system(self):
        """districts → schools → classes → students → grades (4 levels)."""
        data = {
            "districts": [
                {
                    "district_id": 1, "name": "North District",
                    "schools": [
                        {
                            "school_id": 10, "name": "Lincoln High",
                            "classes": [
                                {
                                    "class_id": 100, "subject": "Math",
                                    "students": [
                                        {"student_id": 1, "name": "Alice", "grade": "A"},
                                        {"student_id": 2, "name": "Bob", "grade": "B+"},
                                    ]
                                },
                                {
                                    "class_id": 101, "subject": "Science",
                                    "students": [
                                        {"student_id": 3, "name": "Carol", "grade": "A-"},
                                    ]
                                },
                            ],
                        },
                    ],
                },
            ]
        }
        result, out, tmp, log = _run(data)
        try:
            self.assertGreaterEqual(len(result.tables), 4)
            self.assertIn("districts", result.tables)
            self.assertIn("schools", result.tables)
            self.assertIn("classes", result.tables)
            self.assertIn("students", result.tables)
        finally:
            _cleanup(tmp)


# ===========================================================================
# 5. MIXED / POLYMORPHIC RECORDS
# ===========================================================================

class TestPolymorphicRecords(unittest.TestCase):
    """Records with inconsistent schemas, optional fields, type mixing."""

    def test_01_missing_fields(self):
        """Some records have fields others don't."""
        data = [
            {"id": 1, "name": "Alice", "email": "alice@test.com", "phone": "555-0100"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Carol", "email": "carol@test.com"},
            {"id": 4, "phone": "555-0400"},
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 4)
            # Should still get quality analysis
            self.assertGreater(result.overall_quality_score, 0)
        finally:
            _cleanup(tmp)

    def test_02_type_inconsistency(self):
        """Same field has different types across records."""
        data = [
            {"id": 1, "value": 42},
            {"id": 2, "value": "forty-three"},
            {"id": 3, "value": 44.5},
            {"id": 4, "value": True},
            {"id": 5, "value": None},
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 5)
        finally:
            _cleanup(tmp)

    def test_03_nested_sometimes_present(self):
        """Some records have nested objects, others have the same field as null/string."""
        data = [
            {"id": 1, "metadata": {"source": "web", "version": 2}},
            {"id": 2, "metadata": None},
            {"id": 3, "metadata": {"source": "api", "version": 3}},
            {"id": 4, "metadata": None},
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 4)
        finally:
            _cleanup(tmp)


# ===========================================================================
# 6. REAL-WORLD SHAPED DATA
# ===========================================================================

class TestEcommerceDataset(unittest.TestCase):
    """Realistic e-commerce order system."""

    @classmethod
    def setUpClass(cls):
        random.seed(2024)
        cls.data = {
            "customers": [],
            "orders": [],
            "products": [],
        }
        # Products
        categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]
        for i in range(30):
            cls.data["products"].append({
                "product_id": f"PROD-{i+1:04d}",
                "name": f"Product {i+1}",
                "category": random.choice(categories),
                "price": round(random.uniform(5, 500), 2),
                "weight_kg": round(random.uniform(0.1, 25), 2),
                "in_stock": random.random() > 0.1,
                "rating": round(random.uniform(1, 5), 1),
                "reviews_count": random.randint(0, 5000),
            })
        # Customers
        for i in range(50):
            cls.data["customers"].append({
                "customer_id": i + 1,
                "name": f"Customer_{i+1}",
                "email": f"cust{i+1}@{'gmail' if i%2==0 else 'yahoo'}.com",
                "country": random.choice(["US", "UK", "DE", "JP", "IN", "BR"]),
                "account_type": random.choice(["free", "premium", "enterprise"]),
                "created_at": (datetime(2021, 1, 1) + timedelta(days=random.randint(0, 1000))).isoformat(),
                "lifetime_value": round(random.uniform(0, 10000), 2),
            })
        # Orders with nested line_items
        for i in range(100):
            n_items = random.randint(1, 5)
            cls.data["orders"].append({
                "order_id": f"ORD-{i+1:05d}",
                "customer_id": random.randint(1, 50),
                "status": random.choice(["pending", "shipped", "delivered", "cancelled", "returned"]),
                "order_date": (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 500))).strftime("%Y-%m-%d"),
                "shipping_address": {
                    "street": f"{random.randint(1,999)} Main St",
                    "city": random.choice(["New York", "London", "Berlin", "Tokyo"]),
                    "postal_code": f"{random.randint(10000,99999)}",
                },
                "line_items": [
                    {
                        "product_id": f"PROD-{random.randint(1,30):04d}",
                        "quantity": random.randint(1, 10),
                        "unit_price": round(random.uniform(5, 500), 2),
                        "discount": round(random.uniform(0, 0.3), 2),
                    }
                    for _ in range(n_items)
                ],
                "total": round(random.uniform(10, 2000), 2),
                "payment_method": random.choice(["credit_card", "paypal", "bank_transfer"]),
            })

    def test_01_table_count(self):
        result, out, tmp, log = _run(self.data)
        try:
            # customers, products, orders (flattened), line_items (child table)
            self.assertGreaterEqual(len(result.tables), 4)
            self.assertIn("customers", result.tables)
            self.assertIn("products", result.tables)
            self.assertIn("orders", result.tables)
            self.assertIn("line_items", result.tables)
        finally:
            _cleanup(tmp)

    def test_02_record_counts(self):
        result, out, tmp, log = _run(self.data)
        try:
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            counts = raw["summary"]["table_record_counts"]
            self.assertEqual(counts["customers"], 50)
            self.assertEqual(counts["products"], 30)
            self.assertEqual(counts["orders"], 100)
            # line_items count should be > 100 (1-5 items per order)
            self.assertGreater(counts["line_items"], 100)
        finally:
            _cleanup(tmp)

    def test_03_quality_scores_all_tables(self):
        result, out, tmp, log = _run(self.data)
        try:
            for tname in result.tables:
                self.assertIn(tname, result.quality_scores)
                self.assertGreaterEqual(result.quality_scores[tname], 0)
                self.assertLessEqual(result.quality_scores[tname], 100)
        finally:
            _cleanup(tmp)

    def test_04_output_files_created(self):
        result, out, tmp, log = _run(self.data)
        try:
            self.assertGreater(len(result.output_files), 0)
            for f in result.output_files:
                self.assertTrue(os.path.exists(f), f"Missing output: {f}")
            # Check enriched_metadata.json exists
            enriched_path = os.path.join(out, "enriched_metadata.json")
            self.assertTrue(os.path.exists(enriched_path))
        finally:
            _cleanup(tmp)

    def test_05_shipping_address_flattened(self):
        """shipping_address object should be flattened into orders table."""
        result, out, tmp, log = _run(self.data)
        try:
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            order_attrs = list(raw["tables"]["orders"]["table_metadata"]["attributes"].keys())
            # Should have flattened shipping_address__city, etc.
            has_shipping = any("shipping_address" in a for a in order_attrs)
            self.assertTrue(has_shipping, f"Expected shipping_address columns in {order_attrs}")
        finally:
            _cleanup(tmp)


class TestIoTDataset(unittest.TestCase):
    """IoT sensor data — time series with nested device info."""

    def test_iot_sensor_readings(self):
        random.seed(777)
        data = {
            "devices": [
                {
                    "device_id": f"DEV-{i:03d}",
                    "type": random.choice(["temperature", "humidity", "pressure", "motion"]),
                    "location": {"building": f"B{random.randint(1,5)}", "floor": random.randint(1, 10), "room": f"R{random.randint(100,500)}"},
                    "firmware": f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,20)}",
                    "readings": [
                        {
                            "timestamp": (datetime(2024, 1, 1) + timedelta(hours=h)).isoformat(),
                            "value": round(random.gauss(25, 5), 2),
                            "unit": "celsius",
                            "quality": random.choice(["good", "fair", "poor"]),
                        }
                        for h in range(24)
                    ],
                }
                for i in range(10)
            ]
        }
        result, out, tmp, log = _run(data)
        try:
            self.assertGreaterEqual(len(result.tables), 2)  # devices + readings
            self.assertIn("devices", result.tables)
            self.assertIn("readings", result.tables)
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            # 10 devices × 24 readings = 240 reading records
            self.assertEqual(raw["summary"]["table_record_counts"]["readings"], 240)
        finally:
            _cleanup(tmp)


class TestGitHubLikeDataset(unittest.TestCase):
    """GitHub-style repository data with multiple nested entities."""

    def test_repos_with_issues_and_prs(self):
        data = {
            "repositories": [
                {
                    "repo_id": 1,
                    "name": "parseiq",
                    "owner": "alice",
                    "language": "Python",
                    "stars": 150,
                    "forks": 23,
                    "issues": [
                        {
                            "issue_id": 1, "title": "Bug in parser", "state": "open",
                            "labels": ["bug", "critical"],
                            "created_at": "2024-01-10",
                            "comments": [
                                {"author": "bob", "body": "I can reproduce this", "created_at": "2024-01-11"},
                                {"author": "carol", "body": "Fix incoming", "created_at": "2024-01-12"},
                            ],
                        },
                        {
                            "issue_id": 2, "title": "Feature request: XML support", "state": "closed",
                            "labels": ["enhancement"],
                            "created_at": "2024-02-01",
                            "comments": [],
                        },
                    ],
                    "pull_requests": [
                        {
                            "pr_id": 10, "title": "Add XML parser", "state": "merged",
                            "author": "carol", "additions": 250, "deletions": 10,
                            "files_changed": [
                                {"path": "src/parser.py", "changes": 150},
                                {"path": "tests/test_parser.py", "changes": 100},
                            ],
                        },
                    ],
                },
                {
                    "repo_id": 2,
                    "name": "dataflow",
                    "owner": "alice",
                    "language": "Go",
                    "stars": 45,
                    "forks": 8,
                    "issues": [
                        {"issue_id": 3, "title": "Slow performance", "state": "open", "labels": ["perf"], "created_at": "2024-03-01", "comments": []},
                    ],
                    "pull_requests": [],
                },
            ]
        }
        result, out, tmp, log = _run(data)
        try:
            # Should have: repositories, issues, comments, pull_requests, files_changed
            self.assertGreaterEqual(len(result.tables), 4)
            self.assertIn("repositories", result.tables)
            self.assertIn("issues", result.tables)
            self.assertIn("pull_requests", result.tables)
            # FK columns are in raw data but filtered from metadata attributes.
            # Verify issues table has its own columns.
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            issue_attrs = list(raw["tables"]["issues"]["table_metadata"]["attributes"].keys())
            self.assertIn("issue_id", issue_attrs)
            self.assertIn("title", issue_attrs)
        finally:
            _cleanup(tmp)


class TestHealthcareDataset(unittest.TestCase):
    """Healthcare-like data: patients with visits, diagnoses, medications."""

    def test_patient_records(self):
        random.seed(555)
        data = {
            "patients": [
                {
                    "patient_id": f"P-{i:04d}",
                    "name": f"Patient {i}",
                    "dob": f"{random.randint(1940, 2005)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                    "gender": random.choice(["M", "F", "Other"]),
                    "blood_type": random.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]),
                    "insurance": {"provider": random.choice(["Blue Cross", "Aetna", "United"]),
                                  "policy_number": f"POL-{random.randint(100000, 999999)}"},
                    "visits": [
                        {
                            "visit_id": f"V-{i}-{v}",
                            "date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                            "type": random.choice(["routine", "emergency", "follow_up"]),
                            "doctor": f"Dr. {random.choice(['Smith', 'Jones', 'Lee', 'Patel'])}",
                            "diagnoses": [
                                {"code": f"ICD-{random.randint(100,999)}", "description": random.choice(["Hypertension", "Diabetes", "Fracture", "Flu"])}
                                for _ in range(random.randint(1, 3))
                            ],
                            "medications": [
                                {"name": random.choice(["Aspirin", "Metformin", "Lisinopril", "Amoxicillin"]),
                                 "dosage": f"{random.choice([5,10,20,50,100])}mg",
                                 "frequency": random.choice(["daily", "twice_daily", "as_needed"])}
                                for _ in range(random.randint(0, 3))
                            ],
                            "vitals": {
                                "bp_systolic": random.randint(90, 180),
                                "bp_diastolic": random.randint(60, 120),
                                "heart_rate": random.randint(55, 110),
                                "temperature": round(random.uniform(97, 104), 1),
                            },
                        }
                        for v in range(random.randint(1, 4))
                    ],
                }
                for i in range(1, 21)
            ]
        }
        result, out, tmp, log = _run(data)
        try:
            # Should decompose into multiple tables
            self.assertGreaterEqual(len(result.tables), 4)
            self.assertIn("patients", result.tables)
            self.assertIn("visits", result.tables)
            self.assertIn("diagnoses", result.tables)
            self.assertIn("medications", result.tables)
            # Verify reasonable record counts
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            self.assertEqual(raw["summary"]["table_record_counts"]["patients"], 20)
            self.assertGreater(raw["summary"]["table_record_counts"]["visits"], 20)
            self.assertGreater(raw["summary"]["table_record_counts"]["diagnoses"], 20)
            # Each table should have a quality score
            for tname in result.tables:
                self.assertIn(tname, result.quality_scores)
        finally:
            _cleanup(tmp)


# ===========================================================================
# 7. EDGE CASES
# ===========================================================================

class TestEdgeCases(unittest.TestCase):
    """Unusual inputs that stress the parser and pipeline."""

    def test_01_all_nulls(self):
        """Records where every value is null."""
        data = [
            {"a": None, "b": None, "c": None},
            {"a": None, "b": None, "c": None},
            {"a": None, "b": None, "c": None},
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 3)
        finally:
            _cleanup(tmp)

    def test_02_unicode_heavy(self):
        """Fields with unicode: CJK, emoji, Arabic, Cyrillic."""
        data = [
            {"id": 1, "name": "田中太郎", "city": "東京", "notes": "🎉 Great!"},
            {"id": 2, "name": "Мария Иванова", "city": "Москва", "notes": "Отлично 👍"},
            {"id": 3, "name": "محمد علي", "city": "القاهرة", "notes": "ممتاز"},
            {"id": 4, "name": "José García", "city": "São Paulo", "notes": "café ñ"},
            {"id": 5, "name": "Ελένη", "city": "Αθήνα", "notes": "Τέλεια"},
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 5)
            self.assertGreater(result.overall_quality_score, 0)
        finally:
            _cleanup(tmp)

    def test_03_empty_nested_arrays(self):
        """Child arrays that are empty []."""
        data = {
            "groups": [
                {"id": 1, "name": "Empty Group", "members": []},
                {"id": 2, "name": "Has Members", "members": [
                    {"uid": 1, "role": "admin"},
                    {"uid": 2, "role": "user"},
                ]},
                {"id": 3, "name": "Also Empty", "members": []},
            ]
        }
        result, out, tmp, log = _run(data)
        try:
            self.assertIn("groups", result.tables)
            # members table should have only 2 records from group 2
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            if "members" in raw["summary"]["table_record_counts"]:
                self.assertEqual(raw["summary"]["table_record_counts"]["members"], 2)
        finally:
            _cleanup(tmp)

    def test_04_deeply_nested_single_path(self):
        """A single chain 5 levels deep."""
        data = {
            "level1": [
                {
                    "id": 1,
                    "level2": [
                        {
                            "id": 10,
                            "level3": [
                                {
                                    "id": 100,
                                    "level4": [
                                        {
                                            "id": 1000,
                                            "level5": [
                                                {"id": 10000, "value": "deep_leaf"}
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        result, out, tmp, log = _run(data)
        try:
            # Should create 5 tables
            self.assertGreaterEqual(len(result.tables), 5)
        finally:
            _cleanup(tmp)

    def test_05_very_long_string_values(self):
        """Records with very long strings (10K+ chars)."""
        long_text = "x" * 10000
        data = [
            {"id": 1, "content": long_text, "short": "a"},
            {"id": 2, "content": long_text[:5000], "short": "b"},
            {"id": 3, "content": "normal text", "short": "c"},
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 3)
        finally:
            _cleanup(tmp)

    def test_06_numeric_extremes(self):
        """Very large and very small numbers."""
        data = [
            {"id": 1, "big": 9999999999999999, "small": 0.000000001, "negative": -9999999},
            {"id": 2, "big": 1e15, "small": 1e-10, "negative": -1e8},
            {"id": 3, "big": 0, "small": 0, "negative": 0},
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 3)
        finally:
            _cleanup(tmp)

    def test_07_boolean_heavy(self):
        """Mostly boolean data."""
        data = [
            {"id": i, "flag_a": i % 2 == 0, "flag_b": i % 3 == 0,
             "flag_c": i % 5 == 0, "flag_d": True, "flag_e": False}
            for i in range(20)
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 20)
        finally:
            _cleanup(tmp)

    def test_08_root_level_flat_object(self):
        """A single flat dict at root (not array, not dict-of-arrays)."""
        data = {"name": "Alice", "age": 30, "city": "NYC"}
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 1)
        finally:
            _cleanup(tmp)

    def test_09_dates_in_various_formats(self):
        """Dates in different string formats."""
        data = [
            {"id": 1, "date": "2024-01-15", "event": "ISO format"},
            {"id": 2, "date": "01/15/2024", "event": "US format"},
            {"id": 3, "date": "15-Jan-2024", "event": "abbreviated"},
            {"id": 4, "date": "2024-01-15T10:30:00Z", "event": "ISO with time"},
            {"id": 5, "date": "January 15, 2024", "event": "long format"},
            {"id": 6, "date": "2024/01/15", "event": "slash format"},
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 6)
        finally:
            _cleanup(tmp)

    def test_10_duplicate_rows(self):
        """All rows are identical."""
        row = {"x": 1, "y": "hello", "z": True}
        data = [row.copy() for _ in range(20)]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 20)
        finally:
            _cleanup(tmp)

    def test_11_special_characters_in_keys(self):
        """Field names with spaces, dots, dashes."""
        data = [
            {"user name": "Alice", "user.email": "a@b.com", "user-id": 1, "order#": 100},
            {"user name": "Bob", "user.email": "b@c.com", "user-id": 2, "order#": 101},
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 2)
        finally:
            _cleanup(tmp)


# ===========================================================================
# 8. STRESS TESTS
# ===========================================================================

class TestStressLargeRowCount(unittest.TestCase):
    """Large number of records."""

    def test_1000_rows(self):
        random.seed(42)
        data = [
            {"id": i, "value": round(random.gauss(100, 25), 2),
             "category": random.choice(["A", "B", "C", "D"]),
             "timestamp": (datetime(2024, 1, 1) + timedelta(minutes=i)).isoformat()}
            for i in range(1000)
        ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 1000)
            self.assertGreater(result.overall_quality_score, 0)
        finally:
            _cleanup(tmp)

    def test_5000_rows_multi_table(self):
        random.seed(123)
        data = {
            "transactions": [
                {"txn_id": i, "amount": round(random.uniform(-1000, 10000), 2),
                 "merchant": f"M-{random.randint(1,50):03d}",
                 "category": random.choice(["food", "travel", "shopping", "utility", "entertainment"]),
                 "timestamp": (datetime(2024, 1, 1) + timedelta(minutes=i*5)).isoformat(),
                 "is_fraud": random.random() < 0.02}
                for i in range(3000)
            ],
            "merchants": [
                {"merchant_id": f"M-{i:03d}", "name": f"Merchant {i}",
                 "city": random.choice(["NYC", "LA", "CHI", "HOU", "PHX"]),
                 "mcc_code": random.randint(1000, 9999)}
                for i in range(1, 51)
            ],
            "alerts": [
                {"alert_id": i, "txn_id": random.randint(0, 2999),
                 "type": random.choice(["velocity", "amount", "geo"]),
                 "severity": random.choice(["low", "medium", "high", "critical"])}
                for i in range(200)
            ],
        }
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 3250)
            self.assertEqual(len(result.tables), 3)
        finally:
            _cleanup(tmp)


class TestStressWideTable(unittest.TestCase):
    """Tables with many columns."""

    def test_50_columns(self):
        random.seed(99)
        data = []
        for i in range(30):
            row = {"id": i}
            for c in range(50):
                row[f"feature_{c:03d}"] = round(random.gauss(0, 1), 4)
            data.append(row)
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 30)
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            tname = list(raw["tables"].keys())[0]
            n_attrs = len(raw["tables"][tname]["table_metadata"]["attributes"])
            self.assertGreaterEqual(n_attrs, 50)
        finally:
            _cleanup(tmp)


class TestStressManyTables(unittest.TestCase):
    """Dataset that produces many tables."""

    def test_10_sibling_tables(self):
        data = {}
        for t in range(10):
            data[f"table_{t:02d}"] = [
                {"id": i, "val": f"t{t}_r{i}"} for i in range(10)
            ]
        result, out, tmp, log = _run(data)
        try:
            self.assertEqual(len(result.tables), 10)
            self.assertEqual(result.raw_metadata["summary"]["total_records"], 100)
        finally:
            _cleanup(tmp)


# ===========================================================================
# 9. COMPLEX REAL-WORLD SCENARIOS
# ===========================================================================

class TestComplexNestedRealWorld(unittest.TestCase):
    """The hardest cases: deeply nested, wide, with mixed types."""

    def test_01_api_response_paginated(self):
        """Typical paginated API response with metadata."""
        data = {
            "meta": {"page": 1, "per_page": 10, "total": 100, "total_pages": 10},
            "data": [
                {
                    "id": i,
                    "type": "article",
                    "attributes": {
                        "title": f"Article {i}",
                        "body": f"Content of article {i} " * 20,
                        "published_at": f"2024-{(i%12)+1:02d}-15",
                        "author": {
                            "id": i * 10,
                            "name": f"Author {i}",
                            "profile": {"avatar_url": f"https://example.com/avatars/{i}.jpg", "bio": f"Bio for author {i}"},
                        },
                        "tags": [f"tag{j}" for j in range(random.randint(1, 5))],
                        "stats": {"views": random.randint(100, 50000), "likes": random.randint(0, 1000), "shares": random.randint(0, 200)},
                    },
                    "relationships": {
                        "comments": [
                            {"id": i * 100 + c, "author": f"user_{c}", "text": f"Comment {c}", "likes": random.randint(0, 50)}
                            for c in range(random.randint(0, 5))
                        ],
                    },
                }
                for i in range(1, 11)
            ]
        }
        result, out, tmp, log = _run(data)
        try:
            # Should extract data + comments as separate tables
            self.assertGreaterEqual(len(result.tables), 2)
            # Verify output files
            for f in result.output_files:
                self.assertTrue(os.path.exists(f), f"Missing: {f}")
        finally:
            _cleanup(tmp)

    def test_02_nested_config_with_arrays_at_every_level(self):
        """Infrastructure-like config: cloud resources with nested arrays everywhere."""
        data = {
            "clusters": [
                {
                    "cluster_id": "prod-us-east",
                    "region": "us-east-1",
                    "nodes": [
                        {
                            "node_id": f"node-{n}",
                            "instance_type": random.choice(["m5.large", "c5.xlarge", "r5.2xlarge"]),
                            "status": "running",
                            "containers": [
                                {
                                    "container_id": f"c-{n}-{c}",
                                    "image": f"app-v{random.randint(1,5)}.{random.randint(0,9)}",
                                    "cpu_limit": random.choice(["0.5", "1.0", "2.0"]),
                                    "memory_mb": random.choice([256, 512, 1024, 2048]),
                                    "ports": [
                                        {"container_port": random.randint(3000, 9000), "protocol": "TCP"}
                                        for _ in range(random.randint(1, 3))
                                    ],
                                    "env_vars": [
                                        {"key": f"VAR_{v}", "value": f"val_{v}"}
                                        for v in range(random.randint(2, 6))
                                    ],
                                }
                                for c in range(random.randint(2, 5))
                            ],
                            "labels": {"tier": random.choice(["frontend", "backend", "data"]), "env": "production"},
                        }
                        for n in range(3)
                    ],
                    "load_balancers": [
                        {"lb_id": f"lb-{l}", "type": "ALB", "target_port": 8080}
                        for l in range(2)
                    ],
                },
                {
                    "cluster_id": "staging-eu",
                    "region": "eu-west-1",
                    "nodes": [
                        {
                            "node_id": "node-staging-1",
                            "instance_type": "t3.medium",
                            "status": "running",
                            "containers": [
                                {
                                    "container_id": "c-s1-1",
                                    "image": "app-v2.0",
                                    "cpu_limit": "0.5",
                                    "memory_mb": 512,
                                    "ports": [{"container_port": 8080, "protocol": "TCP"}],
                                    "env_vars": [{"key": "ENV", "value": "staging"}],
                                }
                            ],
                            "labels": {"tier": "all", "env": "staging"},
                        }
                    ],
                    "load_balancers": [],
                },
            ]
        }
        result, out, tmp, log = _run(data)
        try:
            # clusters, nodes, containers, ports, env_vars, load_balancers
            self.assertGreaterEqual(len(result.tables), 5)
            self.assertIn("clusters", result.tables)
            self.assertIn("nodes", result.tables)
            self.assertIn("containers", result.tables)
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            # Verify correct parent-child count relationships
            self.assertEqual(raw["summary"]["table_record_counts"]["clusters"], 2)
            # 3 nodes in prod + 1 in staging = 4
            self.assertEqual(raw["summary"]["table_record_counts"]["nodes"], 4)
        finally:
            _cleanup(tmp)

    def test_03_analytics_event_stream(self):
        """Web analytics events with nested properties and contexts."""
        random.seed(321)
        events = []
        event_types = ["page_view", "click", "form_submit", "purchase", "error"]
        for i in range(200):
            etype = random.choice(event_types)
            event = {
                "event_id": f"evt-{i:06d}",
                "type": etype,
                "timestamp": (datetime(2024, 6, 1) + timedelta(seconds=i * 30)).isoformat(),
                "user": {
                    "id": f"u-{random.randint(1,50):04d}",
                    "session_id": f"sess-{random.randint(1,200):05d}",
                    "is_authenticated": random.random() > 0.4,
                },
                "context": {
                    "page_url": f"/page/{random.choice(['home','products','about','contact','checkout'])}",
                    "referrer": random.choice([None, "google.com", "facebook.com", "direct"]),
                    "device": random.choice(["desktop", "mobile", "tablet"]),
                    "browser": random.choice(["Chrome", "Firefox", "Safari", "Edge"]),
                    "os": random.choice(["Windows", "macOS", "Linux", "iOS", "Android"]),
                },
                "properties": {},
            }
            if etype == "click":
                event["properties"] = {"element": random.choice(["button", "link", "image"]), "target_url": f"/page/{random.randint(1,20)}"}
            elif etype == "purchase":
                event["properties"] = {"order_id": f"ORD-{i}", "amount": round(random.uniform(10, 500), 2), "currency": "USD"}
            elif etype == "error":
                event["properties"] = {"error_code": random.choice([400, 404, 500, 503]), "message": "Something went wrong"}
            events.append(event)

        result, out, tmp, log = _run(events)
        try:
            self.assertGreaterEqual(len(result.tables), 1)
            self.assertGreaterEqual(result.raw_metadata["summary"]["total_records"], 200)
        finally:
            _cleanup(tmp)

    def test_04_mixed_depth_siblings(self):
        """Some top-level arrays are flat, some are deeply nested."""
        data = {
            "settings": [
                {"key": "theme", "value": "dark"},
                {"key": "lang", "value": "en"},
            ],
            "organizations": [
                {
                    "org_id": 1, "name": "Acme",
                    "departments": [
                        {
                            "dept_id": 10, "name": "R&D",
                            "projects": [
                                {
                                    "project_id": 100, "name": "Project X",
                                    "tasks": [
                                        {"task_id": 1000, "title": "Design", "status": "done", "assignee": "Alice"},
                                        {"task_id": 1001, "title": "Implement", "status": "in_progress", "assignee": "Bob"},
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
            "tags": [
                {"tag_id": 1, "label": "urgent"},
                {"tag_id": 2, "label": "low-priority"},
            ],
        }
        result, out, tmp, log = _run(data)
        try:
            # settings (flat), organizations + departments + projects + tasks (4 levels), tags (flat)
            self.assertGreaterEqual(len(result.tables), 6)
            self.assertIn("settings", result.tables)
            self.assertIn("tags", result.tables)
            self.assertIn("organizations", result.tables)
            self.assertIn("tasks", result.tables)
        finally:
            _cleanup(tmp)

    def test_05_financial_portfolio(self):
        """Investment portfolio: accounts → holdings → transactions."""
        random.seed(888)
        data = {
            "accounts": [
                {
                    "account_id": f"ACC-{a:03d}",
                    "owner": f"Investor {a}",
                    "type": random.choice(["brokerage", "retirement", "savings"]),
                    "balance": round(random.uniform(1000, 1000000), 2),
                    "currency": "USD",
                    "holdings": [
                        {
                            "symbol": random.choice(["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"]),
                            "shares": round(random.uniform(1, 1000), 2),
                            "avg_cost": round(random.uniform(50, 500), 2),
                            "current_price": round(random.uniform(50, 500), 2),
                            "transactions": [
                                {
                                    "txn_id": f"TXN-{a}-{h}-{t}",
                                    "type": random.choice(["buy", "sell", "dividend"]),
                                    "date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                                    "shares": round(random.uniform(1, 100), 2),
                                    "price": round(random.uniform(50, 500), 2),
                                    "fees": round(random.uniform(0, 15), 2),
                                }
                                for t in range(random.randint(2, 8))
                            ],
                        }
                        for h in range(random.randint(2, 6))
                    ],
                    "risk_profile": {
                        "score": random.randint(1, 10),
                        "category": random.choice(["conservative", "moderate", "aggressive"]),
                        "max_drawdown": round(random.uniform(0.05, 0.4), 2),
                    },
                }
                for a in range(1, 11)
            ]
        }
        result, out, tmp, log = _run(data)
        try:
            # accounts, holdings, transactions
            self.assertGreaterEqual(len(result.tables), 3)
            self.assertIn("accounts", result.tables)
            self.assertIn("holdings", result.tables)
            self.assertIn("transactions", result.tables)
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            self.assertEqual(raw["summary"]["table_record_counts"]["accounts"], 10)
            # FK columns are in raw data but filtered from metadata attributes.
            holding_attrs = list(raw["tables"]["holdings"]["table_metadata"]["attributes"].keys())
            self.assertIn("symbol", holding_attrs)
        finally:
            _cleanup(tmp)


# ===========================================================================
# 10. OUTPUT VALIDATION
# ===========================================================================

class TestOutputValidation(unittest.TestCase):
    """Verify the structure and content of ALL output files."""

    def test_all_output_artifacts(self):
        """Run a rich dataset and validate every output file."""
        random.seed(42)
        data = {
            "users": [
                {"id": i, "name": f"User {i}", "email": f"u{i}@test.com",
                 "age": random.randint(18, 65), "score": round(random.gauss(75, 10), 1)}
                for i in range(40)
            ],
            "events": [
                {"event_id": i, "user_id": random.randint(1, 40),
                 "type": random.choice(["login", "click", "purchase"]),
                 "timestamp": (datetime(2024, 1, 1) + timedelta(hours=i)).isoformat()}
                for i in range(80)
            ],
        }
        result, out, tmp, log = _run(data)
        try:
            # 1. raw_metadata.json
            raw_path = os.path.join(out, "raw_metadata.json")
            self.assertTrue(os.path.exists(raw_path))
            raw = _read_json(raw_path)
            self.assertIn("tables", raw)
            self.assertIn("summary", raw)
            self.assertIn("dataset_overview", raw)
            self.assertIn("pipeline_metadata", raw)

            # 2. enriched_metadata.json
            enriched_path = os.path.join(out, "enriched_metadata.json")
            self.assertTrue(os.path.exists(enriched_path))
            enriched = _read_json(enriched_path)
            self.assertIn("pipeline_info", enriched)
            self.assertIn("raw_metadata", enriched)
            self.assertIn("llm_insights", enriched)
            self.assertFalse(enriched["pipeline_info"]["llm_used"])

            # 3. Check summary structure for each table
            for tname in ["users", "events"]:
                self.assertIn(tname, raw["tables"])
                tmeta = raw["tables"][tname]
                self.assertIn("table_metadata", tmeta)
                self.assertIn("data_quality_score", tmeta["table_metadata"])
                self.assertIn("attributes", tmeta["table_metadata"])
                score = tmeta["table_metadata"]["data_quality_score"]
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 100)

            # 4. PipelineResult integrity
            self.assertEqual(len(result.tables), 2)
            self.assertEqual(len(result.quality_scores), 2)
            self.assertIsNone(result.llm_insights)
            self.assertIsInstance(result.alerts_fired, list)
            self.assertEqual(len(result.alerts_fired), 0)
            self.assertIsInstance(result.pipeline_info, dict)
            self.assertEqual(result.pipeline_info["total_tables"], 2)
            self.assertEqual(result.pipeline_info["total_records"], 120)
        finally:
            _cleanup(tmp)

    def test_quality_score_per_table_metadata(self):
        """Each table's metadata must contain data_quality_score and attributes."""
        data = {
            "t1": [{"a": 1, "b": 2}] * 5,
            "t2": [{"x": "hello", "y": None}] * 5,
            "t3": [{"d": "2024-01-01", "v": 99.9}] * 5,
        }
        result, out, tmp, log = _run(data)
        try:
            raw = _read_json(os.path.join(out, "raw_metadata.json"))
            for tname in ["t1", "t2", "t3"]:
                tmeta = raw["tables"][tname]["table_metadata"]
                self.assertIn("data_quality_score", tmeta)
                self.assertIn("attributes", tmeta)
                self.assertIn("top_issues", tmeta)
                self.assertIn("anomaly_summary", tmeta)
        finally:
            _cleanup(tmp)


# ===========================================================================
# 11. INCREMENTAL / CACHE BEHAVIOR
# ===========================================================================

class TestIncrementalBehavior(unittest.TestCase):
    """Test that re-running on unchanged data uses cache."""

    def test_second_run_uses_cache(self):
        data = {
            "items": [{"id": i, "val": i * 10} for i in range(20)]
        }
        tmp = tempfile.mkdtemp()
        json_path = os.path.join(tmp, "data.json")
        out_dir = os.path.join(tmp, "out")
        with open(json_path, "w") as f:
            json.dump(data, f)

        try:
            # First run
            buf1 = io.StringIO()
            with redirect_stdout(buf1):
                r1 = Pipeline.from_file(json_path, output_dir=out_dir).run(llm=False)

            # Second run — should detect cache
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                r2 = Pipeline.from_file(json_path, output_dir=out_dir).run(llm=False)

            self.assertIn("unchanged", buf2.getvalue())

            # Third run with force=True — should bypass cache
            buf3 = io.StringIO()
            with redirect_stdout(buf3):
                r3 = Pipeline.from_file(json_path, output_dir=out_dir).run(llm=False, force=True)

            self.assertNotIn("unchanged", buf3.getvalue())
        finally:
            _cleanup(tmp)


if __name__ == "__main__":
    unittest.main()
