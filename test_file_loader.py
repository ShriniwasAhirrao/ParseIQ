import unittest
from unittest.mock import patch, mock_open
from file_loader.loader import FileLoader
from pathlib import Path

class TestFileLoader(unittest.TestCase):
    def setUp(self):
        self.loader = FileLoader()

    def test_supported_formats(self):
        self.assertIn('.json', self.loader.supported_formats)
        self.assertIn('.csv', self.loader.supported_formats)
        self.assertIn('.xml', self.loader.supported_formats)
        self.assertIn('.xlsx', self.loader.supported_formats)

    @patch('file_loader.loader.Path.exists', return_value=True)
    @patch('file_loader.loader.Path.stat')
    def test_load_file_unsupported_format(self, mock_stat, mock_exists):
        mock_stat.return_value.st_size = 1024
        with self.assertRaises(ValueError):
            self.loader.load_file('file.unsupported')

    @patch('file_loader.loader.Path.exists', return_value=False)
    def test_load_file_not_found(self, mock_exists):
        with self.assertRaises(FileNotFoundError):
            self.loader.load_file('missing.json')

    @patch('file_loader.loader.FileLoader._load_json')
    @patch('file_loader.loader.Path.exists', return_value=True)
    @patch('file_loader.loader.Path.stat')
    def test_load_file_json(self, mock_stat, mock_exists, mock_load_json):
        mock_stat.return_value.st_size = 1024
        mock_load_json.return_value = {"key": "value"}
        result = self.loader.load_file('test.json')
        self.assertEqual(result, [{"key": "value"}])

    @patch('file_loader.loader.FileLoader._detect_encoding', return_value='utf-8')
    @patch('builtins.open', new_callable=mock_open, read_data='[{"a":1}]')
    def test_load_json(self, mock_file, mock_detect_encoding):
        result = self.loader._load_json(Path('test.json'))
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]['a'], 1)

    def test_extract_nested_tables_complex(self):
        complex_data = {
            "id": "root123",
            "name": "Root Item",
            "details": {
                "version": "1.0",
                "status": "active"
            },
            "users": [
                {
                    "userId": "u1",
                    "username": "alice",
                    "roles": [
                        {"roleId": "r1", "name": "admin"},
                        {"roleId": "r2", "name": "editor"}
                    ],
                    "preferences": {"theme": "dark"}
                },
                {
                    "userId": "u2",
                    "username": "bob",
                    "roles": [
                        {"roleId": "r3", "name": "viewer"}
                    ]
                }
            ],
            "products": [
                {
                    "productId": "p1",
                    "productName": "Laptop",
                    "specs": [
                        {"specId": "s1", "name": "CPU", "value": "i7"},
                        {"specId": "s2", "name": "RAM", "value": "16GB"}
                    ]
                },
                {
                    "productId": "p2",
                    "productName": "Mouse"
                }
            ],
            "empty_list": [],
            "mixed_list": [
                {"item": 1},
                "string_item",
                {"item": 2}
            ]
        }

        extracted_tables = self.loader._extract_all_nested_tables(complex_data)

        # Assert that the main table is extracted correctly
        self.assertIn("main_table", extracted_tables)
        self.assertEqual(len(extracted_tables["main_table"]), 1)
        self.assertEqual(extracted_tables["main_table"][0]["id"], "root123")

        # Assert 'users' table
        self.assertIn("users", extracted_tables)
        self.assertEqual(len(extracted_tables["users"]), 2)
        self.assertEqual(extracted_tables["users"][0]["username"], "alice")
        self.assertEqual(extracted_tables["users"][1]["username"], "bob")

        # Assert 'users_roles' table (nested within users)
        self.assertIn("users_roles", extracted_tables)
        self.assertEqual(len(extracted_tables["users_roles"]), 3)
        self.assertIn({"roleId": "r1", "name": "admin"}, extracted_tables["users_roles"])
        self.assertIn({"roleId": "r2", "name": "editor"}, extracted_tables["users_roles"])
        self.assertIn({"roleId": "r3", "name": "viewer"}, extracted_tables["users_roles"])

        # Assert 'products' table
        self.assertIn("products", extracted_tables)
        self.assertEqual(len(extracted_tables["products"]), 2)
        self.assertEqual(extracted_tables["products"][0]["productName"], "Laptop")

        # Assert 'products_specs' table (nested within products)
        self.assertIn("products_specs", extracted_tables)
        self.assertEqual(len(extracted_tables["products_specs"]), 2)
        self.assertIn({"specId": "s1", "name": "CPU", "value": "i7"}, extracted_tables["products_specs"])
        self.assertIn({"specId": "s2", "name": "RAM", "value": "16GB"}, extracted_tables["products_specs"])

        # Assert that empty_list does not create a table
        self.assertNotIn("empty_list", extracted_tables)

        # Assert that mixed_list does not create a table (or creates one with only dicts)
        # Depending on the 80% threshold, it might or might not.
        # Given the current implementation, it should create a table with only the dicts.
        self.assertIn("mixed_list", extracted_tables)
        self.assertEqual(len(extracted_tables["mixed_list"]), 2)
        self.assertIn({"item": 1}, extracted_tables["mixed_list"])
        self.assertIn({"item": 2}, extracted_tables["mixed_list"])
        self.assertNotIn("string_item", extracted_tables["mixed_list"])

        # Test with a list as root
        list_root_data = [
            {"id": 1, "name": "itemA", "tags": [{"tagId": "t1"}]},
            {"id": 2, "name": "itemB", "tags": [{"tagId": "t2"}]}
        ]
        extracted_from_list_root = self.loader._extract_all_nested_tables(list_root_data)
        self.assertIn("main_table", extracted_from_list_root)
        self.assertEqual(len(extracted_from_list_root["main_table"]), 2)
        self.assertIn("main_table_tags", extracted_from_list_root)
        self.assertEqual(len(extracted_from_list_root["main_table_tags"]), 2)

if __name__ == "__main__":
    unittest.main()
