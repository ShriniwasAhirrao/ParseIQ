import unittest
from step1_metadata_extractor.extractor import MetadataExtractor

class TestMetadataExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = MetadataExtractor()

    def test_is_multi_table_dataset(self):
        data = {
            "table1": [{"a": 1}],
            "table2": [{"b": 2}],
            "not_table": "string"
        }
        self.assertTrue(self.extractor._is_multi_table_dataset(data))

    def test_extract_single_table_metadata(self):
        data = [{"a": 1}, {"a": 2}]
        metadata = self.extractor._extract_single_table_metadata(data)
        self.assertIn("table_metadata", metadata)
        self.assertEqual(metadata["summary"]["total_records"], 2)

    def test_extract_metadata_single_table(self):
        data = [{"a": 1}]
        metadata = self.extractor.extract_metadata(data)
        self.assertEqual(metadata["dataset_type"], "single_table")

    def test_extract_metadata_multi_table(self):
        data = {
            "table1": [{"a": 1}],
            "table2": [{"b": 2}]
        }
        metadata = self.extractor.extract_metadata(data)
        self.assertEqual(metadata["dataset_type"], "multi_table")
        self.assertIn("tables", metadata)
        self.assertIn("table1", metadata["tables"])
        self.assertIn("table2", metadata["tables"])

if __name__ == "__main__":
    unittest.main()
