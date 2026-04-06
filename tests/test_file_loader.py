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

    @patch('parseiq.file_loader.loader.Path.exists', return_value=True)
    @patch('parseiq.file_loader.loader.Path.stat')
    def test_load_file_unsupported_format(self, mock_stat, mock_exists):
        mock_stat.return_value.st_size = 1024
        with self.assertRaises(ValueError):
            self.loader.load_file('file.unsupported')

    @patch('parseiq.file_loader.loader.Path.exists', return_value=False)
    def test_load_file_not_found(self, mock_exists):
        with self.assertRaises(FileNotFoundError):
            self.loader.load_file('missing.json')

    @patch('file_loader.loader.FileLoader._load_json')
    @patch('parseiq.file_loader.loader.Path.exists', return_value=True)
    @patch('parseiq.file_loader.loader.Path.stat')
    def test_load_file_json(self, mock_stat, mock_exists, mock_load_json):
        mock_stat.return_value.st_size = 1024
        # load_file passes the raw dict through _flatten_nested_json.
        # A dict with list-of-dicts values produces a table dict; a plain
        # scalar dict with no array children becomes {"main_table": [data]}.
        mock_load_json.return_value = [{"key": "value"}]
        result = self.loader.load_file('test.json')
        # After flattening a list-of-dicts, result is a table dict.
        # Root table is named after the file stem ('test' for 'test.json')
        self.assertIsInstance(result, dict)
        root_table = next(k for k in result if not k.startswith('__path__'))
        self.assertEqual(result[root_table][0]['key'], 'value')

    @patch('file_loader.loader.FileLoader._detect_encoding', return_value='utf-8')
    @patch('builtins.open', new_callable=mock_open, read_data='[{"a":1}]')
    def test_load_json(self, mock_file, mock_detect_encoding):
        result = self.loader._load_json(Path('test.json'))
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]['a'], 1)

if __name__ == "__main__":
    unittest.main()
