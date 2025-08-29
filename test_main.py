import unittest
from unittest.mock import patch, MagicMock
from main import MetadataEnrichmentAgent

class TestMetadataEnrichmentAgent(unittest.TestCase):
    @patch('main.Config.get_api_key', return_value='dummy_api_key')
    def setUp(self, mock_get_api_key):
        self.agent = MetadataEnrichmentAgent(debug=False)

    @patch('main.FileLoader')
    @patch('main.MetadataExtractor')
    @patch('main.LLMEnricher')
    @patch('main.Config.get_api_key', return_value='dummy_api_key')
    def test_initialization(self, mock_get_api_key, mock_llm_enricher, mock_metadata_extractor, mock_file_loader):
        mock_llm_enricher.return_value.test_connection.return_value = True
        agent = MetadataEnrichmentAgent(debug=True)
        self.assertTrue(agent.llm_connection_ok)
        self.assertIsNotNone(agent.config)
        self.assertIsNotNone(agent.file_loader)
        self.assertIsNotNone(agent.metadata_extractor)
        self.assertIsNotNone(agent.llm_enricher)

    def test_ensure_directories_creates_dirs(self):
        import os
        for directory in ["output", "input", "logs", "debug_output"]:
            self.assertTrue(os.path.exists(directory))

    def test_run_pipeline_skip_llm(self):
        with patch.object(self.agent.file_loader, 'load_file', return_value=[{"a":1}]), \
             patch.object(self.agent.metadata_extractor, 'extract_metadata', return_value={"summary": {"total_records": 1}}), \
             patch.object(self.agent.llm_enricher, 'enrich_metadata', return_value={"overall_assessment": {}}):
            result = self.agent.run_pipeline("input/input_data.json", skip_llm=True)
            self.assertIn("pipeline_info", result)
            self.assertIn("raw_metadata", result)
            self.assertIn("summary", result)

    def test_run_pipeline_with_llm(self):
        with patch.object(self.agent.file_loader, 'load_file', return_value=[{"a":1}]), \
             patch.object(self.agent.metadata_extractor, 'extract_metadata', return_value={"summary": {"total_records": 1}}), \
             patch.object(self.agent.llm_enricher, 'enrich_metadata', return_value={"overall_assessment": {}}):
            result = self.agent.run_pipeline("input/input_data.json", skip_llm=False)
            self.assertIn("pipeline_info", result)
            self.assertIn("raw_metadata", result)
            self.assertIn("summary", result)

    def test_create_fallback_enrichment(self):
        raw_metadata = {"summary": {"total_records": 1}}
        fallback = self.agent._create_fallback_enrichment(raw_metadata)
        self.assertIn("overall_assessment", fallback)
        self.assertIn("quality_grade", fallback["overall_assessment"])

if __name__ == "__main__":
    unittest.main()
