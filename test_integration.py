import unittest
from unittest.mock import patch, MagicMock
from main import MetadataEnrichmentAgent

class TestIntegrationMetadataEnrichmentAgent(unittest.TestCase):
    @patch('main.Config.get_api_key', return_value='dummy_api_key')
    @patch('main.FileLoader')
    @patch('main.MetadataExtractor')
    @patch('main.LLMEnricher')
    def test_run_pipeline_full_flow(self, mock_llm_enricher, mock_metadata_extractor, mock_file_loader, mock_get_api_key):
        # Arrange
        mock_file_loader.return_value.load_file.return_value = [{"field1": "value1"}]
        mock_metadata_extractor.return_value.extract_metadata.return_value = {
            "summary": {"total_records": 1},
            "tables": {"main_table": {"data_quality_score": 90}}
        }
        mock_llm_enricher.return_value.enrich_metadata.return_value = {
            "overall_assessment": {"quality_grade": "A", "corrected_score": 95}
        }
        mock_llm_enricher.return_value.test_connection.return_value = True

        agent = MetadataEnrichmentAgent(debug=False)

        # Act
        result = agent.run_pipeline("input/input_data.json", skip_llm=False)

        # Assert
        self.assertIn("pipeline_info", result)
        self.assertIn("raw_metadata", result)
        self.assertIn("summary", result)
        self.assertEqual(result["pipeline_info"]["llm_used"], True)
        self.assertEqual(result["llm_insights"]["overall_assessment"]["quality_grade"], "A")

    @patch('main.Config.get_api_key', return_value='dummy_api_key')
    @patch('main.FileLoader')
    @patch('main.MetadataExtractor')
    @patch('main.LLMEnricher')
    def test_run_pipeline_fallback_on_llm_failure(self, mock_llm_enricher, mock_metadata_extractor, mock_file_loader, mock_get_api_key):
        # Arrange
        mock_file_loader.return_value.load_file.return_value = [{"field1": "value1"}]
        mock_metadata_extractor.return_value.extract_metadata.return_value = {
            "summary": {"total_records": 1},
            "tables": {"main_table": {"data_quality_score": 90}}
        }
        mock_llm_enricher.return_value.enrich_metadata.side_effect = Exception("API failure")

        agent = MetadataEnrichmentAgent(debug=False)

        # Act
        result = agent.run_pipeline("input/input_data.json", skip_llm=False)

        # Assert fallback enrichment is used
        self.assertIn("pipeline_info", result)
        self.assertIn("raw_metadata", result)
        self.assertIn("summary", result)
        self.assertIn("overall_assessment", result["llm_insights"])
        self.assertIn("quality_grade", result["llm_insights"]["overall_assessment"])

if __name__ == "__main__":
    unittest.main()
