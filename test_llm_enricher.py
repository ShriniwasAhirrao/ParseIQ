import unittest
from unittest.mock import patch, MagicMock
from step2_llm_enricher.llm_agent import LLMEnricher

class TestLLMEnricher(unittest.TestCase):
    def setUp(self):
        config = {
            'api_key': 'testkey',
            'base_url': 'https://api.test.com',
            'model': 'test-model',
            'max_tokens': 1000,
            'temperature': 0.5,
            'debug': False,
            'prompt_template_path': 'step2_llm_enricher/prompt_template.txt'
        }
        self.enricher = LLMEnricher(config)

    @patch('step2_llm_enricher.llm_agent.open', create=True)
    def test_load_prompt_template_success(self, mock_open):
        mock_open.return_value.__enter__.return_value.read.return_value = "Prompt {metadata_summary}"
        prompt = self.enricher._load_prompt_template()
        self.assertIn("Prompt", prompt)

    def test_is_valid_email_format(self):
        self.assertTrue(self.enricher._is_valid_email_format("test@example.com"))
        self.assertFalse(self.enricher._is_valid_email_format("invalid-email"))

    def test_parse_date(self):
        date = self.enricher._parse_date("2023-01-01")
        self.assertIsNotNone(date)
        self.assertEqual(str(date), "2023-01-01")

    @patch('step2_llm_enricher.llm_agent.requests.post')
    def test_make_api_request_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'choices': [{'message': {'content': '{"status":"ok"}'}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        response = self.enricher._make_api_request("test prompt")
        self.assertIn("status", response)

    def test_test_connection_success(self):
        with patch.object(self.enricher, '_make_api_request', return_value='{"status": "ok", "test": true}'):
            self.assertTrue(self.enricher.test_connection())

if __name__ == "__main__":
    unittest.main()
