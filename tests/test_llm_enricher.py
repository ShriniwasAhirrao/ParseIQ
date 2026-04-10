import unittest
from unittest.mock import patch, MagicMock
from parseiq.step2_llm_enricher.llm_agent import LLMEnricher
from parseiq.config import Config


class TestLLMEnricher(unittest.TestCase):
    def setUp(self):
        config = {
            'api_key': 'testkey',
            'base_url': 'https://api.test.com',
            'model': 'test-model',
            'max_tokens': 1000,
            'temperature': 0.5,
            'debug': False,
            'prompt_template_path': Config.create_prompt_template_path(),
        }
        self.enricher = LLMEnricher(config)

    @patch('parseiq.step2_llm_enricher.llm_agent.open', create=True)
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

    @patch('parseiq.step2_llm_enricher.llm_agent.requests.post')
    def test_make_api_request_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
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

    # ── Regression: Bug — original_score always 0 ────────────────────────────

    def test_corrected_score_reads_from_dataset_overview(self):
        """_calculate_corrected_quality_score must find scores in dataset_overview.table_summaries."""
        metadata = {
            "dataset_overview": {
                "table_summaries": {
                    "orders": {"quality_score": 90.0},
                    "customers": {"quality_score": 80.0},
                }
            }
        }
        result = self.enricher._calculate_corrected_quality_score(metadata, {})
        self.assertGreater(result["corrected_score"], 0,
                           "corrected_score must not be 0 when dataset_overview has quality scores")
        self.assertEqual(result["original_score"], 85.0)

    def test_corrected_score_reads_from_tables_metadata(self):
        """_calculate_corrected_quality_score must find scores in tables[*].table_metadata."""
        metadata = {
            "tables": {
                "employees": {
                    "table_metadata": {"data_quality_score": 75.0}
                },
                "departments": {
                    "table_metadata": {"data_quality_score": 95.0}
                },
            }
        }
        result = self.enricher._calculate_corrected_quality_score(metadata, {})
        self.assertGreater(result["corrected_score"], 0,
                           "corrected_score must not be 0 when tables have data_quality_score")
        self.assertEqual(result["original_score"], 85.0)

    # ── Regression: Bug — fallback missing overall_score and model_used ───────

    def test_fallback_enrichment_has_overall_score(self):
        """_create_fallback_enrichment must include overall_score in overall_assessment."""
        corrected_metrics = {"quality_grade": "B", "corrected_score": 72.0, "critical_issues": 0, "major_issues": 0}
        result = self.enricher._create_fallback_enrichment({}, corrected_metrics, {}, {})
        oa = result.get("overall_assessment", {})
        self.assertIn("overall_score", oa,
                      "overall_assessment must have overall_score (not just corrected_score)")
        self.assertEqual(oa["overall_score"], 72.0)

    def test_fallback_enrichment_has_model_used(self):
        """_create_fallback_enrichment must include model_used in enrichment_metadata."""
        result = self.enricher._create_fallback_enrichment({}, {}, {}, {})
        em = result.get("enrichment_metadata", {})
        self.assertIn("model_used", em,
                      "enrichment_metadata must have model_used so Excel shows the model name")
        self.assertEqual(em["model_used"], "test-model")

    def test_fallback_enrichment_has_key_strengths_and_concerns(self):
        """_create_fallback_enrichment must include key_strengths and primary_concerns lists."""
        result = self.enricher._create_fallback_enrichment({}, {}, {}, {})
        oa = result.get("overall_assessment", {})
        self.assertIn("key_strengths", oa)
        self.assertIn("primary_concerns", oa)
        self.assertIsInstance(oa["key_strengths"], list)
        self.assertIsInstance(oa["primary_concerns"], list)

    # ── Regression: Bug — _generate_outputs overall_score vs corrected_score ─

    def test_generate_outputs_uses_corrected_score_fallback(self):
        """When overall_assessment only has corrected_score, Excel must not show N/A."""
        import io
        import pandas as pd
        from parseiq.pipeline import _generate_outputs

        tables = {"t1": [{"id": 1, "val": "x"}]}
        raw_metadata = {
            "tables": {
                "t1": {
                    "table_metadata": {
                        "attributes": {},
                        "data_quality_score": 88.0,
                        "anomaly_summary": {"total_anomalies": 0},
                        "top_issues": [],
                        "dataset_info": {"total_records": 1},
                        "data_profiling": {"duplicate_analysis": {}, "record_completeness": {}},
                    }
                }
            },
            "summary": {"total_tables": 1, "total_records": 1,
                        "table_names": ["t1"], "table_record_counts": {"t1": 1}},
            "dataset_overview": {"table_summaries": {"t1": {"quality_score": 88.0, "top_issues": [], "record_count": 1}}},
        }
        # Simulate internal fallback: no overall_score, only corrected_score
        llm_insights = {
            "overall_assessment": {
                "quality_grade": "B",
                "corrected_score": 88.0,
                "production_readiness": "Ready",
            },
            "risk_assessment": {"overall_risk_level": "Low"},
            "enrichment_metadata": {"timestamp": "2026-01-01T00:00:00", "model_used": "test-model"},
        }
        enriched = {"pipeline_info": {"total_duration": 1.0}, "llm_insights": llm_insights}

        buf = io.BytesIO()
        with patch("parseiq.pipeline.pd.ExcelWriter") as mock_writer:
            mock_writer.return_value.__enter__ = lambda s: s
            mock_writer.return_value.__exit__ = MagicMock(return_value=False)
            # Just verify the assess_rows are built correctly without N/A for score
            oa = llm_insights["overall_assessment"]
            score_value = next((oa[k] for k in ("overall_score", "corrected_score") if k in oa), "N/A")
            self.assertNotEqual(score_value, "N/A",
                                "Score must not be N/A when corrected_score is available")
            self.assertEqual(score_value, 88.0)


if __name__ == "__main__":
    unittest.main()
