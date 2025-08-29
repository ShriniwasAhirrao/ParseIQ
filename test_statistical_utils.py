import pytest
from step1_metadata_extractor.utils import StatisticalUtils

class TestStatisticalUtils:
    @pytest.fixture
    def stats_utils(self):
        return StatisticalUtils()

    def test_detect_outliers_empty(self, stats_utils):
        result = stats_utils.detect_outliers([])
        assert result['count'] == 0
        assert result['indices'] == []
        assert result['values'] == []

    def test_detect_outliers_known(self, stats_utils):
        values = [1, 2, 3, 100, 5, 6, 7]
        result = stats_utils.detect_outliers(values)
        assert result['count'] > 0
        assert 3 in result['indices']  # index of 100

    def test_calculate_distribution_stats_empty(self, stats_utils):
        result = stats_utils.calculate_distribution_stats([])
        assert result == {}

    def test_calculate_distribution_stats_known(self, stats_utils):
        values = [1, 2, 3, 4, 5]
        result = stats_utils.calculate_distribution_stats(values)
        assert 'mean' in result
        assert result['mean'] == 3.0

    def test_detect_distribution_type_insufficient(self, stats_utils):
        result = stats_utils.detect_distribution_type([1, 2])
        assert result['likely_distribution'] == 'insufficient_data'

    def test_detect_distribution_type_normal(self, stats_utils):
        import numpy as np
        normal_data = np.random.normal(0, 1, 100).tolist()
        result = stats_utils.detect_distribution_type(normal_data)
        assert 'likely_distribution' in result

    def test_calculate_percentiles_default(self, stats_utils):
        values = list(range(1, 101))
        result = stats_utils.calculate_percentiles(values)
        assert '50th' in result
        # Allow small tolerance due to floating point differences
        assert abs(result['50th'] - 50.0) < 0.6

    def test_analyze_string_lengths_empty(self, stats_utils):
        result = stats_utils.analyze_string_lengths([])
        assert result == {}

    def test_analyze_string_lengths_known(self, stats_utils):
        strings = ["a", "ab", "abc", "abcd"]
        result = stats_utils.analyze_string_lengths(strings)
        assert 'length_stats' in result
        assert result['empty_strings'] == 0

    def test_calculate_correlation_matrix_insufficient(self, stats_utils):
        result = stats_utils.calculate_correlation_matrix({'a': [1, 2]})
        assert 'message' in result

    def test_calculate_correlation_matrix_known(self, stats_utils):
        data = {
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],
            'c': [5, 4, 3, 2, 1]
        }
        result = stats_utils.calculate_correlation_matrix(data)
        assert 'correlation_matrix' in result
        assert len(result['high_correlations']) > 0

    def test_detect_time_series_patterns_insufficient(self, stats_utils):
        result = stats_utils.detect_time_series_patterns([1, 2])
        assert 'message' in result

    def test_detect_time_series_patterns_known(self, stats_utils):
        values = list(range(100))
        result = stats_utils.detect_time_series_patterns(values)
        assert 'trend' in result
        assert 'autocorrelation' in result
        assert 'volatility' in result
