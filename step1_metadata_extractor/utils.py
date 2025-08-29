import numpy as np
from scipy import stats
from typing import List, Dict, Any, Union, Tuple


class StatisticalUtils:
    """Statistical analysis utilities for metadata extraction"""
    
    def __init__(self):
        self.z_score_threshold = 3.0
        self.iqr_multiplier = 1.5
    
    def detect_outliers(self, values: List[Union[int, float]]) -> Dict[str, Any]:
        """
        Detect outliers using multiple methods
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary with outlier information
        """
        if not values or len(values) < 3:
            return {"count": 0, "indices": [], "values": [], "methods": {}}
        
        np_values = np.array([float(v) for v in values])
        
        # Z-score method
        z_scores = np.abs(stats.zscore(np_values))
        z_outliers = np.where(z_scores > self.z_score_threshold)[0]
        
        # IQR method
        q1, q3 = np.percentile(np_values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (self.iqr_multiplier * iqr)
        upper_bound = q3 + (self.iqr_multiplier * iqr)
        iqr_outliers = np.where((np_values < lower_bound) | (np_values > upper_bound))[0]
        
        # Modified Z-score method (using median)
        median = np.median(np_values)
        mad = np.median(np.abs(np_values - median))  # Median Absolute Deviation
        modified_z_scores = 0.6745 * (np_values - median) / mad if mad != 0 else np.zeros_like(np_values)
        modified_z_outliers = np.where(np.abs(modified_z_scores) > 3.5)[0]
        
        # Combine all methods
        all_outliers = np.unique(np.concatenate([z_outliers, iqr_outliers, modified_z_outliers]))
        
        return {
            "count": len(all_outliers),
            "indices": all_outliers.tolist(),
            "values": np_values[all_outliers].tolist(),
            "methods": {
                "z_score": {
                    "count": len(z_outliers),
                    "threshold": self.z_score_threshold,
                    "indices": z_outliers.tolist()
                },
                "iqr": {
                    "count": len(iqr_outliers),
                    "lower_bound": round(lower_bound, 4),
                    "upper_bound": round(upper_bound, 4),
                    "indices": iqr_outliers.tolist()
                },
                "modified_z_score": {
                    "count": len(modified_z_outliers),
                    "threshold": 3.5,
                    "indices": modified_z_outliers.tolist()
                }
            }
        }
    
    def calculate_distribution_stats(self, values: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate comprehensive distribution statistics
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary with distribution statistics
        """
        if not values:
            return {}
        
        np_values = np.array([float(v) for v in values])
        
        return {
            "mean": round(float(np.mean(np_values)), 4),
            "median": round(float(np.median(np_values)), 4),
            "mode": round(float(stats.mode(np_values)[0]), 4) if len(np_values) > 1 else float(np_values[0]),
            "std": round(float(np.std(np_values)), 4),
            "variance": round(float(np.var(np_values)), 4),
            "skewness": round(float(stats.skew(np_values)), 4),
            "kurtosis": round(float(stats.kurtosis(np_values)), 4),
            "range": round(float(np.max(np_values) - np.min(np_values)), 4),
            "coefficient_of_variation": round(float(np.std(np_values) / np.mean(np_values)), 4) if np.mean(np_values) != 0 else 0
        }
    
    def detect_distribution_type(self, values: List[Union[int, float]]) -> Dict[str, Any]:
        """
        Attempt to identify the distribution type of the data
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary with distribution analysis
        """
        if not values or len(values) < 10:
            return {"likely_distribution": "insufficient_data", "confidence": 0}
        
        np_values = np.array([float(v) for v in values])
        
        # Test for normality
        shapiro_stat, shapiro_p = stats.shapiro(np_values) if len(np_values) <= 5000 else (0, 1)
        
        # Test for different distributions
        distributions = {
            'normal': lambda x: stats.normaltest(x)[1],
            'exponential': lambda x: stats.kstest(x, 'expon')[1],
            'uniform': lambda x: stats.kstest(x, 'uniform')[1]
        }
        
        dist_results = {}
        for dist_name, test_func in distributions.items():
            try:
                p_value = test_func(np_values)
                dist_results[dist_name] = p_value
            except:
                dist_results[dist_name] = 0
        
        # Find the distribution with highest p-value (best fit)
        best_dist = max(dist_results.items(), key=lambda x: x[1])
        
        return {
            "likely_distribution": best_dist[0],
            "confidence": round(best_dist[1], 4),
            "shapiro_wilk": {
                "statistic": round(shapiro_stat, 4),
                "p_value": round(shapiro_p, 4),
                "is_normal": shapiro_p > 0.05
            },
            "all_tests": {k: round(v, 4) for k, v in dist_results.items()}
        }
    
    def calculate_percentiles(self, values: List[Union[int, float]], 
                            percentiles: List[int] = None) -> Dict[str, float]:
        """
        Calculate specified percentiles
        
        Args:
            values: List of numeric values
            percentiles: List of percentile values to calculate
            
        Returns:
            Dictionary with percentile values
        """
        if not values:
            return {}
        
        if percentiles is None:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        
        np_values = np.array([float(v) for v in values])
        
        result = {}
        for p in percentiles:
            result[f"{p}th"] = round(float(np.percentile(np_values, p)), 4)
        
        return result
    
    def analyze_string_lengths(self, strings: List[str]) -> Dict[str, Any]:
        """
        Analyze string length distribution
        
        Args:
            strings: List of strings
            
        Returns:
            Dictionary with string length analysis
        """
        if not strings:
            return {}
        
        lengths = [len(s) for s in strings]
        
        return {
            "length_stats": self.calculate_distribution_stats(lengths),
            "length_percentiles": self.calculate_percentiles(lengths),
            "length_outliers": self.detect_outliers(lengths),
            "empty_strings": lengths.count(0),
            "single_char_strings": lengths.count(1)
        }
    
    def calculate_correlation_matrix(self, data_dict: Dict[str, List[Union[int, float]]]) -> Dict[str, Any]:
        """
        Calculate correlation matrix for numeric attributes
        
        Args:
            data_dict: Dictionary mapping attribute names to numeric values
            
        Returns:
            Correlation analysis results
        """
        if len(data_dict) < 2:
            return {"message": "Need at least 2 numeric attributes for correlation"}
        
        # Filter numeric data
        numeric_data = {k: v for k, v in data_dict.items() if v and all(isinstance(x, (int, float)) for x in v)}
        
        if len(numeric_data) < 2:
            return {"message": "Need at least 2 numeric attributes with valid data"}
        
        # Create correlation matrix
        import pandas as pd
        df = pd.DataFrame(numeric_data)
        correlation_matrix = df.corr()
        
        # Find high correlations
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    high_correlations.append({
                        "attribute1": correlation_matrix.columns[i],
                        "attribute2": correlation_matrix.columns[j],
                        "correlation": round(corr_value, 4),
                        "strength": "strong" if abs(corr_value) > 0.8 else "moderate"
                    })
        
        return {
            "correlation_matrix": correlation_matrix.round(4).to_dict(),
            "high_correlations": high_correlations,
            "attributes_analyzed": list(numeric_data.keys())
        }
    
    def detect_time_series_patterns(self, values: List[Union[int, float]], 
                                  timestamps: List[str] = None) -> Dict[str, Any]:
        """
        Detect patterns in time series data
        
        Args:
            values: List of numeric values
            timestamps: Optional list of timestamps
            
        Returns:
            Time series analysis results
        """
        if not values or len(values) < 10:
            return {"message": "Insufficient data for time series analysis"}
        
        np_values = np.array([float(v) for v in values])
        
        # Basic trend analysis
        x = np.arange(len(np_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, np_values)
        
        # Detect seasonality (simple autocorrelation check)
        autocorr_results = []
        for lag in [7, 30, 365]:  # Weekly, monthly, yearly patterns
            if len(np_values) > lag:
                autocorr = np.corrcoef(np_values[:-lag], np_values[lag:])[0, 1]
                autocorr_results.append({
                    "lag": lag,
                    "autocorrelation": round(autocorr, 4) if not np.isnan(autocorr) else 0
                })
        
        return {
            "trend": {
                "slope": round(slope, 6),
                "r_squared": round(r_value**2, 4),
                "p_value": round(p_value, 6),
                "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            },
            "autocorrelation": autocorr_results,
            "volatility": round(float(np.std(np.diff(np_values))), 4),
            "change_points": self._detect_change_points(np_values)
        }
    
    def _detect_change_points(self, values: np.ndarray) -> List[int]:
        """
        Simple change point detection using variance
        """
        if len(values) < 20:
            return []
        
        change_points = []
        window_size = max(5, len(values) // 10)
        
        for i in range(window_size, len(values) - window_size):
            left_var = np.var(values[i-window_size:i])
            right_var = np.var(values[i:i+window_size])
            
            # If variance changes significantly
            if abs(left_var - right_var) > np.var(values) * 0.5:
                change_points.append(i)
        
        return change_points