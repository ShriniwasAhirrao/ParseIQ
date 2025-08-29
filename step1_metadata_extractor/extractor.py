import json
import re
import numpy as np
from collections import Counter, defaultdict
from typing import Any, Dict, List, Union, Set
from datetime import datetime
from .utils import StatisticalUtils

class MetadataExtractor:
    """Comprehensive metadata extraction with detailed nested attribute analysis"""
    
    def __init__(self):
        self.stats_utils = StatisticalUtils()
        self.regex_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
            'date_iso': r'^\d{4}-\d{2}-\d{2}$',
            'url': r'^https?://[^\s]+$',
            'ip_address': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$',
            'credit_card': r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$'
        }

    def extract_metadata(self, data: Union[List[Dict], Dict]) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from JSON data with deep nested analysis
        Args:
            data: JSON data (list of dicts or single dict)
        Returns:
            Comprehensive metadata dictionary with detailed table analysis
        """
        print(f"🔍 Starting enhanced metadata extraction...")
        
        # Check if input is a structured dataset with multiple tables
        if isinstance(data, dict) and self._is_multi_table_dataset(data):
            return self._extract_multi_table_metadata(data)
        else:
            # Handle as single table
            return self._extract_single_table_metadata(data)

    def _is_multi_table_dataset(self, data: Dict) -> bool:
        """Check if the data contains multiple tables (arrays of objects)"""
        array_fields = 0
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                array_fields += 1
        return array_fields > 1

    def _extract_multi_table_metadata(self, data: Dict) -> Dict[str, Any]:
        """Extract detailed metadata for multi-table datasets"""
        print(f"🗂️  Detected multi-table dataset with {len(data)} tables")
        
        metadata = {
            "extraction_timestamp": datetime.now().isoformat(),
            "dataset_type": "multi_table",
            "tables": {},
            "summary": {
                "total_tables": 0,
                "total_records": 0,
                "table_names": [],
                "table_record_counts": {}
            },
            "cross_table_analysis": {},
            "overall_quality_score": 0,
            "global_anomaly_summary": {
                "total_anomalies": 0,
                "affected_tables": 0,
                "anomaly_types": {},
                "critical_issues": []
            }
        }

        valid_tables = {}
        
        # Extract each table with detailed attribute analysis
        for table_name, table_data in data.items():
            if isinstance(table_data, list) and len(table_data) > 0:
                print(f"📊 Analyzing table: {table_name} ({len(table_data)} records)")
                
                # Extract detailed metadata for this table
                table_metadata = self._analyze_table_detailed(table_data, table_name)
                metadata["tables"][table_name] = table_metadata
                valid_tables[table_name] = table_data
                
                # Update summary
                metadata["summary"]["table_names"].append(table_name)
                metadata["summary"]["table_record_counts"][table_name] = len(table_data)
                metadata["summary"]["total_records"] += len(table_data)

        metadata["summary"]["total_tables"] = len(metadata["tables"])
        
        # Perform cross-table analysis if multiple tables exist
        if len(valid_tables) > 1:
            metadata["cross_table_analysis"] = self._analyze_cross_table_relationships(valid_tables)
        
        # Calculate overall quality score
        table_scores = [t["data_quality_score"] for t in metadata["tables"].values()]
        metadata["overall_quality_score"] = round(np.mean(table_scores), 2) if table_scores else 0
        
        # Aggregate anomaly summary
        metadata["global_anomaly_summary"] = self._aggregate_anomaly_summary(metadata["tables"])
        
        print(f"✅ Multi-table analysis completed")
        print(f"📈 Overall Quality Score: {metadata['overall_quality_score']:.2f}/100")
        print(f"⚠️  Total Anomalies: {metadata['global_anomaly_summary']['total_anomalies']}")
        
        return metadata

    def _analyze_table_detailed(self, records: List[Dict], table_name: str) -> Dict[str, Any]:
        """Analyze a single table with detailed attribute-level analysis"""
        print(f"  🔍 Deep analysis of {table_name} with detailed attribute extraction...")
        
        # Initialize table metadata structure
        table_metadata = {
            "table_name": table_name,
            "extraction_timestamp": datetime.now().isoformat(),
            "dataset_info": self._get_dataset_info(records),
            "attributes": {},  # This will contain detailed analysis of each attribute within this table
            "schema_analysis": {},
            "data_quality_score": 0,
            "anomaly_summary": {},
            "top_issues": [],
            "data_profiling": {
                "record_completeness": {},
                "duplicate_analysis": {},
                "data_freshness": {}
            }
        }
        
        # Extract all unique attributes from all records in this specific table
        all_attributes = self._collect_all_attributes_from_records(records)
        print(f"    📋 Found {len(all_attributes)} unique attributes in {table_name}: {list(all_attributes)}")
        
        # Analyze each attribute in detail
        for attr_name in all_attributes:
            print(f"    📊 Analyzing attribute: {table_name}.{attr_name}")
            attr_metadata = self._analyze_attribute(records, attr_name)
            table_metadata["attributes"][attr_name] = attr_metadata
        
        # Perform schema analysis
        table_metadata["schema_analysis"] = self._analyze_schema(records)
        
        # Perform additional data profiling
        table_metadata["data_profiling"] = self._perform_data_profiling(records)
        
        # Calculate overall data quality score
        table_metadata["data_quality_score"] = self._calculate_quality_score(table_metadata["attributes"])
        
        # Generate anomaly summary
        table_metadata["anomaly_summary"] = self._generate_anomaly_summary(table_metadata["attributes"])
        
        # Identify top issues
        table_metadata["top_issues"] = self._identify_top_issues(table_metadata["attributes"])
        
        print(f"    ✅ {table_name} analysis completed")
        print(f"    📈 Quality Score: {table_metadata['data_quality_score']:.2f}/100")
        print(f"    ⚠️  Anomalies: {table_metadata['anomaly_summary']['total_anomalies']}")
        
        return table_metadata

    def _collect_all_attributes_from_records(self, records: List[Dict]) -> Set[str]:
        """Collect all unique attribute names from a list of records (not table names)"""
        all_attributes = set()
        
        for record in records:
            if isinstance(record, dict):
                all_attributes.update(record.keys())
        
        return all_attributes

    def _extract_single_table_metadata(self, data: Union[List[Dict], Dict]) -> Dict[str, Any]:
        """Extract metadata for single table dataset"""
        # Normalize input to list
        if isinstance(data, dict):
            data = [data]
        
        print(f"📋 Analyzing single table with {len(data)} records...")
        table_metadata = self._analyze_table_detailed(data, "main_table")
        
        return {
            "extraction_timestamp": datetime.now().isoformat(),
            "dataset_type": "single_table",
            "table_metadata": table_metadata,
            "summary": {
                "total_records": len(data),
                "data_quality_score": table_metadata["data_quality_score"],
                "total_anomalies": table_metadata["anomaly_summary"]["total_anomalies"]
            }
        }

    def _perform_data_profiling(self, records: List[Dict]) -> Dict[str, Any]:
        """Perform additional data profiling on table records"""
        profiling = {
            "record_completeness": self._analyze_record_completeness(records),
            "duplicate_analysis": self._analyze_duplicates(records),
            "data_freshness": self._analyze_data_freshness(records)
        }
        return profiling

    def _analyze_record_completeness(self, records: List[Dict]) -> Dict[str, Any]:
        """Analyze completeness at record level within a table"""
        if not records:
            return {}
        
        all_fields = self._collect_all_attributes_from_records(records)
        completeness_scores = []
        
        for record in records:
            if isinstance(record, dict):
                filled_fields = sum(1 for field in all_fields 
                                if field in record and record[field] is not None and record[field] != "")
                score = (filled_fields / len(all_fields)) * 100 if all_fields else 0
                completeness_scores.append(score)
        
        return {
            "avg_completeness": round(np.mean(completeness_scores), 2) if completeness_scores else 0,
            "min_completeness": round(min(completeness_scores), 2) if completeness_scores else 0,
            "max_completeness": round(max(completeness_scores), 2) if completeness_scores else 0,
            "records_100_complete": sum(1 for s in completeness_scores if s == 100),
            "records_below_50_complete": sum(1 for s in completeness_scores if s < 50)
        }

    def _analyze_cross_table_relationships(self, tables: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze relationships between tables"""
        cross_analysis = {
            "potential_foreign_keys": {},
            "referential_integrity": {},
            "common_attributes": {},
            "relationship_suggestions": []
        }
        
        table_names = list(tables.keys())
        
        # Find potential relationships between tables
        for i, table1_name in enumerate(table_names):
            for j, table2_name in enumerate(table_names[i+1:], i+1):
                table1_data = tables[table1_name]
                table2_data = tables[table2_name]
                
                # Get attributes from both tables
                table1_attrs = self._collect_all_attributes_from_records(table1_data)
                table2_attrs = self._collect_all_attributes_from_records(table2_data)
                
                # Find common attributes
                common_attrs = table1_attrs.intersection(table2_attrs)
                if common_attrs:
                    cross_analysis["common_attributes"][f"{table1_name}__{table2_name}"] = list(common_attrs)
                
                # Look for potential foreign key relationships
                potential_fks = self._find_potential_foreign_keys(table1_name, table1_data, table2_name, table2_data)
                if potential_fks:
                    cross_analysis["potential_foreign_keys"][f"{table1_name}__{table2_name}"] = potential_fks
        
        return cross_analysis

    def _find_potential_foreign_keys(self, table1_name: str, table1_data: List[Dict], 
                                    table2_name: str, table2_data: List[Dict]) -> List[Dict]:
        """Find potential foreign key relationships between two tables"""
        potential_fks = []
        
        table1_attrs = self._collect_all_attributes_from_records(table1_data)
        table2_attrs = self._collect_all_attributes_from_records(table2_data)
        
        # Look for ID-like fields that might be foreign keys
        for attr1 in table1_attrs:
            if 'id' in attr1.lower():
                for attr2 in table2_attrs:
                    if 'id' in attr2.lower() and attr1 != attr2:
                        # Check if values in attr1 appear in attr2 (potential FK relationship)
                        table1_values = {record.get(attr1) for record in table1_data if record.get(attr1)}
                        table2_values = {record.get(attr2) for record in table2_data if record.get(attr2)}
                        
                        if table1_values and table2_values:
                            overlap = table1_values.intersection(table2_values)
                            if overlap:
                                potential_fks.append({
                                    "source_table": table1_name,
                                    "source_field": attr1,
                                    "target_table": table2_name,
                                    "target_field": attr2,
                                    "matching_values": len(overlap),
                                    "confidence": len(overlap) / max(len(table1_values), len(table2_values))
                                })
        
        return potential_fks

    def _analyze_duplicates(self, records: List[Dict]) -> Dict[str, Any]:
        """Analyze duplicate records"""
        if not records:
            return {}
        
        # Convert records to strings for comparison
        record_strings = []
        for record in records:
            if isinstance(record, dict):
                # Sort keys for consistent comparison
                sorted_items = sorted(record.items())
                record_strings.append(str(sorted_items))
        
        duplicate_count = len(record_strings) - len(set(record_strings))
        
        return {
            "total_duplicates": duplicate_count,
            "duplicate_percentage": round((duplicate_count / len(records)) * 100, 2) if records else 0,
            "unique_records": len(set(record_strings))
        }

    def _analyze_data_freshness(self, records: List[Dict]) -> Dict[str, Any]:
        """Analyze data freshness based on date fields"""
        date_fields = []
        
        # Find potential date fields
        for record in records[:10]:  # Sample first 10 records
            if isinstance(record, dict):
                for key, value in record.items():
                    if isinstance(value, str) and ('date' in key.lower() or re.match(r'^\d{4}-\d{2}-\d{2}', str(value))):
                        date_fields.append(key)
        
        date_fields = list(set(date_fields))
        freshness_analysis = {}
        
        for field in date_fields:
            dates = []
            for record in records:
                if isinstance(record, dict) and field in record:
                    try:
                        date_str = str(record[field])
                        if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
                            dates.append(date_str)
                    except:
                        continue
            
            if dates:
                freshness_analysis[field] = {
                    "earliest_date": min(dates),
                    "latest_date": max(dates),
                    "date_range_days": self._calculate_date_range(min(dates), max(dates))
                }
        
        return {
            "date_fields_found": date_fields,
            "field_analysis": freshness_analysis
        }

    def _calculate_date_range(self, start_date: str, end_date: str) -> int:
        """Calculate days between two dates"""
        try:
            from datetime import datetime
            start = datetime.strptime(start_date[:10], '%Y-%m-%d')
            end = datetime.strptime(end_date[:10], '%Y-%m-%d')
            return (end - start).days
        except:
            return 0

    def _aggregate_anomaly_summary(self, tables: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate anomalies across all tables"""
        total_anomalies = 0
        affected_tables = 0
        all_anomaly_types = Counter()
        critical_issues = []
        
        for table_name, table_data in tables.items():
            table_anomalies = table_data["anomaly_summary"]["total_anomalies"]
            if table_anomalies > 0:
                affected_tables += 1
            total_anomalies += table_anomalies
            
            # Aggregate anomaly types
            for anomaly_type, count in table_data["anomaly_summary"]["anomaly_types"].items():
                all_anomaly_types[anomaly_type] += count
            
            # Collect critical issues
            for issue in table_data["top_issues"]:
                critical_issues.append(f"{table_name}: {issue}")
        
        return {
            "total_anomalies": total_anomalies,
            "affected_tables": affected_tables,
            "anomaly_types": dict(all_anomaly_types),
            "critical_issues": critical_issues[:10]  # Top 10 critical issues
        }

    def _get_dataset_info(self, data: List[Dict]) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            "total_records": len(data),
            "total_attributes": len(self._collect_all_attributes(data)),
            "memory_usage_bytes": len(json.dumps(data, default=str).encode('utf-8')),
            "has_nested_objects": any(
                any(isinstance(v, (dict, list)) for v in record.values())
                for record in data if isinstance(record, dict)
            )
        }

    def _collect_all_attributes(self, data: List[Dict]) -> set:
        """Collect all unique attribute names from the dataset"""
        attributes = set()
        for record in data:
            if isinstance(record, dict):
                attributes.update(record.keys())
        return attributes

    def _analyze_attribute(self, data: List[Dict], attr_name: str) -> Dict[str, Any]:
        """Analyze a single attribute comprehensively"""
        values = []
        null_count = 0
        
        # Extract values
        for record in data:
            if isinstance(record, dict):
                if attr_name in record:
                    value = record[attr_name]
                    if value is None or value == "":
                        null_count += 1
                    else:
                        values.append(value)
                else:
                    null_count += 1
        
        total_records = len(data)
        present_count = len(values)
        null_percentage = (null_count / total_records) * 100 if total_records > 0 else 0
        
        # Determine data type
        data_type = self._determine_data_type(values)
        
        # Base metadata
        attr_metadata = {
            "data_type": data_type,
            "present_count": present_count,
            "null_count": null_count,
            "null_percentage": round(null_percentage, 2),
            "unique_count": len(set(str(v) for v in values)) if values else 0,
            "unique_ratio": round(len(set(str(v) for v in values)) / len(values), 3) if values else 0
        }
        
        # Type-specific analysis
        if data_type == "string":
            attr_metadata.update(self._analyze_string_attribute(values))
        elif data_type in ["integer", "float", "numeric"]:
            attr_metadata.update(self._analyze_numeric_attribute(values))
        elif data_type == "boolean":
            attr_metadata.update(self._analyze_boolean_attribute(values))
        elif data_type == "array":
            attr_metadata.update(self._analyze_array_attribute(values))
        elif data_type == "object":
            attr_metadata.update(self._analyze_object_attribute(values))
        
        # Pattern analysis
        attr_metadata["pattern_analysis"] = self._analyze_patterns(values)
        
        # Anomaly detection
        attr_metadata["anomaly_flags"] = self._detect_anomalies(attr_metadata, values)
        
        # Quality assessment
        attr_metadata["quality_score"] = self._calculate_attribute_quality_score(attr_metadata)
        
        return attr_metadata

    def _determine_data_type(self, values: List[Any]) -> str:
        """Determine the predominant data type of values"""
        if not values:
            return "unknown"
        
        type_counts = defaultdict(int)
        
        for value in values:
            if isinstance(value, bool):
                type_counts["boolean"] += 1
            elif isinstance(value, int):
                type_counts["integer"] += 1
            elif isinstance(value, float):
                type_counts["float"] += 1
            elif isinstance(value, str):
                # Try to determine if it's a numeric string
                try:
                    float(value)
                    if '.' in value:
                        type_counts["float"] += 1
                    else:
                        type_counts["integer"] += 1
                except ValueError:
                    type_counts["string"] += 1
            elif isinstance(value, list):
                type_counts["array"] += 1
            elif isinstance(value, dict):
                type_counts["object"] += 1
            else:
                type_counts["other"] += 1
        
        # Return the most common type
        return max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "unknown"

    def _analyze_string_attribute(self, values: List[str]) -> Dict[str, Any]:
        """Analyze string attribute"""
        if not values:
            return {}
        
        str_values = [str(v) for v in values]
        lengths = [len(s) for s in str_values]
        
        return {
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": round(np.mean(lengths), 2),
            "median_length": round(np.median(lengths), 2),
            "std_length": round(np.std(lengths), 2),
            "length_percentiles": {
                "25th": int(np.percentile(lengths, 25)),
                "75th": int(np.percentile(lengths, 75)),
                "95th": int(np.percentile(lengths, 95))
            },
            "common_values": dict(Counter(str_values).most_common(5)),
            "charset_analysis": self._analyze_charset(str_values)
        }

    def _analyze_numeric_attribute(self, values: List[Union[int, float]]) -> Dict[str, Any]:
        """Analyze numeric attribute"""
        if not values:
            return {}
        
        # Convert to numeric
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                continue
        
        if not numeric_values:
            return {}
        
        np_values = np.array(numeric_values)
        
        analysis = {
            "min_value": float(np.min(np_values)),
            "max_value": float(np.max(np_values)),
            "mean": round(float(np.mean(np_values)), 4),
            "median": round(float(np.median(np_values)), 4),
            "std": round(float(np.std(np_values)), 4),
            "variance": round(float(np.var(np_values)), 4),
            "percentiles": {
                "1st": round(float(np.percentile(np_values, 1)), 4),
                "5th": round(float(np.percentile(np_values, 5)), 4),
                "25th": round(float(np.percentile(np_values, 25)), 4),
                "75th": round(float(np.percentile(np_values, 75)), 4),
                "95th": round(float(np.percentile(np_values, 95)), 4),
                "99th": round(float(np.percentile(np_values, 99)), 4)
            }
        }
        
        # Outlier detection
        analysis["outliers"] = self.stats_utils.detect_outliers(numeric_values)
        
        return analysis

    def _analyze_boolean_attribute(self, values: List[bool]) -> Dict[str, Any]:
        """Analyze boolean attribute"""
        if not values:
            return {}
        
        bool_values = [bool(v) for v in values]
        true_count = sum(bool_values)
        false_count = len(bool_values) - true_count
        
        return {
            "true_count": true_count,
            "false_count": false_count,
            "true_percentage": round((true_count / len(bool_values)) * 100, 2),
            "false_percentage": round((false_count / len(bool_values)) * 100, 2)
        }

    def _analyze_array_attribute(self, values: List[list]) -> Dict[str, Any]:
        """Analyze array attribute"""
        if not values:
            return {}
        
        lengths = [len(v) if isinstance(v, list) else 0 for v in values]
        
        return {
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": round(np.mean(lengths), 2),
            "median_length": round(np.median(lengths), 2),
            "empty_arrays": lengths.count(0),
            "element_types": self._analyze_array_element_types(values)
        }

    def _analyze_object_attribute(self, values: List[dict]) -> Dict[str, Any]:
        """Analyze object attribute"""
        if not values:
            return {}
        
        all_keys = set()
        key_frequencies = Counter()
        
        for obj in values:
            if isinstance(obj, dict):
                keys = set(obj.keys())
                all_keys.update(keys)
                key_frequencies.update(keys)
        
        return {
            "unique_keys": len(all_keys),
            "common_keys": dict(key_frequencies.most_common(10)),
            "avg_keys_per_object": round(np.mean([len(obj.keys()) if isinstance(obj, dict) else 0 for obj in values]), 2),
            "key_consistency": round(len(key_frequencies) / len(all_keys) if all_keys else 0, 3)
        }

    def _analyze_charset(self, str_values: List[str]) -> Dict[str, Any]:
        """Analyze character set usage in strings"""
        char_types = {
            'alphabetic': 0,
            'numeric': 0,
            'alphanumeric': 0,
            'special_chars': 0,
            'whitespace': 0,
            'unicode': 0
        }
        
        for s in str_values:
            for char in s:
                if char.isalpha():
                    char_types['alphabetic'] += 1
                elif char.isdigit():
                    char_types['numeric'] += 1
                elif char.isalnum():
                    char_types['alphanumeric'] += 1
                elif char.isspace():
                    char_types['whitespace'] += 1
                elif ord(char) > 127:
                    char_types['unicode'] += 1
                else:
                    char_types['special_chars'] += 1
        
        total_chars = sum(char_types.values())
        if total_chars == 0:
            return char_types
        
        # Convert to percentages
        return {k: round((v / total_chars) * 100, 2) for k, v in char_types.items()}

    def _analyze_array_element_types(self, values: List[list]) -> Dict[str, int]:
        """Analyze types of elements in arrays"""
        element_types = Counter()
        
        for arr in values:
            if isinstance(arr, list):
                for element in arr:
                    element_types[type(element).__name__] += 1
        
        return dict(element_types.most_common())

    def _analyze_patterns(self, values: List[Any]) -> Dict[str, Any]:
        """Analyze patterns in values using regex"""
        if not values:
            return {}
        
        str_values = [str(v) for v in values if v is not None]
        pattern_matches = {}
        
        for pattern_name, pattern in self.regex_patterns.items():
            matches = [bool(re.match(pattern, s)) for s in str_values]
            match_count = sum(matches)
            pattern_matches[pattern_name] = {
                "matches": match_count,
                "percentage": round((match_count / len(str_values)) * 100, 2) if str_values else 0
            }
        
        return {
            "regex_patterns": pattern_matches,
            "dominant_pattern": max(pattern_matches.items(), key=lambda x: x[1]["matches"])[0] if pattern_matches else None
        }

    def _detect_anomalies(self, attr_metadata: Dict[str, Any], values: List[Any]) -> List[str]:
        """Detect anomalies in attribute data"""
        anomalies = []
        
        # High null rate
        if attr_metadata.get("null_percentage", 0) > 30:
            anomalies.append("HIGH_NULL_RATE")
        
        # String length anomalies
        if attr_metadata.get("data_type") == "string":
            avg_length = attr_metadata.get("avg_length", 0)
            if avg_length < 2:
                anomalies.append("AVG_LENGTH_TOO_SHORT")
            elif avg_length > 500:
                anomalies.append("AVG_LENGTH_TOO_LONG")
        
        # Numeric anomalies
        if attr_metadata.get("data_type") in ["integer", "float", "numeric"]:
            outliers = attr_metadata.get("outliers", {})
            if outliers.get("count", 0) > 0:
                anomalies.append("NUMERIC_OUTLIERS_DETECTED")
            
            # Check for negative values where they shouldn't be
            if "min_value" in attr_metadata and attr_metadata["min_value"] < 0:
                anomalies.append("NEGATIVE_VALUES_DETECTED")
        
        # Low uniqueness
        unique_ratio = attr_metadata.get("unique_ratio", 0)
        if unique_ratio < 0.1 and len(values) > 10:
            anomalies.append("LOW_UNIQUENESS")
        
        # Pattern inconsistency
        pattern_analysis = attr_metadata.get("pattern_analysis", {})
        if pattern_analysis.get("regex_patterns"):
            max_pattern_pct = max(
                p["percentage"] for p in pattern_analysis["regex_patterns"].values()
            )
            if max_pattern_pct > 50 and max_pattern_pct < 90:
                anomalies.append("PATTERN_INCONSISTENCY")
        
        return anomalies

    def _calculate_attribute_quality_score(self, attr_metadata: Dict[str, Any]) -> float:
        """Calculate quality score for an attribute (0-100)"""
        score = 100.0
        
        # Penalize high null rates
        null_pct = attr_metadata.get("null_percentage", 0)
        score -= min(null_pct * 0.5, 30)  # Max 30 point penalty
        
        # Penalize anomalies
        anomaly_count = len(attr_metadata.get("anomaly_flags", []))
        score -= anomaly_count * 10  # 10 points per anomaly
        
        # Reward good uniqueness (but not too high for some cases)
        unique_ratio = attr_metadata.get("unique_ratio", 0)
        if 0.1 <= unique_ratio <= 0.9:
            score += 10
        elif unique_ratio > 0.95:
            score += 5  # Very high uniqueness might indicate good data but could also be IDs
        
        # Ensure score is within bounds
        return max(0.0, min(100.0, round(score, 2)))

    def _analyze_schema(self, records: List[Dict]) -> Dict[str, Any]:
        """Analyze schema consistency and structure"""
        if not records:
            return {}
        
        # Collect schema variations
        schema_variations = []
        attribute_presence = defaultdict(int)
        
        for record in records:
            if isinstance(record, dict):
                keys = set(record.keys())
                schema_variations.append(keys)
                for key in keys:
                    attribute_presence[key] += 1
        
        # Find common schema patterns
        schema_counter = Counter()
        for schema in schema_variations:
            schema_counter[frozenset(schema)] += 1
        
        total_records = len(records)
        most_common_schema = schema_counter.most_common(1)[0] if schema_counter else (set(), 0)
        
        return {
            "schema_consistency": round((most_common_schema[1] / total_records) * 100, 2) if total_records > 0 else 0,
            "unique_schemas": len(schema_counter),
            "most_common_schema_attributes": list(most_common_schema[0]),
            "attribute_presence_rates": {
                attr: round((count / total_records) * 100, 2)
                for attr, count in attribute_presence.items()
            },
            "optional_attributes": [
                attr for attr, count in attribute_presence.items()
                if count < total_records * 0.9  # Less than 90% presence
            ],
            "required_attributes": [
                attr for attr, count in attribute_presence.items()
                if count == total_records  # 100% presence
            ]
        }

    def _calculate_quality_score(self, attributes: Dict[str, Any]) -> float:
        """Calculate overall quality score for the table"""
        if not attributes:
            return 0.0
        
        # Get individual attribute scores
        attr_scores = [attr_data.get("quality_score", 0) for attr_data in attributes.values()]
        
        # Calculate weighted average (could be enhanced with attribute importance weights)
        base_score = np.mean(attr_scores) if attr_scores else 0
        
        # Apply table-level adjustments
        total_attributes = len(attributes)
        
        # Bonus for having a good number of attributes
        if 5 <= total_attributes <= 20:
            base_score += 5
        elif total_attributes > 50:
            base_score -= 5  # Too many attributes might indicate poor design
        
        return max(0.0, min(100.0, round(base_score, 2)))

    def _generate_anomaly_summary(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of anomalies across all attributes"""
        anomaly_types = Counter()
        total_anomalies = 0
        affected_attributes = 0
        
        for attr_name, attr_data in attributes.items():
            anomaly_flags = attr_data.get("anomaly_flags", [])
            if anomaly_flags:
                affected_attributes += 1
                total_anomalies += len(anomaly_flags)
                for anomaly in anomaly_flags:
                    anomaly_types[anomaly] += 1
        
        return {
            "total_anomalies": total_anomalies,
            "affected_attributes": affected_attributes,
            "anomaly_types": dict(anomaly_types),
            "anomaly_rate": round((affected_attributes / len(attributes)) * 100, 2) if attributes else 0
        }

    def _identify_top_issues(self, attributes: Dict[str, Any]) -> List[str]:
        """Identify the top data quality issues"""
        issues = []
        
        for attr_name, attr_data in attributes.items():
            # High null rate issues
            null_pct = attr_data.get("null_percentage", 0)
            if null_pct > 50:
                issues.append(f"'{attr_name}' has very high null rate ({null_pct:.1f}%)")
            elif null_pct > 30:
                issues.append(f"'{attr_name}' has high null rate ({null_pct:.1f}%)")
            
            # Low quality score
            quality_score = attr_data.get("quality_score", 100)
            if quality_score < 50:
                issues.append(f"'{attr_name}' has low quality score ({quality_score:.1f}/100)")
            
            # Specific anomalies
            anomalies = attr_data.get("anomaly_flags", [])
            for anomaly in anomalies:
                if anomaly == "HIGH_NULL_RATE":
                    continue  # Already covered above
                elif anomaly == "NUMERIC_OUTLIERS_DETECTED":
                    outlier_count = attr_data.get("outliers", {}).get("count", 0)
                    issues.append(f"'{attr_name}' has {outlier_count} numeric outliers")
                elif anomaly == "PATTERN_INCONSISTENCY":
                    issues.append(f"'{attr_name}' has inconsistent data patterns")
                elif anomaly == "LOW_UNIQUENESS":
                    unique_ratio = attr_data.get("unique_ratio", 0)
                    issues.append(f"'{attr_name}' has low uniqueness ({unique_ratio:.2f})")
                else:
                    issues.append(f"'{attr_name}' has {anomaly.lower().replace('_', ' ')}")
        
        # Sort by severity and return top 10
        return sorted(issues)[:10]

    def export_metadata_report(self, metadata: Dict[str, Any], format: str = "json") -> str:
        """Export metadata as formatted report"""
        if format.lower() == "json":
            return json.dumps(metadata, indent=2, default=str)
        elif format.lower() == "summary":
            return self._generate_summary_report(metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_summary_report(self, metadata: Dict[str, Any]) -> str:
        """Generate a human-readable summary report"""
        report = []
        report.append("=" * 60)
        report.append("DATA QUALITY METADATA REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {metadata.get('extraction_timestamp', 'N/A')}")
        report.append(f"Dataset Type: {metadata.get('dataset_type', 'N/A').upper()}")
        report.append("")
        
        if metadata.get("dataset_type") == "multi_table":
            # Multi-table summary
            summary = metadata.get("summary", {})
            report.append(f"📊 DATASET OVERVIEW")
            report.append(f"   Total Tables: {summary.get('total_tables', 0)}")
            report.append(f"   Total Records: {summary.get('total_records', 0):,}")
            report.append(f"   Overall Quality Score: {metadata.get('overall_quality_score', 0):.1f}/100")
            report.append("")
            
            # Table breakdown
            report.append(f"📋 TABLE BREAKDOWN")
            for table_name in summary.get('table_names', []):
                record_count = summary.get('table_record_counts', {}).get(table_name, 0)
                table_data = metadata.get('tables', {}).get(table_name, {})
                quality_score = table_data.get('data_quality_score', 0)
                report.append(f"   • {table_name}: {record_count:,} records (Quality: {quality_score:.1f}/100)")
            report.append("")
            
            # Global issues
            anomaly_summary = metadata.get("global_anomaly_summary", {})
            if anomaly_summary.get("total_anomalies", 0) > 0:
                report.append(f"⚠️  QUALITY ISSUES")
                report.append(f"   Total Anomalies: {anomaly_summary.get('total_anomalies', 0)}")
                report.append(f"   Affected Tables: {anomaly_summary.get('affected_tables', 0)}")
                
                # Top issues
                critical_issues = anomaly_summary.get("critical_issues", [])
                if critical_issues:
                    report.append(f"   Critical Issues:")
                    for issue in critical_issues[:5]:  # Top 5
                        report.append(f"     - {issue}")
                report.append("")
        
        else:
            # Single table summary
            table_metadata = metadata.get("table_metadata", {})
            dataset_info = table_metadata.get("dataset_info", {})
            
            report.append(f"📊 DATASET OVERVIEW")
            report.append(f"   Records: {dataset_info.get('total_records', 0):,}")
            report.append(f"   Attributes: {dataset_info.get('total_attributes', 0)}")
            report.append(f"   Quality Score: {table_metadata.get('data_quality_score', 0):.1f}/100")
            report.append(f"   Memory Usage: {dataset_info.get('memory_usage_bytes', 0):,} bytes")
            report.append("")
            
            # Top issues
            top_issues = table_metadata.get("top_issues", [])
            if top_issues:
                report.append(f"⚠️  TOP QUALITY ISSUES")
                for i, issue in enumerate(top_issues[:5], 1):
                    report.append(f"   {i}. {issue}")
                report.append("")
        
        report.append("=" * 60)
        return "\n".join(report)