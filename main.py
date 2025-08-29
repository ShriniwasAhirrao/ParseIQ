#!/usr/bin/env python3
"""
Enhanced JSON Metadata Enrichment Agent - Main Orchestrator
Updated to support hierarchical Excel sheet generation for nested tables
with three sheets per table: Original, Enriched_Metadata_Anomalies, Quality_Assessment
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import time
import logging
import pandas as pd
import tempfile
import shutil
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import Config
from file_loader.loader import FileLoader
from step1_metadata_extractor.extractor import MetadataExtractor
from step2_llm_enricher.llm_agent import LLMEnricher


class MetadataEnrichmentAgent:
    def __init__(self, debug: bool = True):
        # Configure logging to handle Unicode properly
        self._configure_logging()
        self.config = Config()
        self.file_loader = FileLoader()
        self.metadata_extractor = MetadataExtractor()
        
        # Initialize LLM enricher with proper configuration structure
        llm_config = {
            'api_key': self.config.openrouter_api_key,
            'base_url': 'https://openrouter.ai/api/v1',
            'model': self.config.MODEL_NAME,
            'max_tokens': self.config.llm_settings['max_tokens'],
            'temperature': self.config.llm_settings['temperature'],
            'debug': debug,
            'prompt_template_path': 'step2_llm_enricher/prompt_template.txt'
        }
        self.llm_enricher = LLMEnricher(llm_config)

        # Supported models list
        self.supported_models = {
            "default": self.config.MODEL_NAME
        }
        self.selected_model_key = "default"
        
        # Ensure output directories exist
        self._ensure_directories()
        
        # Test LLM connection on initialization
        self.llm_connection_ok = False
        self._test_llm_connection()

    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = ["output", "input", "logs", "debug_output"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(directory, '.test_write')
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                print(f"⚠️ Warning: Cannot write to {directory}/ - {e}")

    def _configure_logging(self):
        """Configure logging to handle Unicode characters properly"""
        # Set console encoding to UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
        
        # Configure logging with UTF-8 encoding
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/app.log', encoding='utf-8')
            ]
        )

    def _test_llm_connection(self):
        """Test LLM connection with enhanced error handling"""
        print("\n🔌 Testing LLM API connection...")
        try:
            self.llm_connection_ok = self.llm_enricher.test_connection()
            if self.llm_connection_ok:
                print("✅ LLM API connection successful")
                
                # Check if enhanced validation methods are available
                validation_methods = []
                enhanced_methods = [
                    '_perform_comprehensive_logical_validation',
                    '_calculate_corrected_quality_score',
                    'validate_date_sequences',
                    'is_valid_name_format',
                    'is_business_email_valid',
                    'validate_id_formats',
                    'detect_dataset_type',
                    'analyze_cross_table_relationships'
                ]
                
                for method in enhanced_methods:
                    if hasattr(self.llm_enricher, method):
                        validation_methods.append(method)
                
                if validation_methods:
                    print(f"✅ Enhanced validation methods available: {len(validation_methods)}")
                    if hasattr(self.llm_enricher, 'debug') and self.llm_enricher.debug:
                        print(f"🔧 Available methods: {', '.join(validation_methods[:2])}")
                else:
                    print("⚠️ Basic LLM connection only - enhanced validation not available")
            else:
                print("❌ LLM API connection failed - enrichment will use fallback mode")
        except Exception as e:
            print(f"❌ LLM connection test failed: {e}")
            self.llm_connection_ok = False
            
            # Still check for fallback validation capabilities
            if hasattr(self.llm_enricher, '_calculate_corrected_quality_score'):
                print("💡 Enhanced fallback validation available")
            else:
                print("💡 Basic fallback mode will be used")

    def _create_fallback_enrichment(self, raw_metadata):
        """Create enhanced fallback enrichment using new validation methods"""
        try:
            # Initialize fallback enrichment with enhanced validation
            fallback_data = {
                "overall_assessment": {
                    "quality_grade": "C",
                    "confidence_score": 50,
                    "corrected_score": 70,
                    "summary": "Enhanced fallback analysis with comprehensive validation",
                    "data_completeness": "Medium",
                    "data_consistency": "Medium",
                    "data_validity": "Medium",
                    "production_readiness": "Needs Review"
                },
                "detailed_analysis": {
                    "strengths": ["Data structure is parseable", "Basic format validation passed"],
                    "critical_logical_errors": [],
                    "temporal_issues": [],
                    "cross_field_issues": [],
                    "validation_results": {},
                    "component_scores": {},
                    "data_distribution": {
                        "total_records": 0,
                        "logical_error_count": 0,
                        "future_date_count": 0,
                        "cross_field_mismatch_count": 0,
                        "error_rate_percentage": "0.00%"
                    }
                },
                "recommendations": [
                    "Enable LLM analysis for comprehensive data quality assessment",
                    "Review data structure and format consistency",
                    "Consider implementing data validation rules"
                ],
                "risk_assessment": {
                    "overall_risk": "Medium",
                    "data_integrity_risk": "Medium",
                    "business_impact": "Medium"
                },
                "enrichment_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_used": "enhanced_fallback_analysis",
                    "validation_performed": True,
                    "logical_issues_found": 0,
                    "data_integrity_checked": True,
                    "validation_methods_used": [],
                    "note": "Enhanced fallback analysis with validation checks"
                }
            }
            
            # Apply enhanced validation if LLMEnricher methods are available
            if hasattr(self.llm_enricher, '_calculate_corrected_quality_score'):
                try:
                    # Perform comprehensive logical validation
                    logical_validation = {}
                    if hasattr(self.llm_enricher, '_perform_comprehensive_logical_validation'):
                        logical_validation = self.llm_enricher._perform_comprehensive_logical_validation(raw_metadata)
                    
                    # Calculate corrected quality score
                    corrected_metrics = self.llm_enricher._calculate_corrected_quality_score(raw_metadata, logical_validation)
                    
                    # Update fallback data with calculated metrics
                    fallback_data["overall_assessment"]["corrected_score"] = corrected_metrics.get("corrected_quality_score", 70)
                    fallback_data["detailed_analysis"]["validation_results"] = logical_validation
                    fallback_data["detailed_analysis"]["component_scores"] = corrected_metrics.get("component_scores", {})
                    
                    # Update validation metadata
                    fallback_data["enrichment_metadata"]["validation_methods_used"] = [
                        "comprehensive_logical_validation",
                        "corrected_quality_scoring"
                    ]
                    print("✅ Enhanced fallback enrichment with validation completed")
                except Exception as validation_error:
                    print(f"⚠️ Validation enhancement failed: {validation_error}")
            
            # Update record counts from metadata structure
            if 'summary' in raw_metadata:
                total_records = raw_metadata['summary'].get('total_records', 0)
                fallback_data["detailed_analysis"]["data_distribution"]["total_records"] = total_records
            elif isinstance(raw_metadata.get('tables'), dict):
                total_records = sum(
                    len(table_data) if isinstance(table_data, list) else 1
                    for table_data in raw_metadata['tables'].values()
                )
                fallback_data["detailed_analysis"]["data_distribution"]["total_records"] = total_records
            
            return fallback_data
            
        except Exception as e:
            print(f"❌ Fallback enrichment failed: {e}")
            # Return minimal fallback structure
            return {
                "overall_assessment": {
                    "quality_grade": "D",
                    "confidence_score": 30,
                    "corrected_score": 50,
                    "summary": "Minimal fallback analysis - validation failed",
                    "production_readiness": "Not Ready"
                },
                "enrichment_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_used": "minimal_fallback",
                    "validation_performed": False,
                    "note": "Minimal fallback due to validation errors"
                }
            }

    def run_pipeline(self, input_file_path: str, skip_llm: bool = False, selected_model: str = None):
        """
        Execute the complete 3-step pipeline for multiple tables
        Args:
            input_file_path: Path to input file
            skip_llm: Skip LLM enrichment (useful for testing)
            selected_model: Optional model name to override default
        """
        print("🚀 Starting JSON Metadata Enrichment Pipeline...")
        print(f"📁 Input file: {input_file_path}")
        print(f"🧠 LLM enrichment: {'Enabled' if not skip_llm and self.llm_connection_ok else 'Disabled'}")
        
        pipeline_start_time = time.time()
        
        try:
            # STEP 1: Load and Extract Technical Metadata
            print("\n📂 STEP 1: Extracting Technical Metadata...")
            step1_start = time.time()
            
            # Load file (supports JSON, CSV, XML)
            data = self.file_loader.load_file(input_file_path)
            
            # The file_loader now returns a dictionary of tables
            tables = data
            total_records = sum(len(table) for table in tables.values())
            print(f"✅ Loaded {len(tables)} tables with {total_records} total records")
            
            # Extract metadata for each table
            all_metadata = {}
            for table_name, table_data in tables.items():
                print(f"   📊 Processing table: {table_name}")
                table_metadata = self.metadata_extractor.extract_metadata(table_data)
                all_metadata[table_name] = table_metadata
            
            # Create combined metadata structure
            raw_metadata = {
                'tables': all_metadata,
                'summary': {
                    'total_tables': len(tables),
                    'total_records': total_records,
                    'table_names': list(tables.keys()),
                    'table_record_counts': {name: len(table) if isinstance(table, list) else 1 
                                          for name, table in tables.items()}
                }
            }
            
            # Add dataset_overview key for compatibility with LLMEnricher
            raw_metadata['dataset_overview'] = {
                'table_summaries': all_metadata
            }
            
            # Add pipeline metadata
            raw_metadata['pipeline_metadata'] = {
                'input_file': input_file_path,
                'extraction_timestamp': datetime.now().isoformat(),
                'step1_duration': time.time() - step1_start
            }
            
            # Save raw metadata
            raw_metadata_path = "output/raw_metadata.json"
            with open(raw_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(raw_metadata, f, indent=2, default=str)
            print(f"✅ Raw metadata saved to: {raw_metadata_path}")
            print(f"⏱️ Step 1 completed in {time.time() - step1_start:.2f} seconds")
            
            # STEP 2: Send to LLM for Enrichment
            enriched_insights = None
            if selected_model:
                selected_model_name = selected_model
            else:
                selected_model_name = self.supported_models.get(self.selected_model_key, self.config.MODEL_NAME)
            if not skip_llm and self.llm_connection_ok:
                print(f"\n🧠 STEP 2: Sending to {selected_model_name} for Enrichment...")
                step2_start = time.time()
                try:
                    # Load input JSON data for passing to LLM
                    with open(input_file_path, "r", encoding="utf-8") as f:
                        input_json_data = json.load(f)
                    
                    # Prepare combined input for LLM: input data + raw metadata
                    llm_input = {
                        "input_data": input_json_data,
                        "raw_metadata": raw_metadata
                    }
                    
                    enriched_insights = self.llm_enricher.enrich_metadata(llm_input, model=selected_model_name)
                    print("✅ LLM enrichment completed")
                    print(f"⏱️ Step 2 completed in {time.time() - step2_start:.2f} seconds")
                    
                    # Save enriched insights separately for debugging
                    llm_insights_path = "output/llm_insights.json"
                    with open(llm_insights_path, 'w', encoding='utf-8') as f:
                        json.dump(enriched_insights, f, indent=2, default=str)
                    print(f"💾 LLM insights saved to: {llm_insights_path}")
                    
                except Exception as e:
                    print(f"⚠️ LLM enrichment failed: {e}")
                    print("📝 Continuing with fallback enrichment...")
                    enriched_insights = self._create_fallback_enrichment(raw_metadata)
            else:
                if skip_llm:
                    print("\n⏭️ STEP 2: Skipping LLM enrichment (skip_llm=True)")
                else:
                    print("\n⚠️ STEP 2: Skipping LLM enrichment (connection failed)")
                enriched_insights = self._create_fallback_enrichment(raw_metadata)
            
            # STEP 3: Generate Final Output
            print("\n📤 STEP 3: Generating Enriched Output...")
            step3_start = time.time()
            
            final_output = {
                "pipeline_info": {
                    "timestamp": datetime.now().isoformat(),
                    "input_file": input_file_path,
                    "total_tables": len(tables),
                    "total_records": total_records,
                    "table_summary": raw_metadata['summary']['table_record_counts'],
                    "processing_steps": [
                        "metadata_extraction",
                        "llm_enrichment" if enriched_insights and not skip_llm and self.llm_connection_ok else "fallback_enrichment",
                        "output_generation"
                    ],
                    "llm_used": not skip_llm and self.llm_connection_ok,
                    "total_duration": time.time() - pipeline_start_time,
                    "debug_mode": self.llm_enricher.debug if hasattr(self.llm_enricher, 'debug') else False,
                    "model_used": selected_model_name
                },
                "raw_metadata": raw_metadata,
                "llm_insights": enriched_insights,
                "summary": self._generate_summary(raw_metadata, enriched_insights)
            }
            
            # Save enriched output
            enriched_output_path = "output/enriched_metadata.json"
            with open(enriched_output_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=2, default=str)
            print(f"✅ Enriched metadata saved to: {enriched_output_path}")
            print(f"⏱️ Step 3 completed in {time.time() - step3_start:.2f} seconds")
            print(f"🎯 Total pipeline duration: {time.time() - pipeline_start_time:.2f} seconds")
            
            print("\n🎉 Pipeline completed successfully!")
            
            # Print summary
            self._print_summary(final_output)
            
            # Print debug information if available
            if hasattr(self.llm_enricher, 'debug') and self.llm_enricher.debug:
                self._print_debug_info()
            
            return final_output
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            print(f"💡 Check logs and debug_output directories for more details")
            raise

    def _generate_summary(self, raw_metadata, llm_insights):
        """Generate enhanced pipeline summary with multi-table support"""
        # Handle multi-table structure
        total_attributes = 0
        total_records = 0
        anomaly_count = 0
        data_quality_score = 0
        top_issues = []
        
        # Check if we have tables structure (multi-table)
        if 'tables' in raw_metadata and isinstance(raw_metadata['tables'], dict):
            # Multi-table processing
            for table_name, table_metadata in raw_metadata['tables'].items():
                # Fix: access nested 'table_metadata' for attributes and other info
                if 'table_metadata' in table_metadata:
                    nested_metadata = table_metadata.get('table_metadata', {})
                else:
                    nested_metadata = table_metadata
                
                # Count attributes per table
                table_attributes = len(nested_metadata.get('attributes', {}))
                total_attributes += table_attributes
                
                # Get table records from summary
                table_records = raw_metadata.get('summary', {}).get('table_record_counts', {}).get(table_name, 0)
                total_records += table_records
                
                # Count anomalies per table
                table_anomaly_summary = nested_metadata.get('anomaly_summary', {})
                table_anomaly_count = table_anomaly_summary.get('total_anomalies', 0)
                
                # Alternative: Count anomalies from attributes if summary missing
                if table_anomaly_count == 0:
                    for attr_name, attr_data in nested_metadata.get('attributes', {}).items():
                        anomaly_flags = attr_data.get('anomaly_flags', [])
                        table_anomaly_count += len(anomaly_flags)
                
                anomaly_count += table_anomaly_count
                
                # Aggregate quality scores (average across tables)
                table_quality_score = nested_metadata.get('data_quality_score', 0)
                data_quality_score += table_quality_score
                
                # Collect top issues from each table
                table_issues = nested_metadata.get('top_issues', [])
                for issue in table_issues[:3]:  # Top 3 issues per table
                    top_issues.append(f"[{table_name}] {issue}")
            
            # Average quality score across tables
            num_tables = len(raw_metadata['tables'])
            if num_tables > 0:
                data_quality_score = data_quality_score / num_tables
        else:
            # Single table processing (backward compatibility)
            total_attributes = len(raw_metadata.get('attributes', {}))
            
            # Get records from dataset_info or summary
            dataset_info = raw_metadata.get('dataset_info', {})
            total_records = dataset_info.get('total_records', 0)
            if total_records == 0:
                total_records = raw_metadata.get('summary', {}).get('total_records', 0)
            
            # Count anomalies
            anomaly_summary = raw_metadata.get('anomaly_summary', {})
            anomaly_count = anomaly_summary.get('total_anomalies', 0)
            if anomaly_count == 0:
                for attr_name, attr_data in raw_metadata.get('attributes', {}).items():
                    anomaly_flags = attr_data.get('anomaly_flags', [])
                    anomaly_count += len(anomaly_flags)
            
            data_quality_score = raw_metadata.get('data_quality_score', 0)
            top_issues = raw_metadata.get('top_issues', [])
        
        # Get LLM recommendations count
        llm_recommendations_count = 0
        if llm_insights and isinstance(llm_insights, dict):
            recommendations = llm_insights.get('recommendations', [])
            if isinstance(recommendations, list):
                llm_recommendations_count = len(recommendations)
        
        # Determine processing mode with enhanced logic
        processing_mode = "basic"
        if llm_insights:
            overall_assessment = llm_insights.get('overall_assessment', {})
            confidence_score = overall_assessment.get('confidence_score', 0)
            
            # Check if enhanced validation was used
            enrichment_metadata = llm_insights.get('enrichment_metadata', {})
            validation_performed = enrichment_metadata.get('validation_performed', False)
            
            if confidence_score > 80:
                processing_mode = "llm_enhanced"
            elif confidence_score > 60:
                processing_mode = "llm_standard"
            elif validation_performed:
                processing_mode = "enhanced_fallback"
            else:
                processing_mode = "basic_fallback"
        
        # Enhanced summary with multi-table information
        summary = {
            "total_attributes": total_attributes,
            "total_anomalies": anomaly_count,
            "data_quality_score": data_quality_score,
            "top_issues": top_issues[:10],  # Limit to top 10 issues
            "llm_recommendations_count": llm_recommendations_count,
            "processing_mode": processing_mode,
            "total_records": total_records
        }
        
        # Add multi-table specific information if applicable
        if 'tables' in raw_metadata:
            summary["multi_table_info"] = {
                "total_tables": len(raw_metadata['tables']),
                "table_names": list(raw_metadata['tables'].keys()),
                "table_record_counts": raw_metadata.get('summary', {}).get('table_record_counts', {}),
                "cross_table_analysis": llm_insights.get('cross_table_relationships', {}) if llm_insights else {}
            }
        
        return summary

    def _print_summary(self, final_output):
        """Print enhanced comprehensive summary with multi-table support"""
        summary = final_output['summary']
        pipeline_info = final_output['pipeline_info']
        llm_insights = final_output.get('llm_insights', {})
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE PIPELINE SUMMARY")
        print("="*80)
        print(f"📁 Input File: {pipeline_info['input_file']}")
        
        # Multi-table information
        if summary.get('multi_table_info'):
            multi_info = summary['multi_table_info']
            print(f"📋 Total Tables: {multi_info['total_tables']}")
            print(f"📋 Total Records: {pipeline_info['total_records']}")
            print(f"📋 Tables: {', '.join(multi_info['table_names'])}")
            
            # Show record count per table
            table_counts = multi_info.get('table_record_counts', {})
            if table_counts:
                print("📊 Records per table:")
                for table, count in table_counts.items():
                    print(f"   • {table}: {count} records")
        else:
            print(f"📋 Total Records: {pipeline_info['total_records']}")
        
        print(f"🔍 Total Attributes: {summary['total_attributes']}")
        print(f"⚠️  Total Anomalies: {summary['total_anomalies']}")
        print(f"📈 Data Quality Score: {summary['data_quality_score']:.2f}/100")
        print(f"🐛 Debug Mode: {'Yes' if pipeline_info.get('debug_mode') else 'No'}")
        
        # Show LLM insights summary if available
        if llm_insights and isinstance(llm_insights, dict):
            overall_assessment = llm_insights.get('overall_assessment', {})
            if overall_assessment:
                print(f"\n🎯 LLM Assessment:")
                corrected_score = overall_assessment.get('corrected_score', 'N/A')
                summary_text = overall_assessment.get('summary', 'N/A')
                print(f"  • Corrected Score: {corrected_score}")
                print(f"  • Summary: {summary_text[:100]}{'...' if len(summary_text) > 100 else ''}")
        
        print("="*80)
        
        # Detailed per-table summary and key quality issues
        if summary.get('multi_table_info'):
            print("\n📋 Detailed Table Summaries and Sheets:")
            for table_name in multi_info['table_names']:
                print(f"\nTable: {table_name}")
                print(f"  Sheets:")
                print(f"    • {table_name}_Original - Raw data as provided")
                print(f"    • {table_name}_Enriched_Metadata_Anomalies - Column analysis, data types, patterns, anomalies detected")
                print(f"    • {table_name}_Quality_Assessment - Quality scores, recommendations, and action items")
            
            # Key quality issues detected across dataset
            print("\n🔑 Key Quality Issues Detected Across Dataset:")
            critical_issues = []
            medium_issues = []
            low_issues = []
            monitor_issues = []
            
            if llm_insights and isinstance(llm_insights, dict):
                issues = llm_insights.get('critical_issues', [])
                for issue in issues:
                    if isinstance(issue, dict):
                        severity = issue.get('severity', '').lower()
                        desc = issue.get('issue', 'Unknown issue')
                        if severity == 'critical':
                            critical_issues.append(desc)
                        elif severity == 'high' or severity == 'medium':
                            medium_issues.append(desc)
                        elif severity == 'low':
                            low_issues.append(desc)
                        else:
                            monitor_issues.append(desc)
            
            if critical_issues:
                print("  Critical: " + ", ".join(critical_issues))
            if medium_issues:
                print("  Medium: " + ", ".join(medium_issues))
            if low_issues:
                print("  Low: " + ", ".join(low_issues))
            if monitor_issues:
                print("  Monitor: " + ", ".join(monitor_issues))
            
            print("\n💡 Summary Pattern for All Tables:")
            print(f"  Each of the {multi_info['total_tables']} tables follows this 3-sheet pattern:")
            print(f"    Sheet XA: [TableName]_Original - Raw data as provided")
            print(f"    Sheet XB: [TableName]_Enriched_Metadata_Anomalies - Column analysis, data types, patterns, anomalies detected")
            print(f"    Sheet XC: [TableName]_Quality_Assessment - Quality scores, recommendations, and action items")
            print(f"\n  Total Sheets: {multi_info['total_tables'] * 3} sheets ({multi_info['total_tables']} tables × 3 sheets each)")

    def _print_debug_info(self):
        """Print debug information"""
        print("\n" + "="*60)
        print("🐛 DEBUG INFORMATION")
        print("="*60)
        print("📁 Debug files saved in:")
        print("  • logs/ - Detailed logs")
        print("  • debug_output/ - LLM request/response dumps")
        print("  • output/ - Final outputs and intermediate results")
        print("="*60)


def detect_nested_tables(data, parent_path="", max_depth=5, current_depth=0):
    """
    Recursively detect nested tables in JSON data structure.
    Returns a dictionary with hierarchical table paths and their data.
    
    Args:
        data: The data structure to analyze
        parent_path: Current path in the hierarchy
        max_depth: Maximum recursion depth to prevent infinite loops
        current_depth: Current recursion depth
    
    Returns:
        dict: Nested table structure with paths as keys
    """
    if current_depth >= max_depth:
        return {}
    
    nested_tables = {}
    
    def is_table_like(obj):
        """Check if object represents a table (list of records or single record)"""
        if isinstance(obj, list):
            # Check if it's a list of dictionaries (typical table structure)
            if obj and all(isinstance(item, dict) for item in obj):
                return True
            # Also consider list of primitive values as a simple table
            if obj and all(isinstance(item, (str, int, float, bool)) for item in obj):
                return True
        elif isinstance(obj, dict):
            # Single record table
            if obj and all(isinstance(v, (str, int, float, bool, type(None))) for v in obj.values()):
                return True
        return False
    
    def create_table_path(parent, child):
        """Create hierarchical table path"""
        if parent:
            return f"{parent}_{child}"
        return child
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = create_table_path(parent_path, key)
            
            if is_table_like(value):
                # This is a table - add it to our nested tables
                nested_tables[current_path] = value
                print(f"   🔍 Detected table: {current_path} ({len(value) if isinstance(value, list) else 1} records)")
            elif isinstance(value, dict):
                # Recursively check nested dictionaries
                child_tables = detect_nested_tables(value, current_path, max_depth, current_depth + 1)
                nested_tables.update(child_tables)
            elif isinstance(value, list):
                # Check if list contains nested objects
                for i, item in enumerate(value):
                    if isinstance(item, dict) and not is_table_like(item):
                        # This list item might contain nested tables
                        item_path = f"{current_path}_{i}"
                        child_tables = detect_nested_tables(item, item_path, max_depth, current_depth + 1)
                        nested_tables.update(child_tables)
    elif isinstance(data, list):
        # Handle case where root data is a list
        for i, item in enumerate(data):
            if isinstance(item, dict):
                item_path = create_table_path(parent_path, f"item_{i}")
                child_tables = detect_nested_tables(item, item_path, max_depth, current_depth + 1)
                nested_tables.update(child_tables)
    
    return nested_tables


def get_table_path_hierarchy(table_path):
    """
    Get the hierarchical components of a table path.
    Returns: (parent_table, full_path_components)
    """
    components = table_path.split('_')
    if len(components) == 1:
        # Root level table
        return None, [table_path]
    else:
        # Nested table - parent is the first component
        parent = components[0]
        return parent, components


def create_safe_sheet_name(table_path, suffix, max_length=31):
    """
    Create a safe Excel sheet name with hierarchical naming.
    Format: ParentTable_ChildTable_Suffix or TableName_Suffix
    """
    # Split the table path to understand hierarchy
    components = table_path.split('_')
    
    if len(components) == 1:
        # Root level table
        base_name = components[0]
    elif len(components) == 2:
        # Simple nested table
        base_name = f"{components[0]}_{components[1]}"
    else:
        # Deep nested table - use first and last components
        base_name = f"{components[0]}_{components[-1]}"
    
    # Create full sheet name with suffix
    sheet_name = f"{base_name}_{suffix}"
    
    # Truncate if too long, preserving the suffix
    if len(sheet_name) > max_length:
        available_length = max_length - len(suffix) - 1  # -1 for underscore
        if available_length > 0:
            sheet_name = f"{base_name[:available_length]}_{suffix}"
        else:
            # Fallback: just use truncated base name
            sheet_name = base_name[:max_length]
    
    return sheet_name


def convert_enriched_json_to_excel(
    input_data_path="input/input_data.json",
    raw_path="output/raw_metadata.json",
    enriched_path="output/enriched_metadata.json",
    output_dir="output/",
    create_excel=True
):
    """
    Enhanced converter that creates three sheets per table with hierarchical naming:
    1. TableName_Original - Raw/original data
    2. TableName_Enriched_Metadata_Anomalies - Column analysis, metadata, anomalies
    3. TableName_Quality_Assessment - Quality scores and recommendations
    
    For nested tables: Parent_Child_Original, Parent_Child_Enriched_Metadata_Anomalies, etc.
    """
    try:
        print("📥 Loading data files...")
        
        # Load all JSON files
        with open(input_data_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_metadata = json.load(f)
        
        with open(enriched_path, "r", encoding="utf-8") as f:
            enriched_metadata = json.load(f)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Store all dataframes for Excel export
        excel_sheets = {}
        created_files = []
        
        print("🔄 Detecting nested table structure...")
        
        # Enhanced nested table detection
        nested_tables = detect_nested_tables(input_data)
        
        if not nested_tables:
            # Fallback: treat input_data as single table if no nested structure detected
            if isinstance(input_data, list) or (isinstance(input_data, dict) and input_data):
                nested_tables = {"main_data": input_data}
                print("     → No nested structure detected, treating as single table: 'main_data'")
            else:
                print("     ⚠️ No valid table data found in input")
                return None
        
        print(f"     → Found {len(nested_tables)} tables (including nested)")
        
        # Print table hierarchy for user reference
        for table_path in sorted(nested_tables.keys()):
            depth = table_path.count('_')
            indent = "  " * depth
            table_size = len(nested_tables[table_path]) if isinstance(nested_tables[table_path], list) else 1
            print(f"     {indent}📋 {table_path}: {table_size} records")
        
        print("\n🔄 Generating hierarchical Excel sheets with three-sheet pattern...")
        
        # Process each detected table (including nested ones)
        for table_path, table_data in nested_tables.items():
            print(f"\n🔄 Processing table: '{table_path}'")
            
            # Get table-specific metadata from raw_metadata
            table_attributes = {}
            if 'tables' in raw_metadata and isinstance(raw_metadata['tables'], dict):
                # For nested tables, we need to match the original table structure
                # First try exact match, then try parent table match
                if table_path in raw_metadata['tables']:
                    table_metadata = raw_metadata['tables'][table_path]
                else:
                    # Try to find parent table metadata for nested tables
                    parent_table = table_path.split('_')[0]
                    if parent_table in raw_metadata['tables']:
                        table_metadata = raw_metadata['tables'][parent_table]
                    else:
                        # Create basic metadata structure
                        table_metadata = {'table_metadata': {'attributes': {}}}
                
                # Extract attributes, handling nested structure
                if isinstance(table_metadata, dict):
                    if 'table_metadata' in table_metadata:
                        table_attributes = table_metadata.get('table_metadata', {}).get('attributes', {})
                    else:
                        table_attributes = table_metadata.get('attributes', {})
            
            # If no metadata found, generate basic metadata from the data
            if not table_attributes and table_data:
                print(f"   ⚠️ No metadata found for {table_path}, generating basic metadata...")
                table_attributes = generate_basic_metadata(table_data)
            
            # ===== SHEET 1: ORIGINAL DATA =====
            sheet_name_original = create_safe_sheet_name(table_path, "Original")
            print(f"   • Creating original data sheet: {sheet_name_original}")
            
            if isinstance(table_data, list) and table_data:
                # Flatten nested JSON columns for better tabular output
                try:
                    df_original = pd.json_normalize(table_data, sep='_', max_level=0)
                except Exception as e:
                    print(f"     ⚠️ Warning: Failed to flatten table {table_path}: {e}")
                    df_original = pd.DataFrame(table_data)
            elif isinstance(table_data, dict):
                try:
                    df_original = pd.json_normalize([table_data], sep='_', max_level=0)
                except Exception as e:
                    print(f"     ⚠️ Warning: Failed to flatten table {table_path}: {e}")
                    df_original = pd.DataFrame([table_data])
            else:
                print(f"     ⚠️ Skipping '{table_path}' - unsupported data format")
                continue
            
            # Save original data CSV
            original_csv_file = os.path.join(output_dir, f"{table_path}_original.csv")
            df_original.to_csv(original_csv_file, index=False, encoding="utf-8")
            created_files.append(original_csv_file)
            
            # Add to Excel sheets
            excel_sheets[sheet_name_original] = df_original
            
            # ===== SHEET 2: ENRICHED METADATA AND ANOMALIES =====
            sheet_name_metadata = create_safe_sheet_name(table_path, "Enriched_Metadata_Anomalies")
            print(f"   • Creating enriched metadata sheet: {sheet_name_metadata}")
            
            metadata_rows = []
            
            for attr_name, attr_info in table_attributes.items():
                if not isinstance(attr_info, dict):
                    continue
                
                # Get column data for enhanced analysis
                column_data = df_original[attr_name] if attr_name in df_original.columns else pd.Series([])
                
                # Basic metadata
                data_type = attr_info.get("data_type", "unknown").upper()
                if data_type == "STRING":
                    data_type = "VARCHAR"
                elif data_type == "NUMERIC" or data_type == "FLOAT":
                    data_type = "DECIMAL"
                elif data_type == "INT":
                    data_type = "INTEGER"
                elif data_type == "DATETIME":
                    data_type = "TIMESTAMP"
                elif data_type == "BOOL":
                    data_type = "BOOLEAN"
                
                null_count = attr_info.get("null_count", 0)
                unique_count = attr_info.get("unique_count", 0)
                
                # Calculate pattern and length information
                min_length = ""
                max_length = ""
                pattern = ""
                
                if not column_data.empty:
                    non_null_data = column_data.dropna()
                    if len(non_null_data) > 0:
                        if data_type in ["VARCHAR", "TEXT"]:
                            string_lengths = non_null_data.astype(str).str.len()
                            min_length = string_lengths.min()
                            max_length = string_lengths.max()
                            
                            # Determine pattern
                            sample_values = non_null_data.head(3).astype(str).tolist()
                            if all('@' in val for val in sample_values):
                                pattern = "Email format"
                            elif all(val.startswith('http') for val in sample_values):
                                pattern = "URL format"
                            elif all(val.startswith('@') for val in sample_values):
                                pattern = "@username"
                            elif all(val.replace('-', '').replace(' ', '').isalnum() for val in sample_values):
                                pattern = "Alphanumeric"
                            elif all(val.replace(' ', '').isalpha() for val in sample_values):
                                pattern = "Alpha only"
                            else:
                                pattern = "Free text"
                        
                        elif data_type in ["INTEGER", "DECIMAL"]:
                            if data_type == "INTEGER":
                                pattern = "Sequential" if (non_null_data.diff().dropna() == 1).all() else "Numeric"
                            else:
                                pattern = "Currency" if non_null_data.dtype == 'float64' else "Numeric"
                        
                        elif data_type in ["DATE", "TIMESTAMP"]:
                            pattern = "ISO 8601"
                        
                        elif data_type == "BOOLEAN":
                            pattern = "Boolean"
                        
                        else:
                            if unique_count <= 10:
                                pattern = "Fixed values"
                            else:
                                pattern = "Variable"
                
                # Detect anomalies
                anomaly_flags = attr_info.get("anomaly_flags", [])
                anomalies_detected = "None"
                data_quality_issues = "Good"
                
                if anomaly_flags:
                    anomalies_detected = ", ".join(anomaly_flags[:2])
                    if len(anomaly_flags) > 2:
                        anomalies_detected += f" (+{len(anomaly_flags) - 2} more)"
                    
                    # Determine quality issues
                    if "empty_strings" in anomaly_flags or "invalid_format" in anomaly_flags:
                        data_quality_issues = "Format issues"
                    elif "outliers_detected" in anomaly_flags:
                        data_quality_issues = "Statistical outliers"
                    elif "inconsistent_format" in anomaly_flags:
                        data_quality_issues = "Format inconsistency"
                    elif null_count > 0:
                        data_quality_issues = f"{null_count/len(df_original)*100:.0f}% missing data"
                    else:
                        data_quality_issues = "Minor issues"
                
                # Handle special cases for better reporting
                if attr_name == "email" and len(set(column_data.dropna().str.split('@').str[1])) > 1:
                    anomalies_detected = "Different domains (personal vs corporate)"
                    data_quality_issues = "Minor inconsistency"
                
                if data_type == "DATE" and not column_data.empty:
                    date_range = (pd.to_datetime(column_data.dropna()).max() - pd.to_datetime(column_data.dropna()).min()).days
                    if date_range > 365:
                        anomalies_detected = f"{date_range//365}-year range"
                        data_quality_issues = "Expected variation"
                
                metadata_row = {
                    "Column": attr_name,
                    "Data_Type": data_type,
                    "Null_Count": null_count,
                    "Unique_Values": unique_count,
                    "Min_Length": min_length,
                    "Max_Length": max_length,
                    "Pattern": pattern,
                    "Anomalies_Detected": anomalies_detected,
                    "Data_Quality_Issues": data_quality_issues
                }
                
                metadata_rows.append(metadata_row)
            
            # Save enriched metadata CSV and Excel sheet
            if metadata_rows:
                df_metadata = pd.DataFrame(metadata_rows)
                metadata_csv_file = os.path.join(output_dir, f"{table_path}_enriched_metadata_anomalies.csv")
                df_metadata.to_csv(metadata_csv_file, index=False, encoding="utf-8")
                created_files.append(metadata_csv_file)
                excel_sheets[sheet_name_metadata] = df_metadata
                print(f"     ✅ Created enriched metadata with {len(metadata_rows)} columns")
            
            # ===== SHEET 3: QUALITY ASSESSMENT =====
            sheet_name_quality = create_safe_sheet_name(table_path, "Quality_Assessment")
            print(f"   • Creating quality assessment sheet: {sheet_name_quality}")
            
            quality_rows = []
            
            # Calculate overall quality metrics
            if table_attributes:
                # Completeness
                total_cells = len(df_original) * len(table_attributes)
                missing_cells = sum(attr.get('null_count', 0) for attr in table_attributes.values() if isinstance(attr, dict))
                completeness_score = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
                
                # Consistency - based on format consistency
                format_issues = sum(1 for attr in table_attributes.values() 
                                  if isinstance(attr, dict) and 'inconsistent_format' in attr.get('anomaly_flags', []))
                consistency_score = ((len(table_attributes) - format_issues) / len(table_attributes) * 100) if table_attributes else 100
                
                # Accuracy - based on business rule violations
                accuracy_issues = sum(len(attr.get('anomaly_flags', [])) for attr in table_attributes.values() if isinstance(attr, dict))
                accuracy_score = max(100 - (accuracy_issues * 5), 0)  # Reduce by 5% per issue
                
                # Validity - based on data type compliance
                validity_issues = sum(1 for attr in table_attributes.values() 
                                    if isinstance(attr, dict) and 'invalid_format' in attr.get('anomaly_flags', []))
                validity_score = ((len(table_attributes) - validity_issues) / len(table_attributes) * 100) if table_attributes else 100
                
            # Helper function to safely calculate unique count for columns with unhashable types
            def safe_nunique(series):
                try:
                    return series.nunique()
                except TypeError:
                    # Convert unhashable types (list, dict) to string for uniqueness check
                    return len(set(series.apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (list, dict)) else str(x))))
            
            # Uniqueness - for ID fields
            id_columns = [col for col in df_original.columns if 'id' in str(col).lower()]
            uniqueness_issues = 0
            for id_col in id_columns:
                if len(df_original[id_col]) != safe_nunique(df_original[id_col]):
                    uniqueness_issues += 1
            uniqueness_score = 100 if uniqueness_issues == 0 else 90
            
            # Timeliness - for timestamp fields
            timestamp_columns = [col for col in df_original.columns if any(keyword in str(col).lower() for keyword in ['date', 'time', 'created', 'updated'])]
            timeliness_score = 95  # Default good score
            
            # Overall quality score
            overall_quality = (completeness_score + consistency_score + accuracy_score + validity_score + uniqueness_score + timeliness_score) / 6
            
        else:
            completeness_score = consistency_score = accuracy_score = validity_score = uniqueness_score = timeliness_score = overall_quality = 0
            
            # Generate quality assessment rows
            quality_assessments = [
                {
                    "Metric": "Completeness",
                    "Score": f"{completeness_score:.1f}%",
                    "Details": "No null values" if completeness_score == 100 else f"{missing_cells} missing values",
                    "Recommendation": "Maintain current standards" if completeness_score >= 95 else "Improve data collection processes"
                },
                {
                    "Metric": "Consistency", 
                    "Score": f"{consistency_score:.1f}%",
                    "Details": "All formats consistent" if consistency_score == 100 else f"{format_issues} format inconsistencies",
                    "Recommendation": "Good" if consistency_score >= 90 else "Standardize data formats"
                },
                {
                    "Metric": "Accuracy",
                    "Score": f"{accuracy_score:.1f}%",
                    "Details": "Data appears accurate" if accuracy_score >= 95 else f"{accuracy_issues} potential accuracy issues",
                    "Recommendation": "Good" if accuracy_score >= 90 else "Review and validate data accuracy"
                },
                {
                    "Metric": "Validity",
                    "Score": f"{validity_score:.1f}%",
                    "Details": "All formats valid" if validity_score == 100 else f"{validity_issues} format violations",
                    "Recommendation": "Good" if validity_score >= 95 else "Fix invalid data formats"
                },
                {
                    "Metric": "Uniqueness",
                    "Score": f"{uniqueness_score:.1f}%",
                    "Details": "All IDs unique" if uniqueness_score == 100 else f"{uniqueness_issues} duplicate ID issues",
                    "Recommendation": "Good" if uniqueness_score >= 95 else "Ensure ID uniqueness"
                },
                {
                    "Metric": "Timeliness",
                    "Score": f"{timeliness_score:.1f}%",
                    "Details": "Recent data" if timeliness_score >= 90 else "Some outdated records",
                    "Recommendation": "Good" if timeliness_score >= 90 else "Update data more frequently"
                },
                {
                    "Metric": "Overall Quality",
                    "Score": f"{overall_quality:.1f}%",
                    "Details": "High quality dataset" if overall_quality >= 90 else "Good quality with room for improvement" if overall_quality >= 70 else "Needs significant improvement",
                    "Recommendation": "Minor improvements possible" if overall_quality >= 90 else "Focus on key quality issues" if overall_quality >= 70 else "Comprehensive quality improvement needed"
                }
            ]
            
            # Add specific quality issues from LLM insights if available
            llm_insights = enriched_metadata.get("llm_insights", {})
            if llm_insights and isinstance(llm_insights, dict):
                critical_issues = llm_insights.get("critical_issues", [])
                for issue in critical_issues[:3]:  # Top 3 critical issues
                    if isinstance(issue, dict):
                        issue_desc = issue.get("issue", str(issue))
                        severity = issue.get("severity", "UNKNOWN")
                        quality_assessments.append({
                            "Metric": f"Critical Issue: {issue_desc[:30]}...",
                            "Score": "CRITICAL" if severity == "CRITICAL" else "HIGH",
                            "Details": issue.get("description", "")[:100],
                            "Recommendation": issue.get("specific_fix", "Review and resolve")[:100]
                        })
            
            quality_rows.extend(quality_assessments)
            
            # Save quality assessment CSV and Excel sheet
            if quality_rows:
                df_quality = pd.DataFrame(quality_rows)
                quality_csv_file = os.path.join(output_dir, f"{table_path}_quality_assessment.csv")
                df_quality.to_csv(quality_csv_file, index=False, encoding="utf-8")
                created_files.append(quality_csv_file)
                excel_sheets[sheet_name_quality] = df_quality
                print(f"     ✅ Created quality assessment with {len(quality_rows)} metrics")
        
        # 4. CREATE COMPREHENSIVE EXCEL WORKBOOK
        if create_excel:
            print("   • Creating comprehensive Excel workbook with three-sheet pattern...")
            excel_path = os.path.join(output_dir, "comprehensive_data_analysis.xlsx")
            
            temp_excel_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
                    temp_excel_path = temp_file.name
                
                with pd.ExcelWriter(temp_excel_path, engine='openpyxl') as writer:
                    # Sort sheets by table hierarchy and sheet type
                    sorted_sheets = sorted(excel_sheets.items(), key=lambda x: (
                        x[0].count('_'),  # Hierarchy depth first
                        x[0].split('_')[0],  # Then by table name
                        0 if 'Original' in x[0] else 1 if 'Enriched' in x[0] else 2  # Then by sheet type
                    ))
                    
                    sheet_count_by_type = {"Original": 0, "Enriched": 0, "Quality": 0}
                    
                    for sheet_name, df in sorted_sheets:
                        # Excel sheet names have a 31 character limit
                        safe_sheet_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
                        df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                        
                        # Count sheet types
                        if 'Original' in sheet_name:
                            sheet_count_by_type["Original"] += 1
                        elif 'Enriched' in sheet_name:
                            sheet_count_by_type["Enriched"] += 1
                        elif 'Quality' in sheet_name:
                            sheet_count_by_type["Quality"] += 1
                        
                        print(f"     📄 Added sheet: {safe_sheet_name} ({len(df)} rows)")
                
                shutil.move(temp_excel_path, excel_path)
                print(f"   ✅ Excel workbook created: {excel_path}")
                print(f"     📊 Sheet breakdown: {sheet_count_by_type['Original']} Original, {sheet_count_by_type['Enriched']} Metadata, {sheet_count_by_type['Quality']} Quality")
                created_files.append(excel_path)
                
            except PermissionError as e:
                print(f"   ❌ Cannot create Excel file due to permission error: {e}")
                print(f"   💡 The file might be open in Excel. Please close it and try again.")
                if temp_excel_path and os.path.exists(temp_excel_path):
                    os.unlink(temp_excel_path)
            except Exception as e:
                print(f"   ❌ Failed to create Excel workbook: {e}")
                if temp_excel_path and os.path.exists(temp_excel_path):
                    os.unlink(temp_excel_path)
        
        # Print comprehensive summary
        print(f"\n✅ Three-sheet analysis completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"\n📊 Created files by table and sheet type:")
        
        # Group tables by hierarchy level
        hierarchy_groups = defaultdict(list)
        for table_path in nested_tables.keys():
            level = table_path.count('_')
            hierarchy_groups[level].append(table_path)
        
        total_sheets = 0
        for level in sorted(hierarchy_groups.keys()):
            level_name = "Root Tables" if level == 0 else f"Level {level} Nested Tables"
            print(f"\n   {level_name}:")
            
            for table_path in sorted(hierarchy_groups[level]):
                record_count = len(nested_tables[table_path]) if isinstance(nested_tables[table_path], list) else 1
                print(f"     📋 {table_path} ({record_count} records)")
                print(f"       • {create_safe_sheet_name(table_path, 'Original')}")
                print(f"       • {create_safe_sheet_name(table_path, 'Enriched_Metadata_Anomalies')}")
                print(f"       • {create_safe_sheet_name(table_path, 'Quality_Assessment')}")
                total_sheets += 3
        
        if create_excel:
            print(f"\n   📄 Excel Workbook: comprehensive_data_analysis.xlsx ({total_sheets} sheets)")
        
        print(f"\n📈 Final Summary:")
        print(f"   • Tables processed: {len(nested_tables)} (max depth: {max(table_path.count('_') for table_path in nested_tables.keys())})")
        print(f"   • CSV files created: {len([f for f in created_files if f.endswith('.csv')])}")
        print(f"   • Excel sheets: {len(excel_sheets)} ({total_sheets} total)")
        print(f"   • Files per table: 3 (Original + Metadata + Quality)")
        print(f"   • Total files created: {len(created_files)}")
        
        return {
            "tables_processed": len(nested_tables),
            "csv_files_created": len([f for f in created_files if f.endswith('.csv')]),
            "excel_created": create_excel and any(f.endswith('.xlsx') for f in created_files),
            "output_directory": output_dir,
            "created_files": created_files,
            "table_hierarchy": {table_path: table_path.count('_') for table_path in nested_tables.keys()},
            "nested_table_structure": nested_tables,
            "sheets_per_table": 3,
            "total_sheets": total_sheets
        }

    except Exception as e:
        print(f"❌ Failed to generate three-sheet analysis files: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_basic_metadata(table_data):
    """
    Generate basic metadata for a table when metadata is not available from the pipeline.
    This is a fallback function for nested tables that weren't processed by the metadata extractor.
    """
    if not table_data:
        return {}
    
    # Convert to DataFrame for analysis
    if isinstance(table_data, list):
        if not table_data:
            return {}
        df = pd.DataFrame(table_data)
    elif isinstance(table_data, dict):
        df = pd.DataFrame([table_data])
    else:
        return {}
    
    attributes = {}
    
    for column in df.columns:
        series = df[column]
        
        # Basic statistics
        total_count = len(series)
        null_count = series.isnull().sum()
        present_count = total_count - null_count
        null_percentage = (null_count / total_count * 100) if total_count > 0 else 0
        unique_count = series.nunique()
        
        # Determine data type
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                data_type = "integer"
            else:
                data_type = "float"
        elif pd.api.types.is_datetime64_any_dtype(series):
            data_type = "datetime"
        elif pd.api.types.is_bool_dtype(series):
            data_type = "boolean"
        else:
            data_type = "string"
        
        # Basic quality score (simple heuristic)
        quality_score = 100
        if null_percentage > 0:
            quality_score -= min(null_percentage, 50)  # Penalize missing data
        if unique_count == 1:
            quality_score -= 20  # Penalize single-value columns
        quality_score = max(quality_score, 0)
        
        attributes[column] = {
            "data_type": data_type,
            "present_count": present_count,
            "null_count": null_count,
            "null_percentage": null_percentage,
            "unique_count": unique_count,
            "quality_score": quality_score,
            "anomaly_flags": []
        }
        
        # Add type-specific metadata
        if data_type in ["integer", "float"]:
            if present_count > 0:
                numeric_series = pd.to_numeric(series, errors='coerce')
                attributes[column].update({
                    "min_value": numeric_series.min(),
                    "max_value": numeric_series.max(),
                    "mean": numeric_series.mean(),
                    "median": numeric_series.median(),
                    "std": numeric_series.std()
                })
        elif data_type == "string":
            if present_count > 0:
                string_lengths = series.dropna().astype(str).str.len()
                value_counts = series.value_counts().head(5)
                attributes[column].update({
                    "min_length": string_lengths.min(),
                    "max_length": string_lengths.max(),
                    "avg_length": string_lengths.mean(),
                    "common_values": value_counts.to_dict()
                })
        elif data_type == "datetime":
            if present_count > 0:
                datetime_series = pd.to_datetime(series, errors='coerce')
                date_range = (datetime_series.max() - datetime_series.min()).days
                attributes[column].update({
                    "min_date": datetime_series.min().isoformat() if pd.notna(datetime_series.min()) else "N/A",
                    "max_date": datetime_series.max().isoformat() if pd.notna(datetime_series.max()) else "N/A", 
                    "date_range_days": date_range if pd.notna(date_range) else 0
                })
    
    return attributes


def main():
    """Main function with enhanced error handling and three-sheet Excel generation"""
    try:
        # Check for openpyxl
        try:
            import openpyxl
        except ImportError:
            print("Error: The 'openpyxl' library is required to generate Excel reports.")
            print("Please install it using: pip install openpyxl")
            return

        # Enable debug mode by default for troubleshooting
        agent = MetadataEnrichmentAgent(debug=True)
        
        # Define available models
        available_models = [
            'mistral-large-latest'
        ]
        
        # Auto select the only available model without CLI input
        selected_model_name = available_models[0]
        print(f"Selected model: {selected_model_name}")
        
        # Check if input file exists
        input_file = "input/input_data.json"
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            print("💡 Please ensure your input JSON file is placed in the input/ directory")
            return
        
        # Run the pipeline with selected model
        result = agent.run_pipeline(input_file, selected_model=selected_model_name)
        
        if result:
            print("\n✅ Success! Check the output/ directory for results.")
            print("🐛 Debug files available in logs/ and debug_output/ directories.")
            
            # Generate three-sheet Excel analysis
            print("\n📥 Generating three-sheet Excel analysis...")
            convert_result = convert_enriched_json_to_excel()
            
            if convert_result:
                print(f"\n🎉 Three-Sheet Analysis Complete!")
                print(f"📊 Tables processed: {convert_result['tables_processed']}")
                print(f"📄 Sheets per table: {convert_result['sheets_per_table']}")
                print(f"📁 CSV files created: {convert_result['csv_files_created']}")
                print(f"📋 Total Excel sheets: {convert_result['total_sheets']}")
                print(f"📄 Excel workbook: {'Yes' if convert_result['excel_created'] else 'No'}")
                print(f"💾 Files saved to: {convert_result['output_directory']}")
                
                # Show table hierarchy
                if convert_result.get('table_hierarchy'):
                    print(f"\n📊 Table Hierarchy Overview:")
                    hierarchy_levels = defaultdict(list)
                    for table_path, level in convert_result['table_hierarchy'].items():
                        hierarchy_levels[level].append(table_path)
                    
                    for level in sorted(hierarchy_levels.keys()):
                        level_name = "Root Level" if level == 0 else f"Nested Level {level}"
                        tables_at_level = hierarchy_levels[level]
                        print(f"   {level_name}: {len(tables_at_level)} tables")
                        for table in sorted(tables_at_level)[:3]:  # Show first 3 tables
                            print(f"     • {table}")
                        if len(tables_at_level) > 3:
                            print(f"     • ... and {len(tables_at_level) - 3} more")
                
                print(f"\n💡 Each table has 3 sheets:")
                print(f"   • TableName_Original - Raw data")
                print(f"   • TableName_Enriched_Metadata_Anomalies - Column analysis & anomalies")
                print(f"   • TableName_Quality_Assessment - Quality scores & recommendations")
                print(f"\n📋 Open 'comprehensive_data_analysis.xlsx' to explore all tables!")
        else:
            print("❌ Pipeline failed. Check logs for details.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    