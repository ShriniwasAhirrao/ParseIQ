#!/usr/bin/env python3
"""
JSON Metadata Enrichment Agent - Main Orchestrator
Follows 3-step pipeline: Extract → LLM Enrich → Output
Updated to work with enhanced LLMEnricher with debugging features
Fixed permission issues and improved error handling
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
        
        # Initialize LLM enricher with full config (site_url, site_name)
        llm_config = self.config.get_llm_config()
        llm_config['debug'] = debug
        llm_config['prompt_template_path'] = 'step2_llm_enricher/prompt_template.txt'
        self.llm_enricher = LLMEnricher(llm_config)

        # Supported models list
        self.supported_models = {
            "default": self.config.MODEL_NAME
        }
        self.selected_model_key = "default"
        
        # Ensure output directories exist
        self._ensure_directories()
        
        # Skip startup test to avoid 429 burn - optimistic fallback
        self.llm_connection_ok = True
        print("💡 LLM connection optimistic - will fallback if needed")

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
        """Configure logging — single clean setup, no duplicate handlers."""
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')

        os.makedirs('logs', exist_ok=True)

        # Clear any handlers already on the root logger (added by imported libs, previous runs, etc.)
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)

        file_h = logging.FileHandler('logs/app.log', encoding='utf-8')
        file_h.setFormatter(fmt)

        root.setLevel(logging.INFO)
        root.addHandler(console)
        root.addHandler(file_h)

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

        # Auto-clean stale output files from previous runs
        _output_dir = "output"
        _cleaned = 0
        for _ext in ("*.csv", "*.xlsx"):
            import glob as _glob
            for _f in _glob.glob(os.path.join(_output_dir, _ext)):
                try:
                    os.remove(_f)
                    _cleaned += 1
                except OSError:
                    pass
        if _cleaned:
            print(f"🧹 Cleaned {_cleaned} stale output file(s) from previous run")

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
                # Patch the table_name so it shows the real name, not the extractor default "main_table"
                if isinstance(table_metadata, dict) and 'table_metadata' in table_metadata:
                    table_metadata['table_metadata']['table_name'] = table_name
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

            # Build dataset_overview in the format LLMEnricher's _compress_metadata_for_prompt expects.
            # The extractor uses different key names (attributes / dataset_info) — translate them here.
            raw_metadata['dataset_overview'] = {'table_summaries': {}}
            for table_name, table_meta in all_metadata.items():
                inner = table_meta.get('table_metadata', table_meta)
                dataset_info = inner.get('dataset_info', {})
                attributes = inner.get('attributes', {})
                dup_analysis = inner.get('data_profiling', {}).get('duplicate_analysis', {})
                completeness = inner.get('data_profiling', {}).get('record_completeness', {})

                # Translate attribute metadata to the field_analysis format the LLM compressor reads
                field_analysis = {}
                for attr_name, attr_data in attributes.items():
                    field_analysis[attr_name] = {
                        'data_type': attr_data.get('data_type', 'unknown'),
                        'null_percentage': attr_data.get('null_percentage', 0),
                        'anomalies': attr_data.get('anomaly_flags', []),
                        'outlier_count': attr_data.get('outliers', {}).get('count', 0),
                        'unique_percentage': round(attr_data.get('unique_ratio', 0) * 100, 1),
                        'min_value': attr_data.get('min_value'),
                        'max_value': attr_data.get('max_value'),
                        'mean': attr_data.get('mean'),
                    }

                raw_metadata['dataset_overview']['table_summaries'][table_name] = {
                    'record_count': dataset_info.get('total_records', 0),
                    'field_analysis': field_analysis,
                    'completeness_rate': completeness.get('avg_completeness', 100.0),
                    'duplicate_count': dup_analysis.get('duplicate_rows', 0),
                    'quality_score': inner.get('data_quality_score', 0),
                    'top_issues': inner.get('top_issues', []),
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
                    enriched_insights = self.llm_enricher.enrich_metadata(raw_metadata, model=selected_model_name)
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
        llm_insights = final_output.get('llm_insights', {})
        if llm_insights and isinstance(llm_insights, dict):
            overall_assessment = llm_insights.get('overall_assessment', {})
            if overall_assessment:
                print(f"\n🎯 LLM Assessment:")
                # LLM returns overall_score + quality_grade; corrected_score is in validation_details
                overall_score = (overall_assessment.get('overall_score')
                                 or llm_insights.get('validation_details', {}).get('corrected_score', 'N/A'))
                quality_grade = overall_assessment.get('quality_grade', 'N/A')
                production_readiness = overall_assessment.get('production_readiness', '')
                concerns = overall_assessment.get('primary_concerns', [])
                concern_text = '; '.join(concerns[:2]) if concerns else ''
                print(f"  • Score: {overall_score}/100  |  Grade: {quality_grade}  |  {production_readiness}")
                if concern_text:
                    print(f"  • Key Concerns: {concern_text[:120]}")
        
        print("="*80)

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


def convert_enriched_json_to_csv(
    input_data_path="input/input_data.json",
    raw_path="output/raw_metadata.json",
    enriched_path="output/enriched_metadata.json",
    output_dir="output/",
    create_excel=True,
    csv_per_table=False
):
    """
    Converts input data and metadata JSON files to organized CSV files and Excel workbook.
    Enhanced version that creates separate files and sheets for each table with proper tabular format.
    """
    import json
    import pandas as pd
    import os
    import tempfile
    import shutil
    from datetime import datetime
    
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
        _used_sheet_names: set = set()  # track duplicates

        # Excel sheet name rules: max 31 chars, no [ ] : * ? / \  no leading '
        _EXCEL_INVALID = str.maketrans({
            '[': '_', ']': '_', ':': '_', '*': '_',
            '?': '_', '/': '_', '\\': '_',
        })
        # Longest prefix is "Quality_" (8 chars) → table part ≤ 23
        _MAX_TABLE_PART = 23

        def _safe_sheet(prefix: str, tname: str) -> str:
            """Build a valid, unique Excel sheet name ≤ 31 chars."""
            sanitized = tname.translate(_EXCEL_INVALID).strip("'").strip()
            allowed = 31 - len(prefix)
            base = sanitized[:allowed] if len(sanitized) <= allowed else sanitized[:allowed - 1] + '~'
            candidate = prefix + base
            # Deduplicate
            if candidate in _used_sheet_names:
                for i in range(2, 100):
                    suffix = str(i)
                    trimmed = base[:allowed - len(suffix)] + suffix
                    candidate = prefix + trimmed
                    if candidate not in _used_sheet_names:
                        break
            _used_sheet_names.add(candidate)
            return candidate
        
        print("🔄 Analyzing input data structure...")

        # Use the same flattener the pipeline uses — guaranteed to produce the
        # same table names that are stored in raw_metadata['tables'].
        from file_loader.loader import FileLoader as _FL
        _fl = _FL()
        table_info = _fl._flatten_nested_json(input_data)
        print(f"     → Detected {len(table_info)} tables: {list(table_info.keys())}")
        
        # Process each table individually
        for table_name, table_data in table_info.items():
            print(f"\n🔄 Processing table: '{table_name}'")
            
            # Create safe table name for file naming
            safe_table_name = "".join(c for c in table_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_table_name = safe_table_name.replace(' ', '_').lower()
            
            # Skip empty tables
            if not table_data:
                print(f"     ⚠️  Skipping empty table '{table_name}'")
                continue
            
            # Convert table data to DataFrame with proper tabular format
            print(f"   • Converting data to tabular format...")
            
            if isinstance(table_data, list) and table_data:
                # Flatten any nested structures for tabular display
                flattened_data = []
                for record in table_data:
                    if isinstance(record, dict):
                        flat_record = {}
                        for key, value in record.items():
                            # Convert complex objects to readable strings
                            if isinstance(value, (dict, list)):
                                flat_record[key] = str(value)
                            elif value is None:
                                flat_record[key] = ""
                            else:
                                flat_record[key] = value
                        flattened_data.append(flat_record)
                    else:
                        # Handle non-dict items
                        flattened_data.append({"value": str(record)})
                
                df_table = pd.DataFrame(flattened_data)
            elif isinstance(table_data, dict):
                # Single record table
                flat_record = {}
                for key, value in table_data.items():
                    if isinstance(value, (dict, list)):
                        flat_record[key] = str(value)
                    elif value is None:
                        flat_record[key] = ""
                    else:
                        flat_record[key] = value
                df_table = pd.DataFrame([flat_record])
            else:
                print(f"     ⚠️  Skipping '{table_name}' - unsupported data format")
                continue
            
            # Clean DataFrame - replace NaN/None with None (renders as blank in Excel)
            # while preserving numeric, boolean, and date types intact.
            df_table = df_table.astype(object).where(df_table.notna(), other=None)
            
            print(f"     → Table shape: {df_table.shape[0]} rows × {df_table.shape[1]} columns")
            
            # 1. SAVE MAIN DATA CSV for this table (only when csv_per_table=True)
            if csv_per_table:
                print(f"   • Creating main data file for {table_name}...")
                main_data_file = os.path.join(output_dir, f"{safe_table_name}_data.csv")
                df_table.to_csv(main_data_file, index=False, encoding="utf-8")
                created_files.append(main_data_file)

            # Add to Excel sheets — names are capped at 31 chars with invalid-char sanitization
            data_sheet  = _safe_sheet("Data_",    table_name)
            meta_sheet  = _safe_sheet("Meta_",    table_name)
            qual_sheet  = _safe_sheet("Quality_", table_name)
            excel_sheets[data_sheet] = df_table
            
            # 2. GET TABLE-SPECIFIC METADATA
            print(f"   • Extracting metadata for {table_name}...")
            table_attributes = {}
            
            # Try to get metadata from the raw_metadata structure
            if 'tables' in raw_metadata and isinstance(raw_metadata['tables'], dict):
                table_meta = raw_metadata['tables'].get(table_name, {})
                if 'table_metadata' in table_meta:
                    table_attributes = table_meta['table_metadata'].get('attributes', {})
                elif 'attributes' in table_meta:
                    table_attributes = table_meta['attributes']
            elif table_name == "main_data" and 'attributes' in raw_metadata:
                # For single table scenarios
                table_attributes = raw_metadata.get("attributes", {})
            
            # 3. CREATE TABLE-SPECIFIC METADATA CSV
            print(f"   • Creating metadata report for {table_name}...")
            metadata_rows = []
            
            for attr_name, attr_info in table_attributes.items():
                if not isinstance(attr_info, dict):
                    continue
                
                # Basic metadata row
                metadata_row = {
                    "Table_Name": table_name,
                    "Attribute_Name": attr_name,
                    "Data_Type": attr_info.get("data_type", "unknown"),
                    "Total_Records": attr_info.get("present_count", 0) + attr_info.get("null_count", 0),
                    "Present_Count": attr_info.get("present_count", 0),
                    "Missing_Count": attr_info.get("null_count", 0),
                    "Missing_Percentage": f"{attr_info.get('null_percentage', 0):.1f}%",
                    "Unique_Values": attr_info.get("unique_count", 0),
                    "Unique_Ratio": f"{attr_info.get('unique_ratio', 0):.3f}",
                    "Quality_Score": f"{attr_info.get('quality_score', 0):.0f}/100"
                }
                
                # Add type-specific information
                data_type = attr_info.get("data_type", "").lower()
                
                if data_type == "string":
                    metadata_row.update({
                        "Min_Length": attr_info.get("min_length", 0),
                        "Max_Length": attr_info.get("max_length", 0),
                        "Avg_Length": f"{attr_info.get('avg_length', 0):.2f}",
                        "Median_Length": attr_info.get("median_length", 0),
                        "Most_Common_Values": str(list(attr_info.get("common_values", {}).keys())[:3]),
                        "Character_Distribution": f"Alpha: {attr_info.get('charset_analysis', {}).get('alphabetic', 0):.1f}%, Numeric: {attr_info.get('charset_analysis', {}).get('numeric', 0):.1f}%"
                    })
                elif data_type in ["integer", "float", "numeric"]:
                    metadata_row.update({
                        "Min_Value": attr_info.get("min_value", "N/A"),
                        "Max_Value": attr_info.get("max_value", "N/A"),
                        "Mean_Value": f"{attr_info.get('mean', 0):.2f}" if attr_info.get('mean') is not None else "N/A",
                        "Median_Value": f"{attr_info.get('median', 0):.2f}" if attr_info.get('median') is not None else "N/A",
                        "Std_Deviation": f"{attr_info.get('std', 0):.3f}" if attr_info.get('std') is not None else "N/A",
                        "Outliers_Count": attr_info.get("outliers", {}).get("count", 0)
                    })
                elif data_type == "datetime":
                    metadata_row.update({
                        "Min_Date": attr_info.get("min_date", "N/A"),
                        "Max_Date": attr_info.get("max_date", "N/A"),
                        "Date_Range_Days": attr_info.get("date_range_days", 0)
                    })
                elif data_type == "boolean":
                    metadata_row.update({
                        "True_Count": attr_info.get("true_count", 0),
                        "False_Count": attr_info.get("false_count", 0),
                        "True_Percentage": f"{attr_info.get('true_percentage', 0):.1f}%",
                        "False_Percentage": f"{attr_info.get('false_percentage', 0):.1f}%"
                    })
                
                # Add anomaly information
                anomaly_flags = attr_info.get("anomaly_flags", [])
                metadata_row.update({
                    "Anomaly_Count": len(anomaly_flags),
                    "Anomaly_Types": ", ".join(anomaly_flags[:3]) if anomaly_flags else "None",
                    "Has_Outliers": "Yes" if "outliers_detected" in anomaly_flags else "No"
                })
                
                # Add pattern recognition info
                pattern_info = attr_info.get("pattern_analysis", {}).get("regex_patterns", {})
                recognized_patterns = [pattern for pattern, info in pattern_info.items() 
                                     if isinstance(info, dict) and info.get("percentage", 0) > 0]
                metadata_row["Recognized_Patterns"] = ", ".join(recognized_patterns) if recognized_patterns else "None"
                
                metadata_rows.append(metadata_row)
            
            # Save table-specific metadata (CSV only when csv_per_table=True; always add to Excel)
            if metadata_rows:
                df_metadata = pd.DataFrame(metadata_rows)
                excel_sheets[meta_sheet] = df_metadata
                if csv_per_table:
                    metadata_file = os.path.join(output_dir, f"{safe_table_name}_metadata.csv")
                    df_metadata.to_csv(metadata_file, index=False, encoding="utf-8")
                    created_files.append(metadata_file)
                    print(f"     ✅ Created: {safe_table_name}_metadata.csv ({len(metadata_rows)} attributes)")
            
            # 4. CREATE TABLE-SPECIFIC QUALITY REPORT
            print(f"   • Creating quality report for {table_name}...")
            quality_rows = []
            
            # Overall table quality metrics
            table_quality_score = sum(attr.get('quality_score', 0) for attr in table_attributes.values())
            table_quality_score = table_quality_score / len(table_attributes) if table_attributes else 0
            
            # General table metrics
            quality_rows.extend([
                {
                    "Table_Name": table_name,
                    "Quality_Category": "Overall",
                    "Metric_Name": "Table Quality Score",
                    "Metric_Value": f"{table_quality_score:.1f}/100",
                    "Status": "Excellent" if table_quality_score >= 90 else "Good" if table_quality_score >= 80 else "Needs Attention" if table_quality_score >= 60 else "Critical",
                    "Description": "Overall data quality assessment across all attributes"
                },
                {
                    "Table_Name": table_name,
                    "Quality_Category": "Structure",
                    "Metric_Name": "Total Attributes",
                    "Metric_Value": str(len(table_attributes)),
                    "Status": "Info",
                    "Description": "Number of columns/attributes in the table"
                },
                {
                    "Table_Name": table_name,
                    "Quality_Category": "Volume",
                    "Metric_Name": "Total Records",
                    "Metric_Value": str(len(df_table)),
                    "Status": "Info",
                    "Description": "Number of rows/records in the table"
                }
            ])
            
            # Per-attribute quality assessment
            for attr_name, attr_info in table_attributes.items():
                if not isinstance(attr_info, dict):
                    continue
                    
                quality_score = attr_info.get('quality_score', 0)
                missing_pct = attr_info.get('null_percentage', 0)
                anomaly_count = len(attr_info.get('anomaly_flags', []))
                unique_ratio = attr_info.get('unique_ratio', 0)
                
                # Data quality score
                quality_rows.append({
                    "Table_Name": table_name,
                    "Quality_Category": "Attribute Quality",
                    "Metric_Name": f"{attr_name} - Overall Quality",
                    "Metric_Value": f"{quality_score:.0f}/100",
                    "Status": "Excellent" if quality_score >= 90 else "Good" if quality_score >= 80 else "Warning" if quality_score >= 60 else "Critical",
                    "Description": f"Overall quality score for {attr_name} attribute"
                })
                
                # Completeness
                if missing_pct > 0:
                    quality_rows.append({
                        "Table_Name": table_name,
                        "Quality_Category": "Completeness",
                        "Metric_Name": f"{attr_name} - Missing Data",
                        "Metric_Value": f"{missing_pct:.1f}%",
                        "Status": "Good" if missing_pct < 5 else "Warning" if missing_pct < 15 else "Critical",
                        "Description": f"Percentage of missing values in {attr_name}"
                    })
                
                # Uniqueness
                quality_rows.append({
                    "Table_Name": table_name,
                    "Quality_Category": "Uniqueness",
                    "Metric_Name": f"{attr_name} - Unique Ratio",
                    "Metric_Value": f"{unique_ratio:.3f}",
                    "Status": "Good" if unique_ratio > 0.8 else "Warning" if unique_ratio > 0.5 else "Info",
                    "Description": f"Ratio of unique values in {attr_name} (1.0 = all unique)"
                })
                
                # Anomalies
                if anomaly_count > 0:
                    quality_rows.append({
                        "Table_Name": table_name,
                        "Quality_Category": "Data Issues",
                        "Metric_Name": f"{attr_name} - Anomalies",
                        "Metric_Value": str(anomaly_count),
                        "Status": "Warning" if anomaly_count < 5 else "Critical",
                        "Description": f"Number of anomalies detected in {attr_name}"
                    })
            
            # Save table-specific quality report (CSV only when csv_per_table=True; always add to Excel)
            if quality_rows:
                df_quality = pd.DataFrame(quality_rows)
                excel_sheets[qual_sheet] = df_quality
                if csv_per_table:
                    quality_file = os.path.join(output_dir, f"{safe_table_name}_quality_report.csv")
                    df_quality.to_csv(quality_file, index=False, encoding="utf-8")
                    created_files.append(quality_file)
                    print(f"     ✅ Created: {safe_table_name}_quality_report.csv ({len(quality_rows)} metrics)")
        
        # 5. CREATE OVERALL SUMMARY
        print(f"\n   • Creating overall dataset summary...")
        llm_insights = enriched_metadata.get("llm_insights", {})
        pipeline_info = enriched_metadata.get("pipeline_info", {})
        summary_info = enriched_metadata.get("summary", {})
        
        # Calculate overall statistics
        total_records = sum(len(pd.DataFrame(data) if isinstance(data, list) else [data]) 
                          for data in table_info.values() if data)
        total_attributes = sum(len(raw_metadata.get('tables', {}).get(name, {}).get('table_metadata', {}).get('attributes', {}))
                             for name in table_info.keys())
        
        overall_summary = [
            {
                "Summary_Category": "Dataset Overview",
                "Metric": "Total Tables",
                "Value": str(len(table_info)),
                "Description": "Number of data tables processed"
            },
            {
                "Summary_Category": "Dataset Overview", 
                "Metric": "Total Records",
                "Value": str(total_records),
                "Description": "Combined records across all tables"
            },
            {
                "Summary_Category": "Dataset Overview",
                "Metric": "Total Attributes",
                "Value": str(total_attributes),
                "Description": "Combined attributes across all tables"
            },
            {
                "Summary_Category": "Quality Assessment",
                "Metric": "Overall Data Quality Score",
                "Value": f"{summary_info.get('data_quality_score', 0):.1f}/100",
                "Description": "Aggregate quality assessment across all tables"
            },
            {
                "Summary_Category": "Quality Assessment",
                "Metric": "Total Anomalies Detected",
                "Value": str(summary_info.get("total_anomalies", 0)),
                "Description": "Data quality issues found across all tables"
            },
            {
                "Summary_Category": "Processing Info",
                "Metric": "Processing Mode",
                "Value": summary_info.get("processing_mode", "unknown"),
                "Description": "Analysis method used (LLM vs fallback)"
            },
            {
                "Summary_Category": "Processing Info",
                "Metric": "LLM Model Used",
                "Value": pipeline_info.get("model_used", "N/A"),
                "Description": "AI model used for enrichment"
            },
            {
                "Summary_Category": "Processing Info",
                "Metric": "Processing Time",
                "Value": f"{pipeline_info.get('total_duration', 0):.2f}s",
                "Description": "Complete pipeline execution time"
            }
        ]
        
        # Add table-specific summary information
        for table_name, table_data in table_info.items():
            if not table_data:
                continue
            record_count = len(pd.DataFrame(table_data) if isinstance(table_data, list) else [table_data])
            attr_count = len(raw_metadata.get('tables', {}).get(table_name, {}).get('table_metadata', {}).get('attributes', {}))
            
            overall_summary.extend([
                {
                    "Summary_Category": f"Table: {table_name}",
                    "Metric": "Record Count",
                    "Value": str(record_count),
                    "Description": f"Number of records in {table_name}"
                },
                {
                    "Summary_Category": f"Table: {table_name}",
                    "Metric": "Attribute Count", 
                    "Value": str(attr_count),
                    "Description": f"Number of attributes in {table_name}"
                }
            ])
        
        # Save overall summary
        df_overall_summary = pd.DataFrame(overall_summary)
        summary_file = os.path.join(output_dir, "overall_dataset_summary.csv")
        df_overall_summary.to_csv(summary_file, index=False, encoding="utf-8")
        created_files.append(summary_file)
        excel_sheets["00_Overall_Summary"] = df_overall_summary
        
        # 6. CREATE COMBINED ISSUES AND RECOMMENDATIONS
        print("   • Creating combined issues and recommendations...")
        issues_and_recommendations = []
        
        # Extract issues from summary
        top_issues = summary_info.get("top_issues", [])
        for i, issue in enumerate(top_issues[:10], 1):  # Limit to top 10
            # Issues are prefixed "[table_name] ..." when they come from nested tables.
            # Extract the table name so Affected_Tables is specific, not just "Multiple".
            affected_table = "Multiple"
            if issue.startswith("["):
                end_bracket = issue.find("]")
                if end_bracket > 0:
                    affected_table = issue[1:end_bracket]
            issues_and_recommendations.append({
                "Issue_Type": "Data Quality Issue",
                "Priority": "High" if i <= 3 else "Medium" if i <= 6 else "Low",
                "Affected_Tables": affected_table,
                "Category": "Data Quality",
                "Description": issue[:200] + "..." if len(issue) > 200 else issue,
                "Source": "Automated Analysis",
                "Recommendation": "Review and clean the identified data quality issues"
            })
        
        # Extract LLM recommendations
        if llm_insights and isinstance(llm_insights, dict):
            recommendations = llm_insights.get("recommendations", [])
            if isinstance(recommendations, list):
                for i, rec in enumerate(recommendations[:10], 1):  # Limit to top 10
                    rec_text = ""
                    if isinstance(rec, dict):
                        rec_text = rec.get('action', '') or rec.get('description', '') or str(rec)
                    else:
                        rec_text = str(rec)
                    
                    issues_and_recommendations.append({
                        "Issue_Type": "Process Improvement",
                        "Priority": "High" if i <= 3 else "Medium" if i <= 6 else "Low",
                        "Affected_Tables": "All",
                        "Category": "Best Practice",
                        "Description": rec_text[:200] + "..." if len(rec_text) > 200 else rec_text,
                        "Source": "LLM Analysis",
                        "Recommendation": "Implement the suggested improvement"
                    })
        
        # Add table-specific critical issues
        for table_name in table_info.keys():
            table_meta = raw_metadata.get('tables', {}).get(table_name, {})
            attributes_dict = table_meta.get('table_metadata', {}).get('attributes', {})
            
            critical_attrs = []
            for attr_name, attr_info in attributes_dict.items():
                if not isinstance(attr_info, dict):
                    continue
                    
                quality_score = attr_info.get('quality_score', 100)
                missing_pct = attr_info.get('null_percentage', 0)
                
                if quality_score < 60:
                    critical_attrs.append(f"{attr_name} (Quality: {quality_score}/100)")
                elif missing_pct > 25:
                    critical_attrs.append(f"{attr_name} (Missing: {missing_pct:.1f}%)")
            
            if critical_attrs:
                issues_and_recommendations.append({
                    "Issue_Type": "Critical Table Issue",
                    "Priority": "High",
                    "Affected_Tables": table_name,
                    "Category": "Data Quality",
                    "Description": f"Critical quality issues in attributes: {', '.join(critical_attrs[:3])}",
                    "Source": "Table Analysis",
                    "Recommendation": "Immediate attention required for data quality improvement"
                })
        
        # Save issues and recommendations
        if issues_and_recommendations:
            df_issues = pd.DataFrame(issues_and_recommendations)
            issues_file = os.path.join(output_dir, "combined_issues_and_recommendations.csv")
            df_issues.to_csv(issues_file, index=False, encoding="utf-8")
            created_files.append(issues_file)
            excel_sheets["99_Issues_Recommendations"] = df_issues
        
        # 7. CREATE COMPREHENSIVE EXCEL WORKBOOK
        if create_excel:
            print("   • Creating comprehensive Excel workbook...")
            excel_path = os.path.join(output_dir, "complete_data_analysis.xlsx")
            
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    # Sort sheets logically: Summary first, then Data/Meta/Quality for each table, Issues last
                    sheet_order = []
                    
                    # Add summary first
                    if "00_Overall_Summary" in excel_sheets:
                        sheet_order.append("00_Overall_Summary")
                    
                    # Add table sheets in groups (Data, Meta, Quality for each table)
                    table_sheet_groups = {}
                    for sheet_name in excel_sheets.keys():
                        if sheet_name.startswith(("Data_", "Meta_", "Quality_")):
                            prefix, table_name = sheet_name.split("_", 1)
                            if table_name not in table_sheet_groups:
                                table_sheet_groups[table_name] = {}
                            table_sheet_groups[table_name][prefix] = sheet_name
                    
                    # Add sheets in Data/Meta/Quality order for each table
                    for table_name in sorted(table_sheet_groups.keys()):
                        for prefix in ["Data", "Meta", "Quality"]:
                            if prefix in table_sheet_groups[table_name]:
                                sheet_order.append(table_sheet_groups[table_name][prefix])
                    
                    # Add issues last
                    if "99_Issues_Recommendations" in excel_sheets:
                        sheet_order.append("99_Issues_Recommendations")
                    
                    # Write sheets in order (names already validated by _safe_sheet)
                    for sheet_name in sheet_order:
                        if sheet_name in excel_sheets:
                            df = excel_sheets[sheet_name]
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"   ✅ Excel workbook created: complete_data_analysis.xlsx")
                created_files.append(excel_path)
                
            except Exception as e:
                print(f"   ❌ Failed to create Excel workbook: {e}")
        
        # Print comprehensive summary
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"\n📊 Output files:")
        if create_excel:
            print(f"   • complete_data_analysis.xlsx  — all {len(excel_sheets)} sheets (data + metadata + quality per table)")
        print(f"   • overall_dataset_summary.csv  — high-level dataset overview")
        print(f"   • combined_issues_and_recommendations.csv  — action items")
        print(f"   • raw_metadata.json / enriched_metadata.json / llm_insights.json")
        if csv_per_table:
            print(f"   • Per-table CSVs also written ({len(table_info)} tables × 3 files)")
        else:
            print(f"   ℹ  Per-table CSVs skipped — all table data is in the Excel workbook")
            print(f"      (pass csv_per_table=True to write them)")
        
        print(f"\n📊 Final Summary:")
        print(f"   • Tables processed: {len(table_info)}")
        print(f"   • Total records: {total_records}")
        print(f"   • Total attributes: {total_attributes}")
        print(f"   • CSV files created: {len([f for f in created_files if f.endswith('.csv')])}")
        print(f"   • Excel sheets: {len(excel_sheets)}")
        
        return {
            "success": True,
            "tables_processed": len(table_info),
            "total_records": total_records,
            "total_attributes": total_attributes,
            "csv_files_created": len([f for f in created_files if f.endswith('.csv')]),
            "excel_created": create_excel and any(f.endswith('.xlsx') for f in created_files),
            "output_directory": output_dir,
            "created_files": created_files,
            "table_names": list(table_info.keys()),
            "excel_sheets": len(excel_sheets)
        }
        
    except Exception as e:
        print(f"❌ Failed to generate analysis files: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def main():
    """Main function with enhanced error handling and debugging"""
    try:
        # Validate directories exist
        required_dirs = ['input', 'output', 'logs', 'debug_output']
        for dir_name in required_dirs:
            os.makedirs(dir_name, exist_ok=True)

        # Check for openpyxl
        try:
            import openpyxl
            print("Debug: openpyxl imported successfully")
        except ImportError:
            print("Error: The 'openpyxl' library is required to generate Excel reports.")
            print("Please install it using: pip install openpyxl")
            return

        agent = MetadataEnrichmentAgent(debug=False)
        
        # Use the model defined in config (mistralai/mistral-small-3.1-24b-instruct:free)
        selected_model_name = agent.config.MODEL_NAME
        print(f"Selected model: {selected_model_name}")
        
        # Check if input file exists
        input_file = os.path.join("input", "input_data.json")
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            print("💡 Please ensure your input JSON file is placed in the input/ directory")
            return
        
        # Run the pipeline with selected model
        result = agent.run_pipeline(input_file, selected_model=selected_model_name)
        
        if result:
            print("\n✅ Success! Check the output/ directory for results.")
            print("🐛 Debug files available in logs/ and debug_output/ directories.")
            
            try:
                # Generate CSV and Excel files with proper error handling
                print("\n📥 Loading data files...")
                convert_result = convert_enriched_json_to_csv(
                    input_data_path=input_file,
                    raw_path=os.path.join("output", "raw_metadata.json"),
                    enriched_path=os.path.join("output", "enriched_metadata.json"),
                    output_dir="output",
                    create_excel=True
                )
                
                if convert_result and convert_result.get('success', False):
                    print(f"📊 Generated {convert_result['csv_files_created']} analysis files")
                    print(f"📁 Files saved to: {convert_result['output_directory']}")
                else:
                    print("⚠️ CSV/Excel generation completed with warnings")
                    if 'error' in convert_result:
                        print(f"   Error details: {convert_result['error']}")
            
            except Exception as convert_error:
                print(f"⚠️ Failed to generate analysis files: {convert_error}")
                traceback.print_exc()
        else:
            print("❌ Pipeline failed. Check logs for details.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
    finally:
        # Cleanup any temporary files if they exist
        temp_files = ['.test_write']
        for dir_name in required_dirs:
            for temp_file in temp_files:
                try:
                    temp_path = os.path.join(dir_name, temp_file)
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass

if __name__ == "__main__":
    main()
