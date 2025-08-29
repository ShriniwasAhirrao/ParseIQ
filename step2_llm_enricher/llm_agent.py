
import json
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
import requests
from datetime import datetime, date
import os
import sys
import statistics
from collections import Counter, defaultdict
import time
class LLMEnricher:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLM enricher with configuration."""
        self.config = config
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url', 'https://api.mistral.ai/v1')
        self.model = config.get('model', 'mistral-large-latest')
        self.max_tokens = config.get('max_tokens', 4000)
        self.temperature = config.get('temperature', 0.1)
        self.debug = config.get('debug', False)
        self.current_date = datetime.now().date()
        
        # Setup logging with UTF-8 support
        self.logger = self._setup_logging()
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        
        # Enhanced validation weights for scoring
        self.scoring_weights = {
            'completeness': 0.30,
            'validity': 0.25,
            'consistency': 0.25,
            'business_rule_compliance': 0.20
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration with UTF-8 encoding."""
        logger = logging.getLogger('LLMEnricher')
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # File handler with UTF-8 encoding
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/llm_debug_{timestamp}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler with UTF-8 support
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        if self.debug:
            logger.addHandler(console_handler)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        self.logger = logger
        logger.info(f"Debug logging initialized. Log file: {log_file}")
        
        return logger
        
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        template_path = self.config.get('prompt_template_path', 'step2_llm_enricher/prompt_template.txt')
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            self.logger.info(f"Loaded prompt template from {template_path}")
            if self.debug:
                print(f"🔍 DEBUG: Loaded prompt template from {template_path}")
            return template
        except FileNotFoundError:
            error_msg = f"Prompt template not found at {template_path}. Please ensure the template file exists."
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    def _detect_table_structure(self, metadata_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if we have single or multiple tables and analyze structure."""
        table_info = {
            "table_count": 1,
            "table_names": [],
            "dataset_type": "single_table",
            "cross_table_relationships": [],
            "total_records": 0
        }
        
        # Try to detect multiple tables from metadata structure
        if 'tables' in metadata_summary and isinstance(metadata_summary['tables'], dict):
            table_info["table_count"] = len(metadata_summary['tables'])
            table_info["table_names"] = list(metadata_summary['tables'].keys())
            table_info["dataset_type"] = "multi_table" if table_info["table_count"] > 1 else "single_table"
            table_info["total_records"] = sum(
                table.get('record_count', 0) for table in metadata_summary['tables'].values()
            )
        else:
            # Single table scenario
            table_info["table_names"] = [metadata_summary.get('table_name', 'unknown_table')]
            table_info["total_records"] = metadata_summary.get('total_records', 0)
        
        return table_info
    
    def _perform_comprehensive_logical_validation(self, metadata_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive logical validation on the metadata."""
        validation_results = {
            "temporal_issues": [],
            "cross_field_issues": [],
            "logical_inconsistencies": [],
            "future_dates": [],
            "age_inconsistencies": [],
            "email_format_issues": [],
            "impossible_sequences": [],
            "business_logic_issues": [],
            "missing_required_fields": [],
            "data_type_violations": [],
            "referential_integrity_issues": [],
            "statistical_anomalies": []
        }
    
        # Get sample data for validation
        raw_data = self._extract_sample_data(metadata_summary)
        
        if not raw_data:
            self.logger.warning("No sample data available for logical validation")
            validation_results['total_records_validated'] = 0
            return validation_results

        self.logger.info(f"Validating {len(raw_data)} records for logical consistency")

        # Track total records processed
        total_records_processed = 0
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

        # Validate each record
        for i, record in enumerate(raw_data):
            if not isinstance(record, dict):
                continue
                
            total_records_processed += 1

            # Get comprehensive validation for this record
            record_issues = self._validate_record_comprehensive(record, i)

            # Categorize issues by type and severity
            for issue in record_issues:
                issue_lower = issue.lower()
                severity = self._determine_issue_severity(issue)
                severity_counts[severity] += 1
                
                # Categorize by type
                if 'future' in issue_lower:
                    validation_results["future_dates"].append(issue)
                    validation_results["temporal_issues"].append(issue)
                elif 'missing required field' in issue_lower or 'empty' in issue_lower:
                    validation_results["missing_required_fields"].append(issue)
                elif 'email' in issue_lower and ('mismatch' in issue_lower or 'name' in issue_lower):
                    validation_results["cross_field_issues"].append(issue)
                elif 'email' in issue_lower and 'format' in issue_lower:
                    validation_results["email_format_issues"].append(issue)
                elif 'invalid format' in issue_lower or 'non-numeric' in issue_lower:
                    validation_results["data_type_violations"].append(issue)
                elif 'age' in issue_lower:
                    validation_results["age_inconsistencies"].append(issue)
                elif 'sequence' in issue_lower or 'order' in issue_lower:
                    validation_results["impossible_sequences"].append(issue)
                elif 'impossible value' in issue_lower or 'unrealistic value' in issue_lower:
                    validation_results["business_logic_issues"].append(issue)
                elif 'statistical' in issue_lower or 'outlier' in issue_lower:
                    validation_results["statistical_anomalies"].append(issue)
                else:
                    validation_results["logical_inconsistencies"].append(issue)

        # Add metadata about the validation process
        validation_results['total_records_validated'] = total_records_processed
        validation_results['validation_timestamp'] = datetime.now().isoformat()
        validation_results['severity_distribution'] = severity_counts

        # Calculate validation rates
        validation_results['validation_rates'] = self._calculate_validation_rates(validation_results, total_records_processed)

        # Log summary of findings
        total_issues = sum(len(issues) for issues in validation_results.values() 
                          if isinstance(issues, list))

        self.logger.info(f"Logical validation complete: {total_issues} total issues found across {total_records_processed} records")

        if self.debug:
            print(f"🔍 DEBUG: Validation Summary - {total_issues} total issues in {total_records_processed} records:")
            for issue_type, issues in validation_results.items():
                if isinstance(issues, list) and issues:
                    print(f"  📋 {issue_type}: {len(issues)} issues")
                    for issue in issues[:2]:
                        print(f"    - {issue}")
                    if len(issues) > 2:
                        print(f"    ... and {len(issues) - 2} more")

        return validation_results
    
    def _extract_sample_data(self, metadata_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sample data from various possible locations in metadata."""
        raw_data = metadata_summary.get('raw_sample_data', [])
        if not raw_data:
            raw_data = metadata_summary.get('sample_data', [])
            if not raw_data:
                raw_data = metadata_summary.get('records', [])
                if not raw_data:
                    raw_data = metadata_summary.get('data', [])
                    if not raw_data and 'tables' in metadata_summary:
                        # Multi-table scenario
                        all_data = []
                        for table_name, table_data in metadata_summary['tables'].items():
                            table_records = table_data.get('sample_data', [])
                            if not table_records:
                                table_records = table_data.get('records', [])
                            all_data.extend(table_records)
                        raw_data = all_data
        return raw_data
    
    def _determine_issue_severity(self, issue: str) -> str:
        """Determine the severity level of an issue based on its description."""
        issue_lower = issue.lower()
        
        if any(keyword in issue_lower for keyword in ['critical', 'missing required field', 'impossible value', 'referential integrity']):
            return "CRITICAL"
        elif any(keyword in issue_lower for keyword in ['invalid format', 'business logic', 'unrealistic value', 'format error']):
            return "HIGH"
        elif any(keyword in issue_lower for keyword in ['mismatch', 'inconsistency', 'outlier']):
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_validation_rates(self, validation_results: Dict[str, Any], total_records: int) -> Dict[str, float]:
        """Calculate validation success rates for different aspects."""
        if total_records == 0:
            return {
                "completeness_rate": 0.0,
                "validity_rate": 0.0,
                "consistency_rate": 0.0,
                "business_rule_compliance_rate": 0.0
            }
        
        # Count issues by category
        missing_fields = len(validation_results.get("missing_required_fields", []))
        format_issues = len(validation_results.get("data_type_violations", [])) + len(validation_results.get("email_format_issues", []))
        consistency_issues = len(validation_results.get("cross_field_issues", [])) + len(validation_results.get("age_inconsistencies", []))
        business_issues = len(validation_results.get("business_logic_issues", []))
        
        return {
            "completeness_rate": max(0.0, (total_records - missing_fields) / total_records * 100),
            "validity_rate": max(0.0, (total_records - format_issues) / total_records * 100),
            "consistency_rate": max(0.0, (total_records - consistency_issues) / total_records * 100),
            "business_rule_compliance_rate": max(0.0, (total_records - business_issues) / total_records * 100)
        }
    
    def _validate_record_comprehensive(self, record: Dict[str, Any], record_index: int) -> List[str]:
        """Comprehensive validation of a single record."""
        issues = []
        
        # Extract all possible field variations
        user_id = self._extract_user_id(record, record_index)
        name = self._extract_name(record)
        email = self._extract_email(record)
        age = self._extract_age(record)
        signup_date = self._extract_signup_date(record)
        purchase_dates = self._extract_purchase_dates(record)
        
        # 1. TEMPORAL VALIDATION (Critical)
        issues.extend(self._validate_temporal_logic(user_id, signup_date, purchase_dates))
        
        # 2. CROSS-FIELD VALIDATION
        issues.extend(self._validate_cross_field_consistency(user_id, name, email, age, signup_date))
        
        # 3. EMAIL FORMAT VALIDATION
        issues.extend(self._validate_email_format(user_id, email))
        
        # 4. BUSINESS LOGIC VALIDATION (Enhanced)
        issues.extend(self._validate_business_logic(user_id, record))
        
        # 5. STATISTICAL ANOMALY DETECTION
        issues.extend(self._detect_statistical_anomalies(user_id, record))
        
        return issues
    
    def _detect_statistical_anomalies(self, user_id: str, record: Dict[str, Any]) -> List[str]:
        """Detect statistical anomalies in numeric fields."""
        issues = []
        
        # Define numeric fields to check
        numeric_fields = ['salary', 'age', 'years_experience', 'budget', 'revenue', 'performance_rating']
        
        for field in numeric_fields:
            if field in record and record[field] is not None:
                try:
                    value = float(record[field])
                    # Check for extreme values based on field type
                    if field == 'salary' and (value > 1000000 or value < 15000):
                        issues.append(f"STATISTICAL ANOMALY: User {user_id} has unusual {field}: {value}")
                    elif field == 'age' and (value > 100 or value < 16):
                        issues.append(f"STATISTICAL ANOMALY: User {user_id} has unusual {field}: {value}")
                    elif field == 'years_experience' and (value > 50 or value < 0):
                        issues.append(f"STATISTICAL ANOMALY: User {user_id} has unusual {field}: {value}")
                    elif field == 'performance_rating' and (value > 5.0 or value < 0.0):
                        issues.append(f"STATISTICAL ANOMALY: User {user_id} has out-of-range {field}: {value}")
                except (ValueError, TypeError):
                    pass  # Already caught in business logic validation
        
        return issues
    
    def _extract_user_id(self, record: Dict[str, Any], record_index: int) -> str:
        """Extract user ID from various possible fields."""
        possible_keys = ['user_id', 'id', 'userId', 'ID', 'User_ID', 'employee_id', 'emp_id', 'EMP_ID']
        for key in possible_keys:
            if key in record and record[key] is not None:
                return str(record[key])
        return f"Record_{record_index}"
    
    def _extract_name(self, record: Dict[str, Any]) -> str:
        """Extract name from various possible fields."""
        possible_keys = ['name', 'user', 'username', 'first_name', 'fullname', 'user_name', 'employee_name', 'full_name']
        for key in possible_keys:
            if key in record and record[key] is not None:
                return str(record[key]).strip()
        return ""
    
    def _extract_email(self, record: Dict[str, Any]) -> str:
        """Extract email from various possible fields."""
        possible_keys = ['email', 'email_address', 'mail', 'e_mail', 'work_email', 'company_email']
        for key in possible_keys:
            if key in record and record[key] is not None:
                return str(record[key]).strip()
        return ""
    
    def _extract_age(self, record: Dict[str, Any]) -> Optional[int]:
        """Extract age from various possible fields."""
        possible_keys = ['age', 'Age', 'user_age', 'employee_age']
        for key in possible_keys:
            if key in record and record[key] is not None:
                try:
                    return int(record[key])
                except (ValueError, TypeError):
                    continue
        return None
    
    def _extract_signup_date(self, record: Dict[str, Any]) -> Optional[str]:
        """Extract signup date from various possible fields."""
        possible_keys = ['signup_date', 'created_at', 'registration_date', 'join_date', 'created', 'signup', 'hire_date', 'start_date']
        for key in possible_keys:
            if key in record and record[key] is not None:
                return str(record[key])
        return None
    
    def _extract_purchase_dates(self, record: Dict[str, Any]) -> List[str]:
        """Extract purchase dates from various possible fields."""
        possible_keys = ['purchase_date', 'last_purchase', 'purchase_dates', 'order_date', 'transaction_date', 'project_end_date', 'completion_date']
        purchase_dates = []
        
        for key in possible_keys:
            if key in record and record[key] is not None:
                value = record[key]
                if isinstance(value, list):
                    purchase_dates.extend([str(d) for d in value if d is not None])
                else:
                    purchase_dates.append(str(value))
        
        return purchase_dates
    
    def _validate_temporal_logic(self, user_id: str, signup_date: Optional[str], purchase_dates: List[str]) -> List[str]:
        """Validate temporal logic and relationships."""
        issues = []
        
        # Check signup date
        if signup_date:
            signup_parsed = self._parse_date(signup_date)
            if signup_parsed:
                if signup_parsed > self.current_date:
                    issues.append(f"CRITICAL: User {user_id} has future signup date: {signup_date}")
                # Check if signup date is too far in the past (unrealistic)
                years_ago = (self.current_date - signup_parsed).days / 365.25
                if years_ago > 50:
                    issues.append(f"TEMPORAL ANOMALY: User {user_id} has signup date {years_ago:.1f} years ago: {signup_date}")
        
        # Check purchase dates
        for purchase_date in purchase_dates:
            if purchase_date:
                purchase_parsed = self._parse_date(purchase_date)
                if purchase_parsed:
                    if purchase_parsed > self.current_date:
                        issues.append(f"CRITICAL: User {user_id} has future purchase date: {purchase_date}")
                    
                    # Check purchase vs signup sequence
                    if signup_date:
                        signup_parsed = self._parse_date(signup_date)
                        if signup_parsed and purchase_parsed < signup_parsed:
                            issues.append(f"LOGICAL ERROR: User {user_id} purchased ({purchase_date}) before signup ({signup_date})")
        
        return issues
    
    def _validate_cross_field_consistency(self, user_id: str, name: str, email: str, age: Optional[int], signup_date: Optional[str]) -> List[str]:
        """Validate consistency between related fields."""
        issues = []
        
        # Email-Name consistency check
        if email and name and '@' in email:
            email_local = email.split('@')[0].lower()
            name_lower = name.lower()
            
            if len(name_lower) > 2 and len(email_local) > 2:
                # Extract first name if full name provided
                first_name = name_lower.split()[0] if ' ' in name_lower else name_lower
                
                # Check for reasonable similarity
                if not self._names_reasonably_similar(first_name, email_local):
                    issues.append(f"CROSS-FIELD MISMATCH: User {user_id} name '{name}' doesn't match email '{email}' (local: {email_local})")
        
        # Age-Signup date consistency
        if age is not None and signup_date:
            signup_parsed = self._parse_date(signup_date)
            if signup_parsed:
                years_since_signup = (self.current_date - signup_parsed).days / 365.25
                estimated_age_at_signup = age - years_since_signup
                
                # Flag unrealistic ages at signup
                if estimated_age_at_signup < 13:  # Too young for most services
                    issues.append(f"AGE INCONSISTENCY: User {user_id} would have been {estimated_age_at_signup:.1f} years old at signup (too young)")
                elif estimated_age_at_signup > 100:  # Unrealistically old
                    issues.append(f"AGE INCONSISTENCY: User {user_id} would have been {estimated_age_at_signup:.1f} years old at signup (unrealistic)")
        
        return issues
    
    def _names_reasonably_similar(self, name: str, email_local: str) -> bool:
        """Check if name and email local part are reasonably similar."""
        # Remove common separators and numbers
        email_clean = re.sub(r'[0-9._-]', '', email_local)
        name_clean = re.sub(r'[^a-z]', '', name)
        
        # Allow for common variations
        if not name_clean or not email_clean:
            return True  # Can't validate empty strings
        
        # Check if name is contained in email or vice versa
        if name_clean in email_clean or email_clean in name_clean:
            return True
        
        # Check for common prefixes (first 3 characters)
        if len(name_clean) >= 3 and len(email_clean) >= 3:
            if name_clean[:3] in email_clean or email_clean[:3] in name_clean:
                return True
        
        # More strict check - no similarity found
        return False
    
    def _validate_email_format(self, user_id: str, email: str) -> List[str]:
        """Validate email format."""
        issues = []
        
        if email:
            # Basic format check
            if not self._is_valid_email_format(email):
                issues.append(f"EMAIL FORMAT ERROR: User {user_id} has invalid email format: '{email}'")
        
        return issues
    
    def _validate_business_logic(self, user_id: str, record: Dict[str, Any]) -> List[str]:
        """Enhanced business logic validation with comprehensive rule checking."""
        issues = []
        
        # Check for obviously invalid ages
        age = self._extract_age(record)
        if age is not None:
            if age < 0:
                issues.append(f"IMPOSSIBLE VALUE: User {user_id} has negative age: {age}")
            elif age > 150:
                issues.append(f"UNREALISTIC VALUE: User {user_id} has age over 150: {age}")
            elif age < 13:  # Business rule: minimum age for service
                issues.append(f"BUSINESS RULE VIOLATION: User {user_id} below minimum age requirement: {age}")

        # Enhanced salary validation with business rules
        if 'salary' in record:
            salary = record['salary']
            if salary is not None:
                try:
                    salary_num = float(salary)
                    if salary_num < 0:
                        issues.append(f"IMPOSSIBLE VALUE: User {user_id} has negative salary: {salary}")
                    elif salary_num > 10000000:  # 10M threshold
                        issues.append(f"UNREALISTIC VALUE: User {user_id} has extremely high salary: {salary}")
                    elif salary_num < 15000:  # Minimum wage business rule
                        issues.append(f"BUSINESS RULE VIOLATION: User {user_id} salary below minimum threshold: {salary}")
                    elif salary_num > 1000000:  # Flag for review
                        issues.append(f"REVIEW REQUIRED: User {user_id} has unusually high salary requiring validation: {salary}")
                except (ValueError, TypeError):
                    issues.append(f"INVALID FORMAT: User {user_id} has non-numeric salary: {salary}")
            else:
                # Check if salary is required but missing/null
                issues.append(f"MISSING REQUIRED FIELD: User {user_id} has null/missing salary")

        # Enhanced name validation with business rules
        name = self._extract_name(record)
        if not name or str(name).strip() == '' or str(name).lower() in ['null', 'none', 'n/a', 'unknown', 'test', 'admin']:
            issues.append(f"MISSING REQUIRED FIELD: User {user_id} has empty or invalid name")
        elif len(str(name).strip()) < 2:
            issues.append(f"BUSINESS RULE VIOLATION: User {user_id} name too short: '{name}'")
        elif len(str(name).strip()) > 100:
            issues.append(f"BUSINESS RULE VIOLATION: User {user_id} name too long: '{name}'")
        elif not self._is_valid_name_format(str(name)):
            issues.append(f"INVALID FORMAT: User {user_id} has invalid name format: '{name}'")

        # Enhanced date validation with business logic
        date_fields = ['hire_date', 'signup_date', 'created_date', 'birth_date', 'last_login', 'registration_date', 'start_date', 'end_date']
        for date_field in date_fields:
            if date_field in record and record[date_field] is not None:
                date_value = str(record[date_field]).strip()
                # Check for obviously invalid date values
                if date_value.lower() in ['invalid-date', 'invalid_date', 'null', 'none', 'n/a', '', '0000-00-00']:
                    issues.append(f"INVALID DATE FORMAT: User {user_id} has invalid {date_field}: '{record[date_field]}'")
                elif date_value and not self._is_valid_date_format(date_value):
                    issues.append(f"INVALID DATE FORMAT: User {user_id} has malformed {date_field}: '{record[date_field]}'")
                else:
                    # Parse and validate date logic
                    parsed_date = self._parse_date(date_value)
                    if parsed_date:
                        # Check for future dates where inappropriate
                        from datetime import date
                        today = date.today()
                        if date_field in ['hire_date', 'signup_date', 'birth_date', 'created_date'] and parsed_date > today:
                            issues.append(f"BUSINESS RULE VIOLATION: User {user_id} has future {date_field}: {date_value}")
                        
                        # Check for unrealistic birth dates
                        if date_field == 'birth_date':
                            if parsed_date.year < 1900:
                                issues.append(f"UNREALISTIC VALUE: User {user_id} has birth_date before 1900: {date_value}")
                            elif (today - parsed_date).days > 150 * 365:  # Over 150 years
                                issues.append(f"UNREALISTIC VALUE: User {user_id} has birth_date indicating age over 150: {date_value}")

        # Cross-field date validation
        self._validate_date_sequences(user_id, record, issues)

        # Enhanced email validation with business rules
        email = self._extract_email(record)
        if email:
            if not self._is_valid_email_format(email):
                issues.append(f"INVALID EMAIL FORMAT: User {user_id} has invalid email: '{email}'")
            elif not self._is_business_email_valid(email):
                issues.append(f"BUSINESS RULE VIOLATION: User {user_id} has non-business email format: '{email}'")
        elif 'email' in record:  # Email field exists but is empty
            issues.append(f"MISSING REQUIRED FIELD: User {user_id} has empty email")

        # Check for required fields that are unexpectedly null
        required_fields = ['id', 'user_id', 'employee_id', 'emp_id', 'customer_id']
        for field in required_fields:
            if field in record and (record[field] is None or str(record[field]).strip() == ''):
                issues.append(f"MISSING REQUIRED FIELD: User {user_id} has empty {field}")

        # Enhanced boolean field validation
        boolean_fields = ['is_active', 'is_remote', 'is_deleted', 'is_verified', 'is_admin', 'is_enabled']
        for bool_field in boolean_fields:
            if bool_field in record and record[bool_field] is not None:
                bool_value = record[bool_field]
                if not isinstance(bool_value, bool) and str(bool_value).lower() not in ['true', 'false', '1', '0', 'yes', 'no']:
                    issues.append(f"INVALID BOOLEAN VALUE: User {user_id} has invalid {bool_field}: '{bool_value}'")

        # Validate numeric fields that should be positive with business rules
        positive_numeric_fields = ['years_experience', 'employee_count', 'revenue', 'budget', 'hours_worked', 'commission']
        for num_field in positive_numeric_fields:
            if num_field in record and record[num_field] is not None:
                try:
                    num_value = float(record[num_field])
                    if num_value < 0:
                        issues.append(f"IMPOSSIBLE VALUE: User {user_id} has negative {num_field}: {num_value}")
                    elif num_field == 'years_experience' and num_value > 70:
                        issues.append(f"UNREALISTIC VALUE: User {user_id} has excessive {num_field}: {num_value}")
                    elif num_field == 'hours_worked' and num_value > 168:  # Hours in a week
                        issues.append(f"IMPOSSIBLE VALUE: User {user_id} has {num_field} exceeding weekly limit: {num_value}")
                except (ValueError, TypeError):
                    issues.append(f"INVALID FORMAT: User {user_id} has non-numeric {num_field}: '{record[num_field]}'")

        # Enhanced percentage field validation
        percentage_fields = ['completion_rate', 'success_rate', 'discount_rate', 'tax_rate', 'commission_rate']
        for pct_field in percentage_fields:
            if pct_field in record and record[pct_field] is not None:
                try:
                    pct_value = float(record[pct_field])
                    if pct_value < 0 or pct_value > 100:
                        issues.append(f"OUT OF RANGE: User {user_id} has {pct_field} outside 0-100%: {pct_value}")
                except (ValueError, TypeError):
                    issues.append(f"INVALID FORMAT: User {user_id} has non-numeric {pct_field}: '{record[pct_field]}'")

        # Phone number validation
        if 'phone' in record or 'phone_number' in record:
            phone = record.get('phone') or record.get('phone_number')
            if phone and not self._is_valid_phone_format(str(phone)):
                issues.append(f"INVALID PHONE FORMAT: User {user_id} has invalid phone number: '{phone}'")

        # ID format consistency validation
        self._validate_id_formats(user_id, record, issues)

        # Department/category validation
        self._validate_categorical_fields(user_id, record, issues)

        return issues

    def _validate_date_sequences(self, user_id: str, record: Dict[str, Any], issues: List[str]) -> None:
        """Validate logical date sequences (start <= end dates)."""
        date_pairs = [
            ('start_date', 'end_date'),
            ('hire_date', 'termination_date'),
            ('signup_date', 'last_login'),
            ('created_date', 'updated_date'),
            ('birth_date', 'hire_date')
        ]
        
        for start_field, end_field in date_pairs:
            if start_field in record and end_field in record:
                start_date = self._parse_date(str(record[start_field])) if record[start_field] else None
                end_date = self._parse_date(str(record[end_field])) if record[end_field] else None
                
                if start_date and end_date and start_date > end_date:
                    issues.append(f"IMPOSSIBLE SEQUENCE: User {user_id} has {start_field} ({start_date}) after {end_field} ({end_date})")

    def _is_valid_name_format(self, name: str) -> bool:
        """Validate name format with business rules."""
        import re
        
        # Check for valid name pattern (letters, spaces, hyphens, apostrophes)
        name_pattern = r"^[a-zA-Z\s\-'\.]+$"
        if not re.match(name_pattern, name):
            return False
        
        # Check for obvious test/invalid names
        invalid_names = ['test', 'admin', 'user', 'null', 'undefined', 'anonymous']
        if name.lower().strip() in invalid_names:
            return False
        
        # Check for reasonable structure (at least one letter)
        if not re.search(r'[a-zA-Z]', name):
            return False
        
        return True

    def _is_business_email_valid(self, email: str) -> bool:
        """Validate email against business rules."""
        email_lower = email.lower()
        
        # Check for obviously fake or test domains
        invalid_domains = ['test.com', 'example.com', 'fake.com', 'temp.com', 'dummy.com']
        domain = email_lower.split('@')[1] if '@' in email_lower else ''
        
        if domain in invalid_domains:
            return False
        
        # Check for disposable email patterns
        disposable_patterns = ['temp', 'throw', 'fake', 'test', '10min']
        if any(pattern in domain for pattern in disposable_patterns):
            return False
        
        return True

    def _is_valid_phone_format(self, phone: str) -> bool:
        """Validate phone number format."""
        import re
        
        # Remove common formatting characters
        clean_phone = re.sub(r'[\s\-\(\)\+\.]', '', phone)
        
        # Check if it's all digits after cleaning
        if not clean_phone.isdigit():
            return False
        
        # Check reasonable length (7-15 digits)
        if len(clean_phone) < 7 or len(clean_phone) > 15:
            return False
        
        return True

    def _validate_id_formats(self, user_id: str, record: Dict[str, Any], issues: List[str]) -> None:
        """Validate ID field formats for consistency."""
        id_fields = ['employee_id', 'customer_id', 'account_id', 'order_id']
        
        for id_field in id_fields:
            if id_field in record and record[id_field] is not None:
                id_value = str(record[id_field]).strip()
                
                # Check for empty IDs
                if not id_value:
                    issues.append(f"MISSING REQUIRED FIELD: User {user_id} has empty {id_field}")
                
                # Check for obviously invalid ID formats
                elif id_value.lower() in ['null', 'none', 'n/a', 'undefined']:
                    issues.append(f"INVALID ID FORMAT: User {user_id} has invalid {id_field}: '{id_value}'")
                
                # Check ID length constraints
                elif len(id_value) < 2:
                    issues.append(f"INVALID ID FORMAT: User {user_id} has {id_field} too short: '{id_value}'")
                elif len(id_value) > 50:
                    issues.append(f"INVALID ID FORMAT: User {user_id} has {id_field} too long: '{id_value}'")

    def _validate_categorical_fields(self, user_id: str, record: Dict[str, Any], issues: List[str]) -> None:
        """Validate categorical fields against expected values."""
        categorical_validations = {
            'department': ['HR', 'Finance', 'Engineering', 'Sales', 'Marketing', 'Operations', 'IT', 'Legal'],
            'status': ['Active', 'Inactive', 'Pending', 'Suspended', 'Terminated'],
            'priority': ['Low', 'Medium', 'High', 'Critical'],
            'gender': ['Male', 'Female', 'Other', 'Prefer not to say'],
            'country': None,  # Skip validation - too many valid values
            'state': None     # Skip validation - too many valid values
        }
        
        for field, valid_values in categorical_validations.items():
            if field in record and record[field] is not None and valid_values is not None:
                field_value = str(record[field]).strip()
                
                if field_value and field_value not in valid_values:
                    issues.append(f"INVALID CATEGORY: User {user_id} has invalid {field}: '{field_value}' (expected: {', '.join(valid_values)})")

    def _is_valid_date_format(self, date_str: str) -> bool:
        """Check if a date string has a valid format (not necessarily a valid date)."""
        if not date_str or str(date_str).strip() == '':
            return False

        date_str = str(date_str).strip()

        # Common valid date patterns
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY or DD/MM/YYYY
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO datetime
            r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY or MM-DD-YYYY
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$',  # YYYY-MM-DD HH:MM:SS
        ]

        import re
        for pattern in date_patterns:
            if re.match(pattern, date_str):
                return True

        return False

    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse various date formats robustly."""
        if not date_str or str(date_str).strip() == '':
            return None
        
        date_str = str(date_str).strip()
        
        # Common date formats to try
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
            '%d-%m-%Y',
            '%m-%d-%Y'
        ]
        
        from datetime import datetime, date
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt).date()
                return parsed_date
            except ValueError:
                continue
        
        # Try parsing ISO format with timezone
        try:
            if 'T' in date_str:
                # Remove timezone info if present
                clean_date = date_str.split('T')[0]
                return datetime.strptime(clean_date, '%Y-%m-%d').date()
        except ValueError:
            pass
        
        return None

    def _is_valid_email_format(self, email: str) -> bool:
        """Validate email format with comprehensive regex."""
        if not email:
            return False
        
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, str(email)) is not None

    def _calculate_corrected_quality_score(self, metadata_summary: Dict[str, Any], logical_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate corrected quality score with severe penalties for logical errors."""

        # Count logical issues by severity
        critical_issues = len(logical_validation.get('future_dates', []))
        major_issues = (len(logical_validation.get('cross_field_issues', [])) + 
                       len(logical_validation.get('impossible_sequences', [])))
        minor_issues = (len(logical_validation.get('age_inconsistencies', [])) + 
                       len(logical_validation.get('email_format_issues', [])))

        # Enhanced: Count business logic issues from the enhanced validation
        business_issues = len(logical_validation.get('business_logic_issues', []))
        
        # New: Count different types of business rule violations
        impossible_values = len([issue for issue in logical_validation.get('business_logic_issues', []) 
                               if 'IMPOSSIBLE VALUE' in issue])
        unrealistic_values = len([issue for issue in logical_validation.get('business_logic_issues', []) 
                                if 'UNREALISTIC VALUE' in issue])
        missing_required = len([issue for issue in logical_validation.get('business_logic_issues', []) 
                              if 'MISSING REQUIRED FIELD' in issue])
        format_issues = len([issue for issue in logical_validation.get('business_logic_issues', []) 
                           if 'INVALID FORMAT' in issue])

        total_logical_issues = critical_issues + major_issues + minor_issues + business_issues

        # Get original technical score - Enhanced: Handle multiple possible locations
        original_score = metadata_summary.get('data_quality_summary', {}).get('overall_score', 0)
        if original_score == 0:
            # Fallback to different possible locations in the metadata structure
            original_score = metadata_summary.get('overall_score', 0)
            if original_score == 0:
                original_score = metadata_summary.get('data_quality_score', 0)
                if original_score == 0:
                    # Try to find score in raw_metadata if it exists
                    raw_metadata = metadata_summary.get('raw_metadata', {})
                    original_score = raw_metadata.get('data_quality_score', 0)

        # Get total records - Enhanced: Handle multiple possible locations
        total_records = metadata_summary.get('dataset_overview', {}).get('total_records', 0)
        if total_records <= 0:
            total_records = metadata_summary.get('total_records', 0)
            if total_records <= 0:
                total_records = logical_validation.get('total_records_validated', 1)

        # Enhanced penalty calculation with weighted severity
        critical_penalty = critical_issues * 25      # 25 points per future date
        major_penalty = major_issues * 15           # 15 points per cross-field issue
        minor_penalty = minor_issues * 8            # 8 points per format issue
        business_penalty = business_issues * 18     # 18 points per business logic issue (heavy penalty)
        
        # Additional penalties for specific violation types
        impossible_penalty = impossible_values * 20  # Extra penalty for impossible values
        missing_required_penalty = missing_required * 15  # Heavy penalty for missing required fields

        total_penalty = (critical_penalty + major_penalty + minor_penalty + business_penalty + 
                        impossible_penalty + missing_required_penalty)

        # Apply penalties to original score
        corrected_score = max(0, original_score - total_penalty)

        # Enhanced quality grading with more nuanced business rule considerations
        if critical_issues > 0 or impossible_values > 0:
            quality_grade = "F" if (critical_issues > 5 or impossible_values > 3) else "D"
            corrected_score = min(corrected_score, 40)
        elif missing_required > 2 or business_issues > 5:
            quality_grade = "D"
            corrected_score = min(corrected_score, 50)
        elif major_issues > 5 or business_issues > 2:
            quality_grade = "D" if business_issues > 3 else "C"
            corrected_score = min(corrected_score, 60)
        elif major_issues > 0 or minor_issues > 5 or business_issues > 0:
            quality_grade = "C"
            corrected_score = min(corrected_score, 75)
        elif minor_issues > 0:
            quality_grade = "B"
            corrected_score = min(corrected_score, 85)
        else:
            quality_grade = "A" if corrected_score >= 90 else "B"

        # Calculate error rate
        error_rate = (total_logical_issues / max(total_records, 1)) * 100
        
        # Calculate component scores for weighted methodology
        component_scores = self._calculate_component_scores(
            metadata_summary, logical_validation, total_records
        )
        
        return {
            "corrected_score": corrected_score,
            "quality_grade": quality_grade,
            "total_logical_issues": total_logical_issues,
            "critical_issues": critical_issues,
            "major_issues": major_issues,
            "minor_issues": minor_issues,
            "business_issues": business_issues,
            "impossible_values": impossible_values,
            "unrealistic_values": unrealistic_values,
            "missing_required_fields": missing_required,
            "format_issues": format_issues,
            "error_rate_percentage": round(error_rate, 2),
            "original_score": original_score,
            "total_penalty": total_penalty,
            "total_records": total_records,
            "component_scores": component_scores,
            "scoring_methodology": {
                "completeness_weight": 30,
                "validity_weight": 25,
                "consistency_weight": 25,
                "business_rule_compliance_weight": 20
            }
        }

    def _calculate_component_scores(self, metadata_summary: Dict[str, Any], 
                                   logical_validation: Dict[str, Any], 
                                   total_records: int) -> Dict[str, float]:
        """Calculate individual component scores for weighted methodology."""
        
        # Get base completeness from metadata
        base_completeness = metadata_summary.get('data_quality_summary', {}).get('completeness_rate', 80)
        
        # Calculate completeness penalties
        missing_required = len([issue for issue in logical_validation.get('business_logic_issues', []) 
                              if 'MISSING REQUIRED FIELD' in issue])
        completeness_penalty = (missing_required / max(total_records, 1)) * 100 * 2  # 2x multiplier
        completeness_score = max(0, base_completeness - completeness_penalty)
        
        # Calculate validity score
        format_issues = len([issue for issue in logical_validation.get('business_logic_issues', []) 
                           if 'INVALID FORMAT' in issue])
        base_validity = 85  # Assume base validity
        validity_penalty = (format_issues / max(total_records, 1)) * 100 * 3  # 3x multiplier
        validity_score = max(0, base_validity - validity_penalty)
        
        # Calculate consistency score
        cross_field_issues = len(logical_validation.get('cross_field_issues', []))
        impossible_sequences = len(logical_validation.get('impossible_sequences', []))
        consistency_issues = cross_field_issues + impossible_sequences
        base_consistency = 80
        consistency_penalty = (consistency_issues / max(total_records, 1)) * 100 * 4  # 4x multiplier
        consistency_score = max(0, base_consistency - consistency_penalty)
        
        # Calculate business rule compliance score
        business_issues = len(logical_validation.get('business_logic_issues', []))
        impossible_values = len([issue for issue in logical_validation.get('business_logic_issues', []) 
                               if 'IMPOSSIBLE VALUE' in issue])
        unrealistic_values = len([issue for issue in logical_validation.get('business_logic_issues', []) 
                                if 'UNREALISTIC VALUE' in issue])
        
        base_business_compliance = 75
        business_penalty = ((business_issues + impossible_values * 2 + unrealistic_values) / 
                          max(total_records, 1)) * 100 * 5  # 5x multiplier for business rules
        business_compliance_score = max(0, base_business_compliance - business_penalty)
        
        return {
            "completeness": round(completeness_score, 1),
            "validity": round(validity_score, 1),
            "consistency": round(consistency_score, 1),
            "business_rule_compliance": round(business_compliance_score, 1)
        }

    def test_connection(self) -> bool:
        """Test the LLM API connection."""
        try:
            self.logger.info("Testing API connection")
            if self.debug:
                print("🔍 DEBUG: Testing API connection")
                
            response = self._make_api_request("Return only this JSON: {\"status\": \"ok\", \"test\": true}")
            
            self.logger.info(f"Test response received")
            if self.debug:
                print(f"🔍 DEBUG: Test response: '{response[:200]}...'")
                
            # Save test response for debugging
            self._save_debug_file("test_response.txt", response)
            
            # Try to extract and parse JSON
            json_content = self._extract_json_from_response(response)
            if json_content:
                parsed = json.loads(json_content)
                if parsed.get('status') == 'ok':
                    self.logger.info("✅ LLM API connection and JSON parsing successful")
                    if self.debug:
                        print("🔍 DEBUG: ✅ LLM API connection and JSON parsing successful")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"API connection test failed: {e}")
            if self.debug:
                print(f"🔍 DEBUG: API connection test failed: {e}")
            return False
            
    def _make_api_request(self, prompt: str) -> str:
        """Make a request to the LLM API with enhanced error handling and retry logic."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f'{self.base_url}/chat/completions',
                    headers=headers,
                    json=data,
                    timeout=180  # Increased timeout for complex multi-table analysis
                )
                response.raise_for_status()
                
                result = response.json()
                return result['choices'][0]['message']['content']
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
            except KeyError as e:
                self.logger.error(f"Unexpected API response format: {e}")
                raise
                
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON content from LLM response with enhanced parsing methods."""
        if self.debug:
            print("🔧 Starting JSON extraction from response...")
            
        # Method 1: Look for ```json markdown blocks
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, response, re.MULTILINE | re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            if self.debug:
                print("✅ Found JSON in ```json markdown block")
                print(f"📏 Extracted JSON length: {len(json_content)} characters")
            return json_content
            
        # Method 2: Look for plain ``` blocks that might contain JSON
        plain_code_pattern = r'```\s*([\s\S]*?)\s*```'
        match = re.search(plain_code_pattern, response, re.MULTILINE | re.DOTALL)
        if match:
            potential_json = match.group(1).strip()
            # Check if it's valid JSON
            try:
                json.loads(potential_json)
                if self.debug:
                    print("✅ Found JSON in plain ``` block")
                return potential_json
            except json.JSONDecodeError:
                pass
                
        # Method 3: Look for JSON-like structure without markdown
        # Find content that starts with { and ends with }
        brace_pattern = r'\{[\s\S]*\}'
        match = re.search(brace_pattern, response, re.MULTILINE | re.DOTALL)
        if match:
            potential_json = match.group(0).strip()
            try:
                json.loads(potential_json)
                if self.debug:
                    print("✅ Found JSON without markdown blocks")
                return potential_json
            except json.JSONDecodeError:
                pass
                
        # Method 4: Try to find the start of JSON and extract everything after
        json_start = response.find('{')
        if json_start != -1:
            potential_json = response[json_start:].strip()
            # Find the last complete JSON object
            brace_count = 0
            end_pos = -1
            for i, char in enumerate(potential_json):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
                        
            if end_pos > 0:
                potential_json = potential_json[:end_pos]
                try:
                    json.loads(potential_json)
                    if self.debug:
                        print("✅ Found JSON by brace matching")
                    return potential_json
                except json.JSONDecodeError:
                    pass
                    
        # Method 5: Try to extract JSON from text that might have explanatory text
        # Look for phrases that might precede JSON
        json_intro_patterns = [
            r'(?:here is|here\'s|below is|the json is|json output|analysis result|assessment result):\s*(\{[\s\S]*\})',
            r'(?:json|analysis|assessment|result):\s*(\{[\s\S]*\})',
            r'(\{[\s\S]*\})\s*(?:this is|that is|above is)'
        ]
        
        for pattern in json_intro_patterns:
            match = re.search(pattern, response, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            if match:
                potential_json = match.group(1).strip()
                try:
                    json.loads(potential_json)
                    if self.debug:
                        print("✅ Found JSON with intro pattern")
                    return potential_json
                except json.JSONDecodeError:
                    pass
                    
        if self.debug:
            print("❌ No valid JSON found in response")
            print(f"🔍 Response preview: {response[:200]}...")
            
        return None
            
    def _save_debug_file(self, filename: str, content: str):
        """Save debug content to file with UTF-8 encoding and timestamp."""
        if not self.debug:
            return
            
        debug_dir = 'debug_output'
        os.makedirs(debug_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(debug_dir, f"{timestamp}_{filename}")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                if isinstance(content, dict):
                    json.dump(content, f, indent=2, default=str, ensure_ascii=False)
                else:
                    f.write(content)
            self.logger.info(f"Saved debug content to {filepath}")
            if self.debug:
                print(f"🔍 DEBUG: Saved debug content to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save debug file: {e}")
            
    def enrich_metadata(self, metadata_summary: Dict[str, Any], model: str = None) -> Dict[str, Any]:
        """Enrich metadata using LLM analysis with comprehensive multi-table validation.""" 
        try:
            if model:
                self.logger.info(f"Overriding model to: {model}")
                self.model = model
            self.logger.info("Starting comprehensive metadata enrichment process")
            if self.debug:
                print("🔍 DEBUG: Starting comprehensive metadata enrichment process")
            
            # STEP 1: Detect dataset type (single vs multi-table)
            dataset_type = self._detect_dataset_type(metadata_summary)
            
            # STEP 2: Perform comprehensive logical validation
            logical_validation = self._perform_comprehensive_logical_validation(metadata_summary)
            
            # STEP 3: Calculate corrected quality metrics with component scores
            corrected_metrics = self._calculate_corrected_quality_score(metadata_summary, logical_validation)
            
            # STEP 4: Analyze cross-table relationships if multi-table
            cross_table_analysis = {}
            if dataset_type == "multi_table":
                cross_table_analysis = self._analyze_cross_table_relationships(metadata_summary)
            
            # STEP 5: Perform business rule validation
            business_rule_violations = self._validate_business_rules(metadata_summary)
            
            # STEP 6: Create enhanced metadata with all validation context
            enhanced_metadata = metadata_summary.copy()
            enhanced_metadata.update({
                'dataset_type': dataset_type,
                'logical_validation': logical_validation,
                'corrected_quality_metrics': corrected_metrics,
                'cross_table_analysis': cross_table_analysis,
                'business_rule_violations': business_rule_violations,
                'current_date_context': self.current_date.isoformat(),
                'validation_timestamp': datetime.now().isoformat(),
                'analysis_scope': {
                    'multi_table_analysis': dataset_type == "multi_table",
                    'business_rule_validation': True,
                    'cross_field_validation': True,
                    'temporal_validation': True,
                    'format_validation': True
                }
            })
            
            # Adjust prompt metadata_summary for multi-table structure
            prompt_metadata = enhanced_metadata
            if dataset_type == "multi_table":
                # Recursively flatten tables metadata for prompt to avoid nested 'table_metadata' keys
                def recursive_flatten(metadata):
                    if not isinstance(metadata, dict):
                        return metadata
                    if 'table_metadata' in metadata and isinstance(metadata['table_metadata'], dict):
                        # Merge table_metadata content into current dict
                        flattened = metadata['table_metadata'].copy()
                        # Recursively flatten nested keys
                        for k, v in flattened.items():
                            flattened[k] = recursive_flatten(v)
                        return flattened
                    else:
                        # Recursively flatten all dict values
                        return {k: recursive_flatten(v) for k, v in metadata.items()}
                
                tables = prompt_metadata.get('tables', {})
                flattened_tables = {}
                for table_name, table_data in tables.items():
                    flattened_tables[table_name] = recursive_flatten(table_data)
                prompt_metadata['tables'] = flattened_tables
            
            # Format the prompt with enhanced context
            formatted_prompt = self.prompt_template.format(
                metadata_summary=json.dumps(prompt_metadata, indent=2, default=str),
                current_date=self.current_date.isoformat()
            )
            
            # Save enhanced metadata for debugging
            self._save_debug_file("enhanced_metadata.json", enhanced_metadata)
            
            if self.debug:
                print(f"🔍 DEBUG: Enhanced metadata prepared with {corrected_metrics['total_logical_issues']} logical issues")
                print(f"🔍 DEBUG: Dataset type: {dataset_type}")
                print(f"🔍 DEBUG: Quality grade: {corrected_metrics['quality_grade']}, Score: {corrected_metrics['corrected_score']}")
            
            # STEP 7: Make LLM API request for analysis
            self.logger.info("Sending enhanced metadata to LLM for analysis")
            if self.debug:
                print("🔍 DEBUG: Sending request to LLM API...")
                
            try:
                response = self._make_api_request(formatted_prompt)
            except Exception as api_error:
                self.logger.error(f"LLM API request failed: {api_error}")
                if self.debug:
                    print(f"❌ DEBUG: LLM API request failed: {api_error}")
                # Return fallback enrichment on API request failure
                return self._create_fallback_enrichment(enhanced_metadata, corrected_metrics, logical_validation, cross_table_analysis)
            
            # Save raw response for debugging
            self._save_debug_file("llm_raw_response.txt", response)
            
            # STEP 8: Extract and parse JSON response
            json_content = self._extract_json_from_response(response)
            
            if not json_content:
                self.logger.error("Failed to extract JSON from LLM response")
                if self.debug:
                    print("❌ DEBUG: Failed to extract JSON from LLM response")
                    print(f"🔍 DEBUG: Raw response: {response[:500]}...")
                
                # Return a fallback enriched result
                return self._create_fallback_enrichment(enhanced_metadata, corrected_metrics, logical_validation, cross_table_analysis)
            
            # Save extracted JSON for debugging
            self._save_debug_file("extracted_json.json", json_content)
            
            # Parse the JSON response
            try:
                enriched_data = json.loads(json_content)
                self.logger.info("Successfully parsed LLM response JSON")
                if self.debug:
                    print("✅ DEBUG: Successfully parsed LLM response JSON")
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response: {e}")
                if self.debug:
                    print(f"❌ DEBUG: JSON parsing failed: {e}")
                    print(f"🔍 DEBUG: JSON content: {json_content[:300]}...")
                
                # Return fallback enrichment
                return self._create_fallback_enrichment(enhanced_metadata, corrected_metrics, logical_validation, cross_table_analysis)
            
            # STEP 9: Enhance the response with our calculated metrics
            enriched_data = self._merge_calculated_metrics(enriched_data, corrected_metrics, logical_validation, cross_table_analysis)
            
            # Fix: Ensure 'corrected_score' and 'summary' are present in overall_assessment for display
            overall_assessment = enriched_data.get('overall_assessment', {})
            if 'corrected_score' not in overall_assessment or overall_assessment.get('corrected_score') in [None, 'N/A']:
                overall_assessment['corrected_score'] = corrected_metrics.get('corrected_score', 'N/A')
            if 'summary' not in overall_assessment or not overall_assessment.get('summary'):
                overall_assessment['summary'] = enriched_data.get('summary', 'N/A')
            enriched_data['overall_assessment'] = overall_assessment
            
            # Add metadata about the enrichment process
            enriched_data['enrichment_metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model,
                'validation_performed': True,
                'logical_issues_found': corrected_metrics['total_logical_issues'],
                'data_integrity_checked': True,
                'current_date_context': self.current_date.isoformat(),
                'dataset_type': dataset_type,
                'business_rules_validated': len(business_rule_violations) > 0,
                'cross_table_relationships_analyzed': dataset_type == "multi_table"
            }
            
            # Save final enriched result
            self._save_debug_file("final_enriched_result.json", enriched_data)
            
            self.logger.info(f"Metadata enrichment completed successfully. Quality grade: {enriched_data.get('overall_assessment', {}).get('quality_grade', 'Unknown')}")
            if self.debug:
                print(f"✅ DEBUG: Enrichment completed. Final grade: {enriched_data.get('overall_assessment', {}).get('quality_grade', 'Unknown')}")
            
            return enriched_data
            
        except Exception as e:
            self.logger.error(f"Error during metadata enrichment: {e}")
            if self.debug:
                print(f"❌ DEBUG: Enrichment error: {e}")
            
            # Return fallback enrichment on any error
            return self._create_fallback_enrichment(metadata_summary, {}, {}, {})
    def _detect_dataset_type(self, metadata_summary: Dict[str, Any]) -> str:
        """Detect if dataset is single or multi-table"""
        try:
            # Use 'dataset_overview' if present, else fallback to 'summary'
            dataset_overview = metadata_summary.get('dataset_overview', {})
            if not dataset_overview:
                dataset_overview = metadata_summary.get('summary', {})
            if not isinstance(dataset_overview, dict):
                dataset_overview = {}
            table_summaries = dataset_overview.get('table_summaries', {})
            if not isinstance(table_summaries, dict):
                table_summaries = {}
            if not table_summaries and 'table_record_counts' in dataset_overview and isinstance(dataset_overview['table_record_counts'], dict):
                # Create table_summaries from table_record_counts for fallback
                table_summaries = {k: {'record_count': v, 'field_analysis': {}} for k, v in dataset_overview['table_record_counts'].items()}
            
            # Count the number of tables
            table_count = len(table_summaries)
            
            if self.debug:
                print(f"🔍 DEBUG: Detected {table_count} tables in dataset")
                print(f"🔍 DEBUG: Table names: {list(table_summaries.keys())}")
            
            # Determine dataset type
            if table_count <= 1:
                dataset_type = "single_table"
            else:
                dataset_type = "multi_table"
                
            self.logger.info(f"Dataset type detected: {dataset_type} ({table_count} tables)")
            return dataset_type
            
        except Exception as e:
            self.logger.error(f"Error detecting dataset type: {e}")
            return "single_table"  # Default fallback

    def _analyze_cross_table_relationships(self, metadata_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships between tables in multi-table scenarios"""
        try:
            if self.debug:
                print("🔍 DEBUG: Starting cross-table relationship analysis")
                
            dataset_overview = metadata_summary.get('dataset_overview', {})
            table_summaries = dataset_overview.get('table_summaries', {})
            
            relationships = []
            referential_violations = []
            orphaned_records = []
            consistency_issues = []
            
            table_names = list(table_summaries.keys())
            
            # Analyze potential relationships between tables
            for i, table1_name in enumerate(table_names):
                table1_data = table_summaries[table1_name]
                table1_fields = table1_data.get('field_analysis', {})
                
                for j, table2_name in enumerate(table_names):
                    if i >= j:  # Skip self and already compared pairs
                        continue
                        
                    table2_data = table_summaries[table2_name]
                    table2_fields = table2_data.get('field_analysis', {})
                    
                    # Look for potential foreign key relationships
                    potential_relationships = self._find_potential_relationships(
                        table1_name, table1_fields, table2_name, table2_fields
                    )
                    relationships.extend(potential_relationships)
                    
                    # Check for referential integrity issues
                    ref_violations = self._check_referential_integrity(
                        table1_name, table1_fields, table2_name, table2_fields
                    )
                    referential_violations.extend(ref_violations)
                    
                    # Check for cross-table consistency issues
                    consistency_issues_found = self._check_cross_table_consistency(
                        table1_name, table1_fields, table2_name, table2_fields
                    )
                    consistency_issues.extend(consistency_issues_found)
            
            # Calculate referential integrity score
            total_potential_refs = len(relationships)
            violation_count = len(referential_violations)
            
            if total_potential_refs > 0:
                referential_integrity_score = max(0, 100 - (violation_count * 100 / total_potential_refs))
            else:
                referential_integrity_score = 100  # No relationships means perfect integrity
            
            analysis_result = {
                'relationships': relationships,
                'relationship_violations': referential_violations,
                'orphaned_records': orphaned_records,
                'consistency_issues': consistency_issues,
                'referential_integrity_score': round(referential_integrity_score, 2),
                'schema_evolution_detected': self._detect_schema_evolution(table_summaries),
                'cross_table_stats': {
                    'total_relationships_found': len(relationships),
                    'violations_detected': len(referential_violations),
                    'consistency_issues_found': len(consistency_issues),
                    'tables_analyzed': len(table_names)
                }
            }
            
            if self.debug:
                print(f"✅ DEBUG: Cross-table analysis complete. Found {len(relationships)} relationships, {len(referential_violations)} violations")
                
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in cross-table relationship analysis: {e}")
            return {
                'relationships': [],
                'relationship_violations': [],
                'orphaned_records': [],
                'consistency_issues': [],
                'referential_integrity_score': 100,
                'schema_evolution_detected': False,
                'cross_table_stats': {
                    'total_relationships_found': 0,
                    'violations_detected': 0,
                    'consistency_issues_found': 0,
                    'tables_analyzed': 0
                }
            }

    def _validate_business_rules(self, metadata_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate business rules across the dataset"""
        try:
            if self.debug:
                print("🔍 DEBUG: Starting business rule validation")
                
            violations = []
            dataset_overview = metadata_summary.get('dataset_overview', {})
            table_summaries = dataset_overview.get('table_summaries', {})
            
            for table_name, table_data in table_summaries.items():
                field_analysis = table_data.get('field_analysis', {})
                
                # Rule 1: Critical fields should not have high null rates
                for field_name, field_data in field_analysis.items():
                    null_rate = field_data.get('null_percentage', 0)
                    data_type = field_data.get('data_type', '').lower()
                    
                    # Identify critical fields (IDs, dates, amounts, etc.)
                    is_critical = self._is_critical_field(field_name, data_type)
                    
                    if is_critical and null_rate > 5:  # More than 5% null for critical fields
                        violations.append({
                            'rule': 'Critical Field Completeness',
                            'table': table_name,
                            'field': field_name,
                            'severity': 'HIGH',
                            'description': f'Critical field {field_name} has {null_rate}% null values',
                            'expected': 'Less than 5% null values for critical fields',
                            'actual': f'{null_rate}% null values',
                            'business_impact': 'High - affects data reliability and business processes'
                        })
                
                # Rule 2: Date fields should not have future dates (for historical data)
                for field_name, field_data in field_analysis.items():
                    data_type = field_data.get('data_type', '').lower()
                    
                    if 'date' in data_type or 'time' in data_type:
                        future_date_count = field_data.get('future_date_count', 0)
                        if future_date_count > 0:
                            violations.append({
                                'rule': 'Historical Date Validation',
                                'table': table_name,
                                'field': field_name,
                                'severity': 'MEDIUM',
                                'description': f'Found {future_date_count} future dates in historical data field',
                                'expected': 'All dates should be in the past for historical records',
                                'actual': f'{future_date_count} future dates found',
                                'business_impact': 'Medium - may indicate data entry errors or system clock issues'
                            })
                
                # Rule 3: Email fields should follow proper format
                for field_name, field_data in field_analysis.items():
                    if 'email' in field_name.lower():
                        invalid_email_count = field_data.get('invalid_format_count', 0)
                        if invalid_email_count > 0:
                            violations.append({
                                'rule': 'Email Format Validation',
                                'table': table_name,
                                'field': field_name,
                                'severity': 'MEDIUM',
                                'description': f'Found {invalid_email_count} invalid email formats',
                                'expected': 'All email addresses should follow valid format (user@domain.com)',
                                'actual': f'{invalid_email_count} invalid email formats',
                                'business_impact': 'Medium - affects communication and data quality'
                            })
                
                # Rule 4: Numeric ranges should be within business limits
                for field_name, field_data in field_analysis.items():
                    data_type = field_data.get('data_type', '').lower()
                    
                    if 'int' in data_type or 'float' in data_type or 'decimal' in data_type:
                        min_val = field_data.get('min_value')
                        max_val = field_data.get('max_value')
                        
                        # Check for suspicious ranges
                        if self._is_suspicious_numeric_range(field_name, min_val, max_val):
                            violations.append({
                                'rule': 'Numeric Range Validation',
                                'table': table_name,
                                'field': field_name,
                                'severity': 'LOW',
                                'description': f'Suspicious numeric range: {min_val} to {max_val}',
                                'expected': 'Values within reasonable business ranges',
                                'actual': f'Range: {min_val} to {max_val}',
                                'business_impact': 'Low - may indicate outliers or data entry errors'
                            })
            
            # Cross-table business rules
            cross_table_violations = self._validate_cross_table_business_rules(table_summaries)
            violations.extend(cross_table_violations)
            
            if self.debug:
                print(f"✅ DEBUG: Business rule validation complete. Found {len(violations)} violations")
                
            return violations
            
        except Exception as e:
            self.logger.error(f"Error in business rule validation: {e}")
            return []

    # Helper functions for the main validation functions

    def _find_potential_relationships(self, table1_name: str, table1_fields: Dict, 
                                    table2_name: str, table2_fields: Dict) -> List[Dict]:
        """Find potential foreign key relationships between tables"""
        relationships = []
        
        for field1_name, field1_data in table1_fields.items():
            for field2_name, field2_data in table2_fields.items():
                
                # Check for potential FK relationships based on naming patterns
                if (field1_name.lower().endswith('_id') or field2_name.lower().endswith('_id') or
                    field1_name.lower() == 'id' or field2_name.lower() == 'id' or
                    field1_name.lower().replace('_', '') in field2_name.lower() or
                    field2_name.lower().replace('_', '') in field1_name.lower()):
                    
                    # Check data type compatibility
                    type1 = field1_data.get('data_type', '').lower()
                    type2 = field2_data.get('data_type', '').lower()
                    
                    if self._are_types_compatible(type1, type2):
                        relationships.append({
                            'type': 'potential_foreign_key',
                            'parent_table': table2_name,
                            'parent_field': field2_name,
                            'child_table': table1_name,
                            'child_field': field1_name,
                            'confidence': self._calculate_relationship_confidence(
                                field1_name, field2_name, type1, type2
                            )
                        })
        
        return relationships

    def _check_referential_integrity(self, table1_name: str, table1_fields: Dict,
                                table2_name: str, table2_fields: Dict) -> List[Dict]:
        """Check for referential integrity violations"""
        violations = []
        
        # This would need actual data to check referential integrity
        # For now, we'll check for potential issues based on metadata
        
        for field1_name, field1_data in table1_fields.items():
            if field1_name.lower().endswith('_id'):
                null_rate = field1_data.get('null_percentage', 0)
                
                # Foreign keys should generally not be null
                if null_rate > 10:  # More than 10% null
                    violations.append({
                        'type': 'high_null_foreign_key',
                        'table': table1_name,
                        'field': field1_name,
                        'issue': f'Potential foreign key has {null_rate}% null values',
                        'severity': 'MEDIUM'
                    })
        
        return violations

    def _check_cross_table_consistency(self, table1_name: str, table1_fields: Dict,
                                    table2_name: str, table2_fields: Dict) -> List[Dict]:
        """Check for consistency issues across tables"""
        issues = []
        
        # Check for fields with similar names but different data types
        for field1_name, field1_data in table1_fields.items():
            for field2_name, field2_data in table2_fields.items():
                
                # Similar field names
                if (field1_name.lower() == field2_name.lower() or 
                    self._are_field_names_similar(field1_name, field2_name)):
                    
                    type1 = field1_data.get('data_type', '').lower()
                    type2 = field2_data.get('data_type', '').lower()
                    
                    if not self._are_types_compatible(type1, type2):
                        issues.append({
                            'type': 'inconsistent_data_types',
                            'table1': table1_name,
                            'field1': field1_name,
                            'type1': type1,
                            'table2': table2_name,
                            'field2': field2_name,
                            'type2': type2,
                            'severity': 'MEDIUM'
                        })
        
        return issues

    def _detect_schema_evolution(self, table_summaries: Dict) -> bool:
        """Detect if schema evolution has occurred"""
        # This is a simplified check - in reality, you'd compare with historical schema
        field_count_variance = []
        
        for table_name, table_data in table_summaries.items():
            field_count = len(table_data.get('field_analysis', {}))
            field_count_variance.append(field_count)
        
        # If there's significant variance in field counts, might indicate evolution
        if len(field_count_variance) > 1:
            max_fields = max(field_count_variance)
            min_fields = min(field_count_variance)
            
            # If more than 50% difference in field counts
            if max_fields > 0 and (max_fields - min_fields) / max_fields > 0.5:
                return True
        
        return False

    def _is_critical_field(self, field_name: str, data_type: str) -> bool:
        """Determine if a field is critical for business operations"""
        critical_patterns = [
            'id', 'key', 'code', 'number', 'date', 'time', 'amount', 'price', 
            'cost', 'total', 'sum', 'count', 'quantity', 'status', 'state'
        ]
        
        field_lower = field_name.lower()
        
        # Check if field name contains critical patterns
        for pattern in critical_patterns:
            if pattern in field_lower:
                return True
        
        # Primary key-like fields are critical
        if field_lower in ['id', 'pk', 'key', 'uid', 'uuid']:
            return True
        
        return False

    def _is_suspicious_numeric_range(self, field_name: str, min_val, max_val) -> bool:
        """Check if numeric range seems suspicious"""
        if min_val is None or max_val is None:
            return False
        
        try:
            min_num = float(min_val)
            max_num = float(max_val)
            
            # Age fields should be reasonable
            if 'age' in field_name.lower():
                return min_num < 0 or max_num > 150
            
            # Price/amount fields should be positive
            if any(term in field_name.lower() for term in ['price', 'amount', 'cost', 'total']):
                return min_num < 0
            
            # Very large ranges might be suspicious
            if max_num - min_num > 1000000:  # Range > 1 million
                return True
                
        except (ValueError, TypeError):
            pass
        
        return False

    def _validate_cross_table_business_rules(self, table_summaries: Dict) -> List[Dict]:
        """Validate business rules that span multiple tables"""
        violations = []
        
        # Example: Check if customer table exists and has reasonable relationship with orders
        table_names = list(table_summaries.keys())
        
        # Look for common business entity patterns
        has_customer_table = any('customer' in name.lower() for name in table_names)
        has_order_table = any('order' in name.lower() for name in table_names)
        
        if has_customer_table and has_order_table:
            # Should have referential relationship
            customer_tables = [name for name in table_names if 'customer' in name.lower()]
            order_tables = [name for name in table_names if 'order' in name.lower()]
            
            for order_table in order_tables:
                order_fields = table_summaries[order_table].get('field_analysis', {})
                has_customer_ref = any('customer' in field.lower() for field in order_fields.keys())
                
                if not has_customer_ref:
                    violations.append({
                        'rule': 'Customer-Order Relationship',
                        'table': order_table,
                        'field': 'N/A',
                        'severity': 'MEDIUM',
                        'description': 'Order table lacks customer reference field',
                        'expected': 'Order table should reference customer table',
                        'actual': 'No customer reference found',
                        'business_impact': 'Medium - affects order-customer relationship tracking'
                    })
        
        return violations

    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two data types are compatible for relationships"""
        # Normalize type names
        type1 = type1.lower().strip()
        type2 = type2.lower().strip()
        
        # Exact match
        if type1 == type2:
            return True
        
        # Integer types
        int_types = ['int', 'integer', 'bigint', 'smallint', 'tinyint']
        if type1 in int_types and type2 in int_types:
            return True
        
        # String types
        string_types = ['varchar', 'char', 'text', 'string', 'nvarchar']
        if any(t in type1 for t in string_types) and any(t in type2 for t in string_types):
            return True
        
        return False

    def _calculate_relationship_confidence(self, field1: str, field2: str, type1: str, type2: str) -> float:
        """Calculate confidence score for potential relationship"""
        confidence = 0.0
        
        # Name similarity
        if field1.lower() == field2.lower():
            confidence += 0.4
        elif field1.lower().replace('_', '') in field2.lower() or field2.lower().replace('_', '') in field1.lower():
            confidence += 0.3
        
        # Type compatibility
        if self._are_types_compatible(type1, type2):
            confidence += 0.3
        
        # ID patterns
        if field1.lower().endswith('_id') or field2.lower().endswith('_id'):
            confidence += 0.2
        
        # Primary key patterns
        if field1.lower() == 'id' or field2.lower() == 'id':
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _are_field_names_similar(self, name1: str, name2: str) -> bool:
        """Check if two field names are similar enough to be considered related"""
        name1_clean = name1.lower().replace('_', '').replace('-', '')
        name2_clean = name2.lower().replace('_', '').replace('-', '')
        
        # Check if one is contained in the other
        if name1_clean in name2_clean or name2_clean in name1_clean:
            return True
        
        # Check for common prefixes/suffixes
        common_parts = ['id', 'code', 'number', 'key', 'ref', 'date', 'time']
        
        for part in common_parts:
            if (name1_clean.endswith(part) and name2_clean.endswith(part)) or \
            (name1_clean.startswith(part) and name2_clean.startswith(part)):
                return True
        
        return False
    def _create_fallback_enrichment(self, metadata_summary: Dict[str, Any], corrected_metrics: Dict[str, Any], logical_validation: Dict[str, Any], cross_table_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a comprehensive fallback enrichment result when LLM processing fails."""
        self.logger.info("Creating fallback enrichment result")
        
        # Calculate basic metrics if not provided
        if not corrected_metrics:
            corrected_metrics = {
                "corrected_score": 50,
                "quality_grade": "C",
                "total_logical_issues": 0,
                "critical_issues": 0,
                "major_issues": 0,
                "minor_issues": 0,
                "error_rate_percentage": 0.0,
                "original_score": 50,
                "total_penalty": 0,
                "component_scores": {
                    "completeness": 50,
                    "validity": 50,
                    "consistency": 50,
                    "business_rule_compliance": 50
                }
            }
        
        # Detect dataset type
        dataset_overview = metadata_summary.get('dataset_overview', {})
        table_names = list(dataset_overview.get('table_summaries', {}).keys()) if 'table_summaries' in dataset_overview else ['unknown_table']
        table_count = len(table_names)
        dataset_type = "multi_table" if table_count > 1 else "single_table"
        
        total_records = dataset_overview.get('total_records', 0)
        
        # Determine production readiness based on quality grade
        production_readiness = "Not Ready"
        if corrected_metrics['quality_grade'] in ['A', 'B']:
            production_readiness = "Ready"
        elif corrected_metrics['quality_grade'] == 'C':
            production_readiness = "Needs Review"
        
        # Determine risk levels
        overall_risk = "Low"
        if corrected_metrics['critical_issues'] > 0:
            overall_risk = "Critical"
        elif corrected_metrics['major_issues'] > 5:
            overall_risk = "High"
        elif corrected_metrics['major_issues'] > 0:
            overall_risk = "Medium"
        
        # Create table analysis
        table_analysis = []
        for table_name in table_names:
            table_summary = dataset_overview.get('table_summaries', {}).get(table_name, {})
            table_analysis.append({
                "table_name": table_name,
                "record_count": table_summary.get('record_count', 0),
                "table_quality_score": corrected_metrics['corrected_score'],
                "completeness_rate": corrected_metrics['component_scores']['completeness'],
                "validity_rate": corrected_metrics['component_scores']['validity'],
                "consistency_rate": corrected_metrics['component_scores']['consistency'],
                "business_rule_compliance": corrected_metrics['component_scores']['business_rule_compliance'],
                "critical_fields": table_summary.get('critical_fields', []),
                "relationship_role": "standalone" if dataset_type == "single_table" else "unknown",
                "key_issues": [
                    f"Automated analysis found {corrected_metrics['critical_issues']} critical issues",
                    f"Detected {corrected_metrics['major_issues']} major data quality issues",
                    f"Business rule compliance at {corrected_metrics['component_scores']['business_rule_compliance']:.1f}%"
                ]
            })
        
        return {
            "dataset_overview": {
                "table_count": table_count,
                "table_names": table_names,
                "dataset_type": dataset_type,
                "cross_table_relationships": cross_table_analysis.get('relationships', []) if cross_table_analysis else [],
                "total_records": total_records,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "overall_assessment": {
                "quality_grade": corrected_metrics['quality_grade'],
                "corrected_score": corrected_metrics['corrected_score'],
                "overall_score": corrected_metrics['corrected_score'],
                "confidence_score": 75,  # Lower confidence for fallback
                "scoring_methodology": {
                    "completeness_weight": 30,
                    "validity_weight": 25,
                    "consistency_weight": 25,
                    "business_rule_compliance_weight": 20
                },
                "component_scores": corrected_metrics['component_scores'],
                "production_readiness": production_readiness,
                "key_strengths": ["Data structure is parseable", "Basic format validation passed"],
                "primary_concerns": [
                    f"Found {corrected_metrics['total_logical_issues']} logical issues",
                    "Business rule validation needed",
                    "Data quality monitoring required"
                ],
                "improvement_potential": "High - automated analysis suggests significant opportunities for improvement",
                "summary": "Fallback analysis used due to LLM processing issue - consider re-running with LLM for detailed insights"
            },
            "table_analysis": table_analysis,
            "cross_table_analysis": cross_table_analysis or {
                "referential_integrity_score": 100 if dataset_type == "single_table" else 80,
                "relationship_violations": [],
                "orphaned_records": [],
                "consistency_issues": [],
                "schema_evolution_detected": False
            },
            "critical_issues": [
                {
                    "issue": "Automated Data Quality Assessment",
                    "severity": "HIGH" if corrected_metrics['critical_issues'] > 0 else "MEDIUM",
                    "category": "Data_Quality",
                    "table": table_names[0] if table_names else "unknown",
                    "description": f"Automated analysis identified {corrected_metrics['total_logical_issues']} data quality issues requiring attention",
                    "business_impact": "Potential impact on data reliability and business operations",
                    "root_cause": "Comprehensive validation reveals data integrity concerns",
                    "affected_records": [],
                    "affected_records_pct": corrected_metrics['error_rate_percentage'],
                    "potential_consequences": ["Data reliability issues", "Business process impacts", "Reporting accuracy concerns"],
                    "specific_fix": "Implement comprehensive data validation and cleansing process"
                }
            ],
            "field_analysis": [],  # Would be populated with actual field data
            "business_rule_violations": [],
            "recommendations": [
                {
                    "priority": "HIGH",
                    "category": "Data_Validation",
                    "action": "Implement comprehensive data quality validation pipeline",
                    "rationale": "Automated analysis suggests significant data quality improvements needed",
                    "estimated_effort": "Medium",
                    "expected_impact": "Significant improvement in data reliability and business confidence",
                    "timeline": "2-4 weeks",
                    "success_criteria": f"Reduce logical issues from {corrected_metrics['total_logical_issues']} to <10",
                    "dependencies": ["Data quality tools implementation", "Business rule definition"],
                    "cost_benefit": "Medium cost, high business value - essential for data-driven decisions"
                }
            ],
            "validation_pipeline_recommendations": {
                "immediate_validations": [
                    {
                        "rule": "Implement null value checks for critical fields",
                        "description": "Ensure data completeness for business-critical fields",
                        "severity": "HIGH"
                    }
                ],
                "cross_table_validations": [] if dataset_type == "single_table" else [
                    {
                        "rule": "Validate referential integrity across related tables",
                        "description": "Ensure cross-table relationships are maintained",
                        "severity": "HIGH"
                    }
                ],
                "format_validations": [
                    {
                        "rule": "Validate data formats match expected patterns",
                        "description": "Ensure data formats comply with business requirements",
                        "severity": "MEDIUM"
                    }
                ]
            },
            "risk_assessment": {
                "overall_risk_level": overall_risk,
                "data_reliability_risk": overall_risk,
                "business_impact_risk": overall_risk,
                "compliance_risk": "MEDIUM",
                "operational_risk": overall_risk,
                "security_risk": "LOW",
                "financial_impact": "MEDIUM",
                "reputation_risk": "MEDIUM",
                "cross_table_integrity_risk": "LOW" if dataset_type == "single_table" else "MEDIUM",
                "risk_details": f"Automated analysis identified {corrected_metrics['total_logical_issues']} issues requiring attention",
                "mitigation_urgency": "High priority for data quality improvement initiative"
            },
            "enrichment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_used": "fallback_analysis",
                "validation_performed": True,
                "logical_issues_found": corrected_metrics['total_logical_issues'],
                "data_integrity_checked": True,
                "current_date_context": self.current_date.isoformat(),
                "dataset_type": dataset_type,
                "note": "Fallback analysis used due to LLM processing issue - consider re-running with LLM for detailed insights"
            }
        }

    def _merge_calculated_metrics(self, enriched_data: Dict[str, Any], corrected_metrics: Dict[str, Any], logical_validation: Dict[str, Any], cross_table_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Merge our calculated metrics with LLM response for comprehensive analysis."""
        
        # Update overall assessment with our calculated metrics
        if 'overall_assessment' in enriched_data:
            enriched_data['overall_assessment']['quality_grade'] = corrected_metrics['quality_grade']
            enriched_data['overall_assessment']['overall_score'] = corrected_metrics['corrected_score']
            enriched_data['overall_assessment']['component_scores'] = corrected_metrics.get('component_scores', {})
            
            # Update production readiness based on critical issues
            if corrected_metrics['critical_issues'] > 0:
                enriched_data['overall_assessment']['production_readiness'] = "Not Ready"
            elif corrected_metrics['major_issues'] > 3:
                enriched_data['overall_assessment']['production_readiness'] = "Needs Review"
        
        # Update or create cross-table analysis section
        if cross_table_analysis:
            enriched_data['cross_table_analysis'] = cross_table_analysis
        
        # Update detailed analysis with our findings
        if 'detailed_analysis' in enriched_data:
            # Ensure our critical findings are highlighted
            enriched_data['detailed_analysis']['critical_logical_errors'] = logical_validation.get('future_dates', [])
            enriched_data['detailed_analysis']['temporal_issues'] = logical_validation.get('temporal_issues', [])
            enriched_data['detailed_analysis']['cross_field_issues'] = logical_validation.get('cross_field_issues', [])
            
            # Update data distribution with accurate counts
            if 'data_distribution' in enriched_data['detailed_analysis']:
                enriched_data['detailed_analysis']['data_distribution'].update({
                    "logical_error_count": corrected_metrics['total_logical_issues'],
                    "future_date_count": corrected_metrics['critical_issues'],
                    "cross_field_mismatch_count": corrected_metrics['major_issues'],
                    "error_rate_percentage": f"{corrected_metrics['error_rate_percentage']:.2f}%"
                })
        
        # Update risk assessment based on our analysis
        if 'risk_assessment' in enriched_data:
            if corrected_metrics['critical_issues'] > 0:
                enriched_data['risk_assessment']['overall_risk_level'] = "CRITICAL"
                enriched_data['risk_assessment']['data_reliability_risk'] = "CRITICAL"
                enriched_data['risk_assessment']['business_impact_risk'] = "HIGH"
            elif corrected_metrics['major_issues'] > 5:
                enriched_data['risk_assessment']['overall_risk_level'] = "HIGH"
                enriched_data['risk_assessment']['data_reliability_risk'] = "HIGH"
            
            # Add cross-table specific risks if applicable
            if cross_table_analysis and cross_table_analysis.get('relationship_violations'):
                enriched_data['risk_assessment']['cross_table_integrity_risk'] = "HIGH"
        
        # Add our validation details
        enriched_data['validation_details'] = {
            'original_score': corrected_metrics['original_score'],
            'corrected_score': corrected_metrics['corrected_score'],
            'total_penalty': corrected_metrics['total_penalty'],
            'critical_issues': corrected_metrics['critical_issues'],
            'major_issues': corrected_metrics['major_issues'],
            'minor_issues': corrected_metrics['minor_issues'],
            'component_breakdown': corrected_metrics.get('component_scores', {}),
            'logical_validation_summary': {
                'future_dates_found': len(logical_validation.get('future_dates', [])),
                'cross_field_mismatches': len(logical_validation.get('cross_field_issues', [])),
                'email_format_issues': len(logical_validation.get('email_format_issues', [])),
                'age_inconsistencies': len(logical_validation.get('age_inconsistencies', [])),
                'impossible_sequences': len(logical_validation.get('impossible_sequences', [])),
                'business_rule_violations': len(logical_validation.get('business_rule_violations', []))
            }
        }
        
        # Add cross-table analysis summary if available
        if cross_table_analysis:
            enriched_data['validation_details']['cross_table_summary'] = {
                'referential_integrity_score': cross_table_analysis.get('referential_integrity_score', 100),
                'relationship_violations': len(cross_table_analysis.get('relationship_violations', [])),
                'orphaned_records': len(cross_table_analysis.get('orphaned_records', [])),
                'consistency_issues': len(cross_table_analysis.get('consistency_issues', []))
            }
        
        return enriched_data
    