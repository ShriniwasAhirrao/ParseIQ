import json
import csv
import pandas as pd
import xmltodict
from pathlib import Path
from typing import Any, Dict, List, Union, Set
import chardet


class FileLoader:
    """Handles loading various file formats and converting them to JSON-like structures"""
    
    def __init__(self):
        self.supported_formats = ['.json', '.csv', '.xml', '.xlsx', '.xls']
    
    def load_file(self, file_path: str) -> Union[List[Dict], Dict]:
        """
        Load file and convert to JSON-like structure
        
        Args:
            file_path: Path to the input file
            
        Returns:
            JSON-like data structure
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size (100MB limit)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            raise ValueError(f"File too large: {file_size_mb:.2f}MB (max 100MB)")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        print(f"Loading {file_extension} file...")
        
        if file_extension == '.json':
            data = self._load_json(file_path)
            # Extract all nested tables from JSON
            flattened_data = self._extract_all_nested_tables(data)
            if len(flattened_data) == 1 and "main_table" in flattened_data:
                return flattened_data["main_table"]
            return flattened_data
        elif file_extension == '.csv':
            return self._load_csv(file_path)
        elif file_extension == '.xml':
            return self._load_xml(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return self._load_excel(file_path)
        else:
            raise ValueError(f"Handler not implemented for: {file_extension}")
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding"""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    
    def _load_json(self, file_path: Path) -> Union[List[Dict], Dict]:
        """Load JSON file"""
        encoding = self._detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            # Return dict as is, or list as is
            if isinstance(data, dict):
                return data
            elif isinstance(data, list):
                return data
            else:
                return [{"value": data}]
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")

    def _extract_all_nested_tables(self, data: Union[Dict, List], parent_path: str = "") -> Dict[str, List[Dict]]:
        """
        Recursively extract all nested tables from JSON data structure
        
        Args:
            data: JSON data structure to process
            parent_path: Path to current level (for naming tables)
            
        Returns:
            Dictionary mapping table names to their data
        """
        # Use the improved flatten method with path elements from parent_path
        path_elements = parent_path.split('.') if parent_path else []
        return self._flatten_nested_json(data, path_elements)

    def _flatten_nested_json(self, data: Union[Dict, List], path_elements: List[str] = None, tables: Dict[str, List[Dict]] = None, _processed_objects: set = None) -> Dict[str, List[Dict]]:
        """
        Recursively extract all nested tables from JSON data structure.
        
        This method comprehensively traverses the entire JSON structure to find all
        lists of dictionaries (tables) regardless of their nesting depth or position.
        
        Args:
            data: JSON data structure to process (dict or list)
            path_elements: Current path in the JSON structure (for table naming)
            tables: Accumulated dictionary of extracted tables
            _processed_objects: Set of object IDs to prevent infinite recursion
            
        Returns:
            Dictionary mapping table names to their data (List[Dict])
            
        Key Improvements:
        - Iterates through ALL records in each list, not just the first one
        - Prevents infinite recursion with object ID tracking
        - Handles edge cases like empty lists and mixed-type lists
        - Optimizes duplicate table extraction
        - Maintains memory efficiency through selective processing
        """
        # Initialize default parameters
        if tables is None:
            tables = {}
        if path_elements is None:
            path_elements = []
        if _processed_objects is None:
            _processed_objects = set()

        # Prevent infinite recursion by tracking processed objects
        obj_id = id(data)
        if obj_id in _processed_objects:
            return tables
        _processed_objects.add(obj_id)

        try:
            if isinstance(data, dict):
                # If this is the initial call (root level) and the data is a dict,
                # add it as 'main_table'
                if not path_elements:
                    if "main_table" not in tables:
                        tables["main_table"] = []
                    tables["main_table"].append(data)

                # Process each key-value pair in the dictionary
                for key, value in data.items():
                    if value is None:  # Handle None values gracefully
                        continue

                    # Recursively process the value with updated path
                    self._flatten_nested_json(
                        value,
                        path_elements + [key],
                        tables,
                        _processed_objects
                    )

            elif isinstance(data, list):
                # Handle empty lists gracefully
                if not data:
                    return tables

                # Check if this list represents a table (all elements are dictionaries)
                dict_items = [item for item in data if isinstance(item, dict)]

                # If we have dictionary items, this could be a table
                if dict_items:
                    # Always treat as table if it contains dictionary items
                    # Generate unique table name from path
                    table_name = self._generate_table_name(path_elements, tables)

                    # Store the dictionary items as a table
                    # Filter out non-dictionary items for table consistency
                    if table_name not in tables:
                        tables[table_name] = []
                    tables[table_name].extend(dict_items)

                    # Process ALL dictionary items in the list
                    for index, item in enumerate(data):
                        if isinstance(item, dict):
                            # Create path for this specific record
                            item_path = path_elements + [f"[{index}]"]

                            # Recursively process each dictionary record to find nested tables
                            self._flatten_nested_json(
                                item,
                                item_path,
                                tables,
                                _processed_objects
                            )
                        elif isinstance(item, list):
                            # Handle nested lists that might contain tables
                            item_path = path_elements + [f"[{index}]"]
                            self._flatten_nested_json(
                                item,
                                item_path,
                                tables,
                                _processed_objects
                            )
                # This 'else' block was causing the IndentationError. It should be aligned with the 'if dict_items:' block.
                # If the list contains no dictionaries, but might contain nested lists
                else:
                    for index, item in enumerate(data):
                        if isinstance(item, (dict, list)):
                            item_path = path_elements + [f"[{index}]"]
                            self._flatten_nested_json(
                                item,
                                item_path,
                                tables,
                                _processed_objects
                            )

            # Handle primitive types gracefully (shouldn't recurse further)
            # This covers strings, numbers, booleans, None

        except (AttributeError, TypeError) as e:
            # Handle malformed nested structures gracefully
            # Log the error but continue processing other parts
            print(f"Warning: Malformed structure at path {'.'.join(path_elements)}: {e}")

        finally:
            # Remove from processed set when done to allow reprocessing in different contexts
            # This is important for memory efficiency in large datasets
            _processed_objects.discard(obj_id)

        return tables

    def _generate_table_name(self, path_elements: List[str], existing_tables: Dict[str, Any]) -> str:
        """
        Generate a unique table name from path elements.
        
        Args:
            path_elements: List of path components
            existing_tables: Dictionary of already existing table names
            
        Returns:
            Unique table name string
        """
        if not path_elements:
            base_name = 'main_table'
        else:
            # Use the last meaningful path element as base name
            # Filter out array indices like [0], [1], etc.
            meaningful_parts = [part for part in path_elements if not (part.startswith('[') and part.endswith(']'))]
            base_name = '_'.join(meaningful_parts) if meaningful_parts else 'table'
        
        # Clean the base name - remove special characters and spaces
        base_name = ''.join(c if c.isalnum() else '_' for c in base_name).strip('_')
        
        # Ensure base name is not empty
        if not base_name:
            base_name = 'table'
        
        # Generate unique name by appending counter if needed
        table_name = base_name
        
        return table_name
    

    
    def _load_csv(self, file_path: Path) -> List[Dict]:
        """Load CSV file"""
        encoding = self._detect_encoding(file_path)
        
        try:
            # Try to detect delimiter
            with open(file_path, 'r', encoding=encoding) as f:
                sample = f.read(1024)
                sniffer = csv.Sniffer()
                try:
                    delimiter = sniffer.sniff(sample).delimiter
                except csv.Error:
                    # Fall back to comma if detection fails
                    delimiter = ','
            
            # Read CSV with better error handling
            df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, 
                           on_bad_lines='skip', low_memory=False)
            
            # Handle empty CSV
            if df.empty:
                return []
            
            # Clean column names (remove leading/trailing whitespace)
            df.columns = df.columns.str.strip()
            
            # Convert to JSON-like structure, handling NaN values
            return df.fillna('').to_dict('records')
            
        except Exception as e:
            raise ValueError(f"Error loading CSV: {e}")
    
    def _load_xml(self, file_path: Path) -> Union[List[Dict], Dict[str, List[Dict]]]:
        """Load XML file and extract nested tables"""
        encoding = self._detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                xml_content = f.read()
            
            # Convert XML to dict
            data = xmltodict.parse(xml_content)
            
            # Extract nested tables from XML structure
            tables = self._extract_all_nested_tables(data)
            
            # If only one table, return it directly as a list
            if len(tables) == 1:
                return list(tables.values())[0]
            
            return tables
            
        except Exception as e:
            raise ValueError(f"Error loading XML: {e}")
    
    def _load_excel(self, file_path: Path) -> Union[List[Dict], Dict[str, List[Dict]]]:
        """Load Excel file, handling multiple sheets"""
        try:
            # Get all sheet names
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) == 1:
                # Single sheet - return as simple list
                df = pd.read_excel(file_path, sheet_name=0)
                
                if df.empty:
                    return []
                
                # Clean column names
                df.columns = df.columns.str.strip()
                return df.fillna('').to_dict('records')
            else:
                # Multiple sheets - return as dictionary of tables
                tables = {}
                for sheet_name in sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    if not df.empty:
                        # Clean column names
                        df.columns = df.columns.str.strip()
                        clean_sheet_name = sheet_name.replace(" ", "_").replace("-", "_")
                        tables[clean_sheet_name] = df.fillna('').to_dict('records')
                
                return tables if tables else []
            
        except Exception as e:
            raise ValueError(f"Error loading Excel: {e}")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        info = {
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "is_supported": file_path.suffix.lower() in self.supported_formats,
        }
        
        # Add encoding info for text-based files
        if file_path.suffix.lower() in ['.json', '.csv', '.xml']:
            try:
                info["encoding"] = self._detect_encoding(file_path)
            except Exception:
                info["encoding"] = "unknown"
        
        # Add additional info for specific formats
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            try:
                excel_file = pd.ExcelFile(file_path)
                info["sheet_count"] = len(excel_file.sheet_names)
                info["sheet_names"] = excel_file.sheet_names
            except Exception:
                info["sheet_count"] = "unknown"
                info["sheet_names"] = []
        
        return info
    
    def preview_file(self, file_path: str, max_records: int = 5) -> Dict[str, Any]:
        """
        Preview file contents without loading everything into memory
        
        Args:
            file_path: Path to the file
            max_records: Maximum number of records to preview per table
            
        Returns:
            Dictionary with preview information
        """
        try:
            data = self.load_file(file_path)
            
            preview = {
                "file_info": self.get_file_info(file_path),
                "structure": {}
            }
            
            if isinstance(data, dict):
                # Multiple tables
                for table_name, table_data in data.items():
                    if isinstance(table_data, list) and table_data:
                        preview["structure"][table_name] = {
                            "record_count": len(table_data),
                            "columns": list(table_data[0].keys()) if table_data[0] else [],
                            "sample_records": table_data[:max_records]
                        }
            elif isinstance(data, list):
                # Single table
                preview["structure"]["main_table"] = {
                    "record_count": len(data),
                    "columns": list(data[0].keys()) if data and data[0] else [],
                    "sample_records": data[:max_records]
                }
            
            return preview
            
        except Exception as e:
            return {
                "error": str(e),
                "file_info": self.get_file_info(file_path) if Path(file_path).exists() else None
            }