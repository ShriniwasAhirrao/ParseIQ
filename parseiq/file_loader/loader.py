import json
import csv
import pandas as pd
import xmltodict
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
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
            # Flatten nested JSON into multiple tables
            flattened_data = self._flatten_nested_json(data)
            # Rename 'main_table' (used when root is an unnamed list) to the
            # filename stem so users see e.g. "input_data" instead of "main_table"
            stem = file_path.stem
            if 'main_table' in flattened_data and stem and stem != 'main_table':
                flattened_data[stem] = flattened_data.pop('main_table')
                path_key = '__path__main_table'
                if path_key in flattened_data:
                    flattened_data[f'__path__{stem}'] = flattened_data.pop(path_key)
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

    def _flatten_nested_json(self, data: Union[Dict, List], path_elements: List[str] = None,
                              tables: Dict[str, List[Dict]] = None) -> Dict[str, List[Dict]]:
        """
        Improved nested JSON flattener.

        Algorithm (based on pandas json_normalize + dbt/Airbyte FK-injection pattern):

        1. Traverse the JSON tree depth-first.
        2. Every list-of-dicts found becomes a named table. The table name is the
           array's key in the parent object.
        3. Same-name tables from sibling records are MERGED into one table
           (e.g., division[0].departments and division[1].departments both land
           in 'departments'). A name collision at a DIFFERENT path gets a path-
           aware suffix (parent__child).
        4. FK injection: before recursing into a child array, inject
           '_ref_<parent_table>_id' into each child record using the parent
           record's own *_id field (or positional index as fallback).
        5. Embedded objects (dicts with no nested arrays/dicts) are flattened
           inline with '__' separator: {"address": {"city": "NY"}} →
           {"address__city": "NY"}.  Deeply nested objects are JSON-stringified.
        6. Primitive arrays (["a","b"]) are joined as comma-separated strings.
        7. null / None values pass through as-is for downstream null-rate analysis.
        """
        if tables is None:
            tables = {}
        if path_elements is None:
            path_elements = []

        if isinstance(data, dict):
            # Check if this dict has any list-of-dicts children; if not AND we're at
            # root level, wrap the whole object as a single-row table (Bug 4 fix).
            has_table_children = any(
                isinstance(v, list) and v and all(isinstance(i, dict) for i in v)
                for v in data.values()
            )
            has_nested_dict = any(isinstance(v, dict) for v in data.values())
            if not path_elements and not has_table_children and not has_nested_dict:
                # Plain flat object at root — treat as single-row table
                tables['main_table'] = [data]
                return tables
            for key, value in data.items():
                self._flatten_nested_json(value, path_elements + [key], tables)

        elif isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
            # ── Determine table name ──────────────────────────────────────────
            table_name = path_elements[-1] if path_elements else 'main_table'

            # If the name is already taken by a table from a DIFFERENT path,
            # use a path-aware name (last-two path segments joined).
            if table_name in tables:
                existing_path = tables.get(f'__path__{table_name}', None)
                current_path  = '__'.join(path_elements[-2:]) if len(path_elements) >= 2 else table_name
                if existing_path and existing_path != current_path:
                    # Genuinely different table — use path-qualified name
                    table_name = current_path
                    if table_name in tables:
                        counter = 2
                        while f"{table_name}_{counter}" in tables:
                            counter += 1
                        table_name = f"{table_name}_{counter}"
                # else: same path → merge (just extend below)

            # Initialise slot and record the originating path for collision detection
            if table_name not in tables:
                tables[table_name] = []
                tables[f'__path__{table_name}'] = '__'.join(path_elements[-2:]) if len(path_elements) >= 2 else table_name

            # ── Pre-scan: classify each field's dominant type across all records ──
            # This prevents artifact columns: e.g. when 5 records have
            # "address": {dict} (→ flattened to address__*) and 1 has "address": null,
            # we should NOT create a spurious "address" = None column. (Bug 7 fix)
            field_dominant: Dict[str, str] = {}
            for rec in data:
                for k, v in rec.items():
                    if isinstance(v, dict):
                        field_dominant[k] = 'dict'
                    elif (isinstance(v, list) and v
                          and all(isinstance(i, dict) for i in v)):
                        field_dominant[k] = 'list_of_dicts'
                    elif k not in field_dominant:
                        field_dominant[k] = 'scalar'

            # ── Process each record ───────────────────────────────────────────
            for record in data:
                flat_rec: Dict[str, Any] = {}
                child_arrays: Dict[str, List[Dict]] = {}

                for key, value in record.items():
                    dominant = field_dominant.get(key, 'scalar')

                    if value is None:
                        # Only keep null if this field is a plain scalar across the
                        # table; null dict/list-of-dicts fields become NaN via the
                        # missing __ sub-columns — no extra artifact column needed.
                        if dominant == 'scalar':
                            flat_rec[key] = None
                        # else: skip — sub-columns will be NaN for this row
                    elif isinstance(value, dict):
                        has_list_of_dicts = any(
                            isinstance(v, list) and v and all(isinstance(i, dict) for i in v)
                            for v in value.values()
                        )
                        has_nested_dict = any(isinstance(v, dict) for v in value.values())
                        if not has_nested_dict and not has_list_of_dicts:
                            # Shallow object — flatten inline with __ separator
                            for sub_key, sub_val in value.items():
                                flat_rec[f"{key}__{sub_key}"] = sub_val
                        else:
                            # Deep/complex object — recurse so child arrays become tables
                            self._flatten_nested_json(
                                value,
                                path_elements + [key],
                                tables
                            )
                            # Recursively flatten ALL scalar leaves into this record
                            for leaf_key, leaf_val in self._deep_flatten_scalars(value, key).items():
                                flat_rec[leaf_key] = leaf_val
                    elif isinstance(value, list):
                        if value and all(isinstance(v, dict) for v in value):
                            # Array-of-objects → becomes a child table
                            child_arrays[key] = value
                        elif dominant == 'list_of_dicts':
                            # Empty array for a field that is list-of-dicts elsewhere
                            # → skip entirely (no artifact empty-string column)
                            pass
                        else:
                            # Primitive array → join as comma-separated string
                            flat_rec[key] = ', '.join(str(v) for v in value) if value else ''
                    else:
                        flat_rec[key] = value

                tables[table_name].append(flat_rec)

                # ── FK injection into child tables ────────────────────────────
                rec_id = self._get_record_id(record, table_name)

                for child_key, child_data in child_arrays.items():
                    fk_col = f'_ref_{table_name}_id'
                    enriched_children = [
                        {fk_col: rec_id, **child_rec}
                        for child_rec in child_data
                    ]
                    self._flatten_nested_json(
                        enriched_children,
                        path_elements + [child_key],
                        tables
                    )

        # Remove internal path-tracking sentinel keys before returning
        if not path_elements:
            tables = {k: v for k, v in tables.items() if not k.startswith('__path__')}

        return tables

    def _deep_flatten_scalars(self, obj: Any, prefix: str) -> Dict[str, Any]:
        """Recursively collect all scalar leaves from a nested dict/list into
        a flat dict with '__'-joined keys.  List-of-dicts are skipped (they
        become child tables).  Primitive lists are joined as comma-separated strings."""
        result: Dict[str, Any] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                full_key = f"{prefix}__{k}"
                if isinstance(v, dict):
                    result.update(self._deep_flatten_scalars(v, full_key))
                elif isinstance(v, list):
                    if v and all(isinstance(i, dict) for i in v):
                        pass  # list-of-dicts → child table, skip inlining
                    else:
                        result[full_key] = ', '.join(str(i) for i in v) if v else ''
                else:
                    result[full_key] = v
        return result

    def _get_record_id(self, record: Dict, table_name: str) -> Any:
        """Return the best candidate ID value from a record for FK injection.
        Injected FK columns (starting with '_ref_') are never used as the record's own ID.
        """
        # Try common patterns: exact match, singular form, generic 'id'
        candidates = [
            f'{table_name}_id',
            f'{table_name[:-1]}_id' if table_name.endswith('s') else None,
            'id',
        ]
        for field in candidates:
            if field and field in record and not isinstance(record[field], (dict, list)):
                return record[field]
        # Fallback: first *_id field that is NOT an injected FK column
        for key, val in record.items():
            if key.endswith('_id') and not key.startswith('_ref_') and not isinstance(val, (dict, list)):
                return val
        return None

    
    def _load_csv(self, file_path: Path) -> List[Dict]:
        """Load CSV file"""
        encoding = self._detect_encoding(file_path)
        
        try:
            # Try to detect delimiter
            with open(file_path, 'r', encoding=encoding) as f:
                sample = f.read(1024)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
            
            # Read CSV
            df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)

            # Convert NaN → None so downstream null-rate analysis works correctly (Bug 6 fix)
            # Must cast to object dtype first — float64 columns cannot store Python None,
            # they silently keep NaN even after .where(). astype(object) forces Python objects.
            df = df.astype(object).where(df.notna(), other=None)

            # Convert to JSON-like structure
            return df.to_dict('records')
            
        except Exception as e:
            raise ValueError(f"Error loading CSV: {e}")
    
    def _load_xml(self, file_path: Path) -> List[Dict]:
        """Load XML file"""
        encoding = self._detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                xml_content = f.read()
            
            # Convert XML to dict
            data = xmltodict.parse(xml_content)
            
            # Try to extract records if it's a typical XML structure
            if len(data) == 1:
                root_key = list(data.keys())[0]
                root_data = data[root_key]
                
                # Look for common patterns
                if isinstance(root_data, dict):
                    for key, value in root_data.items():
                        if isinstance(value, list):
                            return value
                    return [root_data]
                elif isinstance(root_data, list):
                    return root_data
            
            return [data]
            
        except Exception as e:
            raise ValueError(f"Error loading XML: {e}")
    
    def _load_excel(self, file_path: Path) -> List[Dict]:
        """Load Excel file"""
        try:
            # Read Excel file
            df = pd.read_excel(file_path, sheet_name=0)  # Read first sheet
            
            # Convert to JSON-like structure
            return df.to_dict('records')
            
        except Exception as e:
            raise ValueError(f"Error loading Excel: {e}")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic file information"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        return {
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "is_supported": file_path.suffix.lower() in self.supported_formats,
            "encoding": self._detect_encoding(file_path) if file_path.suffix.lower() in ['.json', '.csv', '.xml'] else None
        }