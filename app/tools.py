"""Tool definitions and dispatcher functions."""

import json
import os
from typing import List, Dict, Any, Optional
from .schemas import SearchDatasetsArgs, DownloadDatasetArgs, DatasetSummary
from .portal_client import DataGovClient

# In-memory cache for multi-step workflows
_dataset_cache: Dict[str, Dict[str, Any]] = {}


def _prepare_column_info_for_llm(data: List[Dict], sample_size: int = 3) -> str:
    """
    Prepare comprehensive column information for LLM-driven column selection.
    Returns a formatted string containing column names, sample values, and data types.
    """
    if not data:
        return "No data available for column analysis."

    column_info = []
    first_row = data[0]

    for col_name in first_row.keys():
        # Gather sample values (up to sample_size)
        sample_values = []
        data_types = set()

        for i, row in enumerate(data[:sample_size]):
            value = row.get(col_name)
            if value is not None:
                sample_values.append(str(value))
                data_types.add(type(value).__name__)

        # Format column information
        sample_str = ", ".join(sample_values) if sample_values else "No data"
        types_str = ", ".join(data_types) if data_types else "unknown"

        column_info.append(f"- {col_name}: [{types_str}] Sample values: {sample_str}")

    return "\n".join(column_info)


async def _search_datasets(client: DataGovClient, query: str, limit: int = 5):
    resp = await client.search(query, limit)
    results = []
    for pkg in resp.get("datasets", [])[:limit]:
        results.append({"type": "text", "text": f'{pkg.get("title")} (id={pkg.get("resource_id")})'})
    return results or [{"type": "text", "text": "No datasets found"}]


async def _download_dataset(
    client: DataGovClient,
    resource_id: str,
    limit: int = 100,
    include_full_data: bool = False,
    filters: Optional[Dict[str, Any]] = None,
):
    data = await client.download_resource(resource_id, limit)

    # Check if we got data
    records = data.get("records", [])
    if not records:
        return [{"type": "text", "text": f"No data found for resource_id: {resource_id}"}]

    # Apply semantic filters if provided
    original_count = len(records)
    if filters:
        filter_result = _apply_semantic_filters(records, filters)
        records = filter_result["filtered_data"]
        filtered_count = len(records)
        column_guidance = filter_result["guidance"]
    else:
        filtered_count = original_count
        column_guidance = _prepare_column_info_for_llm(records) if records else "No data available"

    # Get metadata about the dataset
    fields = data.get("field", [])
    field_names = [field.get("id", "unknown") for field in fields] if fields else []

    # Format the response for LLM consumption
    response_parts = []

    # 1. Dataset summary with smart filter info
    summary = f"Dataset Summary:\n"
    summary += f"- Resource ID: {resource_id}\n"
    summary += f"- Original records: {original_count}\n"
    if filters:
        summary += f"- Records after filtering: {filtered_count}\n"
        summary += f"- Filter guidance: Use exact column names for filtering\n"
    else:
        summary += f"- Total records retrieved: {filtered_count}\n"
    summary += f"- Fields: {', '.join(field_names) if field_names else 'No field info available'}\n"
    response_parts.append({"type": "text", "text": summary})

    # If no records after filtering
    if not records:
        response_parts.append(
            {
                "type": "text",
                "text": "No records match the specified filters. Try using the 'inspect_dataset_structure' tool to see available column names and sample values.",
            }
        )
        return response_parts

    # 2. Field descriptions (if available)
    if fields:
        field_info = "Field Descriptions:\n"
        for field in fields:
            field_name = field.get("id", "unknown")
            field_type = field.get("type", "unknown")
            field_info += f"- {field_name} ({field_type})\n"
        response_parts.append({"type": "text", "text": field_info})

    # 3. Sample data (first few records)
    if filtered_count > 0:
        sample_size = min(5, filtered_count)
        sample_data = "Sample Data (first {} records):\n".format(sample_size)

        for i, record in enumerate(records[:sample_size]):
            sample_data += f"\nRecord {i+1}:\n"
            if isinstance(record, dict):
                for key, value in record.items():
                    sample_data += f"  {key}: {value}\n"
            else:
                sample_data += f"  {record}\n"

        response_parts.append({"type": "text", "text": sample_data})

    # 4. Full dataset as structured data (if not too large)
    if include_full_data or filtered_count <= 50:  # Include full data for smaller datasets or when explicitly requested
        import json

        try:
            full_data = json.dumps(records, indent=2, ensure_ascii=False)
            response_parts.append(
                {"type": "text", "text": f"Complete Dataset (JSON format):\n```json\n{full_data}\n```"}
            )
        except (TypeError, ValueError):
            response_parts.append({"type": "text", "text": "Full dataset available but contains non-serializable data"})
    else:
        response_parts.append(
            {
                "type": "text",
                "text": f"Dataset is large ({filtered_count} records). Use include_full_data=true or add more specific filters for full data export.",
            }
        )

    # 2.5. Column guidance for semantic analysis (if filters were requested)
    if filters:
        response_parts.append({"type": "text", "text": f"Column Analysis Guidance:\n{column_guidance}"})

    return response_parts


async def _inspect_dataset_structure(client: DataGovClient, resource_id: str, sample_size: int = 5):
    """
    Inspect dataset structure to understand column names, types, and sample values.
    This helps the LLM understand what filtering options are available.
    """
    data = await client.download_resource(resource_id, min(sample_size, 20))  # Small sample for inspection

    records = data.get("records", [])
    if not records:
        return [{"type": "text", "text": f"No data found for resource_id: {resource_id}"}]

    fields = data.get("field", [])

    # Build comprehensive structure info
    structure_info = f"Dataset Structure Analysis:\n"
    structure_info += f"- Resource ID: {resource_id}\n"
    structure_info += f"- Sample size analyzed: {len(records)} records\n\n"

    # Field information with sample values
    structure_info += "Column Details:\n"

    for field in fields:
        field_name = field.get("id", "unknown")
        field_type = field.get("type", "unknown")

        # Get sample values for this field
        sample_values = []
        for record in records[:5]:
            if field_name in record and record[field_name] not in sample_values:
                sample_values.append(record[field_name])

        structure_info += f"- {field_name} ({field_type})\n"
        if sample_values:
            structure_info += f"  Sample values: {sample_values[:3]}\n"
        structure_info += "\n"

    # Suggest common filtering patterns
    suggestions = "\nSuggested Filtering Patterns:\n"

    # Look for common column name patterns
    column_names = [field.get("id", "").lower() for field in fields]

    if any("state" in col for col in column_names):
        state_cols = [field.get("id") for field in fields if "state" in field.get("id", "").lower()]
        suggestions += f"- Geographic filtering: Use columns {state_cols} for state-based filtering\n"

    if any("year" in col or "date" in col for col in column_names):
        time_cols = [
            field.get("id")
            for field in fields
            if any(t in field.get("id", "").lower() for t in ["year", "date", "time"])
        ]
        suggestions += f"- Time-based filtering: Use columns {time_cols} for temporal filtering\n"

    # Look for numeric fields
    numeric_fields = [field.get("id") for field in fields if field.get("type") in ["double", "int", "float", "number"]]
    if numeric_fields:
        suggestions += f"- Numeric range filtering: Use columns {numeric_fields[:5]} for value-based filtering\n"

    response_parts = [{"type": "text", "text": structure_info}, {"type": "text", "text": suggestions}]

    # Include a few sample records to show actual data
    if records:
        sample_data = "Sample Records:\n"
        for i, record in enumerate(records[:3], 1):
            sample_data += f"\nRecord {i}:\n"
            for key, value in record.items():
                sample_data += f"  {key}: {value}\n"

        response_parts.append({"type": "text", "text": sample_data})

    return response_parts


def _apply_semantic_filters(data: List[Dict], filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply filters to data with LLM-driven semantic column matching.
    Returns both filtered data and guidance for the LLM about column selection.
    """
    if not filters or not data:
        return {
            "filtered_data": data,
            "column_info": _prepare_column_info_for_llm(data) if data else "No data available",
            "guidance": "No filters applied. All data returned.",
        }

    column_info = _prepare_column_info_for_llm(data)
    guidance_parts = []

    # Instead of automatically matching columns, provide information for the LLM
    guidance_parts.append(f"Available columns in the dataset:\n{column_info}")
    guidance_parts.append(f"\nFilters requested: {filters}")
    guidance_parts.append(
        "\nTo apply filters semantically, the LLM should:"
        "\n1. Identify which columns best match the filter criteria based on column names and sample data"
        "\n2. Use the analyze_loaded_dataset tool with specific column names for filtering"
        "\n3. Consider synonyms and related concepts (e.g., 'state' might match 'state_name', 'states_uts', etc.)"
    )

    return {
        "filtered_data": data,  # Return unfiltered data - let LLM decide on column mapping
        "column_info": column_info,
        "guidance": "\n".join(guidance_parts),
    }


def _prepare_grouping_and_aggregation_guidance(
    data: List[Dict], group_by: Optional[str] = None, aggregations: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Prepare guidance for LLM-driven grouping and aggregation instead of automatic column matching.
    Returns column information and guidance for semantic selection.
    """
    if not data:
        return {"data": [], "column_info": "No data available", "guidance": "No data to group or aggregate"}

    column_info = _prepare_column_info_for_llm(data)
    guidance_parts = []

    guidance_parts.append(f"Available columns in the dataset:\n{column_info}")

    if group_by:
        guidance_parts.append(f"\nGrouping requested by: '{group_by}'")
        guidance_parts.append(
            "To group semantically, the LLM should:"
            "\n1. Identify which column best represents the grouping concept"
            "\n2. Consider synonyms and variations in column names"
            "\n3. Use the analyze_loaded_dataset tool with the exact column name for grouping"
        )

    if aggregations:
        guidance_parts.append(f"\nAggregations requested: {aggregations}")
        guidance_parts.append(
            "For aggregations, the LLM should:"
            "\n1. Map each aggregation field to the most appropriate column"
            "\n2. Ensure numeric columns are used for mathematical operations (sum, avg, min, max)"
            "\n3. Use count for any column type"
        )

    return {
        "data": data,  # Return original data - let LLM decide on column mapping
        "column_info": column_info,
        "guidance": "\n".join(guidance_parts),
    }


# Tool: Load dataset for analysis
async def _load_dataset_for_analysis(
    client: DataGovClient, resource_id: str, limit: int = 1000, cache_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load a dataset into memory for multi-step analysis.

    Args:
        client: DataGovClient instance
        resource_id: The dataset resource ID
        limit: Maximum number of records to load
        cache_key: Optional key for caching (defaults to resource_id)
    """
    if not cache_key:
        cache_key = resource_id

    try:
        response = await client.download_resource(resource_id, limit)
        data = response.get("records", [])

        if not data:
            return [
                {"type": "text", "text": f"Failed to load dataset or dataset is empty (resource_id: {resource_id})"}
            ]

        # Cache the dataset
        _dataset_cache[cache_key] = {
            "data": data,
            "resource_id": resource_id,
            "total_rows": len(data),
            "response_meta": response,
        }

        # Analyze structure for LLM
        columns = list(data[0].keys()) if data else []
        column_info = {}

        for col in columns:
            values = [row.get(col) for row in data[:100]]  # Sample first 100 rows
            non_null_values = [v for v in values if v is not None and str(v).strip()]

            column_info[col] = {
                "sample_values": list(set(str(v) for v in non_null_values))[:10],  # First 10 unique values
                "null_count": len(values) - len(non_null_values),
                "data_type": (
                    "numeric"
                    if any(str(v).replace(".", "").replace("-", "").isdigit() for v in non_null_values[:5])
                    else "text"
                ),
            }

        result_text = f"""Dataset loaded successfully for analysis:

Cache Key: {cache_key}
Total Rows: {len(data)}
Available Columns: {', '.join(columns)}

Column Analysis:
"""
        for col, info in column_info.items():
            result_text += f"- {col} ({info['data_type']}): {info['sample_values'][:5]}\n"

        result_text += f"""
Sample Data (first 3 rows):
"""
        for i, row in enumerate(data[:3], 1):
            result_text += f"\nRow {i}: {dict(list(row.items())[:5])}..."

        result_text += f"""

Next Steps:
- Use 'analyze_loaded_dataset' with resource_id='{resource_id}' to filter, group, or aggregate this data
- Available smart column matching (e.g., 'state' matches 'state_name', 'state_ut', etc.)
- Supports filters, grouping, and aggregations for complex analysis
"""

        return [{"type": "text", "text": result_text}]

    except Exception as e:
        return [{"type": "text", "text": f"Failed to load dataset: {str(e)} (resource_id: {resource_id})"}]


# Tool: Analyze loaded dataset with LLM-driven semantic column selection
async def _analyze_loaded_dataset(
    client: DataGovClient,
    resource_id: str,
    analysis_query: str,
    filters: Optional[Dict[str, Any]] = None,
    group_by: Optional[str] = None,
    aggregate: Optional[Dict[str, str]] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """Analyze a loaded dataset with LLM-driven semantic column selection.

    This function returns column information and guidance for the LLM to make
    intelligent decisions about column mapping, rather than using static matching.

    Args:
        client: DataGovClient instance
        resource_id: The resource ID (used as cache key)
        analysis_query: Natural language description of what analysis to perform
        filters: Dict of column filters (LLM should map to actual column names)
        group_by: Column concept to group by (LLM should map to actual column name)
        aggregate: Dict of {column_concept: function} for aggregations
        limit: Maximum number of results to return
    """
    cache_key = resource_id  # Use resource_id as cache key

    if cache_key not in _dataset_cache:
        return [
            {
                "type": "text",
                "text": f"Dataset with resource_id '{resource_id}' not loaded. Use 'load_dataset_for_analysis' first.\nAvailable cached datasets: {list(_dataset_cache.keys())}",
            }
        ]

    try:
        cached_dataset = _dataset_cache[cache_key]
        data = cached_dataset["data"].copy()

        if not data:
            return [{"type": "text", "text": "No data available in the loaded dataset."}]

        column_info = _prepare_column_info_for_llm(data)
        guidance_parts = []

        guidance_parts.append(f"Analysis Query: {analysis_query}")
        guidance_parts.append(f"Dataset has {len(data)} records loaded in memory.")
        guidance_parts.append(f"\nAvailable columns:\n{column_info}")

        # Provide guidance for filters
        if filters:
            guidance_parts.append(f"\nFilters requested: {filters}")
            guidance_parts.append(
                "To apply these filters semantically:"
                "\n1. Map each filter key to the most appropriate actual column name"
                "\n2. Consider synonyms and context (e.g., 'state' → 'state_name', 'year' → 'time_period')"
                "\n3. Use this analyze_loaded_dataset tool again with exact column names in filters"
                "\n4. For example: instead of {'equals': {'state': 'Karnataka'}}, use {'equals': {'state_name': 'Karnataka'}}"
            )

        # Provide guidance for grouping
        if group_by:
            guidance_parts.append(f"\nGrouping requested by: '{group_by}'")
            guidance_parts.append(
                "To group semantically:"
                "\n1. Identify which column best represents the grouping concept"
                "\n2. Use this analyze_loaded_dataset tool again with the exact column name for group_by"
            )

        # Provide guidance for aggregations
        if aggregate:
            guidance_parts.append(f"\nAggregations requested: {aggregate}")
            guidance_parts.append(
                "To aggregate semantically:"
                "\n1. Map each aggregation field to the most appropriate numeric column"
                "\n2. Ensure the column contains numeric data for mathematical operations"
                "\n3. Use this analyze_loaded_dataset tool again with exact column names"
            )

        # If this looks like the LLM is providing exact column names, try to apply the operations
        exact_column_names = list(data[0].keys()) if data else []
        can_apply_directly = True

        # Check if filters use exact column names
        if filters:
            for filter_type, filter_dict in filters.items():
                for col_name in filter_dict.keys():
                    if col_name not in exact_column_names:
                        can_apply_directly = False
                        break

        # Check if group_by uses exact column name
        if group_by and group_by not in exact_column_names:
            can_apply_directly = False

        # Check if aggregate uses exact column names
        if aggregate:
            for agg_field in aggregate.keys():
                if agg_field not in exact_column_names:
                    can_apply_directly = False
                    break

        if can_apply_directly and (filters or group_by or aggregate):
            # Apply the operations since exact column names were provided
            return await _execute_analysis_with_exact_columns(data, analysis_query, filters, group_by, aggregate, limit)

        # Otherwise, provide guidance for semantic mapping
        result_text = "\n".join(guidance_parts)
        result_text += f"\n\nTo proceed with analysis, use exact column names from the list above."
        result_text += f"\nYou can also inspect a sample of the data:\n"

        # Show sample data
        for i, row in enumerate(data[:3], 1):
            result_text += f"\nSample record {i}:\n"
            for key, value in row.items():
                result_text += f"  {key}: {value}\n"

        return [{"type": "text", "text": result_text}]

    except Exception as e:
        return [{"type": "text", "text": f"Failed to analyze dataset: {str(e)} (cache_key: {cache_key})"}]


async def _execute_analysis_with_exact_columns(
    data: List[Dict],
    analysis_query: str,
    filters: Optional[Dict[str, Any]] = None,
    group_by: Optional[str] = None,
    aggregate: Optional[Dict[str, str]] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """Execute analysis operations when exact column names are provided."""
    original_count = len(data)
    applied_filters = []

    # Apply filters
    if filters:
        for filter_type, filter_dict in filters.items():
            if filter_type == "equals":
                for col_name, value in filter_dict.items():
                    data = [row for row in data if str(row.get(col_name, "")).lower() == str(value).lower()]
                    applied_filters.append(f"{col_name} equals '{value}'")

            elif filter_type == "contains":
                for col_name, value in filter_dict.items():
                    data = [row for row in data if str(value).lower() in str(row.get(col_name, "")).lower()]
                    applied_filters.append(f"{col_name} contains '{value}'")

            elif filter_type == "range":
                for col_name, range_dict in filter_dict.items():
                    filtered_data = []
                    for row in data:
                        try:
                            val = float(row.get(col_name, 0))
                            if range_dict.get("min") is not None and val < range_dict["min"]:
                                continue
                            if range_dict.get("max") is not None and val > range_dict["max"]:
                                continue
                            filtered_data.append(row)
                        except (ValueError, TypeError):
                            continue
                    data = filtered_data
                    applied_filters.append(f"{col_name} in range {range_dict}")

    filtered_count = len(data)

    # Apply grouping and aggregations
    if group_by and data:
        # Group data by the specified column
        groups = {}
        for row in data:
            group_value = row.get(group_by)
            if group_value not in groups:
                groups[group_value] = []
            groups[group_value].append(row)

        # Apply aggregations
        grouped_data = []
        for group_value, group_rows in groups.items():
            result_row = {group_by: group_value}

            if aggregate:
                for agg_field, agg_func in aggregate.items():
                    values = []
                    for row in group_rows:
                        val = row.get(agg_field)
                        if val is not None:
                            try:
                                values.append(float(val))
                            except (ValueError, TypeError):
                                pass

                    if values:
                        if agg_func.lower() == "sum":
                            result_row[f"{agg_field}_{agg_func}"] = sum(values)
                        elif agg_func.lower() in ["avg", "mean"]:
                            result_row[f"{agg_field}_{agg_func}"] = sum(values) / len(values)
                        elif agg_func.lower() == "max":
                            result_row[f"{agg_field}_{agg_func}"] = max(values)
                        elif agg_func.lower() == "min":
                            result_row[f"{agg_field}_{agg_func}"] = min(values)
                        elif agg_func.lower() == "count":
                            result_row[f"{agg_field}_{agg_func}"] = len(values)
            else:
                # No aggregations, just count
                result_row["count"] = len(group_rows)

            grouped_data.append(result_row)

        data = grouped_data

    # Apply limit
    if limit and limit > 0:
        data = data[:limit]

    # Format results
    result_text = f"""Analysis Results:

Query: {analysis_query}
Original records: {original_count}
After filtering: {filtered_count}
Final results: {len(data)}

Filters applied: {', '.join(applied_filters) if applied_filters else 'None'}
Grouping: {group_by if group_by else 'None'}
Aggregations: {aggregate if aggregate else 'None'}

Results:
"""

    if data:
        columns = list(data[0].keys()) if data else []
        result_text += f"\nColumns: {', '.join(columns)}\n\n"

        # Show up to 20 results
        for i, row in enumerate(data[:20], 1):
            result_text += f"Result {i}: {row}\n"

        if len(data) > 20:
            result_text += f"\n... and {len(data) - 20} more results (showing first 20)"
    else:
        result_text += "\nNo results found matching the criteria."

    return [{"type": "text", "text": result_text}]


# Tool: Inspect dataset structure
async def _inspect_dataset_structure_new(
    client: DataGovClient, resource_id: str, sample_size: int = 100
) -> List[Dict[str, Any]]:
    """Quickly inspect dataset structure without full loading.

    Args:
        client: DataGovClient instance
        resource_id: The dataset resource ID
        sample_size: Number of rows to sample for analysis
    """
    try:
        response = await client.download_resource(resource_id, min(sample_size, 100))
        data = response.get("records", [])

        if not data:
            return [
                {"type": "text", "text": f"Failed to fetch dataset or dataset is empty (resource_id: {resource_id})"}
            ]

        columns = list(data[0].keys()) if data else []
        column_analysis = {}

        for col in columns:
            values = [row.get(col) for row in data]
            non_null_values = [v for v in values if v is not None and str(v).strip()]

            column_analysis[col] = {
                "sample_values": list(set(str(v) for v in non_null_values))[:10],
                "null_percentage": round((len(values) - len(non_null_values)) / len(values) * 100, 2) if values else 0,
                "unique_values": len(set(str(v) for v in non_null_values)),
                "data_type": (
                    "numeric"
                    if any(str(v).replace(".", "").replace("-", "").isdigit() for v in non_null_values[:5])
                    else "text"
                ),
            }

        result_text = f"""Dataset Structure Inspection:

Resource ID: {resource_id}
Total sample rows: {len(data)}
Available columns: {len(columns)}

Column Details:
"""

        for col, analysis in column_analysis.items():
            result_text += f"""
- {col} ({analysis['data_type']})
  Sample values: {analysis['sample_values'][:5]}
  Null%: {analysis['null_percentage']}%, Unique values: {analysis['unique_values']}
"""

        result_text += f"""
Sample Records (first 3):
"""
        for i, record in enumerate(data[:3], 1):
            result_text += f"\nRecord {i}: {dict(list(record.items())[:5])}..."

        result_text += f"""

Recommended Next Steps:
1. Use 'load_dataset_for_analysis' to cache this dataset for multi-step analysis
2. Use 'download_dataset' with filters for simple one-time downloads
3. Available smart column matching: 'state' → state_name/states_uts, 'year' → year_code/time_period, etc.
"""

        return [{"type": "text", "text": result_text}]

    except Exception as e:
        return [{"type": "text", "text": f"Failed to inspect dataset: {str(e)} (resource_id: {resource_id})"}]


TOOLS: List[Dict[str, Any]] = [
    {
        "name": "search_datasets",
        "description": "Search public datasets on data.gov.in by keyword",
        "inputSchema": SearchDatasetsArgs.model_json_schema(),
    },
    {
        "name": "load_dataset_for_analysis",
        "description": "STEP 1: Load a dataset into memory for multi-step LLM-driven analysis. Returns comprehensive column information with sample values and data types to enable intelligent semantic column selection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "resource_id": {"type": "string", "description": "The resource ID to load for analysis"},
                "limit": {"type": "integer", "default": 1000, "description": "Maximum number of records to load"},
            },
            "required": ["resource_id"],
        },
    },
    {
        "name": "analyze_loaded_dataset",
        "description": "STEP 2: Analyze a previously loaded dataset with LLM-driven semantic column selection. The system will provide column information and guidance for intelligent mapping of filter/grouping concepts to actual column names.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "resource_id": {"type": "string", "description": "The resource ID (must match the loaded dataset)"},
                "analysis_query": {
                    "type": "string",
                    "description": "Natural language description of the analysis being performed",
                },
                "filters": {
                    "type": "object",
                    "description": "Semantic filters - provide concepts that will be mapped to actual columns (e.g., 'state', 'year', 'value'). Use exact column names for direct application.",
                    "properties": {
                        "contains": {
                            "type": "object",
                            "description": "Text contains filters: {concept_or_column: value}",
                        },
                        "equals": {"type": "object", "description": "Exact match filters: {concept_or_column: value}"},
                        "range": {
                            "type": "object",
                            "description": "Numeric range filters: {concept_or_column: {min: X, max: Y}}",
                        },
                    },
                },
                "group_by": {
                    "type": "string",
                    "description": "Column concept or exact column name to group results by (e.g., 'state', 'year', or exact name like 'state_name')",
                },
                "aggregate": {
                    "type": "object",
                    "description": "Aggregation functions: {concept_or_column: function}. Functions: sum, avg, max, min, count. Use concepts like 'value', 'amount' or exact column names.",
                },
                "limit": {
                    "type": "integer",
                    "default": 1000,
                    "description": "Maximum number of results to return",
                },
            },
            "required": ["resource_id", "analysis_query"],
        },
    },
    {
        "name": "inspect_dataset_structure",
        "description": "Quick inspection of dataset structure with comprehensive column analysis. Returns column names, data types, sample values, and semantic guidance for intelligent column selection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "resource_id": {"type": "string", "description": "The resource ID to inspect"},
                "sample_size": {"type": "integer", "default": 5, "description": "Number of sample records to analyze"},
            },
            "required": ["resource_id"],
        },
    },
    {
        "name": "download_dataset",
        "description": "Dataset download with LLM-driven semantic column guidance. When filters are provided, returns column analysis to help map filtering concepts to actual column names.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "resource_id": {"type": "string", "description": "The resource ID to download"},
                "limit": {"type": "integer", "default": 100, "description": "Maximum number of records to retrieve"},
                "include_full_data": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include full dataset even if large",
                },
                "filters": {
                    "type": "object",
                    "description": "Semantic filters - provide concepts for column mapping guidance (e.g., 'state', 'year'). For actual filtering, use exact column names or load dataset for analysis.",
                    "properties": {
                        "contains": {"type": "object", "description": "Text contains filters: {concept: value}"},
                        "equals": {"type": "object", "description": "Exact match filters: {concept: value}"},
                        "range": {
                            "type": "object",
                            "description": "Numeric range filters: {concept: {min: X, max: Y}}",
                        },
                    },
                },
            },
            "required": ["resource_id"],
        },
    },
]

# Mapping tool name -> callable
TOOL_MAP = {
    "search_datasets": _search_datasets,
    "load_dataset_for_analysis": _load_dataset_for_analysis,
    "analyze_loaded_dataset": _analyze_loaded_dataset,
    "inspect_dataset_structure": _inspect_dataset_structure_new,
    "download_dataset": _download_dataset,
}
