#!/usr/bin/env python3
"""
Simple standalone MCP server for data.gov.in using FastMCP with semantic search.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import httpx
from mcp.server import FastMCP

# Import semantic search functionality
try:
    from semantic_search import initialize_semantic_search, DatasetSemanticSearch

    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Semantic search not available: {e}", file=sys.stderr)
    print(
        "Install required packages with: micromamba install -c conda-forge sentence-transformers faiss-cpu numpy",
        file=sys.stderr,
    )
    SEMANTIC_SEARCH_AVAILABLE = False


# ============================================================================
# Configuration and Utility Functions
# ============================================================================


def load_env_file() -> None:
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


def load_dataset_registry() -> List[Dict[str, Any]]:
    """Load the dataset registry from the JSON file."""
    registry_path = Path(__file__).parent / "data" / "data_gov_in_api_registry.json"
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠ WARNING: Registry file not found at {registry_path}", file=sys.stderr)
        return []
    except json.JSONDecodeError as e:
        print(f"⚠ WARNING: Invalid JSON in registry file: {e}", file=sys.stderr)
        return []


async def download_api(
    resource_id: str, api_key: str, limit: int = 100, filters: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Download a dataset from data.gov.in API with optional filtering."""
    params = {
        "resource_id": resource_id,
        "api-key": api_key,
        "format": "json",
        "limit": limit,
    }

    # Add filters as query parameters
    if filters:
        params.update(filters)

    async with httpx.AsyncClient() as client:
        url = f"https://api.data.gov.in/resource/{resource_id}"
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def filter_dataset_records(data: Dict[str, Any], column_filters: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Filter dataset records based on column values."""
    if not column_filters or not data.get("records"):
        return data

    filtered_records = []
    for record in data.get("records", []):
        match = True
        for column, filter_value in column_filters.items():
            record_value = str(record.get(column, "")).lower()
            filter_value_lower = filter_value.lower()

            # Simple substring matching (case-insensitive)
            if filter_value_lower not in record_value:
                match = False
                break

        if match:
            filtered_records.append(record)

    # Return filtered data with original structure
    filtered_data = data.copy()
    filtered_data["records"] = filtered_records
    filtered_data["total"] = len(filtered_records)

    return filtered_data


# ============================================================================
# Server Initialization
# ============================================================================

# Initialize environment and configuration
load_env_file()
API_KEY = os.getenv("DATA_GOV_API_KEY")
DATASET_REGISTRY = load_dataset_registry()

# Initialize semantic search if available
SEMANTIC_SEARCH_ENGINE = None
if SEMANTIC_SEARCH_AVAILABLE and DATASET_REGISTRY:
    try:
        print("🔄 Initializing semantic search...", file=sys.stderr)
        SEMANTIC_SEARCH_ENGINE = initialize_semantic_search(DATASET_REGISTRY)
        print("✅ Semantic search initialized", file=sys.stderr)
    except Exception as e:
        print(f"❌ Failed to initialize semantic search: {e}", file=sys.stderr)
        print("❌ Server requires semantic search to function properly", file=sys.stderr)
        # Don't exit here, let the tool function handle the error gracefully

# Create the FastMCP server instance
mcp = FastMCP("data-gov-in-mcp")

# Log initialization status
print("Creating MCP server...", file=sys.stderr)
if not API_KEY:
    print("⚠ WARNING: DATA_GOV_API_KEY environment variable not set", file=sys.stderr)
    print("Either set it as an environment variable or add it to a .env file", file=sys.stderr)
print(f"✓ Loaded {len(DATASET_REGISTRY)} datasets from registry", file=sys.stderr)

if SEMANTIC_SEARCH_ENGINE and SEMANTIC_SEARCH_ENGINE.is_ready():
    print("✅ Semantic search is ready for AI-powered dataset discovery", file=sys.stderr)
else:
    print("❌ Semantic search not available - server functionality will be limited", file=sys.stderr)
    if not SEMANTIC_SEARCH_AVAILABLE:
        print(
            "❌ Install semantic search packages: micromamba install -c conda-forge sentence-transformers faiss-cpu numpy",
            file=sys.stderr,
        )
    else:
        print("❌ Run 'python build_embeddings.py' to build semantic search embeddings", file=sys.stderr)


# ============================================================================
# MCP Resources
# ============================================================================


@mcp.resource("dataset://registry")
async def get_dataset_registry() -> str:
    """Expose the dataset registry as an MCP resource for metadata browsing."""
    try:
        # Return the full registry with enhanced metadata
        registry_with_metadata = {
            "total_datasets": len(DATASET_REGISTRY),
            "last_updated": "2025-07-08",
            "source": "data.gov.in API registry",
            "datasets": DATASET_REGISTRY,
            "sectors": list({d.get("sector", "Unknown") for d in DATASET_REGISTRY}),
            "ministries": list({d.get("ministry", "Unknown") for d in DATASET_REGISTRY}),
        }

        return json.dumps(registry_with_metadata, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error accessing registry: {str(e)}"


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
async def search_datasets(query: str, limit: int = 20) -> dict:
    """
    Search public datasets on data.gov.in by keyword using AI-powered semantic search.

    This tool uses semantic search to find relevant datasets based on meaning,
    not just keyword matching. The search prioritizes dataset titles over ministry/sector names.

    MANDATORY WORKFLOW: Unless the user's query is simply to search for datasets, you MUST:
    1. Identify ALL datasets that seem relevant to the user's query
    2. For EACH relevant dataset, call inspect_dataset_structure() to understand its contents and identify if it can add value to the user's query
    3. For EACH dataset found to be useful, call download_filtered_dataset() to get actual data
    4. Combine information from ALL relevant datasets to provide a comprehensive answer
    5. DO NOT stop after examining just one dataset - this is a critical error

    EXAMPLE WORKFLOW:
    - If 5 datasets are returned and 3 seem relevant, inspect ALL 3 structures.
    - If 2 of these inspected datasets have complementary information or seem useful, then download data from ALL 2 datasets.
    - Finally, synthesize findings from ALL 2 datasets in your response.

    CRITICAL: The user expects comprehensive analysis using multiple data sources.
    Using only one dataset when multiple are available provides incomplete answers.
    """
    try:
        # Check if semantic search is available and ready
        if not SEMANTIC_SEARCH_ENGINE:
            return {
                "error": "Semantic search is not initialized. Please ensure the required packages are installed and embeddings are built.",
                "suggestion": "Run 'python build_embeddings.py' to initialize semantic search.",
                "available_packages": SEMANTIC_SEARCH_AVAILABLE,
            }

        if not SEMANTIC_SEARCH_ENGINE.is_ready():
            return {
                "error": "Semantic search engine is not ready. Embeddings may not be built yet.",
                "suggestion": "Run 'python build_embeddings.py' to build embeddings for semantic search.",
            }

        # Use semantic search
        results = SEMANTIC_SEARCH_ENGINE.search(query, limit=limit)

        if not results:
            return {
                "message": f"No datasets found matching '{query}' using semantic search",
                "suggestion": "Try rephrasing your query or using different keywords. For example: health, petroleum, oil, crude, inflation, taxes, or guarantees",
                "total_datasets": len(DATASET_REGISTRY),
                "search_method": "semantic search (AI-powered)",
            }

        # Prepare response with enhanced guidance for LLM
        response = {
            "query": query,
            "found": len(results),
            "total_available": len(DATASET_REGISTRY),
            "search_method": "semantic search (AI-powered)",
            "datasets": results,
            "note": "Results from curated dataset registry using AI-powered semantic search. API key still required for downloading data.",
            "MANDATORY_NEXT_STEPS": [
                "1. Identify datasets with similarity_score >= 0.25 as potentially relevant",
                "2. For EACH relevant dataset, call inspect_dataset_structure(resource_id) to see if they are useful",
                "3. For EACH useful dataset, call download_filtered_dataset(resource_id, filters)",
                "4. Synthesize information from ALL useful datasets in your final answer",
                "5. Do NOT stop after examining just the first dataset - this is incomplete analysis",
            ],
            "multi_dataset_guidance": (
                "CRITICAL REQUIREMENT: You must examine and use data from ALL relevant datasets found. "
                "Using only one dataset when multiple are available provides incomplete and poor analysis. "
                "Each dataset contains different aspects of the information. Your goal is comprehensive coverage."
            ),
            "workflow_example": (
                "Suppose you find 4 datasets using the semantic search. "
                "Call inspect_dataset_structure() for each, then download_filtered_dataset() for each that was found useful (e.g. 3 of them). "
                "Your final answer should integrate findings from all 3 datasets."
            ),
        }

        # Add guidance for LLM based on relevance scores and count
        if results:
            relevant_datasets = [r for r in results if r.get("similarity_score", 0) >= 0.25]
            highly_relevant = [r for r in results if r.get("similarity_score", 0) >= 0.5]

            response["relevance_analysis"] = {
                "total_returned": len(results),
                "potentially_relevant": len(relevant_datasets),
                "highly_relevant": len(highly_relevant),
                "relevance_threshold": 0.25,
                "datasets_to_examine": [d["resource_id"] for d in relevant_datasets],
            }

            if len(relevant_datasets) == 0:
                response["action_required"] = (
                    "No datasets meet the relevance threshold (0.25). "
                    "Consider refining your search query or examining the top results manually."
                )
            elif len(relevant_datasets) == 1:
                response["action_required"] = (
                    f"Found 1 relevant dataset. Call inspect_dataset_structure('{relevant_datasets[0]['resource_id']}') "
                    f"then download_filtered_dataset('{relevant_datasets[0]['resource_id']}', filters) to get the data."
                )
            else:
                response["action_required"] = (
                    f"Found {len(relevant_datasets)} relevant datasets. YOU MUST examine ALL of them:\n"
                    + "\n".join(
                        [
                            f"- inspect_dataset_structure('{d['resource_id']}') then download_filtered_dataset('{d['resource_id']}', filters)"
                            for d in relevant_datasets
                        ]
                    )
                    + f"\nThen combine insights from all {len(relevant_datasets)} datasets in your final answer."
                )

        return response

    except Exception as e:
        return {"error": f"Error in semantic search: {str(e)}"}


@mcp.tool()
async def download_dataset(resource_id: str, limit: int = 100) -> dict:
    """
    Download a complete dataset from data.gov.in.

    Warning: This may return large amounts of data. Consider using download_filtered_dataset()
    with specific column filters to get only the data you need and avoid long responses.
    """
    try:
        if not API_KEY:
            return {"error": "DATA_GOV_API_KEY environment variable not set. Please set it to use this tool."}
        result = await download_api(resource_id, API_KEY, limit)
        return result
    except Exception as e:
        return {"error": f"Error downloading dataset: {str(e)}"}


@mcp.tool()
async def download_filtered_dataset(
    resource_id: str, column_filters: Union[str, Dict[str, str]], limit: int = 100
) -> dict:
    """
    Download a filtered dataset from data.gov.in.

    Args:
        resource_id: The dataset resource ID
        column_filters: Column filters as JSON string (e.g., '{"state": "Maharashtra", "year": "2023"}')
                       or as a dictionary (e.g., {"state": "Maharashtra", "year": "2023"})
        limit: Maximum number of records to return

    This function is ideal for getting specific data subsets and avoiding large responses.
    Use inspect_dataset_structure first to see available columns.
    """
    try:
        if not API_KEY:
            return {"error": "DATA_GOV_API_KEY environment variable not set. Please set it to use this tool."}

        # Parse the column filters - handle both string and dict inputs
        try:
            if isinstance(column_filters, dict):
                filters_dict = column_filters
            elif isinstance(column_filters, str):
                filters_dict = json.loads(column_filters) if column_filters else {}
            else:
                filters_dict = dict(column_filters) if column_filters else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {
                "error": f"Invalid column_filters format. Received: {type(column_filters).__name__}. "
                + 'Expected JSON string like "{"column_name": "filter_value"}" or a dictionary object.'
            }

        result = await download_api(resource_id, API_KEY, limit)

        if filters_dict:
            result = filter_dataset_records(result, filters_dict)
            result["applied_filters"] = filters_dict
            result["note"] = f"Data filtered by: {', '.join(f'{k}={v}' for k, v in filters_dict.items())}"

        return result
    except Exception as e:
        return {"error": f"Error downloading filtered dataset: {str(e)}"}


@mcp.tool()
async def inspect_dataset_structure(resource_id: str, sample_size: int = 5) -> str:
    """
    Quick inspection of dataset structure and available columns.

    Use this function first to understand the data structure, then use download_filtered_dataset
    with specific column filters to get only the data you need.
    """
    try:
        if not API_KEY:
            return json.dumps(
                {"error": "DATA_GOV_API_KEY environment variable not set. Please set it to use this tool."}, indent=2
            )

        # Download a small sample to inspect structure
        result = await download_api(resource_id, API_KEY, sample_size)

        # Extract structure info with enhanced guidance
        fields = result.get("field", [])
        sample_records = result.get("records", [])[:sample_size]

        # Extract column names for easier reference
        column_names = [
            field.get("id", field.get("name", "")) for field in fields if field.get("id") or field.get("name")
        ]

        structure = {
            "fields": fields,
            "column_names": column_names,
            "sample_records": sample_records,
            "total_records_available": result.get("total", "unknown"),
            "usage_tip": "Use download_filtered_dataset() with column_filters to get specific data subsets",
            "example_filter": (
                f'{{"column_name": "filter_value"}} or as dict {{"column_name": "filter_value"}}'
                if column_names
                else "No columns available for filtering"
            ),
        }

        return json.dumps(structure, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error inspecting dataset: {str(e)}"


@mcp.tool()
async def get_registry_summary() -> str:
    """Get a summary of the dataset registry including counts by sector and ministry."""
    try:
        # Count datasets by sector and ministry
        sector_counts = {}
        ministry_counts = {}

        for dataset in DATASET_REGISTRY:
            sector = dataset.get("sector", "Unknown")
            ministry = dataset.get("ministry", "Unknown")

            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            ministry_counts[ministry] = ministry_counts.get(ministry, 0) + 1

        return json.dumps(
            {
                "total_datasets": len(DATASET_REGISTRY),
                "sectors_count": len(sector_counts),
                "ministries_count": len(ministry_counts),
                "datasets_by_sector": dict(sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)),
                "top_ministries": dict(sorted(ministry_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                "note": "Use list_sectors() to see all sectors, or search_datasets() to find specific datasets",
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:
        return f"Error getting registry summary: {str(e)}"


@mcp.tool()
async def list_sectors() -> str:
    """List all available sectors in the registry."""
    try:
        sectors = {dataset.get("sector", "Unknown") for dataset in DATASET_REGISTRY}
        sorted_sectors = sorted(list(sectors))

        return json.dumps(
            {
                "total_sectors": len(sorted_sectors),
                "sectors": sorted_sectors,
                "note": "Use search_datasets() with sector names to find datasets in specific sectors",
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:
        return f"Error listing sectors: {str(e)}"


@mcp.tool()
async def plan_multi_dataset_analysis(search_results: List[str], query_context: str) -> dict:
    """
    Plan a comprehensive analysis workflow using multiple datasets.

    Args:
        search_results: List of resource_ids from search_datasets results
        query_context: The original user query or analysis goal

    This tool helps organize a systematic approach to analyzing multiple datasets
    and ensures comprehensive coverage of all relevant data sources.
    """
    try:
        if not search_results:
            return {"error": "No resource IDs provided for analysis planning"}

        # Find dataset metadata for the provided resource IDs
        relevant_datasets = []
        for resource_id in search_results:
            for dataset in DATASET_REGISTRY:
                if dataset.get("resource_id") == resource_id:
                    relevant_datasets.append(dataset)
                    break

        if not relevant_datasets:
            return {"error": "None of the provided resource IDs found in registry"}

        # Create analysis plan
        analysis_plan = {
            "query_context": query_context,
            "total_datasets_to_analyze": len(relevant_datasets),
            "systematic_workflow": [
                "Phase 1: Structure Inspection",
                "Phase 2: Data Collection",
                "Phase 3: Cross-Dataset Analysis",
                "Phase 4: Comprehensive Synthesis",
            ],
            "phase_1_inspect_calls": [f"inspect_dataset_structure('{d['resource_id']}')" for d in relevant_datasets],
            "phase_2_download_calls": [
                f"download_filtered_dataset('{d['resource_id']}', relevant_filters)" for d in relevant_datasets
            ],
            "datasets_overview": [
                {
                    "resource_id": d["resource_id"],
                    "title": d.get("title", "Unknown"),
                    "sector": d.get("sector", "Unknown"),
                    "ministry": d.get("ministry", "Unknown"),
                    "expected_contribution": f"Sector: {d.get('sector', 'Unknown')} data for {query_context}",
                }
                for d in relevant_datasets
            ],
            "analysis_strategy": (
                f"Execute all Phase 1 calls to understand data structures, "
                f"then execute all Phase 2 calls to collect data, "
                f"finally synthesize insights from all {len(relevant_datasets)} datasets "
                f"to provide comprehensive analysis of: {query_context}"
            ),
            "success_criteria": f"Final answer incorporates findings from all {len(relevant_datasets)} datasets",
        }

        return analysis_plan

    except Exception as e:
        return {"error": f"Error creating analysis plan: {str(e)}"}


# ============================================================================
# Main Entry Point
# ============================================================================


async def main() -> None:
    """Run the MCP server."""
    print("Starting MCP server with semantic search...", file=sys.stderr)

    if not API_KEY:
        print("⚠ WARNING: DATA_GOV_API_KEY environment variable not set", file=sys.stderr)
        print("Set DATA_GOV_API_KEY=<your_api_key> for full functionality", file=sys.stderr)
    else:
        print("✓ DATA_GOV_API_KEY is configured", file=sys.stderr)

    # Check semantic search status
    if SEMANTIC_SEARCH_ENGINE and SEMANTIC_SEARCH_ENGINE.is_ready():
        print("✅ Semantic search is ready for AI-powered dataset discovery", file=sys.stderr)
    elif SEMANTIC_SEARCH_AVAILABLE:
        print("❌ Semantic search packages available but not initialized", file=sys.stderr)
        print("❌ Run 'python build_embeddings.py' to build embeddings", file=sys.stderr)
    else:
        print("❌ Semantic search packages not installed", file=sys.stderr)
        print(
            "❌ Install with: micromamba install -c conda-forge sentence-transformers faiss-cpu numpy", file=sys.stderr
        )

    print("Starting stdio transport...", file=sys.stderr)
    await mcp.run_stdio_async()


if __name__ == "__main__":
    print("Script starting...", file=sys.stderr)
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
