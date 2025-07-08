#!/usr/bin/env python3
"""
Simple standalone MCP server for data.gov.in using FastMCP.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import httpx
from mcp.server import FastMCP


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


def search_static_registry(dataset_registry: List[Dict[str, Any]], query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search through the dataset registry."""
    query_lower = query.lower()
    results = []

    for dataset in dataset_registry:
        # Search in title, ministry, and sector
        searchable_text = " ".join(
            [
                dataset.get("title", "").lower(),
                dataset.get("ministry", "").lower(),
                dataset.get("sector", "").lower(),
                dataset.get("catalog", "").lower(),
            ]
        )

        # Simple text matching
        if query_lower in searchable_text:
            results.append(
                {
                    "resource_id": dataset["resource_id"],
                    "title": dataset["title"],
                    "ministry": dataset.get("ministry", "Unknown"),
                    "sector": dataset.get("sector", "Unknown"),
                    "url": dataset.get("url", f"https://www.data.gov.in/resource/{dataset['resource_id']}#api"),
                }
            )

    return results[:limit]


async def download_api(resource_id: str, api_key: str, limit: int = 100) -> Dict[str, Any]:
    """Download a dataset from data.gov.in API."""
    params = {
        "resource_id": resource_id,
        "api-key": api_key,
        "format": "json",
        "limit": limit,
    }
    async with httpx.AsyncClient() as client:
        url = f"https://api.data.gov.in/resource/{resource_id}"
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


# ============================================================================
# Server Initialization
# ============================================================================

# Initialize environment and configuration
load_env_file()
API_KEY = os.getenv("DATA_GOV_API_KEY")
DATASET_REGISTRY = load_dataset_registry()

# Create the FastMCP server instance
mcp = FastMCP("data-gov-in-mcp")

# Log initialization status
print("Creating MCP server...", file=sys.stderr)
if not API_KEY:
    print("⚠ WARNING: DATA_GOV_API_KEY environment variable not set", file=sys.stderr)
    print("Either set it as an environment variable or add it to a .env file", file=sys.stderr)
print(f"✓ Loaded {len(DATASET_REGISTRY)} datasets from registry", file=sys.stderr)


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
async def search_datasets(query: str, limit: int = 5) -> str:
    """Search public datasets on data.gov.in by keyword using static registry."""
    try:
        # Use static registry
        results = search_static_registry(DATASET_REGISTRY, query, limit)

        if not results:
            return json.dumps(
                {
                    "message": f"No datasets found matching '{query}'",
                    "suggestion": "Try searching for: health, petroleum, oil, crude, inflation, taxes, or guarantees",
                    "total_datasets": len(DATASET_REGISTRY),
                },
                indent=2,
            )

        return json.dumps(
            {
                "query": query,
                "found": len(results),
                "total_available": len(DATASET_REGISTRY),
                "datasets": results,
                "note": "Results from curated dataset registry. API key still required for downloading data.",
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:
        return f"Error searching datasets: {str(e)}"


@mcp.tool()
async def download_dataset(resource_id: str, limit: int = 100) -> str:
    """Download a dataset from data.gov.in."""
    try:
        if not API_KEY:
            return json.dumps(
                {"error": "DATA_GOV_API_KEY environment variable not set. Please set it to use this tool."}, indent=2
            )

        result = await download_api(resource_id, API_KEY, limit)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error downloading dataset: {str(e)}"


@mcp.tool()
async def inspect_dataset_structure(resource_id: str, sample_size: int = 5) -> str:
    """Quick inspection of dataset structure."""
    try:
        if not API_KEY:
            return json.dumps(
                {"error": "DATA_GOV_API_KEY environment variable not set. Please set it to use this tool."}, indent=2
            )

        # Download a small sample to inspect structure
        result = await download_api(resource_id, API_KEY, sample_size)

        # Extract just the structure info
        structure = {"fields": result.get("field", []), "sample_records": result.get("records", [])[:sample_size]}
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


# ============================================================================
# Main Entry Point
# ============================================================================


async def main() -> None:
    """Run the MCP server."""
    print("Starting simple MCP server...", file=sys.stderr)

    if not API_KEY:
        print("⚠ WARNING: DATA_GOV_API_KEY environment variable not set", file=sys.stderr)
        print("Set DATA_GOV_API_KEY=<your_api_key> for full functionality", file=sys.stderr)
    else:
        print("✓ DATA_GOV_API_KEY is configured", file=sys.stderr)

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
