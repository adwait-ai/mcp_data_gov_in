#!/usr/bin/env python3
"""
Simple standalone MCP server for data.gov.in using FastMCP.
"""

import asyncio
import json
import os
import sys
from typing import Optional, Dict, Any

import httpx
from mcp.server import FastMCP

print("Creating MCP server...", file=sys.stderr)


# Try to load .env file if it exists
def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


load_env_file()

# Create the FastMCP server instance
mcp = FastMCP("data-gov-in-mcp")

# Get API key from environment (now includes .env file)
API_KEY = os.getenv("DATA_GOV_API_KEY")
if not API_KEY:
    print("⚠ WARNING: DATA_GOV_API_KEY environment variable not set", file=sys.stderr)
    print("Either set it as an environment variable or add it to a .env file", file=sys.stderr)


# Simple data.gov.in API functions
async def search_api(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search for datasets on data.gov.in"""
    params = {
        "q": query,
        "api-key": API_KEY,
        "format": "json",
        "limit": limit,
    }
    async with httpx.AsyncClient() as client:
        url = "https://api.data.gov.in/catalog/1.0/search"
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def download_api(resource_id: str, limit: int = 100) -> Dict[str, Any]:
    """Download a dataset from data.gov.in"""
    params = {
        "resource_id": resource_id,
        "api-key": API_KEY,
        "format": "json",
        "limit": limit,
    }
    async with httpx.AsyncClient() as client:
        url = f"https://api.data.gov.in/resource/{resource_id}"
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def search_datasets(query: str, limit: int = 5) -> str:
    """Search public datasets on data.gov.in by keyword."""
    try:
        if not API_KEY:
            return json.dumps(
                {"error": "DATA_GOV_API_KEY environment variable not set. Please set it to use this tool."}, indent=2
            )

        result = await search_api(query, limit)
        return json.dumps(result, indent=2, ensure_ascii=False)
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

        result = await download_api(resource_id, limit)
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
        result = await download_api(resource_id, sample_size)

        # Extract just the structure info
        structure = {"fields": result.get("field", []), "sample_records": result.get("records", [])[:sample_size]}
        return json.dumps(structure, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error inspecting dataset: {str(e)}"


async def main():
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
