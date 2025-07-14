# Learning MCP Server Development

This document uses the `mcp_data_gov_in` project as a case study to explain how to build Model Context Protocol (MCP) servers using the official Python SDK.

## 📚 Table of Contents

1. [MCP Fundamentals](#mcp-fundamentals)
2. [Project Setup and Structure](#project-setup-and-structure)
3. [Core MCP Components](#core-mcp-components)
4. [Tools Implementation](#tools-implementation)
5. [Resources Implementation](#resources-implementation)
6. [Configuration and Environment](#configuration-and-environment)
7. [Error Handling Patterns](#error-handling-patterns)
8. [Advanced Features](#advanced-features)
9. [Best Practices](#best-practices)

## 🔍 MCP Fundamentals

### What is MCP?

The Model Context Protocol (MCP) is an open standard for connecting AI assistants with external tools and data sources. It defines a client-server architecture where:
- **MCP Client**: AI assistants like Claude Desktop
- **MCP Server**: Your application that provides tools and resources

### Key Concepts

1. **Tools**: Functions that the AI can call to perform actions
2. **Resources**: Static or dynamic data that the AI can read
3. **Prompts**: Pre-defined conversation starters (not used in this project)

## 🏗️ Project Setup and Structure

### Dependencies

Looking at our `environment.yml`, an MCP server with semantic search needs:

```yaml
name: mcp-data-gov-in
channels:
  - conda-forge
dependencies:
  - python=3.11
  - httpx                    # Async HTTP client
  - sentence-transformers    # Embedding models for semantic search
  - faiss-cpu               # Fast similarity search
  - numpy                   # Numerical operations
  - pip
  - pip:
    - mcp==1.0.0            # Official MCP SDK
```

**Key Dependencies Explained:**
- `mcp`: Official Model Context Protocol SDK
- `httpx`: Async HTTP client for API calls
- `sentence-transformers`: Provides pre-trained embedding models
- `faiss-cpu`: Facebook's library for efficient similarity search
- `numpy`: Required for vector operations

### Project Architecture

Our project follows a modular architecture with semantic search capabilities:

```
mcp_server.py           # Main MCP server implementation
├── Configuration       # Environment and data loading
├── Semantic Search     # AI-powered dataset discovery
├── Utility Functions   # Helper functions for data processing
├── Server Init         # FastMCP instance and setup
├── MCP Resources       # Data sources for the AI
├── MCP Tools          # Functions the AI can execute
└── Main Entry Point    # Server startup logic

semantic_search.py      # Semantic search implementation
├── DatasetSemanticSearch  # Main search class
├── Embedding Model     # sentence-transformers integration
├── FAISS Index        # Vector similarity search
└── Local Caching      # Model and embeddings storage

Supporting Scripts:
├── download_model.py   # Download and cache embedding model
├── build_embeddings.py # Precompute dataset embeddings
└── embeddings/        # Cached model and embeddings directory
```

## 🛠️ Core MCP Components

### 1. Server Initialization

The foundation of any MCP server is the FastMCP instance:

```python
from mcp.server import FastMCP

# Create the FastMCP server instance
mcp = FastMCP("data-gov-in-mcp")
```

**Key Learning Points:**
- Use a descriptive server name
- FastMCP handles all MCP protocol details
- The server name appears in Claude Desktop's MCP server list

### 2. Server Lifecycle

The main function demonstrates the standard MCP server lifecycle:

```python
async def main() -> None:
    """Run the MCP server."""
    print("Starting simple MCP server...", file=sys.stderr)
    
    # Configuration checks
    if not API_KEY:
        print("⚠ WARNING: DATA_GOV_API_KEY environment variable not set", file=sys.stderr)
    
    # Start the server
    await mcp.run_stdio_async()
```

**Key Learning Points:**
- Use `run_stdio_async()` for Claude Desktop integration
- Print status messages to `sys.stderr` (not stdout)
- Perform configuration validation before starting

## 🧠 Semantic Search Architecture

This project implements AI-powered semantic search using sentence transformers and FAISS for efficient vector similarity search.

### Components

1. **Embedding Model**: `all-MiniLM-L6-v2` from sentence-transformers
2. **Vector Database**: FAISS for fast similarity search
3. **Precomputed Embeddings**: Built once, reused for all searches
4. **Local Caching**: Model downloaded and cached locally

### Implementation Structure

```
semantic_search.py       # Main semantic search module
download_model.py        # Download and cache the embedding model
build_embeddings.py      # Precompute dataset embeddings
embeddings/             # Directory for cached embeddings and model
```

### Key Features

- **Semantic Understanding**: Finds datasets by meaning, not just keywords
- **Title Prioritization**: Dataset titles weighted higher than descriptions
- **Similarity Scores**: Returns confidence scores for each result
- **Local Operation**: No external API calls required for search
- **Fast Performance**: Precomputed embeddings enable instant search

### Initialization Pattern

```python
# Import semantic search functionality
try:
    from semantic_search import initialize_semantic_search, DatasetSemanticSearch
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Semantic search not available: {e}", file=sys.stderr)
    SEMANTIC_SEARCH_AVAILABLE = False

# Initialize semantic search if available
SEMANTIC_SEARCH_ENGINE = None
if SEMANTIC_SEARCH_AVAILABLE and DATASET_REGISTRY:
    try:
        print("🔄 Initializing semantic search...", file=sys.stderr)
        SEMANTIC_SEARCH_ENGINE = initialize_semantic_search(DATASET_REGISTRY)
        print("✅ Semantic search initialized", file=sys.stderr)
    except Exception as e:
        print(f"❌ Failed to initialize semantic search: {e}", file=sys.stderr)
```

### Setup Requirements

```bash
# Install semantic search dependencies
micromamba install -c conda-forge sentence-transformers faiss-cpu numpy

# Download the embedding model
python download_model.py

# Build precomputed embeddings
python build_embeddings.py
```

## 🔧 Tools Implementation

### Tool Definition Pattern

**2025 Update: Return Python objects, not JSON strings**

Every MCP tool should return a Python dictionary (or list) for structured data. FastMCP will handle JSON serialization automatically. Only return a string for plain error messages or human-readable output.

```python
@mcp.tool()
async def tool_name(param1: type, param2: type = default) -> dict:
    """Tool description that the AI sees."""
    try:
        # Input validation
        if not some_condition:
            return {"error": "Error message"}
        # Business logic
        result = await some_async_operation()
        return result  # Return as dict, not as JSON string
    except Exception as e:
        return {"error": str(e)}
```

### Example: Semantic Search Tool

```python
@mcp.tool()
async def search_datasets(query: str, limit: int = 5) -> dict:
    """
    Search public datasets on data.gov.in by keyword using AI-powered semantic search.

    This tool uses semantic search to find relevant datasets based on meaning,
    not just keyword matching. The search prioritizes dataset titles over ministry/sector names.
    """
    try:
        # Check if semantic search is available and ready
        if not SEMANTIC_SEARCH_ENGINE:
            return {
                "error": "Semantic search is not initialized. Please ensure the required packages are installed and embeddings are built.",
                "suggestion": "Run 'python build_embeddings.py' to initialize semantic search.",
                "available_packages": SEMANTIC_SEARCH_AVAILABLE
            }
        
        if not SEMANTIC_SEARCH_ENGINE.is_ready():
            return {
                "error": "Semantic search engine is not ready. Embeddings may not be built yet.",
                "suggestion": "Run 'python build_embeddings.py' to build embeddings for semantic search."
            }

        # Use semantic search
        results = SEMANTIC_SEARCH_ENGINE.search(query, limit=limit)

        if not results:
            return {
                "message": f"No datasets found matching '{query}' using semantic search",
                "suggestion": "Try rephrasing your query or using different keywords. For example: health, petroleum, oil, crude, inflation, taxes, or guarantees",
                "total_datasets": len(DATASET_REGISTRY),
                "search_method": "semantic search (AI-powered)"
            }

        # Return results with semantic similarity scores
        return {
            "query": query,
            "found": len(results),
            "total_available": len(DATASET_REGISTRY),
            "search_method": "semantic search (AI-powered)",
            "datasets": results,
            "note": "Results from curated dataset registry using AI-powered semantic search. API key still required for downloading data.",
            "tip": "Use inspect_dataset_structure() first, then download_filtered_dataset() to get specific data subsets.",
            "semantic_note": (
                "These results are ranked by semantic similarity. The search prioritizes dataset titles. "
                "Each result includes a similarity_score and full metadata for context."
            )
        }
    except Exception as e:
        return {"error": f"Error in semantic search: {str(e)}"}
```

**Key Learning Points:**
- **Always return Python dictionaries or lists for structured data.**
- FastMCP will serialize your return value to JSON automatically.
- Only return a string for plain error messages or human-readable output.
- Include helpful suggestions in responses.
- Provide guidance on follow-up actions.

### Advanced Tool: Flexible Input Types

The `download_filtered_dataset` tool shows how to handle multiple input types:

```python
@mcp.tool()
async def download_filtered_dataset(
    resource_id: str, 
    column_filters: Union[str, Dict[str, str]], 
    limit: int = 100
) -> str:
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
            + 'Expected JSON string like "{\"column_name\": \"filter_value\"}" or a dictionary object.'
        }
```

**Key Learning Points:**
- Use `Union` types for flexible input handling
- Always validate and sanitize inputs
- Provide clear error messages with examples
- Handle edge cases gracefully
- **Return Python objects, not JSON strings, for all structured data.**

## 📊 Resources Implementation

### Resource Definition Pattern

Resources provide static or dynamic data that the AI can read:

```python
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
```

**Key Learning Points:**
- Use descriptive URI schemes (`dataset://registry`)
- Resources return strings, just like tools
- Include metadata about the data (counts, last updated, etc.)
- Resources are read-only - use tools for actions

## ⚙️ Configuration and Environment

### Environment Loading

The project shows a clean pattern for configuration:

```python
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

# Initialize environment and configuration
load_env_file()
API_KEY = os.getenv("DATA_GOV_API_KEY")
```

**Key Learning Points:**
- Use `pathlib.Path` for file operations
- Support both environment variables and `.env` files
- Handle missing configuration gracefully
- Strip whitespace from values

### Data Loading

For performance, load static data once at startup:

```python
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

# Load data once at startup
DATASET_REGISTRY = load_dataset_registry()
```

**Key Learning Points:**
- Use proper encoding (`utf-8`) for international data
- Handle file not found and JSON parsing errors
- Return empty collections as fallbacks
- Load data once, use many times

## 🛡️ Error Handling Patterns

### Tool Error Handling

Every tool should handle errors gracefully:

```python
@mcp.tool()
async def some_tool(param: str) -> str:
    try:
        # Input validation
        if not API_KEY:
            return json.dumps({
                "error": "DATA_GOV_API_KEY environment variable not set. Please set it to use this tool."
            }, indent=2)
        
        # Business logic that might fail
        result = await risky_operation(param)
        
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error in some_tool: {str(e)}"
```

### HTTP Error Handling

For API calls, handle HTTP errors appropriately:

```python
async def download_api(resource_id: str, api_key: str, limit: int = 100) -> Dict[str, Any]:
    """Download a dataset from data.gov.in API with optional filtering."""
    async with httpx.AsyncClient() as client:
        url = f"https://api.data.gov.in/resource/{resource_id}"
        response = await client.get(url, params=params)
        response.raise_for_status()  # Raises HTTPError for bad responses
        return response.json()
```

**Key Learning Points:**
- Use `raise_for_status()` to catch HTTP errors
- Return consistent error JSON structures
- Include helpful error messages
- Don't expose internal errors to users

## 🚀 Advanced Features

### Client-Side Filtering

The project implements client-side filtering for better user experience:

```python
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
```

**Key Learning Points:**
- Implement client-side filtering to reduce response sizes
- Use case-insensitive substring matching
- Preserve original data structure
- Update relevant metadata (like record counts)

### Search and Discovery

The search function demonstrates efficient text search:

```python
def search_static_registry(dataset_registry: List[Dict[str, Any]], query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search through the dataset registry."""
    query_lower = query.lower()
    results = []

    for dataset in dataset_registry:
        # Search in title, ministry, and sector
        searchable_text = " ".join([
            dataset.get("title", "").lower(),
            dataset.get("ministry", "").lower(),
            dataset.get("sector", "").lower(),
            dataset.get("catalog", "").lower(),
        ])

        # Simple text matching
        if query_lower in searchable_text:
            results.append({
                "resource_id": dataset["resource_id"],
                "title": dataset["title"],
                "ministry": dataset.get("ministry", "Unknown"),
                "sector": dataset.get("sector", "Unknown"),
                "url": dataset.get("url", f"https://www.data.gov.in/resource/{dataset['resource_id']}#api"),
            })

    return results[:limit]
```

**Key Learning Points:**
- Create searchable text by combining relevant fields
- Use case-insensitive matching
- Return structured, consistent results
- Implement proper result limiting

## 📋 Best Practices

### 1. Code Organization

```python
# ============================================================================
# Configuration and Utility Functions
# ============================================================================
# Helper functions and configuration

# ============================================================================
# Server Initialization
# ============================================================================
# FastMCP setup and global state

# ============================================================================
# MCP Resources
# ============================================================================
# Data sources for the AI

# ============================================================================
# MCP Tools
# ============================================================================
# Functions the AI can execute

# ============================================================================
# Main Entry Point
# ============================================================================
# Server startup logic
```

**Use clear section dividers** to organize your code logically.

### 2. Function Documentation

```python
@mcp.tool()
async def inspect_dataset_structure(resource_id: str, sample_size: int = 5) -> str:
    """
    Quick inspection of dataset structure and available columns.
    
    Use this function first to understand the data structure, then use download_filtered_dataset
    with specific column filters to get only the data you need.
    """
```

**Include usage guidance** in docstrings to help the AI use tools effectively.

### 3. Response Structure

```python
structure = {
    "fields": fields,
    "column_names": column_names,
    "sample_records": sample_records,
    "total_records_available": result.get("total", "unknown"),
    "usage_tip": "Use download_filtered_dataset() with column_filters to get specific data subsets",
    "example_filter": f'{{"column_name": "filter_value"}} or as dict {{"column_name": "filter_value"}}' if column_names else "No columns available for filtering"
}
```

**Include usage tips and examples** in responses to guide the AI's next actions.

### 4. Type Hints

```python
from typing import Dict, Any, List, Optional, Union

def search_static_registry(
    dataset_registry: List[Dict[str, Any]], 
    query: str, 
    limit: int = 10
) -> List[Dict[str, Any]]:
```

**Use comprehensive type hints** for better code maintainability and IDE support.

### 5. Error Messages

```python
return json.dumps({
    "error": f"Invalid column_filters format. Received: {type(column_filters).__name__}. " +
            "Expected JSON string like '{\"column_name\": \"filter_value\"}' or a dictionary object."
}, indent=2)
```

**Provide specific, actionable error messages** with examples.

## 🎯 Semantic Search Best Practices

### 1. Graceful Degradation

```python
# Always check if semantic search is available
if not SEMANTIC_SEARCH_ENGINE:
    return {
        "error": "Semantic search is not initialized.",
        "suggestion": "Run 'python build_embeddings.py' to initialize semantic search.",
        "available_packages": SEMANTIC_SEARCH_AVAILABLE
    }

if not SEMANTIC_SEARCH_ENGINE.is_ready():
    return {
        "error": "Semantic search engine is not ready. Embeddings may not be built yet.",
        "suggestion": "Run 'python build_embeddings.py' to build embeddings for semantic search."
    }
```

### 2. Local Model Caching

```python
# Download model once, cache locally
def download_model():
    """Download and cache the sentence transformer model locally."""
    model_name = "all-MiniLM-L6-v2"
    cache_dir = Path("./embeddings/model_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model {model_name}...", file=sys.stderr)
    model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
    print(f"✅ Model downloaded and cached at {cache_dir}", file=sys.stderr)
```

### 3. Precomputed Embeddings

```python
# Build embeddings once, reuse for all searches
def build_embeddings(registry_path: str):
    """Build and save embeddings for the dataset registry."""
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    # Create combined text for embedding (prioritize titles)
    texts = []
    for dataset in registry:
        title = dataset.get('title', '')
        desc = dataset.get('desc', '')
        # Title gets more weight in the embedding
        combined_text = f"{title} {title} {desc}"
        texts.append(combined_text)
    
    # Compute embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Save for fast loading
    np.save(embeddings_path, embeddings)
```

### 4. Similarity Score Interpretation

```python
# Provide guidance when similarity scores are low
if results:
    max_score = max(result.get("similarity_score", 0) for result in results)
    if max_score < 0.3:
        response["relevance_warning"] = (
            "Low similarity scores detected. Please verify if these datasets are relevant "
            "to the user's query before proceeding with data download."
        )
```

### 5. Error Handling for Dependencies

```python
# Handle missing dependencies gracefully
try:
    from semantic_search import initialize_semantic_search, DatasetSemanticSearch
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Semantic search not available: {e}", file=sys.stderr)
    print("Install required packages with: micromamba install -c conda-forge sentence-transformers faiss-cpu numpy", file=sys.stderr)
    SEMANTIC_SEARCH_AVAILABLE = False
```

## 🎯 Common Patterns

### 1. Pure Functions

```python
# Good: Pure function that's easy to test
def filter_dataset_records(data: Dict[str, Any], column_filters: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    # No side effects, deterministic output

# Usage in tools
@mcp.tool()
async def download_filtered_dataset(...):
    result = await download_api(resource_id, API_KEY, limit)
    if filters_dict:
        result = filter_dataset_records(result, filters_dict)
```

### 2. Async/Await for I/O

```python
async def download_api(resource_id: str, api_key: str, limit: int = 100) -> Dict[str, Any]:
    """Download a dataset from data.gov.in API with optional filtering."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        return response.json()
```

### 3. Resource Management

```python
async with httpx.AsyncClient() as client:
    # Client automatically closed after use
    response = await client.get(url, params=params)
```

## 🔍 Debugging Tips

### 1. Logging to stderr

```python
print("Creating MCP server...", file=sys.stderr)
print(f"✓ Loaded {len(DATASET_REGISTRY)} datasets from registry", file=sys.stderr)
```

### 2. Error Tracking

```python
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
```

### 3. Validation

```python
# Validate configuration
if not API_KEY:
    print("⚠ WARNING: DATA_GOV_API_KEY environment variable not set", file=sys.stderr)
    print("Set DATA_GOV_API_KEY=<your_api_key> for full functionality", file=sys.stderr)
```

## 📖 Further Learning

### Official Documentation
- [MCP Specification](https://spec.modelcontextprotocol.org/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

### This Project's Key Files
- `mcp_server.py` - Complete MCP server implementation
- `data/data_gov_in_api_registry.json` - Example of curated data
- `environment.yml` - Dependencies and environment setup

This project demonstrates a complete, production-ready MCP server that showcases all the essential patterns and best practices for building robust MCP integrations.
