# MCP Data.gov.in Server

A clean, production-ready MCP server for analyzing Indian government datasets using the official Python MCP SDK. Provides Claude Desktop with intelligent access to data.gov.in through a curated dataset registry and **AI-powered semantic search** capabilities.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   micromamba env create -f environment.yml
   micromamba activate mcp-data-gov-in
   ```

2. **Download model and precompute embeddings:**
   ```bash
   # Download the sentence transformer model locally (self-contained)
   python download_model.py
   
   # Build embeddings for semantic search
   python build_embeddings.py
   
   # Or run the complete setup script
   python setup.py
   ```
   This downloads the all-MiniLM-L6-v2 model to the `models/` directory and creates embeddings for all datasets.

3. **Get API Key:**
   - Sign up at [data.gov.in](https://data.gov.in/user/register)
   - Get your API key from your profile
   - Set it as environment variable: `export DATA_GOV_API_KEY=your_api_key_here`
   - Or create a `.env` file with: `DATA_GOV_API_KEY=your_api_key_here`

4. **Configure Claude Desktop:**
   Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "data-gov-in": {
         "command": "/Users/adwait/micromamba/envs/mcp-data-gov-in/bin/python",
         "args": ["/Users/adwait/projects/mcp_data_gov_in/mcp_server.py"]
       }
     }
   }
   ```

5. **Restart Claude Desktop and test:**
   ```
   "Search for healthcare datasets in India, inspect the structure of one, 
   and show me filtered data by state to understand regional patterns."
   ```

## 🎯 Key Features

- **Pure AI-Powered Semantic Search**: Uses all-MiniLM-L6-v2 and FAISS for intelligent dataset discovery only
- **No Fallback Search**: Ensures consistent, high-quality semantic matching for all queries
- **Precomputed Embeddings**: Fast search with model preloading for optimal performance
- **Title-Prioritized Search**: Dataset titles get higher weight than ministry/sector for better relevance
- **Official MCP SDK**: Uses FastMCP for robust protocol handling
- **Curated Dataset Registry**: 5,673 pre-indexed datasets for comprehensive coverage
- **Smart Filtering**: Hybrid server-side and client-side filtering for optimal performance
- **Complete Data Access**: Automatic pagination to download complete datasets (up to 100K records)
- **6 Core Tools**: Search, download, filter, inspect, summarize, and browse datasets
- **1 Resource**: Full dataset registry accessible as MCP resource
- **Clean Architecture**: Modular design with pure semantic search capabilities

## 🔍 Pure Semantic Search

The server **exclusively** uses **sentence-transformers** with the **all-MiniLM-L6-v2** model:

- **Self-Contained**: Model is downloaded to the local `models/` directory, making the project portable
- **Intelligent Matching**: Finds datasets based on meaning, not just keywords
- **Title Priority**: Dataset titles are weighted 3x higher than ministry/sector for better relevance
- **Similarity Scores**: Each result includes a confidence score
- **Fast Performance**: Embeddings are precomputed and cached using FAISS
- **Required Component**: Semantic search is mandatory - server will indicate errors if not available

## 📊 Available Tools

1. **`search_datasets`** - Pure AI-powered semantic search through curated dataset registry
2. **`download_dataset`** - Download complete datasets (use with caution for large data)
3. **`download_filtered_dataset`** - Download datasets with column-based filtering
4. **`inspect_dataset_structure`** - Examine dataset schema and sample records
5. **`get_registry_summary`** - Get overview of available datasets by sector/ministry
6. **`list_sectors`** - List all available data sectors

## 🏗️ Architecture

The server is a **standalone MCP implementation** that:
- Uses FastMCP for MCP protocol handling
- Loads a curated dataset registry from JSON for fast search
- Connects directly to data.gov.in API for data download
- Provides intelligent hybrid filtering (server-side + client-side)
- Uses automatic pagination for complete dataset downloads
- Supports both string and dictionary inputs for filters

## 📁 Project Structure

```
├── data/                   # Dataset registry and static data
│   └── data_gov_in_api_registry.json  # Curated dataset registry
├── tests/                  # Test suite for MCP server functions
├── examples/               # Usage examples and sample prompts
├── mcp_server.py          # Main MCP server implementation
├── environment.yml        # Dependencies (cleaned up)
├── Dockerfile            # Container configuration
├── learning_mcp.md       # MCP development guide
└── README.md             # This file
```

## 🛠️ Usage


**Note on Tool Return Values (2025 update):**
All MCP tool functions now return Python dictionaries (or lists) for structured data. FastMCP handles JSON serialization automatically. Do not return JSON strings from tool functions—return native Python objects for best compatibility and error handling.

The MCP server runs as a standalone process and connects directly to Claude Desktop. No need to run a separate backend server.

**Example Workflow:**
1. Search for datasets: `"Find datasets about education"`
2. Inspect structure: `"Show me the structure of dataset XYZ"`
3. Download filtered data: `"Get education data for Karnataka state only"`

## 📖 Examples

Check the `examples/` directory for sample prompts that demonstrate:
- Multi-step data discovery workflows
- Intelligent filtering strategies
- Complex data analysis patterns

## 🧪 Testing

```bash
# Run tests
pytest

# Test the MCP server directly (requires API key)
python mcp_server.py
```

## ⚙️ Configuration

The server uses a `config.json` file for easily adjustable parameters. Key settings include:

### Semantic Search Parameters
- `model_name`: Sentence transformer model to use
- `default_search_limit`: Number of datasets to return
- `max_search_limit`: Maximum allowed search limit
- `relevance_threshold`: Minimum similarity score for relevance
- `min_similarity_score`: Minimum similarity score to include in results
- `title_weight`: How much to weight dataset titles in search

### Data API Parameters
- `default_download_limit`: Default number of records to download
- `max_download_limit`: Maximum allowed download limit
- `default_inspect_sample_size`: Sample size for dataset inspection
- `request_timeout`: API request timeout in seconds

### Analysis Parameters
- `high_relevance_threshold`: Threshold for high relevance datasets

### Updating Configuration
You can update configuration values using the MCP tools:
```
"Update the semantic search limit to 30 datasets"
"Change the relevance threshold to 0.3"
"Set the default download limit to 200"
```

Or edit `config.json` directly and restart the server.

## 🆕 Recent Improvements (July 2025)

### Enhanced Filtering and Pagination
- **Server-side Filtering**: Automatically uses data.gov.in API's native filtering for keyword fields
- **Smart Hybrid Approach**: Falls back to client-side filtering for non-keyword fields
- **Complete Dataset Access**: Automatic pagination downloads up to 100,000 records total
- **Optimized Performance**: Reduces data transfer by filtering at the API level when possible
- **Transparent Integration**: Same tool interface with improved backend capabilities

### Configuration Updates
- Added `pagination_limit` (1000 records per API request)
- Added `max_total_records` (100,000 maximum total records)
- Increased `default_download_limit` from 100 to 1000
- Added `enable_server_side_filtering` feature flag

### Technical Implementation
- **Field Analysis**: Automatically detects which fields support server-side filtering
- **Pagination Management**: Handles offset-based pagination transparently
- **Error Resilience**: Graceful fallback to client-side filtering when needed
- **Comprehensive Reporting**: Detailed filtering summaries in tool responses

## 🐛 Recent Bug Fixes (July 2025)

### Fixed Client-Side Date Filtering Issue
- **Problem**: Client-side filtering failed when using display field names (e.g., "Arrival_Date") that differed from actual record field names (e.g., "arrival_date")
- **Solution**: Enhanced client-side filtering with intelligent field name mapping
- **Improvements**:
  - Automatic field name variation handling (case-insensitive, underscore variations)
  - Better date field matching with exact and substring matching
  - Preserved backward compatibility with existing filter syntax
  - Added field mapping debug information for troubleshooting

## 🚀 Advanced Usage

### Effective Search Strategies
The MCP server provides intelligent search guidance to help find relevant datasets:

**Search Strategies:**
- **Specific Queries**: Use when you know exact terminology (e.g., "covid vaccination data")
- **General Queries**: Use for exploration, then filter by columns (e.g., "health" → filter by state/year)
- **Iterative Approach**: Start broad → inspect structures → filter specifically

**Getting Search Help:**
```
"Get search guidance for health domain"
"Help me search for energy-related datasets"
```

The `get_search_guidance` tool provides domain-specific strategies, query suggestions, and filtering tips.

### Intelligent Hybrid Filtering
The server uses **intelligent hybrid filtering** that automatically optimizes performance by combining server-side and client-side approaches:

```python
# Dictionary format (recommended)
column_filters = {"state": "Maharashtra", "commodity": "Tomato"}

# JSON string format (also supported)
column_filters = '{"state": "Maharashtra", "commodity": "Tomato"}'
```

**Automatic Smart Filtering Process:**
1. **Field Analysis**: Automatically detects which fields support server-side filtering from API metadata
2. **Server-side filtering** applied first for optimized fields (e.g., state.keyword, commodity)
   - More efficient, reduces data transfer and processing time
   - Uses data.gov.in API's native filtering capabilities with exact field IDs
3. **Client-side filtering** for remaining fields as transparent fallback
   - Applied after download for comprehensive coverage
   - Ensures no relevant data is missed

**Recommended Usage Pattern:**
1. Search broadly: "agriculture" or "market prices" 
2. Inspect structure: `inspect_dataset_structure()` to see available columns
3. Filter specifically: `{"state": "Karnataka", "commodity": "Rice"}`

**Benefits:**
- **Transparent**: Same tool interface, optimized backend automatically
- **Efficient**: Server-side filtering reduces API calls and data transfer
- **Comprehensive**: Client-side fallback ensures complete filtering coverage with intelligent field name mapping
- **Robust**: Graceful handling when server-side filtering isn't available
- **Flexible**: Supports field name variations (e.g., "Arrival_Date" vs "arrival_date") automatically

### Complete Dataset Downloads with Intelligent Pagination
The server automatically handles large datasets using smart pagination and filtering:

**Automatic Dataset Processing:**
- Intelligently detects which filters can be applied server-side for optimal performance
- Downloads complete datasets using multiple API calls with offset-based pagination
- Can download up to 100,000 records total (configurable via `max_total_records`)
- Each pagination request fetches 1,000 records (API limit, configurable via `pagination_limit`)
- Applies server-side filters during download to minimize data transfer

**Smart Filtering Integration:**
- Server-side filters are applied at the API level during pagination
- Only downloads data that matches server-side filterable criteria
- Client-side filters are applied after complete download for non-server-filterable fields
- Provides detailed filtering summary showing what was filtered where

**Performance Benefits:**
- Dramatically reduced data transfer when server-side filtering is available
- Faster downloads by filtering at the source rather than downloading everything
- Transparent fallback ensures data completeness even when server-side filtering isn't available

### Registry Structure
The curated dataset registry contains 500+ datasets with:
- Resource IDs for API access
- Title, ministry, and sector metadata
- Direct URLs to data.gov.in pages
- Field metadata for intelligent filtering

### Data.gov.in API Integration
The MCP server intelligently works with the API's capabilities:

**Enhanced API Integration:**
- **Intelligent Field Detection**: Automatically analyzes API metadata to identify server-filterable fields
- **Adaptive Pagination**: Uses `offset` parameter to download complete datasets efficiently  
- **Hybrid Filtering**: Uses `filters[field_id]` for server-side filtering when available, client-side for others
- **Performance Optimization**: Combines server and client filtering for optimal speed and completeness

**Filtering Behavior Details:**
- Fields with `.keyword` suffix (e.g., `state.keyword`, `commodity`) are typically server-side filterable
- The system automatically detects filterable fields from each dataset's `field_exposed` metadata
- Server-side filters are built using exact field IDs from the API metadata
- Complete datasets are downloaded when no server-side filters apply
- Maximum 100,000 total records per download (configurable via `max_total_records`)
- For large filtered results, provides sample records and suggestions for additional filters

**Error Resilience:**
- Graceful fallback to client-side filtering when server-side filtering fails
- Intelligent field name mapping handles variations (e.g., "Arrival_Date" vs "arrival_date")
- Enhanced date field matching for exact and partial date filtering
- Comprehensive error handling for network issues and API limits
- Detailed status reporting showing which filters were applied where

## 🤝 Contributing

This codebase serves as a clean example of MCP server implementation. See `learning_mcp.md` for detailed explanations of MCP concepts, patterns, and up-to-date return value conventions for MCP tools.
