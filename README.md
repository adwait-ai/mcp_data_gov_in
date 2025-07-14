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
- **Smart Filtering**: Download only the data you need with column-based filters
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
- Provides client-side filtering to reduce response sizes
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

## 🔧 Configuration

The server supports multiple configuration methods:

1. **Environment Variables:**
   ```bash
   export DATA_GOV_API_KEY=your_api_key_here
   ```

2. **`.env` File:**
   ```
   DATA_GOV_API_KEY=your_api_key_here
   ```

3. **Runtime Detection:**
   The server will warn if no API key is configured but still provide search functionality.

## 🚀 Advanced Usage

### Filtering Data
Use `download_filtered_dataset` with column filters to get specific subsets:

```python
# Dictionary format (recommended)
column_filters = {"state": "Maharashtra", "year": "2023"}

# JSON string format (also supported, but not required)
column_filters = '{"state": "Maharashtra", "year": "2023"}'
```

### Registry Structure
The curated dataset registry contains 500+ datasets with:
- Resource IDs for API access
- Title, ministry, and sector metadata
- Direct URLs to data.gov.in pages
- Optimized for fast text search

## 🤝 Contributing

This codebase serves as a clean example of MCP server implementation. See `learning_mcp.md` for detailed explanations of MCP concepts, patterns, and up-to-date return value conventions for MCP tools.
