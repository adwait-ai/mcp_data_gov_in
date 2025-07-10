# MCP Data.gov.in Server

A clean, production-ready MCP server for analyzing Indian government datasets using the official Python MCP SDK. Provides Claude Desktop with intelligent access to data.gov.in through a curated dataset registry and smart filtering capabilities.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   micromamba env create -f environment.yml
   micromamba activate mcp-data-gov-in
   ```

2. **Get API Key:**
   - Sign up at [data.gov.in](https://data.gov.in/user/register)
   - Get your API key from your profile
   - Set it as environment variable: `export DATA_GOV_API_KEY=your_api_key_here`
   - Or create a `.env` file with: `DATA_GOV_API_KEY=your_api_key_here`

3. **Configure Claude Desktop:**
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

4. **Restart Claude Desktop and test:**
   ```
   "Search for healthcare datasets in India, inspect the structure of one, 
   and show me filtered data by state to understand regional patterns."
   ```

## 🎯 Key Features

- **Official MCP SDK**: Uses FastMCP for robust protocol handling
- **Curated Dataset Registry**: 500+ pre-indexed datasets for fast discovery
- **Smart Filtering**: Download only the data you need with column-based filters
- **6 Core Tools**: Search, download, filter, inspect, summarize, and browse datasets
- **1 Resource**: Full dataset registry accessible as MCP resource
- **Clean Architecture**: Single-file, standalone MCP server

## 📊 Available Tools

1. **`search_datasets`** - Search through curated dataset registry
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
# JSON string format
column_filters = '{"state": "Maharashtra", "year": "2023"}'

# Dictionary format (automatically handled)
column_filters = {"state": "Maharashtra", "year": "2023"}
```

### Registry Structure
The curated dataset registry contains 500+ datasets with:
- Resource IDs for API access
- Title, ministry, and sector metadata
- Direct URLs to data.gov.in pages
- Optimized for fast text search

## 🤝 Contributing

This codebase serves as a clean example of MCP server implementation. See `learning_mcp.md` for detailed explanations of MCP concepts and patterns used in this project.
