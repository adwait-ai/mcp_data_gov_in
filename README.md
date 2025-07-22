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
- **Multi-Query Search**: Simultaneous search with multiple queries to find both specific and general filterable datasets
- **Automatic Server-Side Pagination**: Downloads complete datasets up to 100,000 records without manual pagination
- **Intelligent Hybrid Filtering**: Server-side filtering (faster) + client-side filtering (comprehensive) with full pagination
- **Precomputed Embeddings**: Fast search with model preloading for optimal performance
- **Title-Prioritized Search**: Dataset titles get higher weight than ministry/sector for better relevance
- **Official MCP SDK**: Uses FastMCP for robust protocol handling
- **Curated Dataset Registry**: 83,000+ pre-indexed datasets for comprehensive coverage
- **Transparent Large Dataset Handling**: No size limits - server automatically paginates through any dataset size
- **8 Core Tools**: Multi-query search, download, filter, inspect, plan, summarize, and browse datasets
- **1 Resource**: Full dataset registry accessible as MCP resource
- **Clean Architecture**: Modular design with pagination-optimized data access

## 🔍 **Multi-Query Search Strategy**

The MCP server uses a **multi-query search approach** that finds both specific datasets and general filterable datasets simultaneously:

### **1. Think Multiple Queries Simultaneously**
```
search_datasets(['flight data', 'airline flights', 'Air India flights', 'Delhi airport data'])
→ Top 10 results per query → Comprehensive coverage of specific + general datasets
→ 🔴 MANDATORY: inspect_dataset_structure() on ALL promising datasets → understand structure and filtering options
→ download_filtered_dataset() strategically → combine specific and filtered general data (ONLY after inspection)
```

### **2. Multi-Query Philosophy**
- **Broad to Specific Coverage**: "flight data" finds general datasets, "Air India flights" finds specific ones
- **Filterable General Data**: General datasets from broad queries can be filtered for specific needs
- **Comprehensive Discovery**: Each query returns top 10 most relevant results
- **Low Relevance Filtering**: Results below similarity threshold automatically filtered out

### **3. Strategic Query Design**
- **Domain Level**: "agriculture data", "transport data", "health statistics"
- **Medium Specificity**: "crop production", "airline flights", "hospital data"  
- **High Specificity**: "wheat production Maharashtra", "Air India flights", "Delhi hospitals"
- **Geographic/Entity**: Include location and organization terms when relevant

### **4. Smart Result Analysis**
- **Cross-Query Relevance**: Datasets appearing in multiple queries are highly relevant
- **General vs Specific**: Compare broad query results (filterable) with specific query results (targeted)
- **Ministry Diversity**: Results span multiple government departments for comprehensive coverage

### **Example Multi-Query Workflow**
```
User: "Air India flights landing in Delhi airport"
→ search_datasets(['flight data', 'airline flights', 'Air India flights', 'Delhi airport data'])
→ Results: General flight datasets (filter for Air India + Delhi) + specific Air India datasets
→ 🔴 MANDATORY: inspect_dataset_structure() on ALL promising datasets BEFORE any downloads
→ Strategy: After inspection, filter general "all flights" dataset for Air India + Delhi, combine with specific Air India data
```

**Core Principle**: Multiple angles → simultaneous search → 🔴 **MANDATORY INSPECTION** → strategic downloads!

## 📊 Available Tools

1. **`search_datasets`** - Multi-query semantic search with 2-5 queries simultaneously (10 results per query)
2. **`get_detailed_dataset_info`** - Get complete information for specific datasets by resource ID
3. **`get_search_guidance`** - Comprehensive guide for multi-query search strategy and pagination capabilities
4. **`download_dataset`** - Download complete datasets with automatic pagination (⚠️ **REQUIRES INSPECTION FIRST**)
5. **`download_filtered_dataset`** - Download datasets with intelligent hybrid filtering (⚠️ **REQUIRES INSPECTION FIRST**)
6. **🔴 `inspect_dataset_structure`** - **MANDATORY** before downloads: Examine dataset schema, sample records, and pagination info
7. **`plan_multi_dataset_analysis`** - Plan comprehensive analysis workflows across multiple datasets
8. **`get_registry_summary`** - Get overview of available datasets by sector/ministry
9. **`list_sectors`** - List all available data sectors

## � **CRITICAL: Dataset Inspection Requirement**

**⚠️  MANDATORY WORKFLOW: You MUST inspect datasets before downloading!**

### **Why Inspection is Required**
- **Understand Data Structure**: See available columns, data types, and sample values
- **Plan Optimal Filtering**: Identify which columns can be used for filtering large datasets 
- **Avoid Wasted Resources**: Prevent downloading unnecessary data or missing filtering opportunities
- **Quality Assessment**: Understand data completeness and structure before analysis

### **Correct Workflow**
```
1. search_datasets() → Find promising datasets
2. 🔴 inspect_dataset_structure() → MANDATORY for ALL promising datasets  
3. Plan filtering strategy based on inspection insights
4. download_filtered_dataset() → Download with optimal filters
```

### **Incorrect Workflow** ❌
```
1. search_datasets() → Find datasets
2. download_dataset() → Downloads without understanding structure (WRONG!)
```

## �🔄 **Automatic Pagination & Large Dataset Handling**

The MCP server includes advanced **server-side pagination** that makes working with large datasets seamless:

### **Transparent Pagination**
- **No Manual Pagination**: Server automatically handles all pagination - no offset/limit management needed
- **Complete Dataset Access**: Downloads complete datasets up to 100,000 records automatically
- **Efficient Chunking**: Uses 1000 records per API call for optimal performance
- **Works with Filtering**: Pagination works seamlessly with both server-side and client-side filtering

### **Large Dataset Capabilities**
- **No Size Limits**: Handle datasets ranging from hundreds to tens of thousands of records
- **Automatic Detection**: Server detects dataset size and applies appropriate pagination strategy
- **Memory Efficient**: Streams data efficiently without overwhelming memory usage
- **Progress Transparency**: Returns pagination info showing total available vs downloaded records

### **Example Scenarios**
```
scenario: Dataset has 45,000 records, you want Maharashtra records only
→ download_filtered_dataset(filters={"state": "Maharashtra"}) 
→ Server paginates through all 45,000 records, filters for Maharashtra
→ Returns complete filtered result (e.g., 3,200 Maharashtra records)
→ All handled automatically - no manual pagination code needed
```

**Key Benefit**: Focus on data analysis, not pagination mechanics - the server handles all the complexity!

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
- `results_per_query`: Number of results per query in multi-query search (default: 10)
- `max_queries_per_search`: Maximum queries allowed per search (default: 5)
- `min_similarity_score`: Minimum similarity score for search engine
- `min_display_similarity_score`: Minimum score to display to users (default: 0.15)
- `title_weight`: How much to weight dataset titles in search
- `guidance_high_result_threshold`: Threshold for encouraging aggregation (default: 25)
- `guidance_medium_result_threshold`: Threshold for medium result guidance (default: 10)
- `encourage_aggregation`: Enable multi-dataset aggregation guidance (default: true)
- `emphasize_multi_query_search`: Emphasize multi-query search approach (default: true)

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

### Multi-Query Search Revolution
- **Simultaneous Query Processing**: Search with 2-5 queries at once to find specific + general datasets
- **Smart Result Organization**: Top 10 results per query, organized by search term for clear analysis  
- **General + Specific Discovery**: Find both filterable general datasets and targeted specific datasets
- **Automatic Relevance Filtering**: Low-similarity results filtered out automatically
- **Strategic Query Guidance**: Built-in examples for effective query combinations

### Enhanced Search Intelligence
- **Cross-Query Analysis**: Datasets appearing in multiple queries identified as highly relevant
- **Filterable Dataset Detection**: General datasets highlighted for specific filtering potential
- **Ministry Diversity Analysis**: Results spanning multiple government departments
- **Configurable Similarity Thresholds**: Customizable filtering for result quality

### Practical Use Case Optimization
- **Real-World Query Patterns**: Optimized for questions like "Air India flights landing in Delhi"
- **General-to-Specific Strategy**: Broad queries find filterable datasets, specific queries find targeted data
- **Combined Analysis Approach**: Guidance for using both general (filtered) and specific datasets together

### Automatic Server-Side Pagination & Large Dataset Handling
- **Transparent Pagination**: Complete automation - no manual offset/limit handling required
- **Large Dataset Support**: Seamlessly handle datasets up to 100,000 records with automatic chunking
- **Efficient Data Transfer**: 1000 records per API call for optimal performance
- **Pagination + Filtering Integration**: Server-side filtering applied during pagination for maximum efficiency
- **Memory-Optimized Processing**: Stream large datasets without memory overflow issues

### Enhanced Hybrid Filtering with Pagination
- **Server-side Filtering**: Applied during pagination for keyword fields (faster, less data transfer)
- **Client-side Filtering**: Applied to complete paginated datasets for non-keyword fields
- **Intelligent Field Detection**: Automatically determines optimal filtering approach per field
- **Complete Result Guarantee**: Pagination ensures you get ALL matching records, not just first page
- **Performance Optimization**: Reduces data transfer by filtering during download when possible

### Pagination Configuration & Control
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

## 🌐 Smithery Deployment

This MCP server is ready for deployment on [Smithery.ai](https://smithery.ai) for hosted access.

### Deployment Files

The repository includes:
- `smithery.yaml` - Deployment configuration for Smithery
- `Dockerfile` - Container configuration for deployment
- HTTP transport support via FastMCP's Streamable HTTP

### Configuration Parameters

When deployed on Smithery, the server accepts these configuration parameters:

- **dataGovApiKey** (required): Your data.gov.in API key for accessing datasets
- **maxSearchResults** (optional): Maximum search results to return (default: 20, max: 100)
- **downloadLimit** (optional): Maximum records per download (default: 1000, max: 10000)

### Deploying to Smithery

1. Push your repository to GitHub
2. Connect your GitHub account to Smithery
3. Navigate to the Deployments tab on your server page
4. Click "Deploy" to build and host the container
5. Configure with your data.gov.in API key when connecting

The server automatically detects HTTP mode when the `PORT` environment variable is set (Smithery deployment) and falls back to stdio for local development.
