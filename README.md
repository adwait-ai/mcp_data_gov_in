# MCP Data.gov.in Server

A clean, production-read## 📁 Project Structure

```
├── app/                    # FastAPI server with tool implementations
│   ├── main.py            # FastAPI app and JSON-RPC endpoint
│   ├── tools.py           # Data analysis tools with LLM-driven column selection
│   ├── portal_client.py   # data.gov.in API client
│   └── schemas.py         # Pydantic models
├── tests/                 # Test suite
├── examples/              # Usage examples and sample prompts
├── mcp_server.py          # Clean MCP server using official SDK
├── environment.yml        # Dependencies
├── test.http             # HTTP test requests for FastAPI
└── README.md             # This file
```for analyzing Indian government datasets using the official Python MCP SDK. Provides Claude Desktop with intelligent access to data.gov.in through LLM-driven semantic column selection.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   micromamba env create -f environment.yml
   micromamba activate mcp-data-gov-in
   ```

2. **Start the FastAPI backend:**
   ```bash
   uvicorn app.main:app --host localhost --port 8000
   ```

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
   "Search for healthcare datasets in India, load one for analysis, 
   and show me data filtered by state to understand regional patterns."
   ```

## 🎯 Key Features

- **Official MCP SDK**: Uses the official Python MCP SDK for robust protocol handling
- **LLM-Driven Column Selection**: Intelligent semantic column mapping without static lookups
- **Multi-Step Workflows**: Chain operations for complex data analysis
- **5 Analysis Tools**: Search, load, analyze, inspect, and download datasets
- **Clean Architecture**: Minimal, production-ready codebase

## � Architecture

- **FastAPI Backend** (`app/`): Handles data.gov.in API integration and tool logic
- **MCP Server** (`mcp_server.py`): Official SDK-based MCP protocol server
- **No Custom Protocol Code**: Uses only the official MCP SDK for reliability

## �📁 Project Structure

```
├── app/                    # FastAPI server with tool implementations
│   ├── main.py            # FastAPI app and JSON-RPC endpoint
│   ├── tools.py           # Data analysis tools with LLM-driven column selection
│   ├── portal_client.py   # data.gov.in API client
│   └── schemas.py         # Pydantic models
├── tests/                 # Test suite
├── mcp_server.py          # Clean MCP server using official SDK
├── environment.yml        # Dependencies
└── README.md             # This file
```

## 🛠️ Usage

Keep the FastAPI server running while using Claude Desktop. The MCP server automatically connects to the FastAPI backend and provides Claude with access to all data analysis tools.

## 📖 Examples

Check the `examples/` directory for sample prompts that demonstrate multi-tool workflows and intelligent analysis patterns.

## 🧪 Testing

```bash
# Run tests
pytest

# Test FastAPI directly
curl -X POST http://localhost:8000/ -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```
