import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict
from dotenv import load_dotenv
from .portal_client import DataGovClient
from .tools import TOOLS, TOOL_MAP

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Data.Gov.in MCP Server")

client = DataGovClient(api_key=os.getenv("DATA_GOV_API_KEY", "DEMO_KEY"))


@app.post("/", response_class=JSONResponse)
async def json_rpc_gateway(payload: Dict[str, Any]):
    method = payload.get("method")
    call_id = payload.get("id")
    if not method:
        raise HTTPException(status_code=400, detail="Missing method")

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": call_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = payload.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        if tool_name not in TOOL_MAP:
            return {
                "jsonrpc": "2.0",
                "id": call_id,
                "result": {"isError": True, "content": [{"type": "text", "text": f"Unknown tool {tool_name}"}]},
            }
        try:
            content = await TOOL_MAP[tool_name](client, **args)
            return {"jsonrpc": "2.0", "id": call_id, "result": {"content": content}}
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": call_id,
                "result": {"isError": True, "content": [{"type": "text", "text": f"Error: {exc}"}]},
            }

    raise HTTPException(status_code=400, detail="Unsupported method")
