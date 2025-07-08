import pytest
from httpx import AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_tools_list():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        r = await ac.post("/", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "tools" in data["result"]
        assert any(t["name"] == "search_datasets" for t in data["result"]["tools"])


@pytest.mark.asyncio
async def test_semantic_column_selection():
    """Test that the new LLM-driven semantic column selection provides guidance instead of automatic matching."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Test conceptual filter that should trigger guidance, not automatic filtering
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "analyze_loaded_dataset",
                "arguments": {
                    "resource_id": "test-resource-id",
                    "analysis_query": "Test semantic analysis",
                    "filters": {"equals": {"state": "TestState"}},  # Conceptual filter, should trigger guidance
                },
            },
        }
        r = await ac.post("/", json=payload)
        assert r.status_code == 200
        data = r.json()

        # Should return guidance about column mapping, not automatic filter results
        content_text = data["result"]["content"][0]["text"]
        assert "not loaded" in content_text or "Available columns" in content_text

        # Should provide LLM guidance for semantic mapping
        if "Available columns" in content_text:
            assert "To apply these filters semantically" in content_text
            assert "Map each filter key to the most appropriate actual column name" in content_text


@pytest.mark.asyncio
async def test_tool_descriptions_updated():
    """Test that tool descriptions reflect the new LLM-driven approach."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        r = await ac.post("/", json=payload)
        assert r.status_code == 200
        data = r.json()

        tools = {tool["name"]: tool for tool in data["result"]["tools"]}

        # Check analyze_loaded_dataset description
        analyze_desc = tools["analyze_loaded_dataset"]["description"]
        assert "LLM-driven semantic column selection" in analyze_desc
        assert "intelligent mapping" in analyze_desc

        # Check download_dataset description
        download_desc = tools["download_dataset"]["description"]
        assert "LLM-driven semantic column guidance" in download_desc
        assert "column analysis" in download_desc

        # Check load_dataset_for_analysis description
        load_desc = tools["load_dataset_for_analysis"]["description"]
        assert "LLM-driven analysis" in load_desc
        assert "semantic column selection" in load_desc
