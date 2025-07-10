import pytest
import json
from pathlib import Path
import sys

# Add the parent directory to path so we can import mcp_server
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server import search_static_registry, filter_dataset_records, load_dataset_registry, DATASET_REGISTRY


def test_search_static_registry():
    """Test the static registry search functionality."""
    # Test with a simple query
    results = search_static_registry(DATASET_REGISTRY, "health", limit=5)
    assert isinstance(results, list)
    assert len(results) <= 5

    # Test with empty query
    results = search_static_registry(DATASET_REGISTRY, "", limit=5)
    assert isinstance(results, list)

    # Test with no results
    results = search_static_registry(DATASET_REGISTRY, "xyzzzznonexistent", limit=5)
    assert results == []


def test_filter_dataset_records():
    """Test the dataset record filtering functionality."""
    # Test data
    test_data = {
        "records": [
            {"state": "Maharashtra", "year": "2023", "value": 100},
            {"state": "Gujarat", "year": "2023", "value": 200},
            {"state": "Maharashtra", "year": "2022", "value": 150},
            {"state": "Karnataka", "year": "2023", "value": 180},
        ],
        "total": 4,
        "field": [{"id": "state"}, {"id": "year"}, {"id": "value"}],
    }

    # Test filtering by state
    filtered = filter_dataset_records(test_data, {"state": "Maharashtra"})
    assert len(filtered["records"]) == 2
    assert filtered["total"] == 2
    assert all(record["state"] == "Maharashtra" for record in filtered["records"])

    # Test filtering by multiple criteria
    filtered = filter_dataset_records(test_data, {"state": "Maharashtra", "year": "2023"})
    assert len(filtered["records"]) == 1
    assert filtered["total"] == 1
    assert filtered["records"][0]["state"] == "Maharashtra"
    assert filtered["records"][0]["year"] == "2023"

    # Test no filters
    filtered = filter_dataset_records(test_data, None)
    assert len(filtered["records"]) == 4
    assert filtered["total"] == 4

    # Test empty data
    empty_data = {"records": [], "total": 0}
    filtered = filter_dataset_records(empty_data, {"state": "Maharashtra"})
    assert len(filtered["records"]) == 0
    assert filtered["total"] == 0


def test_load_dataset_registry():
    """Test that the dataset registry loads correctly."""
    registry = load_dataset_registry()
    assert isinstance(registry, list)
    # Should have some datasets (assuming the file exists)
    if registry:
        assert isinstance(registry[0], dict)
        assert "resource_id" in registry[0]
        assert "title" in registry[0]


@pytest.mark.asyncio
async def test_registry_is_loaded():
    """Test that the global registry is loaded at startup."""
    assert isinstance(DATASET_REGISTRY, list)
    # The registry should be loaded from the JSON file
    # This tests that the startup process works correctly


def test_search_registry_case_insensitive():
    """Test that search is case insensitive."""
    results_upper = search_static_registry(DATASET_REGISTRY, "HEALTH", limit=5)
    results_lower = search_static_registry(DATASET_REGISTRY, "health", limit=5)
    results_mixed = search_static_registry(DATASET_REGISTRY, "Health", limit=5)

    # All should return the same results
    assert len(results_upper) == len(results_lower)
    assert len(results_lower) == len(results_mixed)


def test_filter_records_case_insensitive():
    """Test that filtering is case insensitive."""
    test_data = {
        "records": [
            {"state": "Maharashtra", "year": "2023"},
            {"state": "GUJARAT", "year": "2023"},
            {"state": "karnataka", "year": "2023"},
        ],
        "total": 3,
    }

    # Test case insensitive filtering
    filtered = filter_dataset_records(test_data, {"state": "maharashtra"})
    assert len(filtered["records"]) == 1
    assert filtered["records"][0]["state"] == "Maharashtra"

    filtered = filter_dataset_records(test_data, {"state": "gujarat"})
    assert len(filtered["records"]) == 1
    assert filtered["records"][0]["state"] == "GUJARAT"
