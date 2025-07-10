import pytest
import json
from pathlib import Path
import sys

# Add the parent directory to path so we can import mcp_server
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server import search_static_registry, filter_dataset_records, load_dataset_registry, DATASET_REGISTRY

# Try to import semantic search, but don't fail if packages aren't installed
try:
    from semantic_search import DatasetSemanticSearch, initialize_semantic_search
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False


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


@pytest.mark.skipif(not SEMANTIC_SEARCH_AVAILABLE, reason="Semantic search packages not installed")
def test_semantic_search_initialization():
    """Test semantic search initialization."""
    if not DATASET_REGISTRY:
        pytest.skip("No dataset registry loaded")
    
    # Test with a small subset for faster testing
    sample_registry = DATASET_REGISTRY[:10] if len(DATASET_REGISTRY) > 10 else DATASET_REGISTRY
    
    search_engine = DatasetSemanticSearch()
    search_engine.preload_model()
    search_engine.build_embeddings(sample_registry)
    
    assert search_engine.is_ready()
    assert len(search_engine.dataset_metadata) == len(sample_registry)


@pytest.mark.skipif(not SEMANTIC_SEARCH_AVAILABLE, reason="Semantic search packages not installed")
def test_semantic_search_functionality():
    """Test semantic search functionality."""
    if not DATASET_REGISTRY:
        pytest.skip("No dataset registry loaded")
    
    # Test with a small subset for faster testing
    sample_registry = DATASET_REGISTRY[:10] if len(DATASET_REGISTRY) > 10 else DATASET_REGISTRY
    
    search_engine = DatasetSemanticSearch()
    search_engine.preload_model()
    search_engine.build_embeddings(sample_registry)
    
    # Test search
    results = search_engine.search("health", limit=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    
    # Check result structure
    if results:
        result = results[0]
        assert 'resource_id' in result
        assert 'title' in result
        assert 'similarity_score' in result
        assert 'metadata' in result
        assert isinstance(result['similarity_score'], (int, float))


def test_searchable_text_creation():
    """Test the creation of searchable text for semantic search."""
    if not SEMANTIC_SEARCH_AVAILABLE:
        pytest.skip("Semantic search packages not installed")
    
    search_engine = DatasetSemanticSearch()
    
    test_dataset = {
        "title": "Health Data Survey",
        "ministry": "Ministry of Health",
        "sector": "Health",
        "resource_id": "test-123"
    }
    
    searchable_text = search_engine._create_searchable_text(test_dataset)
    
    # Title should appear 3 times for higher weight
    assert searchable_text.count("Health Data Survey") == 3
    assert "Ministry of Health" in searchable_text
    assert "Health" in searchable_text
