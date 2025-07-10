#!/usr/bin/env python3
"""
Test script for semantic search functionality.
"""

import sys
from pathlib import Path
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_semantic_search():
    """Test semantic search functionality."""
    try:
        from semantic_search import DatasetSemanticSearch, SemanticSearchConfig
        
        print("🧪 Testing semantic search functionality...")
        
        # Load a sample of the registry for testing
        registry_path = Path(__file__).parent / "data" / "data_gov_in_api_registry.json"
        with open(registry_path, "r", encoding="utf-8") as f:
            full_registry = json.load(f)
        
        # Use first 100 datasets for quick testing
        sample_registry = full_registry[:100]
        print(f"📊 Testing with {len(sample_registry)} datasets")
        
        # Check if model is available locally
        models_dir = Path(__file__).parent / "models"
        if not models_dir.exists() or not list(models_dir.rglob("*all-MiniLM-L6-v2*")):
            print("📥 Model not found locally, downloading...")
            from download_model import download_model
            if not download_model():
                print("❌ Failed to download model")
                return False
        
        # Initialize search engine with local model
        config = SemanticSearchConfig(show_progress=True)
        search_engine = DatasetSemanticSearch(config)
        search_engine.preload_model()
        search_engine.build_embeddings(sample_registry)
        
        # Test queries
        test_queries = [
            "health HIV AIDS",
            "oil petroleum crude",
            "inflation prices economy",
            "taxes revenue government",
            "guarantees credit loans"
        ]
        
        print("\n🔍 Testing search queries:")
        for query in test_queries:
            print(f"\n  Query: '{query}'")
            results = search_engine.search(query, limit=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    score = result.get('similarity_score', 0)
                    title = result.get('title', 'Unknown')
                    print(f"    {i}. [{score:.3f}] {title[:80]}...")
            else:
                print("    No results found")
        
        print("\n✅ Semantic search test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install required packages with: micromamba install -c conda-forge sentence-transformers faiss-cpu numpy")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_search():
    """Test fallback text search."""
    try:
        from mcp_server import search_static_registry, load_dataset_registry
        
        print("\n🔄 Testing fallback text search...")
        
        registry = load_dataset_registry()
        if not registry:
            print("❌ No registry loaded")
            return False
        
        # Test search
        results = search_static_registry(registry, "health", limit=3)
        print(f"Found {len(results)} results for 'health'")
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Unknown')
            print(f"  {i}. {title[:80]}...")
        
        print("✅ Fallback search test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting semantic search tests...\n")
    
    # Test semantic search
    semantic_success = test_semantic_search()
    
    # Test fallback
    fallback_success = test_fallback_search()
    
    print(f"\n📊 Test Summary:")
    print(f"  Semantic Search: {'✅ PASS' if semantic_success else '❌ FAIL'}")
    print(f"  Fallback Search: {'✅ PASS' if fallback_success else '❌ FAIL'}")
    
    if semantic_success and fallback_success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
