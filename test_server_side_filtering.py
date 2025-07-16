#!/usr/bin/env python3
"""
Test script to demonstrate server-side filtering implementation.
This script shows how the build_server_side_filters function correctly maps
user filter keys to the exact field IDs from field_exposed for API filtering.
"""

import json
import sys
from pathlib import Path

# Add the current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_server import build_server_side_filters


def test_with_real_api_structure():
    """Test with realistic field_exposed structure from Data.gov.in API"""

    print("🧪 Testing Server-Side Filtering Implementation")
    print("=" * 60)

    # Example field_exposed structure from a real Data.gov.in dataset
    field_exposed = [
        {"id": "state.keyword", "name": "State", "type": "string"},
        {"id": "district.keyword", "name": "District", "type": "string"},
        {"id": "year", "name": "Year", "type": "integer"},
        {"id": "scheme_name.keyword", "name": "Scheme Name", "type": "string"},
        {"id": "amount", "name": "Amount", "type": "float"},
        {"id": "beneficiaries", "name": "Number of Beneficiaries", "type": "integer"},
        {"id": "category.keyword", "name": "Category", "type": "string"},
    ]

    print("📋 Available Fields for Server-Side Filtering:")
    for field in field_exposed:
        print(f"  • {field['name']} (ID: {field['id']}, Type: {field['type']})")

    print("\n" + "-" * 60)

    # Test Case 1: Basic mapping
    print("🔍 Test Case 1: Basic Field Mapping")
    user_filters = {"state": "Maharashtra", "year": "2023", "district": "Mumbai"}

    server_filters, client_filters = build_server_side_filters(user_filters, field_exposed)

    print(f"User Filters: {json.dumps(user_filters, indent=2)}")
    print(f"Server-Side Filters: {json.dumps(server_filters, indent=2)}")
    print(f"Client-Side Filters: {json.dumps(client_filters, indent=2)}")

    # Verify correct mapping
    expected_server_filters = {
        "filters[state.keyword]": "Maharashtra",
        "filters[year]": "2023",
        "filters[district.keyword]": "Mumbai",
    }

    assert server_filters == expected_server_filters, f"Expected {expected_server_filters}, got {server_filters}"
    assert len(client_filters) == 0, f"Expected no client filters, got {client_filters}"
    print("✅ Correct mapping: user fields → field_exposed IDs")

    print("\n" + "-" * 60)

    # Test Case 2: Mixed mappable and unmappable fields
    print("🔍 Test Case 2: Mixed Mappable and Unmappable Fields")
    user_filters = {
        "state": "Gujarat",
        "scheme_name": "PM-KISAN",
        "custom_field": "some_value",  # This field doesn't exist in field_exposed
        "year": "2022",
    }

    server_filters, client_filters = build_server_side_filters(user_filters, field_exposed)

    print(f"User Filters: {json.dumps(user_filters, indent=2)}")
    print(f"Server-Side Filters: {json.dumps(server_filters, indent=2)}")
    print(f"Client-Side Filters: {json.dumps(client_filters, indent=2)}")

    # Verify server-side filters use exact field IDs
    assert "filters[state.keyword]" in server_filters
    assert "filters[scheme_name.keyword]" in server_filters
    assert "filters[year]" in server_filters
    assert server_filters["filters[state.keyword]"] == "Gujarat"
    assert server_filters["filters[scheme_name.keyword]"] == "PM-KISAN"
    assert server_filters["filters[year]"] == "2022"

    # Verify unmappable field falls back to client-side
    assert "custom_field" in client_filters
    assert client_filters["custom_field"] == "some_value"
    print("✅ Correct hybrid filtering: mappable → server-side, unmappable → client-side")

    print("\n" + "-" * 60)

    # Test Case 3: Case-insensitive matching
    print("🔍 Test Case 3: Case-Insensitive Field Matching")
    user_filters = {"STATE": "Karnataka", "District": "Bangalore", "YEAR": "2021"}

    server_filters, client_filters = build_server_side_filters(user_filters, field_exposed)

    print(f"User Filters: {json.dumps(user_filters, indent=2)}")
    print(f"Server-Side Filters: {json.dumps(server_filters, indent=2)}")
    print(f"Client-Side Filters: {json.dumps(client_filters, indent=2)}")

    # Should still map correctly despite case differences
    assert len(server_filters) == 3, f"Expected 3 server filters, got {len(server_filters)}"
    assert len(client_filters) == 0, f"Expected no client filters, got {client_filters}"
    print("✅ Case-insensitive matching works correctly")

    print("\n" + "-" * 60)

    # Test Case 4: Empty inputs
    print("🔍 Test Case 4: Edge Cases")

    # Empty filters
    server_filters, client_filters = build_server_side_filters({}, field_exposed)
    assert len(server_filters) == 0 and len(client_filters) == 0
    print("✅ Empty filters handled correctly")

    # Empty field_exposed
    user_filters = {"state": "Punjab"}
    server_filters, client_filters = build_server_side_filters(user_filters, [])
    assert len(server_filters) == 0 and len(client_filters) == 1
    print("✅ Empty field_exposed handled correctly")

    print("\n" + "=" * 60)
    print("🎉 All tests passed! Server-side filtering implementation is correct.")
    print("\n📝 Key Points:")
    print("  • Uses exact field IDs from field_exposed for server-side filtering")
    print("  • Falls back to client-side filtering for unmappable fields")
    print("  • Supports case-insensitive field name matching")
    print("  • Handles edge cases gracefully")
    print("  • Enables efficient data download with server-side filtering")


if __name__ == "__main__":
    test_with_real_api_structure()
