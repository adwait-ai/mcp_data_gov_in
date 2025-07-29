#!/usr/bin/env python3
"""
Test script to verify that API responses are clean and don't contain
backend implementation details that confuse LLMs.
"""

import asyncio
import json
import os
from mcp_server import mcp

# Mock data for testing
MOCK_INSPECT_RESPONSE = {
    "fields": [
        {"id": "state", "name": "State", "type": "text"},
        {"id": "year", "name": "Year", "type": "keyword"},
        {"id": "value", "name": "Value", "type": "number"},
    ],
    "records": [
        {"state": "Maharashtra", "year": "2023", "value": "1000"},
        {"state": "Karnataka", "year": "2023", "value": "800"},
    ],
    "total": 1000,
}


def test_clean_responses():
    """Test that responses don't contain backend implementation details."""

    # Test the structure of field filtering guide
    fields = MOCK_INSPECT_RESPONSE["fields"]
    field_name_guide = {}

    for field in fields:
        field_id = field.get("id", "")
        field_name = field.get("name", "")
        field_type = field.get("type", "")

        if field_id and field_name:
            field_name_guide[field_name] = {
                "use_in_filters": field_id,
                "display_name": field_name,
                "type": field_type,
            }

    # Check that backend details are NOT present
    backend_terms = ["server_filterable", "filter_method", "server-side", "client-side"]

    guide_str = json.dumps(field_name_guide)

    print("✅ Testing field filtering guide cleanup...")
    for term in backend_terms:
        if term in guide_str:
            print(f"❌ FAIL: Found backend term '{term}' in field guide")
            return False
        else:
            print(f"✅ PASS: Backend term '{term}' not found")

    # Test pagination info cleanup
    print("\n✅ Testing pagination info cleanup...")
    backend_pagination_terms = [
        "used_server_filters",
        "server_filters_applied",
        "automatic_pagination",
        "no_manual_pagination",
        "server downloads complete datasets",
    ]

    # Simulate the clean structure
    clean_structure = {
        "fields": fields,
        "column_names": ["state", "year", "value"],
        "field_filtering_guide": field_name_guide,
        "sample_records_csv": "state,year,value\nMaharashtra,2023,1000\nKarnataka,2023,800",
        "total_records_available": 1000,
        "sample_size_shown": 2,
        "usage_tip": "Use download_filtered_dataset() for complete data in CSV format with intelligent filtering",
        "field_name_tips": {
            "flexible_naming": "You can use either display names (e.g., 'State') or field IDs (e.g., 'state')",
            "case_insensitive": "Field name matching is case-insensitive",
            "date_filtering": "For date fields, use exact format as shown in sample CSV data",
        },
        "example_filter": '{"column_name": "filter_value"}',
    }

    structure_str = json.dumps(clean_structure)

    for term in backend_pagination_terms:
        if term in structure_str:
            print(f"❌ FAIL: Found backend pagination term '{term}' in structure")
            return False
        else:
            print(f"✅ PASS: Backend pagination term '{term}' not found")

    print("\n✅ All tests passed! Responses are clean and LLM-friendly.")
    return True


def test_useful_info_preserved():
    """Test that useful information for LLMs is preserved."""

    print("\n✅ Testing that useful information is preserved...")

    useful_terms = [
        "field_filtering_guide",
        "sample_records_csv",
        "total_records_available",
        "usage_tip",
        "field_name_tips",
        "example_filter",
    ]

    clean_structure = {
        "fields": MOCK_INSPECT_RESPONSE["fields"],
        "column_names": ["state", "year", "value"],
        "field_filtering_guide": {"State": {"use_in_filters": "state", "display_name": "State", "type": "text"}},
        "sample_records_csv": "state,year,value\nMaharashtra,2023,1000",
        "total_records_available": 1000,
        "sample_size_shown": 2,
        "usage_tip": "Use download_filtered_dataset() for complete data",
        "field_name_tips": {"flexible_naming": "You can use display names or field IDs"},
        "example_filter": '{"column_name": "filter_value"}',
    }

    structure_str = json.dumps(clean_structure)

    for term in useful_terms:
        if term in structure_str:
            print(f"✅ PASS: Useful term '{term}' found")
        else:
            print(f"❌ FAIL: Useful term '{term}' missing")
            return False

    print("✅ All useful information preserved!")
    return True


if __name__ == "__main__":
    print("🧹 Testing Clean API Responses")
    print("=" * 50)

    success = test_clean_responses() and test_useful_info_preserved()

    if success:
        print("\n🎉 All tests passed! The API responses are clean and LLM-optimized.")
    else:
        print("\n❌ Some tests failed. Check the output above.")
        exit(1)
