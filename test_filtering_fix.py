#!/usr/bin/env python3
"""
Test script to verify the improved client-side filtering logic.
"""

import sys
from pathlib import Path

# Add the parent directory to path so we can import mcp_server
sys.path.insert(0, str(Path(__file__).parent))

from mcp_server import filter_dataset_records


def test_arrival_date_filtering():
    """Test the specific Arrival_Date filtering issue reported by Gemini."""

    # Mock data based on the real dataset structure
    test_data = {
        "records": [
            {
                "state": "Andhra Pradesh",
                "district": "Anantapur",
                "market": "Anantapur",
                "commodity": "Mousambi(Sweet Lime)",
                "variety": "Mousambi",
                "grade": "Medium",
                "arrival_date": "16/07/2025",  # Note: lowercase with underscore
                "min_price": "300",
                "max_price": "1500",
                "modal_price": "1100",
            },
            {
                "state": "Andhra Pradesh",
                "district": "Chittor",
                "market": "Chittoor",
                "commodity": "Gur(Jaggery)",
                "variety": "NO 2",
                "grade": "FAQ",
                "arrival_date": "16/07/2025",
                "min_price": "3400",
                "max_price": "3500",
                "modal_price": "3500",
            },
            {
                "state": "Andhra Pradesh",
                "district": "Chittor",
                "market": "Chittoor",
                "commodity": "Mango",
                "variety": "Totapuri",
                "grade": "Large",
                "arrival_date": "15/07/2025",  # Different date
                "min_price": "400",
                "max_price": "600",
                "modal_price": "500",
            },
        ],
        "total": 3,
        "field": [
            {"name": "State", "id": "state", "type": "keyword"},
            {"name": "Arrival_Date", "id": "arrival_date", "type": "date"},
        ],
    }

    print("Testing client-side filtering improvements...")

    # Test 1: Filter using display name "Arrival_Date" (should work now)
    print("\n1. Testing filter with display name 'Arrival_Date':")
    filters = {"Arrival_Date": "16/07/2025"}
    result = filter_dataset_records(test_data, filters)
    print(f"   Filter: {filters}")
    print(f"   Records found: {len(result['records'])}")
    print(f"   Field mapping: {result.get('field_mapping_used', {})}")
    assert len(result["records"]) == 2, f"Expected 2 records, got {len(result['records'])}"

    # Test 2: Filter using field ID "arrival_date" (should also work)
    print("\n2. Testing filter with field ID 'arrival_date':")
    filters = {"arrival_date": "16/07/2025"}
    result = filter_dataset_records(test_data, filters)
    print(f"   Filter: {filters}")
    print(f"   Records found: {len(result['records'])}")
    print(f"   Field mapping: {result.get('field_mapping_used', {})}")
    assert len(result["records"]) == 2, f"Expected 2 records, got {len(result['records'])}"

    # Test 3: Case-insensitive filtering
    print("\n3. Testing case-insensitive filtering:")
    filters = {"ARRIVAL_DATE": "16/07/2025"}
    result = filter_dataset_records(test_data, filters)
    print(f"   Filter: {filters}")
    print(f"   Records found: {len(result['records'])}")
    print(f"   Field mapping: {result.get('field_mapping_used', {})}")
    assert len(result["records"]) == 2, f"Expected 2 records, got {len(result['records'])}"

    # Test 4: Multiple filters (like the working case)
    print("\n4. Testing multiple filters (commodity + state):")
    filters = {"commodity": "Gur(Jaggery)", "state": "Andhra Pradesh"}
    result = filter_dataset_records(test_data, filters)
    print(f"   Filter: {filters}")
    print(f"   Records found: {len(result['records'])}")
    print(f"   Field mapping: {result.get('field_mapping_used', {})}")
    assert len(result["records"]) == 1, f"Expected 1 record, got {len(result['records'])}"

    # Test 5: Combined date and other filters
    print("\n5. Testing combined date + commodity filter:")
    filters = {"Arrival_Date": "16/07/2025", "commodity": "Mousambi(Sweet Lime)"}
    result = filter_dataset_records(test_data, filters)
    print(f"   Filter: {filters}")
    print(f"   Records found: {len(result['records'])}")
    print(f"   Field mapping: {result.get('field_mapping_used', {})}")
    assert len(result["records"]) == 1, f"Expected 1 record, got {len(result['records'])}"

    print("\n✅ All tests passed! Client-side filtering now handles field name variations correctly.")


if __name__ == "__main__":
    test_arrival_date_filtering()
