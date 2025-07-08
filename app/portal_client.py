import httpx
from typing import Dict, Any


class DataGovClient:
    """Thin async wrapper around data.gov.in endpoints."""

    BASE_URL = "https://api.data.gov.in"

    def __init__(self, api_key: str, timeout: int = 10):
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout)

    async def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for datasets. Try multiple possible endpoints."""
        params = {
            "q": query,
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
        }

        # Try the standard search endpoint first
        url = f"{self.BASE_URL}/catalog/1.0/search"
        try:
            r = await self._client.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 405:
                # Method not allowed - API might have changed
                # Return a helpful response indicating the issue
                return {
                    "datasets": [
                        {
                            "title": f"[API ISSUE] Search for '{query}' - data.gov.in search endpoint currently unavailable (405 Method Not Allowed)",
                            "resource_id": "api-endpoint-unavailable",
                            "description": "The data.gov.in search API endpoint appears to be down or changed. Try using a known resource ID with the download_dataset tool instead.",
                        }
                    ]
                }
            raise

    async def download_resource(self, resource_id: str, limit: int = 100) -> Dict[str, Any]:
        params = {
            "resource_id": resource_id,
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
        }
        url = f"{self.BASE_URL}/resource/{resource_id}"
        r = await self._client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    async def close(self):
        await self._client.aclose()
