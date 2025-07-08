from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any


class SearchDatasetsArgs(BaseModel):
    """Input schema for search_datasets."""

    query: str = Field(..., min_length=3, description="Keyword search query")
    limit: int = Field(5, ge=1, le=20, description="Number of results (max 20)")

    model_config = ConfigDict(extra="forbid")


class DatasetSummary(BaseModel):
    id: str
    title: str
    org: Optional[str] = None
    description: Optional[str] = None


class DownloadDatasetArgs(BaseModel):
    """Input schema for download_dataset."""

    resource_id: str = Field(..., description="Resource ID of the dataset")
    limit: int = Field(100, ge=1, le=10000, description="Row limit")

    model_config = ConfigDict(extra="forbid")
