#!/usr/bin/env python3
"""
Configuration loader for the MCP server and semantic search.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the MCP data.gov.in server."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from JSON file.

        Args:
            config_path: Path to config file. Uses default if None.
        """
        if config_path is None:
            self.config_path = Path(__file__).parent / "config.json"
        else:
            self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Config file not found at {self.config_path}, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"⚠️ Invalid JSON in config file: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file is missing or invalid."""
        return {
            "semantic_search": {
                "model_name": "all-MiniLM-L6-v2",
                "title_weight": 3,
                "min_similarity_score": 0.1,
                "default_search_limit": 20,
                "max_search_limit": 50,
                "cache_embeddings": True,
                "show_progress": True,
                "relevance_threshold": 0.25,
            },
            "data_api": {
                "default_download_limit": 100,
                "max_download_limit": 1000,
                "default_inspect_sample_size": 5,
                "request_timeout": 30,
            },
            "mcp_server": {"server_name": "data-gov-in-mcp", "registry_last_updated": "2025-07-08"},
            "analysis": {
                "multi_dataset_threshold": 2,
                "high_relevance_threshold": 0.5,
                "low_relevance_warning_threshold": 0.3,
            },
        }

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            section: Configuration section (e.g., 'semantic_search')
            key: Configuration key (e.g., 'default_search_limit')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self._config.get(section, {}).get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        return self._config.get(section, {})

    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def save(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save config: {e}")

    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()

    # Convenience properties for commonly used values
    @property
    def semantic_search_limit(self) -> int:
        """Default limit for semantic search results."""
        return self.get("semantic_search", "default_search_limit", 20)

    @property
    def max_search_limit(self) -> int:
        """Maximum allowed search limit."""
        return self.get("semantic_search", "max_search_limit", 50)

    @property
    def relevance_threshold(self) -> float:
        """Threshold for considering datasets relevant."""
        return self.get("semantic_search", "relevance_threshold", 0.25)

    @property
    def download_limit(self) -> int:
        """Default limit for dataset downloads."""
        return self.get("data_api", "default_download_limit", 100)

    @property
    def inspect_sample_size(self) -> int:
        """Default sample size for dataset inspection."""
        return self.get("data_api", "default_inspect_sample_size", 5)

    @property
    def server_name(self) -> str:
        """MCP server name."""
        return self.get("mcp_server", "server_name", "data-gov-in-mcp")


# Global config instance
_config_instance = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def reload_config() -> None:
    """Reload the global configuration from file."""
    global _config_instance
    if _config_instance:
        _config_instance.reload()
