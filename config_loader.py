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
            raise FileNotFoundError(f"Config file not found at {self.config_path}. Please ensure config.json exists.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}. Please fix the config.json file.")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            section: Configuration section (e.g., 'semantic_search')
            key: Configuration key (e.g., 'default_search_limit')
            default: Default value if key not found (only used for backwards compatibility)

        Returns:
            Configuration value

        Raises:
            KeyError: If section or key not found and no default provided
        """
        if section not in self._config:
            if default is not None:
                return default
            raise KeyError(f"Configuration section '{section}' not found")

        if key not in self._config[section]:
            if default is not None:
                return default
            raise KeyError(f"Configuration key '{key}' not found in section '{section}'")

        return self._config[section][key]

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
        return self.get("semantic_search", "default_search_limit")

    @property
    def max_search_limit(self) -> int:
        """Maximum allowed search limit."""
        return self.get("semantic_search", "max_search_limit")

    @property
    def relevance_threshold(self) -> float:
        """Threshold for considering datasets relevant."""
        return self.get("semantic_search", "relevance_threshold")

    @property
    def download_limit(self) -> int:
        """Default limit for dataset downloads."""
        return self.get("data_api", "default_download_limit")

    @property
    def inspect_sample_size(self) -> int:
        """Default sample size for dataset inspection."""
        return self.get("data_api", "default_inspect_sample_size")

    @property
    def server_name(self) -> str:
        """MCP server name."""
        return self.get("mcp_server", "server_name")


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
