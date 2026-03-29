"""
Utility modules for the SOC pipeline.

This subpackage provides:

- :class:`StatsEngine` — demographic sampling engine with support for
  region profiles, population-weighted name/subregion selection, and
  age distribution sampling.

- :mod:`config` — centralised YAML configuration loader that reads
  ``conf/config.yaml`` and exposes helpers for region profiles, model
  presets, language settings, and provider endpoints.
"""

from soc_2602.utils.config import (
    get_language_settings,
    get_model_preset,
    get_provider_config,
    get_region_profile,
    get_section,
    load_config,
)
from soc_2602.utils.sampling import StatsEngine

__all__ = [
    "StatsEngine",
    "load_config",
    "get_region_profile",
    "get_model_preset",
    "get_provider_config",
    "get_language_settings",
    "get_section",
]
