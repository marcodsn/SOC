"""
Centralised configuration loader for the SOC pipeline.

Reads ``conf/config.yaml`` and exposes helpers that the rest of the
codebase can use without touching YAML directly.

Usage
-----
    from soc_2602.utils.config import load_config, get_region_profile, get_model_preset

    cfg = load_config()                          # reads from default path
    cfg = load_config("conf/config.yaml")        # or supply an explicit path

    profile  = get_region_profile(cfg, "west")   # list of region codes
    preset   = get_model_preset(cfg, "kimi_k2")  # dict with model_id, provider, …
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ── Default file locations (relative to the project root) ────────────────────

_DEFAULT_CONFIG_PATH = "conf/config.yaml"


# ── Public helpers ───────────────────────────────────────────────────────────


def load_config(
    config_path: Optional[str] = None,
    *,
    project_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Load the pipeline configuration from ``config.yaml``.

    Parameters
    ----------
    config_path : str | None
        Path to the main ``config.yaml``.  Falls back to
        ``<project_root>/conf/config.yaml``.
    project_root : str | Path | None
        If supplied, relative paths are resolved against this directory.
        Otherwise the current working directory is used.

    Returns
    -------
    dict
        Configuration dictionary with all top-level keys from
        ``config.yaml``, including ``models`` and ``providers``.
    """
    root = Path(project_root) if project_root else Path.cwd()

    cfg_path = Path(config_path) if config_path else root / _DEFAULT_CONFIG_PATH

    cfg: Dict[str, Any] = _read_yaml(cfg_path)
    return cfg


def get_region_profile(
    cfg: Dict[str, Any],
    profile_name: Optional[str] = None,
) -> Optional[List[str]]:
    """Return the list of region codes for *profile_name*, or ``None`` for global.

    ``None`` signals "use all regions with their natural weights".

    Parameters
    ----------
    cfg : dict
        Configuration as returned by :func:`load_config`.
    profile_name : str | None
        Profile name (e.g. ``"west"``, ``"east_asia"``).  ``None`` or
        ``"global"`` both map to all regions.

    Returns
    -------
    list[str] | None
        Region codes, or ``None`` when every region should be used.
    """
    if profile_name is None:
        profile_name = cfg.get("default_region_profile", "global")

    profiles = cfg.get("region_profiles", {})
    profile = profiles.get(profile_name)

    if profile is None:
        available = ", ".join(sorted(profiles.keys())) if profiles else "(none)"
        raise ValueError(
            f"Unknown region profile '{profile_name}'. Available: {available}"
        )

    return profile.get("regions")  # None means global


def get_model_preset(
    cfg: Dict[str, Any],
    preset_name: str,
) -> Dict[str, Any]:
    """Look up a named model preset and return a copy of its settings.

    Searches in ``config.yaml → models.presets``.

    Parameters
    ----------
    cfg : dict
        Configuration as returned by :func:`load_config`.
    preset_name : str
        Preset key (e.g. ``"kimi_k2"``).

    Returns
    -------
    dict
        A dictionary with at least ``model_id`` and ``provider``.

    Raises
    ------
    ValueError
        If the preset is not found.
    """
    config_presets = (cfg.get("models") or {}).get("presets", {})

    preset = config_presets.get(preset_name)

    if preset is None:
        available = (
            ", ".join(sorted(config_presets.keys())) if config_presets else "(none)"
        )
        raise ValueError(
            f"Unknown model preset '{preset_name}'. Available: {available}"
        )

    return copy.deepcopy(preset)


def get_provider_config(
    cfg: Dict[str, Any],
    provider_name: str,
) -> Dict[str, Any]:
    """Return endpoint information for *provider_name*.

    Parameters
    ----------
    cfg : dict
        Configuration as returned by :func:`load_config`.
    provider_name : str
        Provider key (e.g. ``"local"``, ``"huggingface"``).

    Returns
    -------
    dict
        Provider config with ``base_url`` and/or ``api_key_env``.
    """
    cfg_providers = cfg.get("providers") or {}

    provider = cfg_providers.get(provider_name)

    if provider is None:
        available = (
            ", ".join(sorted(cfg_providers.keys())) if cfg_providers else "(none)"
        )
        raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")

    return copy.deepcopy(provider)


def get_language_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the language block with sensible defaults filled in."""
    defaults = {
        "lingua_franca": "English",
        "native_flavor": True,
        "max_native_words_per_message": 2,
    }
    lang = cfg.get("language", {})
    return {**defaults, **lang}


def get_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """Return a config section by name with an empty-dict fallback.

    Convenience wrapper so callers don't have to ``cfg.get(name, {})``
    everywhere.

    Parameters
    ----------
    cfg : dict
        Configuration as returned by :func:`load_config`.
    section : str
        Top-level key, e.g. ``"persona"``, ``"experience"``, ``"conversation"``.
    """
    return cfg.get(section, {})


# ── Internal ─────────────────────────────────────────────────────────────────


def _read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file and return its contents as a dict.

    Returns an empty dict when the file is missing or empty rather than
    raising, so that the pipeline can still run with partial configuration.
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}
