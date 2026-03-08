"""
Centralised configuration loader for the SOC pipeline.

Reads ``conf/config.yaml`` and ``conf/models.yaml``, merges them into a
single namespace, and exposes helpers that the rest of the codebase can use
without touching YAML directly.

Usage
-----
    from soc_2602.utils.config import load_config, get_region_profile, get_model_preset

    cfg = load_config()                          # reads from default paths
    cfg = load_config("conf/config.yaml",        # or supply explicit paths
                      "conf/models.yaml")

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
_DEFAULT_MODELS_PATH = "conf/models.yaml"


# ── Public helpers ───────────────────────────────────────────────────────────


def load_config(
    config_path: Optional[str] = None,
    models_path: Optional[str] = None,
    *,
    project_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Load and merge the pipeline config and model-presets files.

    Parameters
    ----------
    config_path : str | None
        Path to the main ``config.yaml``.  Falls back to
        ``<project_root>/conf/config.yaml``.
    models_path : str | None
        Path to ``models.yaml``.  Falls back to
        ``<project_root>/conf/models.yaml``.
    project_root : str | Path | None
        If supplied, relative paths are resolved against this directory.
        Otherwise the current working directory is used.

    Returns
    -------
    dict
        Merged configuration dictionary with top-level keys from both files.
        The models file is nested under the ``"models_file"`` key to avoid
        collisions with the ``"models"`` section in the main config.
    """
    root = Path(project_root) if project_root else Path.cwd()

    cfg_path = Path(config_path) if config_path else root / _DEFAULT_CONFIG_PATH
    mdl_path = Path(models_path) if models_path else root / _DEFAULT_MODELS_PATH

    cfg: Dict[str, Any] = _read_yaml(cfg_path)
    mdl: Dict[str, Any] = _read_yaml(mdl_path)

    # Merge models file under a dedicated key so we never clobber config.models
    cfg["models_file"] = mdl
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
        Merged configuration as returned by :func:`load_config`.
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

    Searches in both ``config.yaml → models.presets`` and
    ``models.yaml → presets``.

    Parameters
    ----------
    cfg : dict
        Merged configuration as returned by :func:`load_config`.
    preset_name : str
        Preset key (e.g. ``"kimi_k2"``).

    Returns
    -------
    dict
        A dictionary with at least ``model_id`` and ``provider``.

    Raises
    ------
    ValueError
        If the preset is not found in either file.
    """
    # Check models.yaml first, then config.yaml fallback
    models_file_presets = (cfg.get("models_file") or {}).get("presets", {})
    config_presets = (cfg.get("models") or {}).get("presets", {})

    preset = models_file_presets.get(preset_name) or config_presets.get(preset_name)

    if preset is None:
        all_names = set(models_file_presets.keys()) | set(config_presets.keys())
        available = ", ".join(sorted(all_names)) if all_names else "(none)"
        raise ValueError(
            f"Unknown model preset '{preset_name}'. Available: {available}"
        )

    return copy.deepcopy(preset)


def get_provider_config(
    cfg: Dict[str, Any],
    provider_name: str,
) -> Dict[str, Any]:
    """Return endpoint information for *provider_name*.

    Searches both config files; ``models.yaml → providers`` takes priority.

    Parameters
    ----------
    cfg : dict
        Merged configuration.
    provider_name : str
        Provider key (e.g. ``"local"``, ``"huggingface"``).

    Returns
    -------
    dict
        Provider config with ``base_url`` and/or ``api_key_env``.
    """
    mdl_providers = (cfg.get("models_file") or {}).get("providers", {})
    cfg_providers = cfg.get("providers") or {}

    provider = mdl_providers.get(provider_name) or cfg_providers.get(provider_name)

    if provider is None:
        all_names = set(mdl_providers.keys()) | set(cfg_providers.keys())
        available = ", ".join(sorted(all_names)) if all_names else "(none)"
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
        Merged configuration.
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
