from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


class StatsEngine:
    """Demographic sampling engine for the SOC pipeline.

    Supports optional region profiles that restrict sampling to a subset
    of regions (e.g. "west", "east_asia") while re-normalizing population
    weights within that subset.  When no profile is active, all regions
    from ``regions.yaml`` are used with their natural population weights.

    Parameters
    ----------
    stats_dir : Path
        Root directory containing ``demographics/regions.yaml`` and ``names/*.csv``.
    allowed_regions : list[str] | None
        If supplied, only these region codes are eligible for sampling.
        Weights are automatically re-normalized.  ``None`` means global.
    """

    def __init__(
        self,
        stats_dir: Path,
        allowed_regions: Optional[List[str]] = None,
    ):
        self.stats_dir = stats_dir
        self.regions_config = self._load_yaml("demographics/regions.yaml")
        self.name_cache: Dict[str, pd.DataFrame] = {}

        # Build the effective region pool
        all_regions: Dict[str, float] = self.regions_config.get("regions", {})

        if allowed_regions is not None:
            # Keep only the requested regions; warn about unknown codes
            filtered: Dict[str, float] = {}
            for code in allowed_regions:
                if code in all_regions:
                    filtered[code] = all_regions[code]
                else:
                    print(
                        f"⚠ StatsEngine: region code '{code}' not found in regions.yaml, skipping"
                    )
            if not filtered:
                raise ValueError(
                    "No valid regions remain after filtering. "
                    f"Requested: {allowed_regions}"
                )
            self._region_codes: List[str] = list(filtered.keys())
            weights = np.array(list(filtered.values()), dtype=np.float64)
        else:
            self._region_codes = list(all_regions.keys())
            weights = np.array(list(all_regions.values()), dtype=np.float64)

        # Normalize to a proper probability distribution
        weights /= weights.sum()
        self._region_weights: np.ndarray = weights

    # ── YAML loading ─────────────────────────────────────────────────────────

    def _load_yaml(self, relative_path: str) -> dict:
        with open(self.stats_dir / relative_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ── Region sampling ──────────────────────────────────────────────────────

    @property
    def available_regions(self) -> List[str]:
        """Return the list of region codes currently eligible for sampling."""
        return list(self._region_codes)

    def get_random_region(self) -> str:
        """Select a region code weighted by (possibly filtered) population."""
        return np.random.choice(self._region_codes, p=self._region_weights)

    # ── Subregion sampling ───────────────────────────────────────────────────

    def gen_random_subregion(self, region_code: Optional[str] = None) -> str:
        """Given a region code, randomly select a subregion weighted by population.

        Falls back to ``get_random_region()`` when *region_code* is ``None``.
        Returns the region code itself if no subregion data is available.
        """
        if region_code is None:
            region_code = self.get_random_region()

        subregions_all = self.regions_config.get("subregions", {})
        subregion_data = subregions_all.get(region_code)

        if not subregion_data:
            return region_code  # graceful fallback

        subregion_names = list(subregion_data.keys())
        weights = np.array(list(subregion_data.values()), dtype=np.float64)
        weights /= weights.sum()

        return np.random.choice(subregion_names, p=weights)

    # ── Name sampling ────────────────────────────────────────────────────────

    def get_random_name(
        self,
        region_code: str,
        gender: Optional[str] = None,
    ) -> str:
        """Pick a name for *region_code*, weighted by real-world frequency.

        Lazy-loads the CSV on first access for each region.

        Parameters
        ----------
        region_code : str
            Region code matching a CSV file in ``data/stats/names/``.
        gender : str | None
            If supplied, filter to names matching this gender column value.
        """
        if region_code not in self.name_cache:
            file_path = self.stats_dir / f"names/{region_code}.csv"
            if not file_path.exists():
                return "Unknown"
            self.name_cache[region_code] = pd.read_csv(file_path)

        df = self.name_cache[region_code]

        if gender:
            df = df[df["gender"] == gender]

        if df.empty:
            return "Unknown"

        return df.sample(n=1, weights=df["probability"]).iloc[0]["name"]

    # ── Age sampling ─────────────────────────────────────────────────────────

    @staticmethod
    def get_random_age(
        mean: float = 25,
        std: float = 5,
        min_age: int = 14,
        max_age: int = 100,
    ) -> int:
        """Sample an age from a clamped normal distribution."""
        age = int(np.random.normal(loc=mean, scale=std))
        return max(min_age, min(max_age, age))
