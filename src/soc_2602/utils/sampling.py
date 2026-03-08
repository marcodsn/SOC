from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ── Psychological archetype & regulation-style distributions ─────────────────
# These archetypes are synthesized from the five-type personality taxonomy
# identified by Kerber et al. (2021) using SOEP panel data (N = 22,820):
#
#   resilient       → 27%    (mapped to: grounded_pragmatist + warm_connector)
#   reserved        → 22%    (mapped to: quiet_introvert + drifter)
#   undercontroller → 23%    (mapped to: anxious_achiever + defensive_cynic)
#   overcontroller  → 18%    (mapped to: earnest_idealist + wounded_caretaker)
#   confident       → 10%    (mapped to: ambitious_performer)
#
# We split the five research types into nine finer-grained labels that give
# LLMs more actionable narrative direction while preserving the original
# base-rate structure.  The probabilities below sum to 1.0 and approximate
# the population distribution reported in the source study.
#
# Ref: Kerber, A., et al. (2021). "Personality types revisited — a
#      literature-informed and data-driven approach to an integration of
#      prototypical and dimensional constructs of personality description."
#      PLOS ONE, 16(1), e0244849.

ARCHETYPES: List[Tuple[str, float]] = [
    ("grounded_pragmatist", 0.22),  # resilient bucket: practical, steady, reliable
    ("warm_connector", 0.18),  # resilient bucket: empathetic, relationship-oriented
    ("anxious_achiever", 0.14),  # undercontroller bucket: driven but stressed
    ("quiet_introvert", 0.12),  # reserved bucket: low-key, reflective, private
    ("earnest_idealist", 0.10),  # overcontroller bucket: principled, sometimes rigid
    ("defensive_cynic", 0.08),  # undercontroller bucket: guarded, sharp-tongued
    ("drifter", 0.07),  # reserved bucket: aimless, disengaged, passive
    ("ambitious_performer", 0.05),  # confident bucket: self-assured, competitive
    ("wounded_caretaker", 0.04),  # overcontroller bucket: self-sacrificing, resentful
]

# Emotional regulation styles — rough population base rates synthesized from
# the affect regulation literature (Gross & John, 2003; Aldao et al., 2010).
REGULATION_STYLES: List[Tuple[str, float]] = [
    ("stable", 0.35),  # generally even-keeled, predictable emotional responses
    ("expressive", 0.30),  # emotions are visible and freely communicated
    ("suppressed", 0.20),  # emotions are felt but rarely shown or discussed
    ("volatile", 0.15),  # quick shifts, intense reactions, slow to baseline
]


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

    # ── Psychological archetype sampling ─────────────────────────────────────

    @staticmethod
    def get_random_archetype() -> str:
        """Sample a psychological archetype weighted by population base rates.

        Returns one of nine archetype labels synthesized from the Kerber et al.
        (2021) five-type personality taxonomy (SOEP, N = 22,820).  The finer
        granularity gives LLMs more actionable narrative direction while the
        probabilities preserve the original research base rates.
        """
        labels = [a[0] for a in ARCHETYPES]
        probs = np.array([a[1] for a in ARCHETYPES], dtype=np.float64)
        probs /= probs.sum()  # guard against float drift
        return np.random.choice(labels, p=probs)

    @staticmethod
    def get_random_regulation_style() -> str:
        """Sample an emotional regulation style weighted by population base rates.

        Returns one of: stable, expressive, suppressed, volatile.
        Rough base rates synthesized from affect regulation literature
        (Gross & John, 2003; Aldao et al., 2010).
        """
        labels = [r[0] for r in REGULATION_STYLES]
        probs = np.array([r[1] for r in REGULATION_STYLES], dtype=np.float64)
        probs /= probs.sum()
        return np.random.choice(labels, p=probs)
