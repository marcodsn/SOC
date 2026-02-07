from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class StatsEngine:
    def __init__(self, stats_dir: Path):
        self.stats_dir = stats_dir
        self.regions_config = self._load_yaml("demographics/regions.yaml")
        self.name_cache = {}

    def _load_yaml(self, relative_path):
        with open(self.stats_dir / relative_path, "r") as f:
            return yaml.safe_load(f)

    def get_random_region(self) -> str:
        """Selects a region code (e.g., 'it_IT') based on population weights."""
        regions = list(self.regions_config["regions"].keys())
        weights = list(self.regions_config["regions"].values())

        # Normalize weights to sum to 1.0 (just in case)
        weights = np.array(weights)
        weights /= weights.sum()

        return np.random.choice(regions, p=weights)

    def gen_random_subregion(self, region_code: str = None) -> str:
        """Given a region code, randomly selects a subregion (e.g., 'Lombardy')."""
        if region_code is None:
            region_code = self.get_random_region()

        subregions = list(self.regions_config["subregions"][region_code].keys())
        weights = list(self.regions_config["subregions"][region_code].values())

        # Normalize weights to sum to 1.0 (just in case)
        weights = np.array(weights)
        weights /= weights.sum()

        return np.random.choice(subregions, p=weights)

    def get_random_name(self, region_code: str, gender: str = None) -> str:
        """
        Loads the specific CSV for that region and picks a name
        weighted by its real-world frequency.
        """
        # Lazy load the CSV to save memory
        if region_code not in self.name_cache:
            file_path = self.stats_dir / f"names/{region_code}.csv"
            if not file_path.exists():
                # Fallback to a default if file missing
                return "Wallpup"
            self.name_cache[region_code] = pd.read_csv(file_path)

        df = self.name_cache[region_code]

        # Filter by gender if specified
        if gender:
            df = df[df["gender"] == gender]

        if df.empty:
            return "Unknown"

        # Weighted sampling
        return df.sample(n=1, weights=df["probability"]).iloc[0]["name"]

    def get_random_age(
        self, mean: float, std: float, min_age: int = 14, max_age: int = 100
    ) -> int:
        """Generates a random age based on a normal distribution with given mean and standard deviation."""
        age = int(np.random.normal(loc=mean, scale=std))
        return max(min_age, min(max_age, age))  # Clamp age to [min_age, max_age]


# Usage Example
# engine = StatsEngine(Path("data/stats"))
# region = engine.get_random_region()  # -> "it_IT"
# subregion = engine.gen_random_subregion(region)  # -> "Lombardy"
# name = engine.get_random_name(region)  # -> "Giulia"
# age = engine.get_random_age(mean=20, std=18)  # -> 24

# print(f"Selected region: {region}, subregion: {subregion}, name: {name}, age: {age}")
