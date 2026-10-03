"""Generate Wine cultivar samples using the training reference."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


class WineDataGenerator:
    def __init__(self, config_path="config.yaml"):
        path = Path(config_path)
        if not path.exists() and config_path == "config.yaml":
            path = Path(__file__).with_name("config.yaml")
        self.config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        root = Path(__file__).resolve().parents[1]
        self.reference = pd.read_csv(root / "data/reference.csv")
        self.feature_names = self.reference.columns.tolist()
        self.rng = np.random.default_rng(self.config.get("seed", 501))
        self.std = self.reference.std(ddof=0)
        self.features = self.config["features"]

    def get_feature_names(self):
        return self.feature_names.copy()

    def generate_normal_sample(self):
        index = int(self.rng.integers(len(self.reference)))
        return {
            name: float(self.reference.iloc[index][name])
            for name in self.feature_names
        }

    def generate_drifted_sample(
        self, drift_multiplier=1.5, affected_features=None, noise_level=0.0
    ):
        sample = self.generate_normal_sample()
        affected = (
            self.feature_names[:6]
            if affected_features is None else affected_features
        )
        for name in affected:
            if name not in sample:
                raise ValueError("Unknown feature: " + name)
            shift = (drift_multiplier - 1.0) * 2.0 * self.std[name]
            noise = self.rng.normal(0, self.std[name] * noise_level)
            sample[name] = float(sample[name] + shift + noise)
        return sample

    def generate_batch(self, n_samples=100, scenario="normal"):
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        if scenario not in self.config["scenarios"]:
            raise ValueError("Unknown scenario: " + scenario)
        settings = self.config["scenarios"][scenario]
        affected = self.feature_names[:settings.get("affected_features", 0)]
        samples = []
        for _ in range(n_samples):
            sample = self.generate_normal_sample()
            for name in affected:
                sample[name] += (
                    settings.get("shift_std", 0.0) * self.std[name]
                )
            samples.append(sample)
        return samples

    def generate_dataframe(self, n_samples=100, scenario="normal"):
        return pd.DataFrame(
            self.generate_batch(n_samples, scenario),
            columns=self.feature_names,
        )
