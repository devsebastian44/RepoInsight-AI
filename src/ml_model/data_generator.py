"""
ml_model/data_generator.py
============================
Generates a labeled synthetic training dataset for the
repository-level classifier.

Classes:
    0 → Junior    (small, simple, few patterns, many smells)
    1 → Mid-level (moderate complexity, some patterns, decent practices)
    2 → Senior    (large, well-structured, many patterns & practices, few smells)

Each sample is a numpy array matching FeatureVector field order.
The feature distributions are calibrated to realistic repository profiles.
"""

from __future__ import annotations

import numpy as np
from dataclasses import fields

from analysis.feature_engineering import FeatureVector


LABEL_MAP = {0: "Junior", 1: "Mid-level", 2: "Senior"}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Number of synthetic samples per class
SAMPLES_PER_CLASS = 400


def _feature_names() -> list[str]:
    return [f.name for f in fields(FeatureVector)]


def _clamp(value: float, lo: float = 0.0, hi: float = 1e9) -> float:
    return max(lo, min(hi, value))


class SyntheticDataGenerator:
    """
    Generates realistic (but synthetic) training data for the classifier.

    Feature distributions per class are defined as (mean, std) tuples,
    representing plausible values for Junior / Mid-level / Senior repos.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.rng = np.random.default_rng(random_state)

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (X, y) where X.shape == (n_samples, n_features).
        """
        samples: list[np.ndarray] = []
        labels:  list[int]        = []

        for label, generator in [
            (0, self._junior_sample),
            (1, self._mid_sample),
            (2, self._senior_sample),
        ]:
            for _ in range(SAMPLES_PER_CLASS):
                samples.append(generator())
                labels.append(label)

        X = np.stack(samples, axis=0).astype(np.float32)
        y = np.array(labels, dtype=np.int32)

        # Shuffle
        idx = self.rng.permutation(len(y))
        return X[idx], y[idx]

    # ── Generators ─────────────────────────────────────────────────────────────

    def _junior_sample(self) -> np.ndarray:
        r = self.rng
        fv = FeatureVector(
            # Repo metadata — small, young, few stars
            stars_log            = _clamp(r.normal(0.5, 0.5)),
            forks_log            = _clamp(r.normal(0.2, 0.3)),
            repo_age_days        = _clamp(r.normal(120, 90), 1),
            days_since_push      = _clamp(r.normal(90, 60)),
            commits_log          = _clamp(r.normal(2.5, 0.8)),
            branches_count       = _clamp(r.normal(1.5, 0.8), 1, 50),
            contributors_log     = _clamp(r.normal(0.5, 0.3)),
            releases_log         = _clamp(r.normal(0.1, 0.2)),
            has_readme           = float(r.random() > 0.4),
            readme_length        = _clamp(r.normal(3.5, 1.0)),
            has_license          = float(r.random() > 0.7),
            topics_count         = _clamp(r.normal(1, 1)),
            # Code volume — small
            file_count           = _clamp(r.normal(5, 3), 1),
            total_lines_log      = _clamp(r.normal(5, 1)),
            total_code_lines_log = _clamp(r.normal(4.5, 1)),
            functions_log        = _clamp(r.normal(2.5, 0.8)),
            classes_log          = _clamp(r.normal(0.8, 0.6)),
            avg_file_length      = _clamp(r.normal(80, 40), 1),
            max_file_length_log  = _clamp(r.normal(5.0, 0.8)),
            # Quality — low
            comment_density      = _clamp(r.normal(0.05, 0.04)),
            avg_complexity       = _clamp(r.normal(60, 20)),
            duplication_score    = _clamp(r.normal(40, 20)),
            modularity_score     = _clamp(r.normal(20, 15)),
            naming_score         = _clamp(r.normal(45, 20)),
            docstring_ratio      = _clamp(r.normal(0.05, 0.05)),
            type_hint_ratio      = _clamp(r.normal(0.0, 0.05)),
            quality_score        = _clamp(r.normal(30, 15)),
            # Structure — flat
            unique_dirs          = _clamp(r.normal(1.5, 1), 0),
            max_depth            = _clamp(r.normal(1.5, 1), 0),
            language_count       = _clamp(r.normal(1.2, 0.5), 1),
            avg_file_depth       = _clamp(r.normal(0.8, 0.5)),
            # Practices — few
            has_ci               = float(r.random() > 0.85),
            has_tests            = float(r.random() > 0.75),
            has_docker           = float(r.random() > 0.90),
            has_requirements     = float(r.random() > 0.50),
            # Patterns — very few
            design_pattern_count = _clamp(r.normal(0.3, 0.5)),
            best_practice_count  = _clamp(r.normal(1.5, 1.0)),
            code_smell_count     = _clamp(r.normal(5, 2)),
            critical_smell_count = _clamp(r.normal(1.5, 1.0)),
            practice_score       = _clamp(r.normal(15, 10)),
            # Commits — infrequent
            commit_frequency     = _clamp(r.normal(0.03, 0.03)),
            commit_message_quality = _clamp(r.normal(0.3, 0.2)),
        )
        return fv.to_numpy()

    def _mid_sample(self) -> np.ndarray:
        r = self.rng
        fv = FeatureVector(
            stars_log            = _clamp(r.normal(2.5, 1.0)),
            forks_log            = _clamp(r.normal(1.5, 0.8)),
            repo_age_days        = _clamp(r.normal(500, 200), 1),
            days_since_push      = _clamp(r.normal(30, 25)),
            commits_log          = _clamp(r.normal(4.0, 0.8)),
            branches_count       = _clamp(r.normal(4, 2), 1, 50),
            contributors_log     = _clamp(r.normal(1.2, 0.6)),
            releases_log         = _clamp(r.normal(0.8, 0.5)),
            has_readme           = float(r.random() > 0.15),
            readme_length        = _clamp(r.normal(5.5, 1.0)),
            has_license          = float(r.random() > 0.35),
            topics_count         = _clamp(r.normal(4, 2)),
            file_count           = _clamp(r.normal(20, 10), 1),
            total_lines_log      = _clamp(r.normal(7.5, 1)),
            total_code_lines_log = _clamp(r.normal(7.0, 1)),
            functions_log        = _clamp(r.normal(4.5, 0.8)),
            classes_log          = _clamp(r.normal(2.5, 0.8)),
            avg_file_length      = _clamp(r.normal(150, 60), 1),
            max_file_length_log  = _clamp(r.normal(6.0, 0.8)),
            comment_density      = _clamp(r.normal(0.12, 0.05)),
            avg_complexity       = _clamp(r.normal(45, 15)),
            duplication_score    = _clamp(r.normal(20, 15)),
            modularity_score     = _clamp(r.normal(50, 20)),
            naming_score         = _clamp(r.normal(65, 15)),
            docstring_ratio      = _clamp(r.normal(0.35, 0.15)),
            type_hint_ratio      = _clamp(r.normal(0.20, 0.15)),
            quality_score        = _clamp(r.normal(55, 15)),
            unique_dirs          = _clamp(r.normal(5, 3), 0),
            max_depth            = _clamp(r.normal(3, 1.5), 0),
            language_count       = _clamp(r.normal(2, 1), 1),
            avg_file_depth       = _clamp(r.normal(2.0, 0.8)),
            has_ci               = float(r.random() > 0.40),
            has_tests            = float(r.random() > 0.35),
            has_docker           = float(r.random() > 0.55),
            has_requirements     = float(r.random() > 0.20),
            design_pattern_count = _clamp(r.normal(2, 1.5)),
            best_practice_count  = _clamp(r.normal(5, 2)),
            code_smell_count     = _clamp(r.normal(3, 1.5)),
            critical_smell_count = _clamp(r.normal(0.5, 0.5)),
            practice_score       = _clamp(r.normal(45, 15)),
            commit_frequency     = _clamp(r.normal(0.15, 0.08)),
            commit_message_quality = _clamp(r.normal(0.55, 0.15)),
        )
        return fv.to_numpy()

    def _senior_sample(self) -> np.ndarray:
        r = self.rng
        fv = FeatureVector(
            stars_log            = _clamp(r.normal(4.5, 1.5)),
            forks_log            = _clamp(r.normal(3.0, 1.0)),
            repo_age_days        = _clamp(r.normal(1200, 400), 1),
            days_since_push      = _clamp(r.normal(7, 10)),
            commits_log          = _clamp(r.normal(6.0, 1.0)),
            branches_count       = _clamp(r.normal(8, 4), 1, 50),
            contributors_log     = _clamp(r.normal(2.5, 0.8)),
            releases_log         = _clamp(r.normal(2.0, 0.8)),
            has_readme           = float(r.random() > 0.02),
            readme_length        = _clamp(r.normal(7.5, 0.8)),
            has_license          = float(r.random() > 0.05),
            topics_count         = _clamp(r.normal(7, 3)),
            file_count           = _clamp(r.normal(45, 15), 1),
            total_lines_log      = _clamp(r.normal(10, 1)),
            total_code_lines_log = _clamp(r.normal(9.5, 1)),
            functions_log        = _clamp(r.normal(6.5, 1.0)),
            classes_log          = _clamp(r.normal(4.5, 1.0)),
            avg_file_length      = _clamp(r.normal(200, 80), 1),
            max_file_length_log  = _clamp(r.normal(6.5, 0.8)),
            comment_density      = _clamp(r.normal(0.22, 0.06)),
            avg_complexity       = _clamp(r.normal(30, 12)),
            duplication_score    = _clamp(r.normal(8, 8)),
            modularity_score     = _clamp(r.normal(80, 15)),
            naming_score         = _clamp(r.normal(85, 10)),
            docstring_ratio      = _clamp(r.normal(0.75, 0.15)),
            type_hint_ratio      = _clamp(r.normal(0.70, 0.20)),
            quality_score        = _clamp(r.normal(80, 10)),
            unique_dirs          = _clamp(r.normal(12, 5), 0),
            max_depth            = _clamp(r.normal(5, 2), 0),
            language_count       = _clamp(r.normal(3, 1.5), 1),
            avg_file_depth       = _clamp(r.normal(3.5, 1.0)),
            has_ci               = float(r.random() > 0.05),
            has_tests            = float(r.random() > 0.05),
            has_docker           = float(r.random() > 0.20),
            has_requirements     = float(r.random() > 0.05),
            design_pattern_count = _clamp(r.normal(5, 2)),
            best_practice_count  = _clamp(r.normal(9, 2)),
            code_smell_count     = _clamp(r.normal(1, 1)),
            critical_smell_count = _clamp(r.normal(0.1, 0.2)),
            practice_score       = _clamp(r.normal(75, 12)),
            commit_frequency     = _clamp(r.normal(0.5, 0.3)),
            commit_message_quality = _clamp(r.normal(0.80, 0.12)),
        )
        return fv.to_numpy()
