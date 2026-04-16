"""
analysis/feature_engineering.py
=================================
Transforms raw repository data, code metrics, and pattern findings
into a numeric feature vector suitable for scikit-learn.

Feature groups:
  1. Repository metadata     (stars, age, activity)
  2. Code volume             (files, lines, functions, classes)
  3. Quality indicators      (comment density, duplication, complexity)
  4. Project structure       (directories, languages, depth)
  5. Dev practices           (CI, tests, Docker, type hints)
  6. Pattern signals         (design patterns found, practice count, smells)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

from analysis.code_analyzer import CodeMetrics
from analysis.pattern_detector import PatternReport
from config import Config
from data_collection.code_extractor import SourceFile
from data_collection.github_client import RepositoryData

# ── Feature Vector ────────────────────────────────────────────────────────────


@dataclass
class FeatureVector:
    """
    Numeric feature representation of a repository.
    All values are floats; ordinal/binary features are 0.0 or 1.0.
    """

    # ── Repository Metadata ──────────────────────────────────────────────────
    stars_log: float = 0.0  # log1p(stars)
    forks_log: float = 0.0  # log1p(forks)
    repo_age_days: float = 0.0
    days_since_push: float = 0.0
    commits_log: float = 0.0  # log1p(commits_count)
    branches_count: float = 0.0
    contributors_log: float = 0.0  # log1p(contributors)
    releases_log: float = 0.0
    has_readme: float = 0.0
    readme_length: float = 0.0  # log1p(chars)
    has_license: float = 0.0
    topics_count: float = 0.0

    # ── Code Volume ──────────────────────────────────────────────────────────
    file_count: float = 0.0
    total_lines_log: float = 0.0
    total_code_lines_log: float = 0.0
    functions_log: float = 0.0
    classes_log: float = 0.0
    avg_file_length: float = 0.0
    max_file_length_log: float = 0.0

    # ── Quality Metrics ──────────────────────────────────────────────────────
    comment_density: float = 0.0  # 0–1
    avg_complexity: float = 0.0  # 0–100
    duplication_score: float = 0.0  # 0–100
    modularity_score: float = 0.0  # 0–100
    naming_score: float = 0.0  # 0–100
    docstring_ratio: float = 0.0  # 0–1
    type_hint_ratio: float = 0.0  # 0–1
    quality_score: float = 0.0  # 0–100

    # ── Project Structure ─────────────────────────────────────────────────────
    unique_dirs: float = 0.0
    max_depth: float = 0.0
    language_count: float = 0.0
    avg_file_depth: float = 0.0

    # ── Dev Practices ─────────────────────────────────────────────────────────
    has_ci: float = 0.0
    has_tests: float = 0.0
    has_docker: float = 0.0
    has_requirements: float = 0.0

    # ── Pattern & Smell Signals ───────────────────────────────────────────────
    design_pattern_count: float = 0.0
    best_practice_count: float = 0.0
    code_smell_count: float = 0.0
    critical_smell_count: float = 0.0
    practice_score: float = 0.0  # 0–100

    # ── Commit Activity ───────────────────────────────────────────────────────
    commit_frequency: float = 0.0  # commits / age_days
    commit_message_quality: float = 0.0  # avg message length score

    def to_numpy(self) -> np.ndarray:
        """Return a 1D numpy array of all features (in field order)."""
        return np.array(list(asdict(self).values()), dtype=np.float32)

    @property
    def feature_names(self) -> list[str]:
        return list(asdict(self).keys())


# ── Engineer ──────────────────────────────────────────────────────────────────


class FeatureEngineer:
    """Builds a FeatureVector from all collected data."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def build(
        self,
        repo_data: RepositoryData,
        source_files: list[SourceFile],
        code_metrics: CodeMetrics,
        patterns: PatternReport,
    ) -> FeatureVector:
        """Assemble feature vector from all data sources."""
        fv = FeatureVector()
        self._add_repo_features(fv, repo_data)
        self._add_code_volume_features(fv, source_files, code_metrics)
        self._add_quality_features(fv, code_metrics)
        self._add_structure_features(fv, source_files, repo_data)
        self._add_practice_features(fv, repo_data)
        self._add_pattern_features(fv, patterns)
        self._add_commit_features(fv, repo_data)
        return fv

    # ── Feature groups ─────────────────────────────────────────────────────────

    def _add_repo_features(self, fv: FeatureVector, r: RepositoryData) -> None:
        fv.stars_log = math.log1p(r.stars)
        fv.forks_log = math.log1p(r.forks)
        fv.commits_log = math.log1p(r.commits_count)
        fv.branches_count = min(r.branches_count, 50)
        fv.contributors_log = math.log1p(r.contributors_count)
        fv.releases_log = math.log1p(r.releases_count)
        fv.has_readme = 1.0 if r.readme_content else 0.0
        fv.readme_length = math.log1p(len(r.readme_content or ""))
        fv.has_license = 1.0 if r.license_name else 0.0
        fv.topics_count = min(len(r.topics), 20)

        now = datetime.now(timezone.utc)
        if r.created_at:
            try:
                created = datetime.fromisoformat(r.created_at.replace("Z", "+00:00"))
                fv.repo_age_days = (now - created).days
            except ValueError:
                fv.repo_age_days = 0.0

        if r.pushed_at:
            try:
                pushed = datetime.fromisoformat(r.pushed_at.replace("Z", "+00:00"))
                fv.days_since_push = (now - pushed).days
            except ValueError:
                fv.days_since_push = 365.0

    def _add_code_volume_features(self, fv: FeatureVector, sfs: list[SourceFile], m: CodeMetrics) -> None:
        fv.file_count = len(sfs)
        fv.total_lines_log = math.log1p(m.total_lines)
        fv.total_code_lines_log = math.log1p(m.total_code_lines)
        fv.functions_log = math.log1p(m.total_functions)
        fv.classes_log = math.log1p(m.total_classes)
        fv.avg_file_length = min(m.avg_file_length, 1000)
        fv.max_file_length_log = math.log1p(m.max_file_length)

    def _add_quality_features(self, fv: FeatureVector, m: CodeMetrics) -> None:
        fv.comment_density = m.comment_density
        fv.avg_complexity = m.avg_complexity
        fv.duplication_score = m.duplication_score
        fv.modularity_score = m.modularity_score
        fv.naming_score = m.naming_score
        fv.docstring_ratio = m.docstring_ratio
        fv.type_hint_ratio = m.type_hint_ratio
        fv.quality_score = m.quality_score

    def _add_structure_features(self, fv: FeatureVector, sfs: list[SourceFile], r: RepositoryData) -> None:
        if sfs:
            dirs = {sf.path.rsplit("/", 1)[0] for sf in sfs if "/" in sf.path}
            fv.unique_dirs = len(dirs)
            fv.max_depth = max(sf.depth for sf in sfs)
            fv.avg_file_depth = sum(sf.depth for sf in sfs) / len(sfs)
        fv.language_count = len(r.languages)

    def _add_practice_features(self, fv: FeatureVector, r: RepositoryData) -> None:
        fv.has_ci = 1.0 if r.has_ci else 0.0
        fv.has_tests = 1.0 if r.has_tests else 0.0
        fv.has_docker = 1.0 if r.has_docker else 0.0
        fv.has_requirements = 1.0 if r.has_requirements else 0.0

    def _add_pattern_features(self, fv: FeatureVector, p: PatternReport) -> None:
        fv.design_pattern_count = len(p.design_patterns)
        fv.best_practice_count = len(p.best_practices)
        fv.code_smell_count = len(p.code_smells)
        fv.critical_smell_count = sum(1 for s in p.code_smells if s.severity == "critical")
        fv.practice_score = p.practice_score

    def _add_commit_features(self, fv: FeatureVector, r: RepositoryData) -> None:
        if r.repo_age_days_approx > 0:
            fv.commit_frequency = r.commits_count / r.repo_age_days_approx
        else:
            fv.commit_frequency = 0.0

        if r.commit_history:
            msg_lengths = [len(c.get("message", "")) for c in r.commit_history]
            avg_len = sum(msg_lengths) / len(msg_lengths)
            # Score 0–1: ideal commit message 30–72 chars
            fv.commit_message_quality = min(1.0, avg_len / 50)
        else:
            fv.commit_message_quality = 0.0


# ── Patch RepositoryData to add helper property ───────────────────────────────
# We add a computed property without modifying the dataclass.


def _repo_age_days_approx(self) -> float:
    """Approximate repo age in days from created_at."""
    if not self.created_at:
        return 0.0
    try:
        created = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        return max((datetime.now(timezone.utc) - created).days, 1)
    except ValueError:
        return 365.0


# Monkey-patch property onto RepositoryData
from data_collection.github_client import RepositoryData as _RD  # noqa: E402

_RD.repo_age_days_approx = property(_repo_age_days_approx)
