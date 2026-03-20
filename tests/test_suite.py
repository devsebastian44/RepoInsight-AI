"""
tests/test_suite.py
====================
Unit and integration tests for all major modules.
Run with:  pytest tests/test_suite.py -v
"""

from __future__ import annotations

import sys
import os
import math
import numpy as np
import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return Config(verbose=False, max_files=5)


@pytest.fixture
def sample_python_source() -> str:
    return '''"""Module docstring."""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

CONSTANT_VALUE = 42


class UserService:
    """Service for user operations."""

    _instance = None  # Singleton

    def __init__(self, db_connection):
        self.db = db_connection
        self.cache: dict = {}

    def get_user(self, user_id: int) -> Optional[dict]:
        """Fetch user by ID with caching."""
        if user_id in self.cache:
            return self.cache[user_id]
        try:
            user = self.db.query(f"SELECT * FROM users WHERE id = {user_id}")
            self.cache[user_id] = user
            return user
        except Exception as exc:
            logger.error("Failed to fetch user %s: %s", user_id, exc)
            return None

    def create_user(self, name: str, email: str) -> dict:
        """Create and persist a new user."""
        if not name or not email:
            raise ValueError("Name and email are required")
        user = {"name": name, "email": email}
        self.db.insert("users", user)
        logger.info("Created user: %s", email)
        return user
'''


@pytest.fixture
def source_file(sample_python_source):
    from data_collection.code_extractor import CodeExtractor, SourceFile
    extractor = CodeExtractor(Config())
    return extractor._parse("services/user_service.py", ".py", sample_python_source)


@pytest.fixture
def code_metrics(source_file):
    from analysis.code_analyzer import CodeAnalyzer
    return CodeAnalyzer(Config()).analyze([source_file])


@pytest.fixture
def pattern_report(source_file, code_metrics):
    from analysis.pattern_detector import PatternDetector
    return PatternDetector(Config()).detect([source_file], code_metrics)


# ─────────────────────────────────────────────────────────────────────────────
#  Config Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_values(self, config):
        assert config.github_api_base == "https://api.github.com"
        assert config.max_files == 5
        assert config.model_n_estimators == 200

    def test_github_headers_unauthenticated(self, config):
        headers = config.github_headers
        assert "Accept" in headers
        assert "Authorization" not in headers

    def test_github_headers_authenticated(self):
        cfg = Config(github_token="test_token_abc")
        assert cfg.github_headers["Authorization"] == "Bearer test_token_abc"
        assert cfg.is_authenticated is True

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "env_token_xyz")
        cfg = Config()
        assert cfg.github_token == "env_token_xyz"


# ─────────────────────────────────────────────────────────────────────────────
#  CodeExtractor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeExtractor:
    def test_parse_basic(self, source_file):
        assert source_file.path == "services/user_service.py"
        assert source_file.extension == ".py"
        assert source_file.line_count > 0
        assert source_file.code_lines > 0

    def test_functions_extracted(self, source_file):
        assert "get_user" in source_file.functions
        assert "create_user" in source_file.functions

    def test_classes_extracted(self, source_file):
        assert "UserService" in source_file.classes

    def test_imports_extracted(self, source_file):
        assert any("os" in imp or "logging" in imp for imp in source_file.imports)

    def test_depth_computed(self, source_file):
        assert source_file.depth == 1  # "services/user_service.py"

    def test_blank_lines_counted(self, source_file):
        assert source_file.blank_lines >= 0
        assert source_file.blank_lines < source_file.line_count

    def test_comment_lines_counted(self, source_file):
        assert source_file.comment_lines >= 1

    def test_code_lines_invariant(self, source_file):
        total = source_file.code_lines + source_file.comment_lines + source_file.blank_lines
        assert abs(total - source_file.line_count) <= 2   # small rounding allowed

    def test_js_function_extraction(self):
        from data_collection.code_extractor import CodeExtractor
        ext = CodeExtractor(Config())
        js_code = """
        function fetchData(url) { return fetch(url); }
        const handleClick = async (event) => { console.log(event); }
        """
        sf = ext._parse("app.js", ".js", js_code)
        assert "fetchData" in sf.functions

    def test_large_file_respects_token_count(self):
        from data_collection.code_extractor import CodeExtractor
        ext = CodeExtractor(Config())
        large_content = "\n".join([f"x_{i} = {i}" for i in range(2000)])
        sf = ext._parse("big.py", ".py", large_content)
        assert sf.tokens > 0


# ─────────────────────────────────────────────────────────────────────────────
#  CodeAnalyzer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeAnalyzer:
    def test_returns_code_metrics(self, code_metrics):
        from analysis.code_analyzer import CodeMetrics
        assert isinstance(code_metrics, CodeMetrics)

    def test_quality_score_range(self, code_metrics):
        assert 0.0 <= code_metrics.quality_score <= 100.0

    def test_comment_density_range(self, code_metrics):
        assert 0.0 <= code_metrics.comment_density <= 1.0

    def test_file_metrics_populated(self, code_metrics):
        assert len(code_metrics.file_metrics) == 1
        fm = code_metrics.file_metrics[0]
        assert fm.function_count > 0
        assert fm.class_count > 0

    def test_type_hints_detected(self, code_metrics):
        # Our sample code has type hints
        assert code_metrics.type_hint_ratio > 0

    def test_docstring_ratio_detected(self, code_metrics):
        assert code_metrics.docstring_ratio > 0

    def test_empty_input_returns_zero_metrics(self, config):
        from analysis.code_analyzer import CodeAnalyzer
        metrics = CodeAnalyzer(config).analyze([])
        assert metrics.total_files == 0
        assert metrics.quality_score == 0

    def test_duplication_score_range(self, config):
        from data_collection.code_extractor import CodeExtractor
        from analysis.code_analyzer import CodeAnalyzer

        ext = CodeExtractor(config)
        # Two identical files → high duplication
        content = "\n".join([f"line_{i} = {i}" for i in range(100)])
        f1 = ext._parse("a.py", ".py", content)
        f2 = ext._parse("b.py", ".py", content)
        m = CodeAnalyzer(config).analyze([f1, f2])
        assert 0 <= m.duplication_score <= 100

    def test_naming_score_good_python(self, config):
        from data_collection.code_extractor import CodeExtractor
        from analysis.code_analyzer import CodeAnalyzer

        code = "def get_data():\n    pass\nclass DataService:\n    pass"
        sf = CodeExtractor(config)._parse("x.py", ".py", code)
        m = CodeAnalyzer(config).analyze([sf])
        assert m.naming_score >= 50


# ─────────────────────────────────────────────────────────────────────────────
#  PatternDetector Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPatternDetector:
    def test_singleton_detected(self, config):
        from data_collection.code_extractor import CodeExtractor
        from analysis.code_analyzer import CodeAnalyzer
        from analysis.pattern_detector import PatternDetector

        code = "class MyService:\n    _instance = None\n    def getInstance(self): pass"
        sf = CodeExtractor(config)._parse("s.py", ".py", code)
        m = CodeAnalyzer(config).analyze([sf])
        report = PatternDetector(config).detect([sf], m)
        pattern_names = [p.name for p in report.design_patterns]
        assert "Singleton" in pattern_names

    def test_type_annotation_practice_detected(self, pattern_report):
        practice_names = [p.name for p in pattern_report.best_practices]
        assert "Type Annotations" in practice_names

    def test_logging_practice_detected(self, pattern_report):
        practice_names = [p.name for p in pattern_report.best_practices]
        assert "Logging" in practice_names

    def test_todos_smell_detection(self, config):
        from data_collection.code_extractor import CodeExtractor
        from analysis.code_analyzer import CodeAnalyzer
        from analysis.pattern_detector import PatternDetector

        code = "# TODO: refactor this mess\ndef broken(): pass"
        sf = CodeExtractor(config)._parse("t.py", ".py", code)
        m = CodeAnalyzer(config).analyze([sf])
        report = PatternDetector(config).detect([sf], m)
        smell_names = [s.name for s in report.code_smells]
        assert "TODO / FIXME" in smell_names

    def test_bare_except_smell(self, config):
        from data_collection.code_extractor import CodeExtractor
        from analysis.code_analyzer import CodeAnalyzer
        from analysis.pattern_detector import PatternDetector

        code = "try:\n    risky()\nexcept:\n    pass"
        sf = CodeExtractor(config)._parse("b.py", ".py", code)
        m = CodeAnalyzer(config).analyze([sf])
        report = PatternDetector(config).detect([sf], m)
        smell_names = [s.name for s in report.code_smells]
        assert "Bare Except" in smell_names

    def test_pattern_report_practice_score(self, pattern_report):
        assert 0 <= pattern_report.practice_score <= 100

    def test_empty_files_returns_empty_report(self, config):
        from analysis.code_analyzer import CodeAnalyzer
        from analysis.pattern_detector import PatternDetector

        m = CodeAnalyzer(config).analyze([])
        report = PatternDetector(config).detect([], m)
        assert len(report.all_findings) == 0


# ─────────────────────────────────────────────────────────────────────────────
#  FeatureEngineer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineer:
    def _make_repo_data(self):
        from data_collection.github_client import RepositoryData
        return RepositoryData(
            owner="test", name="repo", full_name="test/repo",
            description="A test repo", url="https://github.com/test/repo",
            stars=100, forks=20, watchers=100, open_issues=5,
            default_branch="main", created_at="2022-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z", pushed_at="2024-06-01T00:00:00Z",
            size_kb=1500, language="Python", languages={"Python": 12000, "Shell": 400},
            topics=["python", "ml"], has_wiki=True, has_issues=True,
            has_projects=False, license_name="MIT", contributors_count=5,
            commits_count=200, branches_count=4, releases_count=3,
            commit_history=[{"message": "Add feature X", "sha": "abc123",
                             "author": "dev", "date": "2024-01-01"}],
            tree_entries=[], readme_content="# Test\nA test project.",
            has_ci=True, has_tests=True, has_docker=False, has_requirements=True,
        )

    def test_feature_vector_shape(self, config, source_file, code_metrics, pattern_report):
        from analysis.feature_engineering import FeatureEngineer, FeatureVector
        from dataclasses import fields

        fv = FeatureEngineer(config).build(
            self._make_repo_data(), [source_file], code_metrics, pattern_report
        )
        expected_len = len(fields(FeatureVector))
        assert len(fv.to_numpy()) == expected_len

    def test_feature_vector_all_finite(self, config, source_file, code_metrics, pattern_report):
        from analysis.feature_engineering import FeatureEngineer

        fv = FeatureEngineer(config).build(
            self._make_repo_data(), [source_file], code_metrics, pattern_report
        )
        arr = fv.to_numpy()
        assert np.all(np.isfinite(arr)), "Feature vector contains NaN or Inf"

    def test_stars_log_transform(self, config, source_file, code_metrics, pattern_report):
        from analysis.feature_engineering import FeatureEngineer

        repo = self._make_repo_data()
        repo.stars = 1000
        fv = FeatureEngineer(config).build(repo, [source_file], code_metrics, pattern_report)
        assert abs(fv.stars_log - math.log1p(1000)) < 0.01

    def test_practice_flags_propagated(self, config, source_file, code_metrics, pattern_report):
        from analysis.feature_engineering import FeatureEngineer

        repo = self._make_repo_data()
        fv = FeatureEngineer(config).build(repo, [source_file], code_metrics, pattern_report)
        assert fv.has_ci == 1.0
        assert fv.has_tests == 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  SyntheticDataGenerator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticDataGenerator:
    def test_shape(self):
        from ml_model.data_generator import SyntheticDataGenerator, SAMPLES_PER_CLASS
        gen = SyntheticDataGenerator(random_state=0)
        X, y = gen.generate()
        assert X.shape[0] == SAMPLES_PER_CLASS * 3
        assert len(np.unique(y)) == 3

    def test_no_nan(self):
        from ml_model.data_generator import SyntheticDataGenerator
        gen = SyntheticDataGenerator(random_state=1)
        X, _ = gen.generate()
        assert np.all(np.isfinite(X))

    def test_labels_balanced(self):
        from ml_model.data_generator import SyntheticDataGenerator, SAMPLES_PER_CLASS
        gen = SyntheticDataGenerator()
        _, y = gen.generate()
        counts = np.bincount(y)
        assert all(c == SAMPLES_PER_CLASS for c in counts)

    def test_senior_higher_quality_than_junior(self):
        from ml_model.data_generator import SyntheticDataGenerator
        from analysis.feature_engineering import FeatureVector
        from dataclasses import fields as dc_fields

        gen = SyntheticDataGenerator(random_state=42)
        fn = [f.name for f in dc_fields(FeatureVector)]
        quality_idx = fn.index("quality_score")

        samples_j = np.stack([gen._junior_sample() for _ in range(100)])
        samples_s = np.stack([gen._senior_sample() for _ in range(100)])
        assert samples_s[:, quality_idx].mean() > samples_j[:, quality_idx].mean()


# ─────────────────────────────────────────────────────────────────────────────
#  ModelTrainer + Classifier Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelTrainer:
    def test_train_and_predict(self, tmp_path, config):
        from ml_model.trainer import ModelTrainer
        from ml_model.data_generator import SyntheticDataGenerator

        cfg = Config(
            model_path=str(tmp_path / "model.pkl"),
            model_n_estimators=20,
            verbose=False,
        )
        trainer = ModelTrainer(cfg)
        pipeline = trainer.train_and_save()
        assert trainer.model_exists()

        gen = SyntheticDataGenerator()
        X, y = gen.generate()
        preds = pipeline.predict(X[:10])
        assert len(preds) == 10
        assert all(p in [0, 1, 2] for p in preds)

    def test_cross_val_above_chance(self, tmp_path):
        from ml_model.trainer import ModelTrainer
        from ml_model.data_generator import SyntheticDataGenerator
        from sklearn.model_selection import cross_val_score

        cfg = Config(model_path=str(tmp_path / "m.pkl"), model_n_estimators=30)
        trainer = ModelTrainer(cfg)
        pipeline = trainer.train_and_save()

        gen = SyntheticDataGenerator()
        X, y = gen.generate()
        scores = cross_val_score(pipeline, X, y, cv=3)
        # Must beat random chance (33%) by a comfortable margin
        assert scores.mean() > 0.70, f"CV accuracy too low: {scores.mean():.3f}"


class TestClassifier:
    def test_predict_returns_valid_label(self, tmp_path, config,
                                          source_file, code_metrics, pattern_report):
        from analysis.feature_engineering import FeatureEngineer, FeatureVector
        from ml_model.classifier import RepositoryClassifier
        from ml_model.data_generator import LABEL_MAP
        from data_collection.github_client import RepositoryData

        cfg = Config(model_path=str(tmp_path / "clf.pkl"), model_n_estimators=20)

        repo = RepositoryData(
            owner="u", name="r", full_name="u/r", description=None,
            url="https://github.com/u/r", stars=50, forks=5, watchers=50,
            open_issues=2, default_branch="main",
            created_at="2023-01-01T00:00:00Z", updated_at="2024-01-01T00:00:00Z",
            pushed_at="2024-01-01T00:00:00Z", size_kb=300, language="Python",
            languages={"Python": 5000}, topics=[], has_wiki=False, has_issues=True,
            has_projects=False, license_name=None, contributors_count=1,
            commits_count=30, branches_count=2, releases_count=0,
            commit_history=[], tree_entries=[], readme_content="# Test",
            has_ci=False, has_tests=False, has_docker=False, has_requirements=True,
        )

        fv = FeatureEngineer(cfg).build(repo, [source_file], code_metrics, pattern_report)
        clf = RepositoryClassifier(cfg)
        result = clf.predict(fv)

        assert result.level in LABEL_MAP.values()
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.composite_score <= 100.0
        assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
#  LLMAnalyzer Mock Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMAnalyzerMock:
    def test_mock_returns_insights(self, config, source_file, code_metrics, pattern_report):
        from ml_model.classifier import ClassificationResult
        from llm.llm_analyzer import LLMAnalyzer

        clf_result = ClassificationResult(
            level="Mid-level", level_index=1, confidence=0.72,
            probabilities={"Junior": 0.1, "Mid-level": 0.72, "Senior": 0.18},
            composite_score=62.0, feature_importances={},
        )
        analyzer = LLMAnalyzer(config)
        insights = analyzer._mock_response(code_metrics, pattern_report, clf_result)

        assert insights.summary != ""
        assert len(insights.strengths) > 0
        assert len(insights.improvements) > 0
        assert insights.provider == "mock"
        assert insights.error is None

    def test_parse_valid_json(self, config):
        from llm.llm_analyzer import LLMAnalyzer
        import json

        payload = {
            "summary": "Good project.",
            "strengths": ["Clean code"],
            "improvements": ["Add tests"],
            "architectural_observations": ["Flat structure"],
            "security_concerns": ["None identified"],
            "scalability_notes": "Could use caching.",
        }
        raw = json.dumps(payload)
        insights = LLMAnalyzer(config)._parse_response(raw, "openai")
        assert insights.summary == "Good project."
        assert insights.error is None

    def test_parse_malformed_json_returns_error(self, config):
        from llm.llm_analyzer import LLMAnalyzer
        insights = LLMAnalyzer(config)._parse_response("not json at all !!!", "openai")
        assert insights.error is not None


# ─────────────────────────────────────────────────────────────────────────────
#  ReportBuilder Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReportBuilder:
    def test_report_has_required_keys(self, config, source_file, code_metrics, pattern_report):
        from analysis.feature_engineering import FeatureEngineer
        from ml_model.classifier import ClassificationResult
        from reporting.report_builder import ReportBuilder
        from data_collection.github_client import RepositoryData

        repo = RepositoryData(
            owner="u", name="r", full_name="u/r", description="Test",
            url="https://github.com/u/r", stars=10, forks=2, watchers=10,
            open_issues=1, default_branch="main",
            created_at="2023-06-01T00:00:00Z", updated_at="2024-01-01T00:00:00Z",
            pushed_at="2024-01-01T00:00:00Z", size_kb=100, language="Python",
            languages={"Python": 3000}, topics=["test"], has_wiki=False,
            has_issues=True, has_projects=False, license_name="MIT",
            contributors_count=1, commits_count=20, branches_count=1,
            releases_count=0, commit_history=[], tree_entries=[],
            readme_content="# Hi", has_ci=False, has_tests=False,
            has_docker=False, has_requirements=True,
        )
        fv = FeatureEngineer(config).build(repo, [source_file], code_metrics, pattern_report)
        clf = ClassificationResult(
            level="Mid-level", level_index=1, confidence=0.65,
            probabilities={"Junior": 0.2, "Mid-level": 0.65, "Senior": 0.15},
            composite_score=58.0, feature_importances={},
        )
        report = ReportBuilder(config).build(
            repo_data=repo, source_files=[source_file],
            code_metrics=code_metrics, patterns=pattern_report,
            feature_vector=fv, classification=clf, llm_insights={},
        )

        for key in ("nivel", "calidad", "score", "buenas_practicas", "mejoras",
                    "repositorio", "clasificacion", "metricas_codigo",
                    "patrones_diseno", "code_smells", "recomendaciones"):
            assert key in report, f"Missing key: {key}"

    def test_score_within_range(self, config, source_file, code_metrics, pattern_report):
        from analysis.feature_engineering import FeatureEngineer
        from ml_model.classifier import ClassificationResult
        from reporting.report_builder import ReportBuilder
        from data_collection.github_client import RepositoryData

        repo = RepositoryData(
            owner="u", name="r", full_name="u/r", description=None,
            url="https://github.com/u/r", stars=5, forks=0, watchers=5,
            open_issues=0, default_branch="main",
            created_at="2024-01-01T00:00:00Z", updated_at="2024-06-01T00:00:00Z",
            pushed_at="2024-06-01T00:00:00Z", size_kb=50, language="Python",
            languages={"Python": 1500}, topics=[], has_wiki=False,
            has_issues=False, has_projects=False, license_name=None,
            contributors_count=1, commits_count=5, branches_count=1,
            releases_count=0, commit_history=[], tree_entries=[],
            readme_content=None, has_ci=False, has_tests=False,
            has_docker=False, has_requirements=False,
        )
        fv = FeatureEngineer(config).build(repo, [source_file], code_metrics, pattern_report)
        clf = ClassificationResult(
            level="Junior", level_index=0, confidence=0.80,
            probabilities={"Junior": 0.80, "Mid-level": 0.15, "Senior": 0.05},
            composite_score=28.0, feature_importances={},
        )
        report = ReportBuilder(config).build(
            repo_data=repo, source_files=[source_file],
            code_metrics=code_metrics, patterns=pattern_report,
            feature_vector=fv, classification=clf, llm_insights={},
        )
        assert 0 <= report["score"] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
