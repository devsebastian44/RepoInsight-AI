"""
analysis/code_analyzer.py
===========================
Computes static code metrics over the collection of SourceFile objects.

Metrics produced:
  - Cyclomatic complexity proxy (branch keywords per function)
  - Comment density (comment lines / total lines)
  - Average function length
  - Code duplication estimate (Jaccard similarity between files)
  - Naming quality score (adherence to conventions)
  - Modularity score (number of classes and modules)
  - Long-method smell count
  - Magic-number smell count
  - Deeply-nested code ratio
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import Config
from data_collection.code_extractor import SourceFile

# ── Result Models ─────────────────────────────────────────────────────────────


@dataclass
class FileMetrics:
    """Per-file quality metrics."""

    path: str
    complexity_score: float  # 0–100 (higher = more complex)
    comment_density: float  # 0.0–1.0
    avg_function_length: float  # Lines per function
    function_count: int
    class_count: int
    import_count: int
    max_line_length: int
    long_lines_ratio: float  # Fraction of lines > 120 chars
    magic_numbers: int  # Occurrences of raw numeric literals
    nested_depth_score: float  # Proxy for deep nesting (0–10)
    naming_score: float  # 0–10 (snake_case / camelCase adherence)
    has_docstrings: bool
    has_type_hints: bool  # Python-specific


@dataclass
class CodeMetrics:
    """Aggregate code metrics across the full repository."""

    total_files: int
    total_lines: int
    total_code_lines: int
    total_comment_lines: int
    total_blank_lines: int
    comment_density: float
    total_functions: int
    total_classes: int
    avg_file_length: float
    avg_function_length: float
    avg_complexity: float
    duplication_score: float  # 0–100 estimate
    modularity_score: float  # 0–100
    naming_score: float  # 0–100
    type_hint_ratio: float  # Python: fraction of files with hints
    docstring_ratio: float  # Fraction of files with docstrings
    long_method_count: int
    magic_number_count: int
    max_file_length: int
    deep_nesting_ratio: float
    file_metrics: list[FileMetrics] = field(default_factory=list)
    quality_score: float = 0.0  # Computed composite 0–100


# ── Analyzer ──────────────────────────────────────────────────────────────────


class CodeAnalyzer:
    """
    Static code quality analyzer.

    All metrics are heuristic (regex-based) since we do not execute the code.
    The composite quality_score is a weighted average of sub-metrics.
    """

    # Branch keywords that contribute to complexity
    _BRANCH_KW = re.compile(r"\b(if|elif|else|for|while|try|except|catch|switch|case|&&|\|\|)\b")
    _MAGIC_NUM = re.compile(r"(?<!['\"\w])(?!0[xXbBoO])\d+(?:\.\d+)?(?!['\"\w\d])")
    _NESTED = re.compile(r"^(\s+)(if|for|while|with|try)\b", re.M)
    _SNAKE = re.compile(r"^[a-z_][a-z0-9_]*$")
    _CAMEL = re.compile(r"^[a-z][a-zA-Z0-9]*$")
    _PASCAL = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
    _SCREAMING = re.compile(r"^[A-Z_][A-Z0-9_]*$")

    # Thresholds
    LONG_METHOD_THRESHOLD = 50  # Lines
    LONG_LINE_THRESHOLD = 120  # Characters
    MAGIC_NUM_THRESHOLD = 5  # Per file

    def __init__(self, config: Config) -> None:
        self.config = config

    def analyze(self, source_files: list[SourceFile]) -> CodeMetrics:
        """Compute full metrics for the repository."""
        if not source_files:
            return self._empty_metrics()

        file_metrics = [self._analyze_file(sf) for sf in source_files]

        total_lines = sum(sf.line_count for sf in source_files)
        total_code = sum(sf.code_lines for sf in source_files)
        total_comment = sum(sf.comment_lines for sf in source_files)
        total_blank = sum(sf.blank_lines for sf in source_files)
        total_funcs = sum(sf_m.function_count for sf_m in file_metrics)
        total_classes = sum(sf_m.class_count for sf_m in file_metrics)

        comment_density = total_comment / max(total_lines, 1)
        avg_file_length = total_lines / len(source_files)

        all_fn_lengths = [sf_m.avg_function_length for sf_m in file_metrics if sf_m.function_count > 0]
        avg_fn_length = sum(all_fn_lengths) / len(all_fn_lengths) if all_fn_lengths else 0

        avg_complexity = sum(sf_m.complexity_score for sf_m in file_metrics) / len(file_metrics)
        duplication = self._estimate_duplication(source_files)
        modularity = self._compute_modularity(source_files, file_metrics)
        naming_score = sum(sf_m.naming_score for sf_m in file_metrics) / len(file_metrics)

        py_files = [sf for sf in source_files if sf.extension == ".py"]
        type_hint_ratio = sum(1 for m in file_metrics if m.has_type_hints) / len(py_files) if py_files else 0.0
        docstring_ratio = sum(1 for m in file_metrics if m.has_docstrings) / len(file_metrics)
        long_method_count = sum(1 for m in file_metrics if m.avg_function_length > self.LONG_METHOD_THRESHOLD)
        magic_number_count = sum(m.magic_numbers for m in file_metrics)
        max_file_length = max(sf.line_count for sf in source_files)
        deep_nesting = sum(m.nested_depth_score for m in file_metrics) / len(file_metrics)

        metrics = CodeMetrics(
            total_files=len(source_files),
            total_lines=total_lines,
            total_code_lines=total_code,
            total_comment_lines=total_comment,
            total_blank_lines=total_blank,
            comment_density=comment_density,
            total_functions=total_funcs,
            total_classes=total_classes,
            avg_file_length=avg_file_length,
            avg_function_length=avg_fn_length,
            avg_complexity=avg_complexity,
            duplication_score=duplication,
            modularity_score=modularity,
            naming_score=naming_score * 10,  # scale to 100
            type_hint_ratio=type_hint_ratio,
            docstring_ratio=docstring_ratio,
            long_method_count=long_method_count,
            magic_number_count=magic_number_count,
            max_file_length=max_file_length,
            deep_nesting_ratio=deep_nesting / 10,
            file_metrics=file_metrics,
        )

        metrics.quality_score = self._compute_quality_score(metrics)
        return metrics

    # ── Per-file analysis ──────────────────────────────────────────────────────

    def _analyze_file(self, sf: SourceFile) -> FileMetrics:
        content = sf.content
        lines = sf.lines

        # Complexity: branch keywords / (functions + 1)
        branch_count = len(self._BRANCH_KW.findall(content))
        fn_count = max(len(sf.functions), 1)
        complexity = min(100.0, (branch_count / fn_count) * 5)

        # Average function length (heuristic: code_lines / functions)
        avg_fn_len = sf.code_lines / fn_count if sf.functions else 0.0

        # Long lines ratio
        long_lines = sum(1 for l in lines if len(l) > self.LONG_LINE_THRESHOLD)
        long_lines_ratio = long_lines / max(len(lines), 1)

        # Magic numbers (excluding common: 0, 1, 2, 10, 100)
        common_nums = {"0", "1", "2", "10", "100", "255"}
        magic_nums = [m for m in self._MAGIC_NUM.findall(content) if m not in common_nums]

        # Nesting depth score: count lines with deep indentation (>= 4 levels)
        deep_lines = sum(
            1
            for l in sf.content.splitlines()
            if len(l) - len(l.lstrip()) >= 16  # 4 × 4-space indent
        )
        nested_score = min(10.0, deep_lines / max(len(lines), 1) * 100)

        # Naming score
        naming = self._score_naming(sf.functions + sf.classes, sf.extension)

        # Docstrings (Python)
        has_docstrings = '"""' in content or "'''" in content

        # Type hints (Python)
        has_type_hints = bool(re.search(r"def\s+\w+\([^)]*:\s*\w", content))

        return FileMetrics(
            path=sf.path,
            complexity_score=complexity,
            comment_density=sf.comment_lines / max(sf.line_count, 1),
            avg_function_length=avg_fn_len,
            function_count=len(sf.functions),
            class_count=len(sf.classes),
            import_count=len(sf.imports),
            max_line_length=sf.max_line_len,
            long_lines_ratio=long_lines_ratio,
            magic_numbers=len(magic_nums),
            nested_depth_score=nested_score,
            naming_score=naming,
            has_docstrings=has_docstrings,
            has_type_hints=has_type_hints,
        )

    # ── Scoring helpers ───────────────────────────────────────────────────────

    def _score_naming(self, names: list[str], ext: str) -> float:
        """Score 0–10: how well names follow language conventions."""
        if not names:
            return 5.0
        score = 0
        for name in names:
            if ext == ".py":
                if self._SNAKE.match(name) or self._PASCAL.match(name) or self._SCREAMING.match(name):
                    score += 1
            elif ext in (".js", ".ts"):
                if self._CAMEL.match(name) or self._PASCAL.match(name):
                    score += 1
            elif ext in (".java", ".kt", ".cs"):
                if self._CAMEL.match(name) or self._PASCAL.match(name):
                    score += 1
            else:
                score += 0.7  # Unknown convention — neutral
        return min(10.0, (score / len(names)) * 10)

    def _estimate_duplication(self, source_files: list[SourceFile]) -> float:
        """
        Rough duplication estimate using line-level n-gram fingerprinting.
        Returns a score from 0 (no duplication) to 100 (high duplication).
        """
        if len(source_files) < 2:
            return 0.0

        def ngrams(lines: list[str], n: int = 5) -> set[tuple]:
            stripped = [l.strip() for l in lines if len(l.strip()) > 10]
            return {tuple(stripped[i : i + n]) for i in range(len(stripped) - n + 1)}

        fingerprints = [ngrams(sf.lines) for sf in source_files]
        duplicated = 0
        total = sum(len(fp) for fp in fingerprints)
        if total == 0:
            return 0.0

        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                shared = fingerprints[i] & fingerprints[j]
                duplicated += len(shared)

        return min(100.0, (duplicated / max(total, 1)) * 200)

    def _compute_modularity(self, source_files: list[SourceFile], file_metrics: list[FileMetrics]) -> float:
        """
        Modularity score 0–100.
        High score: many classes, reasonable file sizes, clear separation.
        """
        total_classes = sum(m.class_count for m in file_metrics)
        avg_size = sum(sf.line_count for sf in source_files) / max(len(source_files), 1)

        class_score = min(40, total_classes * 2)
        size_score = max(0, 40 - max(0, avg_size - 200) / 10)
        multi_file = min(20, len(source_files))
        return class_score + size_score + multi_file

    def _compute_quality_score(self, m: CodeMetrics) -> float:
        """
        Composite quality score (0–100).
        Weighted combination of key quality indicators.
        """
        weights = {
            "comment_density": (min(m.comment_density * 5, 1.0), 15),
            "naming": (m.naming_score / 100, 20),
            "complexity": (1 - min(m.avg_complexity / 100, 1), 20),
            "duplication": (1 - m.duplication_score / 100, 15),
            "modularity": (m.modularity_score / 100, 10),
            "docstrings": (m.docstring_ratio, 10),
            "type_hints": (m.type_hint_ratio, 5),
            "long_lines": (1 - sum(fm.long_lines_ratio for fm in m.file_metrics) / max(len(m.file_metrics), 1), 5),
        }

        total_weight = sum(w for _, (_, w) in weights.items())
        score = sum(v * w for _, (v, w) in weights.items()) / total_weight * 100
        return round(min(100.0, max(0.0, score)), 1)

    def _empty_metrics(self) -> CodeMetrics:
        return CodeMetrics(
            total_files=0,
            total_lines=0,
            total_code_lines=0,
            total_comment_lines=0,
            total_blank_lines=0,
            comment_density=0,
            total_functions=0,
            total_classes=0,
            avg_file_length=0,
            avg_function_length=0,
            avg_complexity=0,
            duplication_score=0,
            modularity_score=0,
            naming_score=0,
            type_hint_ratio=0,
            docstring_ratio=0,
            long_method_count=0,
            magic_number_count=0,
            max_file_length=0,
            deep_nesting_ratio=0,
            quality_score=0,
        )
