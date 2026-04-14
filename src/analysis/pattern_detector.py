"""
analysis/pattern_detector.py
==============================
Detects:
  1. Design patterns (Singleton, Factory, Observer, Strategy, Decorator, etc.)
  2. Best practices (SOLID signals, clean code, separation of concerns)
  3. Code smells (God Class, Long Method, Feature Envy, etc.)

All detection is heuristic (regex + structural analysis).
Returns a PatternReport with classified findings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import NamedTuple

from analysis.code_analyzer import CodeMetrics
from config import Config
from data_collection.code_extractor import SourceFile

# ── Result Models ─────────────────────────────────────────────────────────────


class Finding(NamedTuple):
    category: str  # "pattern" | "practice" | "smell"
    name: str  # Human-readable name
    severity: str  # "info" | "warning" | "critical" (smells only)
    confidence: str  # "high" | "medium" | "low"
    files: list[str]  # Files where finding was observed
    description: str  # One-line description


@dataclass
class PatternReport:
    """All pattern-detection findings for a repository."""

    design_patterns: list[Finding] = field(default_factory=list)
    best_practices: list[Finding] = field(default_factory=list)
    code_smells: list[Finding] = field(default_factory=list)

    @property
    def all_findings(self) -> list[Finding]:
        return self.design_patterns + self.best_practices + self.code_smells

    @property
    def smell_count(self) -> int:
        return len(self.code_smells)

    @property
    def practice_score(self) -> float:
        """0–100: ratio of good practices found vs max possible."""
        max_practices = 12
        return min(100.0, len(self.best_practices) / max_practices * 100)


# ── Detector ──────────────────────────────────────────────────────────────────


class PatternDetector:
    """
    Analyzes source files for patterns, practices, and smells.

    Detection strategies:
        - Regex matching on class names, method names, imports
        - Structural heuristics (file count per module, class size)
        - Import-graph proxy (dependency analysis via import names)
    """

    # ── Design Pattern Signatures ─────────────────────────────────────────────
    # Each entry: (pattern_name, regexes_to_match, confidence)
    _PATTERN_SIGNATURES = [
        (
            "Singleton",
            [r"_instance\s*=\s*None", r"getInstance\(\)", r"__new__.*cls\._instance"],
            "high",
            "Restricts class to one instance. Detected via _instance, getInstance patterns.",
        ),
        (
            "Factory / Abstract Factory",
            [r"def\s+create_\w+\(", r"class\s+\w+Factory", r"def\s+factory\("],
            "high",
            "Delegates object creation to a separate factory class/method.",
        ),
        (
            "Observer / Event System",
            [r"def\s+subscribe|def\s+unsubscribe|def\s+notify|on_\w+\s*=", r"EventEmitter|Observable"],
            "medium",
            "Implements publish-subscribe or event-driven notification.",
        ),
        (
            "Strategy Pattern",
            [r"class\s+\w+Strategy", r"self\._strategy", r"set_strategy\("],
            "medium",
            "Encapsulates interchangeable algorithms behind a common interface.",
        ),
        (
            "Decorator Pattern",
            [r"@\w+\n\s*def\s+\w+", r"class\s+\w+Decorator\b", r"functools\.wraps"],
            "medium",
            "Wraps objects/functions to extend behavior dynamically.",
        ),
        (
            "Repository Pattern",
            [r"class\s+\w+Repository", r"def\s+find_by_\w+\(", r"def\s+save\(self"],
            "medium",
            "Abstracts data access behind a collection-like interface.",
        ),
        (
            "Command Pattern",
            [r"class\s+\w+Command\b", r"def\s+execute\(self\)", r"def\s+undo\(self\)"],
            "medium",
            "Encapsulates requests as objects, enabling undo/queue support.",
        ),
        (
            "Builder Pattern",
            [r"class\s+\w+Builder\b", r"def\s+build\(self\)\s*->", r"\.set_\w+\("],
            "medium",
            "Constructs complex objects step by step.",
        ),
        (
            "Adapter Pattern",
            [r"class\s+\w+Adapter\b", r"class\s+\w+Wrapper\b"],
            "low",
            "Wraps incompatible interfaces to make them compatible.",
        ),
        (
            "MVC / MVP / MVVM",
            [r"class\s+\w+Controller\b", r"class\s+\w+View\b", r"class\s+\w+ViewModel\b", r"class\s+\w+Model\b"],
            "medium",
            "Architectural pattern separating UI logic from business logic.",
        ),
        (
            "Dependency Injection",
            [r"def\s+__init__\(self[^)]+:\s*\w+[^)]*\)", r"@inject\b", r"Container\(\)"],
            "medium",
            "Dependencies are passed in rather than created internally.",
        ),
        (
            "Context Manager",
            [r"def\s+__enter__\(self\)", r"def\s+__exit__\(self"],
            "high",
            "Implements the context-manager protocol (__enter__ / __exit__).",
        ),
    ]

    # ── Best Practices ────────────────────────────────────────────────────────
    _PRACTICE_CHECKS = [
        (
            "Type Annotations",
            r"def\s+\w+\([^)]*:\s*\w+[^)]*\)\s*->",
            "Python type hints on function signatures detected.",
        ),
        (
            "Docstrings / JSDoc",
            r'("""|\'\'\'|/\*\*)',
            "Module/function documentation strings present.",
        ),
        (
            "Exception Handling",
            r"\b(try|except|catch|finally)\b",
            "Structured exception/error handling found.",
        ),
        (
            "Logging",
            r"\b(logging\.|logger\.|log\.|console\.log|console\.error)\b",
            "Dedicated logging (not just print/puts) used.",
        ),
        (
            "Environment Variables",
            r"\b(os\.environ|os\.getenv|process\.env|dotenv)\b",
            "Configuration via environment variables — avoids hardcoded secrets.",
        ),
        (
            "Constants / Enums",
            r"\b(Enum|@enum|const\s+[A-Z_]+|[A-Z_]{3,}\s*=)\b",
            "Named constants or enums replace magic literals.",
        ),
        (
            "Dataclasses / Value Objects",
            r"\b(@dataclass|@Data|record\s+class|data\s+class)\b",
            "Immutable value objects or dataclasses in use.",
        ),
        (
            "Unit Tests",
            r"\b(def\s+test_\w+|it\(|describe\(|@Test\b|#\[test\])\b",
            "Unit test functions or test-framework hooks present.",
        ),
        (
            "Linting Config",
            r"(\.flake8|\.eslintrc|\.pylintrc|ruff\.toml|tslint\.json|biome\.json)",
            "Linting/formatting configuration detected.",
        ),
        (
            "Separation of Concerns",
            None,  # Checked structurally, not by regex
            "Multiple modules/packages indicate SoC.",
        ),
        (
            "Async / Concurrent Programming",
            r"\b(async\s+def|await\s+|asyncio\.|Promise\.|goroutine|thread\.start)\b",
            "Asynchronous or concurrent programming constructs used.",
        ),
        (
            "Interface / Protocol / ABC",
            r"\b(ABC|abstractmethod|Protocol|interface\s+\w+|implements\s+\w+)\b",
            "Abstract base classes or interfaces define clear contracts.",
        ),
    ]

    # ── Code Smells ────────────────────────────────────────────────────────────
    # (name, detector_method_name, severity, description)
    _SMELL_DEFINITIONS = [
        ("God Class", "_detect_god_class", "critical", "A class with too many responsibilities (>10 methods or >200 lines)."),
        ("Long Method", "_detect_long_method", "warning", "Functions exceeding 50 lines are hard to read and test."),
        ("Deep Nesting", "_detect_deep_nesting", "warning", "Code nested >4 levels deep — refactor with early returns or helpers."),
        ("Magic Numbers", "_detect_magic_numbers", "warning", "Raw numeric literals in logic — use named constants instead."),
        ("Long Parameter List", "_detect_long_params", "warning", "Functions with >5 parameters — consider a parameter object."),
        ("Commented-Out Code", "_detect_dead_code_comments", "info", "Blocks of commented-out source code should be deleted, not commented."),
        ("print() Debugging", "_detect_print_debugging", "info", "print() statements in production code — use a proper logger."),
        ("TODO / FIXME", "_detect_todos", "info", "Unresolved TODO/FIXME/HACK markers in source."),
        ("Duplicate Code", "_detect_duplication_smell", "warning", "Similar code blocks detected across files."),
        ("Bare Except", "_detect_bare_except", "warning", "Python bare `except:` catches all exceptions — use specific types."),
    ]

    def __init__(self, config: Config) -> None:
        self.config = config

    def detect(self, source_files: list[SourceFile], metrics: CodeMetrics) -> PatternReport:
        """Run all detectors and return a PatternReport."""
        if not source_files:
            return PatternReport()

        design_patterns = self._detect_design_patterns(source_files)
        best_practices = self._detect_best_practices(source_files, metrics)
        code_smells = self._detect_code_smells(source_files, metrics)

        return PatternReport(
            design_patterns=design_patterns,
            best_practices=best_practices,
            code_smells=code_smells,
        )

    # ── Design Patterns ───────────────────────────────────────────────────────

    def _detect_design_patterns(self, source_files: list[SourceFile]) -> list[Finding]:
        findings: list[Finding] = []
        "\n".join(sf.content for sf in source_files)
        [sf.path for sf in source_files]

        for name, regexes, confidence, desc in self._PATTERN_SIGNATURES:
            matched_files: list[str] = []
            for sf in source_files:
                if any(re.search(rx, sf.content, re.I) for rx in regexes):
                    matched_files.append(sf.path)
            if matched_files:
                findings.append(
                    Finding(
                        category="pattern",
                        name=name,
                        severity="info",
                        confidence=confidence,
                        files=matched_files[:5],
                        description=desc,
                    )
                )
        return findings

    # ── Best Practices ────────────────────────────────────────────────────────

    def _detect_best_practices(self, source_files: list[SourceFile], metrics: CodeMetrics) -> list[Finding]:
        findings: list[Finding] = []
        "\n".join(sf.content for sf in source_files)
        {sf.path: sf for sf in source_files}

        for name, regex, desc in self._PRACTICE_CHECKS:
            if regex is None:
                # Structural check: multiple packages = SoC
                dirs = {sf.path.rsplit("/", 1)[0] for sf in source_files if "/" in sf.path}
                if len(dirs) >= 3:
                    findings.append(
                        Finding(
                            category="practice",
                            name=name,
                            severity="info",
                            confidence="medium",
                            files=[],
                            description=desc,
                        )
                    )
                continue

            matched = [sf.path for sf in source_files if re.search(regex, sf.content, re.M)]
            if matched:
                findings.append(
                    Finding(
                        category="practice",
                        name=name,
                        severity="info",
                        confidence="high",
                        files=matched[:5],
                        description=desc,
                    )
                )

        return findings

    # ── Code Smells ────────────────────────────────────────────────────────────

    def _detect_code_smells(self, source_files: list[SourceFile], metrics: CodeMetrics) -> list[Finding]:
        findings: list[Finding] = []
        for smell_name, method_name, severity, desc in self._SMELL_DEFINITIONS:
            detector = getattr(self, method_name)
            affected = detector(source_files, metrics)
            if affected:
                findings.append(
                    Finding(
                        category="smell",
                        name=smell_name,
                        severity=severity,
                        confidence="medium",
                        files=affected[:5],
                        description=desc,
                    )
                )
        return findings

    # Individual smell detectors —————————————————————————————————————————————

    def _detect_god_class(self, sfs: list[SourceFile], _metrics) -> list[str]:
        result = []
        for sf in sfs:
            if len(sf.classes) == 0:
                continue
            # Proxy: methods / class > 10 or file > 200 code lines
            if len(sf.functions) / len(sf.classes) > 10 or sf.code_lines > 250:
                result.append(sf.path)
        return result

    def _detect_long_method(self, sfs: list[SourceFile], metrics) -> list[str]:
        return [m.path for m in metrics.file_metrics if m.avg_function_length > 50]

    def _detect_deep_nesting(self, sfs: list[SourceFile], metrics) -> list[str]:
        return [m.path for m in metrics.file_metrics if m.nested_depth_score > 4]

    def _detect_magic_numbers(self, sfs: list[SourceFile], metrics) -> list[str]:
        return [m.path for m in metrics.file_metrics if m.magic_numbers > 5]

    def _detect_long_params(self, sfs: list[SourceFile], _metrics) -> list[str]:
        result = []
        pattern = re.compile(r"def\s+\w+\(([^)]{60,})\)")
        for sf in sfs:
            if pattern.search(sf.content):
                result.append(sf.path)
        return result

    def _detect_dead_code_comments(self, sfs: list[SourceFile], _metrics) -> list[str]:
        pattern = re.compile(r"^\s*#\s*(if|for|while|def|class|return|import)\s", re.M)
        return [sf.path for sf in sfs if pattern.search(sf.content)]

    def _detect_print_debugging(self, sfs: list[SourceFile], _metrics) -> list[str]:
        pattern = re.compile(r"^\s*print\s*\(", re.M)
        return [sf.path for sf in sfs if pattern.search(sf.content)]

    def _detect_todos(self, sfs: list[SourceFile], _metrics) -> list[str]:
        pattern = re.compile(r"#\s*(TODO|FIXME|HACK|XXX|BUG)\b", re.I)
        return [sf.path for sf in sfs if pattern.search(sf.content)]

    def _detect_duplication_smell(self, sfs: list[SourceFile], metrics) -> list[str]:
        if metrics.duplication_score > 20:
            return [sf.path for sf in sfs[:3]]
        return []

    def _detect_bare_except(self, sfs: list[SourceFile], _metrics) -> list[str]:
        pattern = re.compile(r"^\s*except\s*:", re.M)
        return [sf.path for sf in sfs if sf.extension == ".py" and pattern.search(sf.content)]
