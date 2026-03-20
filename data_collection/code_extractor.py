"""
data_collection/code_extractor.py
===================================
Downloads and structures source code files from a GitHub repository.
Respects size limits, extension filters and max_files cap.
Returns a list of SourceFile objects ready for analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from data_collection.github_client import GitHubClient, RepositoryData


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class SourceFile:
    """
    A single analyzed source file.

    Attributes:
        path        : Relative path inside the repo
        extension   : File extension (e.g. ".py")
        content     : Raw source text
        lines       : All non-empty lines
        line_count  : Total lines (including blank)
        blank_lines : Number of blank lines
        comment_lines: Number of comment lines
        code_lines  : Effective code lines (line_count - blank - comment)
        functions   : Detected function/method names
        classes     : Detected class names
        imports     : Detected import statements
        max_line_len: Longest line character count
        avg_line_len: Average line character count
        tokens      : Rough token count (words)
        depth       : Directory nesting depth
    """
    path: str
    extension: str
    content: str
    lines: list[str] = field(default_factory=list)
    line_count: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    code_lines: int = 0
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    max_line_len: int = 0
    avg_line_len: float = 0.0
    tokens: int = 0
    depth: int = 0


# ── Extractor ────────────────────────────────────────────────────────────────

class CodeExtractor:
    """
    Downloads source files from a repository and produces SourceFile objects.

    Extraction strategy:
        1. Filter tree entries to supported extensions only
        2. Sort by extension frequency (analyze dominant language first)
        3. Download up to max_files files (skipping oversized ones)
        4. Parse each file into a SourceFile dataclass
    """

    # Regex patterns for structural extraction (language-agnostic heuristics)
    _PATTERNS = {
        "functions_py":  re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M),
        "functions_js":  re.compile(r"(?:function\s+([A-Za-z_$][A-Za-z0-9_$]*)|(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s+)?(?:function|\())", re.M),
        "functions_java": re.compile(r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M),
        "classes_py":    re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:(]", re.M),
        "classes_java":  re.compile(r"\b(?:class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M),
        "imports_py":    re.compile(r"^(?:import|from)\s+([^\s]+)", re.M),
        "imports_js":    re.compile(r"^(?:import|require)\s*[\({]?\s*['\"]([^'\"]+)['\"]", re.M),
        "imports_java":  re.compile(r"^import\s+([\w.]+);", re.M),
    }

    def __init__(self, config: Config) -> None:
        self.config = config

    def extract(
        self,
        client: GitHubClient,
        owner: str,
        repo: str,
        repo_data: RepositoryData,
    ) -> list[SourceFile]:
        """
        Main extraction entry point.

        Returns:
            List of parsed SourceFile objects (up to config.max_files).
        """
        cfg = self.config
        supported = set(cfg.supported_extensions)
        max_size = cfg.max_file_size_kb * 1024

        # Filter and sort candidate files
        candidates = [
            entry for entry in repo_data.tree_entries
            if self._get_ext(entry["path"]) in supported
            and entry.get("size", 0) <= max_size
        ]

        # Prioritize dominant language files, then sort by path for consistency
        dominant_ext = self._dominant_extension(repo_data)
        candidates.sort(key=lambda e: (
            0 if self._get_ext(e["path"]) == dominant_ext else 1,
            e["path"],
        ))
        candidates = candidates[: cfg.max_files]

        cfg.log(f"Downloading {len(candidates)} source files...")

        source_files: list[SourceFile] = []
        for entry in candidates:
            path = entry["path"]
            ext = self._get_ext(path)
            content = client.get_file_content(owner, repo, path)
            if content is None:
                continue
            parsed = self._parse(path, ext, content)
            source_files.append(parsed)
            cfg.log(f"  Extracted: {path} ({parsed.line_count} lines)")

        return source_files

    # ── Private ───────────────────────────────────────────────────────────────

    def _parse(self, path: str, ext: str, content: str) -> SourceFile:
        """Parse raw source text into a structured SourceFile."""
        raw_lines = content.splitlines()
        all_lines = raw_lines                               # Keep blank for counts
        non_empty = [l for l in raw_lines if l.strip()]

        blank_lines  = sum(1 for l in raw_lines if not l.strip())
        comment_lines = self._count_comment_lines(raw_lines, ext)
        code_lines = max(0, len(raw_lines) - blank_lines - comment_lines)

        line_lengths = [len(l) for l in non_empty] if non_empty else [0]

        return SourceFile(
            path=path,
            extension=ext,
            content=content,
            lines=non_empty,
            line_count=len(raw_lines),
            blank_lines=blank_lines,
            comment_lines=comment_lines,
            code_lines=code_lines,
            functions=self._extract_functions(content, ext),
            classes=self._extract_classes(content, ext),
            imports=self._extract_imports(content, ext),
            max_line_len=max(line_lengths),
            avg_line_len=sum(line_lengths) / len(line_lengths),
            tokens=len(content.split()),
            depth=path.count("/"),
        )

    def _count_comment_lines(self, lines: list[str], ext: str) -> int:
        """Estimate comment lines using language-specific markers."""
        markers = self.config.comment_markers.get(ext, ["//", "#"])
        single_line_markers = [m for m in markers if len(m) <= 2]
        count = 0
        in_block = False
        for line in lines:
            stripped = line.strip()
            if in_block:
                count += 1
                if "*/" in stripped or '"""' in stripped or "'''" in stripped:
                    in_block = False
                continue
            if any(stripped.startswith(m) for m in single_line_markers if m):
                count += 1
            elif stripped.startswith('"""') or stripped.startswith("'''"):
                count += 1
                # Single-line docstring
                if stripped.count('"""') < 2 and stripped.count("'''") < 2:
                    in_block = True
            elif stripped.startswith("/*"):
                count += 1
                if "*/" not in stripped:
                    in_block = True
        return count

    def _extract_functions(self, content: str, ext: str) -> list[str]:
        """Extract function/method names."""
        names: list[str] = []
        if ext == ".py":
            names = self._PATTERNS["functions_py"].findall(content)
        elif ext in (".js", ".ts"):
            matches = self._PATTERNS["functions_js"].findall(content)
            names = [m[0] or m[1] for m in matches if m[0] or m[1]]
        elif ext in (".java", ".kt"):
            names = self._PATTERNS["functions_java"].findall(content)
        else:
            # Generic: look for word followed by "("
            names = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", content)
        return list(dict.fromkeys(names))[:100]   # Deduplicate, cap at 100

    def _extract_classes(self, content: str, ext: str) -> list[str]:
        """Extract class/interface names."""
        if ext == ".py":
            return self._PATTERNS["classes_py"].findall(content)
        if ext in (".java", ".kt", ".cs"):
            return self._PATTERNS["classes_java"].findall(content)
        # JS/TS
        return re.findall(r"\bclass\s+([A-Za-z_$][A-Za-z0-9_$]*)", content)

    def _extract_imports(self, content: str, ext: str) -> list[str]:
        """Extract import/require statements."""
        if ext == ".py":
            return self._PATTERNS["imports_py"].findall(content)
        if ext in (".js", ".ts"):
            return self._PATTERNS["imports_js"].findall(content)
        if ext in (".java", ".kt"):
            return self._PATTERNS["imports_java"].findall(content)
        return []

    @staticmethod
    def _get_ext(path: str) -> str:
        if "." not in path.rsplit("/", 1)[-1]:
            return ""
        return "." + path.rsplit(".", 1)[-1].lower()

    @staticmethod
    def _dominant_extension(repo_data: RepositoryData) -> str:
        """Return the file extension associated with the primary language."""
        lang_to_ext = {
            "Python": ".py", "JavaScript": ".js", "TypeScript": ".ts",
            "Java": ".java", "Go": ".go", "Ruby": ".rb",
            "C++": ".cpp", "C": ".c", "C#": ".cs",
            "PHP": ".php", "Rust": ".rs", "Kotlin": ".kt", "Swift": ".swift",
        }
        return lang_to_ext.get(repo_data.language or "", ".py")
