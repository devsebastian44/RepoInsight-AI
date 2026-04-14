"""
config.py
=========
Centralized configuration for AI Repository Intelligence Analyzer.
Reads from environment variables with sensible defaults.
All modules receive a Config instance for consistent settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Central configuration object.

    All settings come from constructor kwargs (which in turn can be
    sourced from CLI args or environment variables).
    """

    # ── GitHub ────────────────────────────────────────────────────────────────
    github_token: str | None = field(default=None)
    github_api_base: str = "https://api.github.com"
    max_files: int = 50
    max_file_size_kb: int = 500  # Skip files larger than this
    request_timeout: int = 15  # HTTP timeout in seconds

    # ── Analysis ─────────────────────────────────────────────────────────────
    supported_extensions: tuple = (
        ".py",
        ".js",
        ".ts",
        ".java",
        ".go",
        ".rb",
        ".cpp",
        ".c",
        ".cs",
        ".php",
        ".rs",
        ".kt",
        ".swift",
    )
    comment_markers: dict = field(
        default_factory=lambda: {
            ".py": ["#", '"""', "'''"],
            ".js": ["//", "/*"],
            ".ts": ["//", "/*"],
            ".java": ["//", "/*"],
            ".go": ["//", "/*"],
            ".rb": ["#"],
            ".cpp": ["//", "/*"],
            ".c": ["//", "/*"],
            ".cs": ["//", "/*"],
            ".php": ["//", "#", "/*"],
            ".rs": ["//", "/*"],
            ".kt": ["//", "/*"],
            ".swift": ["//", "/*"],
        }
    )

    # ── ML Model ──────────────────────────────────────────────────────────────
    model_random_state: int = 42
    model_n_estimators: int = 200
    model_path: str = "data/rf_classifier.pkl"
    scaler_path: str = "data/feature_scaler.pkl"

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: str = "mock"  # openai | anthropic | mock
    openai_api_key: str | None = field(default=None)
    anthropic_api_key: str | None = field(default=None)
    llm_model_openai: str = "gpt-4o"
    llm_model_anthropic: str = "claude-3-5-sonnet-20241022"
    llm_max_tokens: int = 2048
    llm_sample_lines: int = 150  # Lines of code sent to LLM

    # ── Reporting ─────────────────────────────────────────────────────────────
    report_version: str = "1.0.0"
    tool_name: str = "AI Repository Intelligence Analyzer"

    # ── Misc ─────────────────────────────────────────────────────────────────
    verbose: bool = False

    def __post_init__(self) -> None:
        """Resolve values from environment variables as fallbacks."""
        if self.github_token is None:
            self.github_token = os.getenv("GITHUB_TOKEN")

        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    @property
    def github_headers(self) -> dict:
        """Headers for GitHub API requests."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return headers

    @property
    def is_authenticated(self) -> bool:
        """Whether a GitHub token has been provided."""
        return self.github_token is not None

    def log(self, message: str) -> None:
        """Verbose-only print."""
        if self.verbose:
            print(f"  [DEBUG] {message}")
