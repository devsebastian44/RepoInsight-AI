"""
data_collection/github_client.py
==================================
Thin wrapper around the GitHub REST API v3.
Handles authentication, rate-limit awareness, pagination,
and structured data models for repository metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests

from config import Config

# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class RepositoryData:
    """Structured representation of GitHub repository metadata."""

    owner: str
    name: str
    full_name: str
    description: str | None
    url: str
    stars: int
    forks: int
    watchers: int
    open_issues: int
    default_branch: str
    created_at: str
    updated_at: str
    pushed_at: str
    size_kb: int
    language: str | None
    languages: dict[str, int]  # {"Python": 12400, "Shell": 340}
    topics: list[str]
    has_wiki: bool
    has_issues: bool
    has_projects: bool
    license_name: str | None
    contributors_count: int
    commits_count: int
    branches_count: int
    releases_count: int
    commit_history: list[dict]  # Last N commits (date, message, author)
    tree_entries: list[dict]  # Flat tree of all paths
    readme_content: str | None
    has_ci: bool
    has_tests: bool
    has_docker: bool
    has_requirements: bool


@dataclass
class FileEntry:
    """A single file entry from the repo tree."""

    path: str
    sha: str
    size: int
    extension: str = field(init=False)

    def __post_init__(self) -> None:
        self.extension = "." + self.path.rsplit(".", 1)[-1] if "." in self.path else ""


# ── Client ───────────────────────────────────────────────────────────────────


class GitHubClient:
    """
    GitHub REST API v3 client.

    Rate limits:
        Unauthenticated  : 60 req/hour
        Authenticated    : 5 000 req/hour  (PAT)
    """

    # Heuristic path patterns
    _CI_PATHS = {
        ".github/workflows",
        ".travis.yml",
        ".circleci",
        "Jenkinsfile",
        "azure-pipelines.yml",
        ".gitlab-ci.yml",
        "bitbucket-pipelines.yml",
        "Makefile",
    }
    _TEST_DIRS = {
        "test",
        "tests",
        "spec",
        "specs",
        "__tests__",
        "test_",
        "_test",
        ".test.",
        ".spec.",
    }
    _DOCKER_FILES = {"Dockerfile", "docker-compose.yml", "docker-compose.yaml", ".dockerignore"}
    _REQUIREMENT_FILES = {
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "Pipfile",
        "package.json",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "Cargo.toml",
    }

    def __init__(self, config: Config) -> None:
        self.config = config
        self._session = requests.Session()
        self._session.headers.update(config.github_headers)

    # ── Public ────────────────────────────────────────────────────────────────

    def fetch_repository(self, owner: str, repo: str) -> RepositoryData:
        """
        Collect all metadata needed for analysis.

        Makes ~8-12 API calls depending on repository size.
        """
        cfg = self.config

        cfg.log(f"Fetching repo metadata: {owner}/{repo}")
        meta = self._get(f"/repos/{owner}/{repo}")

        cfg.log("Fetching languages...")
        languages = self._get(f"/repos/{owner}/{repo}/languages") or {}

        cfg.log("Fetching topics...")
        topics_resp = self._get(f"/repos/{owner}/{repo}/topics") or {}
        topics = topics_resp.get("names", [])

        cfg.log("Fetching contributors count...")
        contributors = self._get_count(f"/repos/{owner}/{repo}/contributors?per_page=1&anon=true")

        cfg.log("Fetching commits (last 100)...")
        commits_raw = self._get_paginated(f"/repos/{owner}/{repo}/commits", per_page=100, max_pages=1)
        commits_count = self._get_count(f"/repos/{owner}/{repo}/commits?per_page=1")

        cfg.log("Fetching branches count...")
        branches_count = self._get_count(f"/repos/{owner}/{repo}/branches?per_page=1")

        cfg.log("Fetching releases count...")
        releases_count = self._get_count(f"/repos/{owner}/{repo}/releases?per_page=1")

        cfg.log("Fetching repository tree...")
        default_branch = meta.get("default_branch", "main")
        tree = self._fetch_tree(owner, repo, default_branch)

        cfg.log("Fetching README...")
        readme = self._fetch_readme(owner, repo)

        # Parse commit history
        commit_history = [
            {
                "sha": c.get("sha", "")[:8],
                "message": (c.get("commit", {}).get("message", "") or "")[:120].split("\n")[0],
                "author": c.get("commit", {}).get("author", {}).get("name", "unknown"),
                "date": c.get("commit", {}).get("author", {}).get("date", ""),
            }
            for c in commits_raw[:30]
        ]

        # Heuristic flags from file tree
        tree_paths = {e.get("path", "") for e in tree}
        has_ci = any(ci in p for ci in self._CI_PATHS for p in tree_paths)
        has_tests = any(any(t in p.lower() for t in self._TEST_DIRS) for p in tree_paths)
        has_docker = any(d in tree_paths for d in self._DOCKER_FILES)
        has_requirements = any(r in tree_paths for r in self._REQUIREMENT_FILES)

        license_info = meta.get("license") or {}
        license_name = license_info.get("spdx_id") if isinstance(license_info, dict) else None

        return RepositoryData(
            owner=owner,
            name=repo,
            full_name=meta.get("full_name", f"{owner}/{repo}"),
            description=meta.get("description"),
            url=meta.get("html_url", f"https://github.com/{owner}/{repo}"),
            stars=meta.get("stargazers_count", 0),
            forks=meta.get("forks_count", 0),
            watchers=meta.get("watchers_count", 0),
            open_issues=meta.get("open_issues_count", 0),
            default_branch=default_branch,
            created_at=meta.get("created_at", ""),
            updated_at=meta.get("updated_at", ""),
            pushed_at=meta.get("pushed_at", ""),
            size_kb=meta.get("size", 0),
            language=meta.get("language"),
            languages=languages,
            topics=topics,
            has_wiki=meta.get("has_wiki", False),
            has_issues=meta.get("has_issues", False),
            has_projects=meta.get("has_projects", False),
            license_name=license_name,
            contributors_count=contributors,
            commits_count=commits_count,
            branches_count=branches_count,
            releases_count=releases_count,
            commit_history=commit_history,
            tree_entries=tree,
            readme_content=readme,
            has_ci=has_ci,
            has_tests=has_tests,
            has_docker=has_docker,
            has_requirements=has_requirements,
        )

    def get_file_content(self, owner: str, repo: str, path: str) -> str | None:
        """Download and decode a single file's content (UTF-8)."""
        import base64

        try:
            data = self._get(f"/repos/{owner}/{repo}/contents/{path}")
            if data and data.get("encoding") == "base64":
                raw = data.get("content", "").replace("\n", "")
                return base64.b64decode(raw).decode("utf-8", errors="replace")
        except Exception as exc:
            self.config.log(f"Could not fetch {path}: {exc}")
        return None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get(self, endpoint: str) -> dict | list | None:
        """GET request with retry on rate-limit (429 / 403)."""
        url = self.config.github_api_base + endpoint
        for attempt in range(3):
            try:
                resp = self._session.get(url, timeout=self.config.request_timeout)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (403, 429):
                    reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                    wait = max(reset - int(time.time()), 5)
                    self.config.log(f"Rate limited — waiting {wait}s (attempt {attempt + 1})")
                    time.sleep(min(wait, 30))
                    continue
                if resp.status_code == 404:
                    self.config.log(f"404 Not Found: {endpoint}")
                    return None
                self.config.log(f"HTTP {resp.status_code} for {endpoint}")
                return None
            except requests.RequestException as exc:
                self.config.log(f"Request error: {exc}")
                time.sleep(2**attempt)
        return None

    def _get_paginated(self, endpoint: str, per_page: int = 100, max_pages: int = 5) -> list:
        """Collect all pages of a paginated endpoint."""
        results = []
        sep = "&" if "?" in endpoint else "?"
        for page in range(1, max_pages + 1):
            data = self._get(f"{endpoint}{sep}per_page={per_page}&page={page}")
            if not data or not isinstance(data, list):
                break
            results.extend(data)
            if len(data) < per_page:
                break
        return results

    def _get_count(self, endpoint: str) -> int:
        """
        Estimate total count via Link header on a per_page=1 request.
        Falls back to len() of items if no Link header.
        """
        url = self.config.github_api_base + endpoint
        try:
            resp = self._session.get(url, timeout=self.config.request_timeout)
            if resp.status_code != 200:
                return 0
            link = resp.headers.get("Link", "")
            if 'rel="last"' in link:
                import re

                m = re.search(r'page=(\d+)>;\s*rel="last"', link)
                if m:
                    return int(m.group(1))
            data = resp.json()
            return len(data) if isinstance(data, list) else 0
        except Exception:
            return 0

    def _fetch_tree(self, owner: str, repo: str, branch: str) -> list[dict]:
        """Fetch the full recursive tree for a branch."""
        data = self._get(f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
        if data and "tree" in data:
            return [
                {"path": e["path"], "sha": e["sha"], "size": e.get("size", 0), "type": e["type"]} for e in data["tree"] if e.get("type") == "blob"
            ]
        return []

    def _fetch_readme(self, owner: str, repo: str) -> str | None:
        """Fetch README content (any variant)."""
        import base64

        data = self._get(f"/repos/{owner}/{repo}/readme")
        if data and data.get("encoding") == "base64":
            raw = data.get("content", "").replace("\n", "")
            try:
                return base64.b64decode(raw).decode("utf-8", errors="replace")[:3000]
            except Exception:
                pass
        return None
