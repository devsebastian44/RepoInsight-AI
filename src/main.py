#!/usr/bin/env python3
"""
AI Repository Intelligence Analyzer
=====================================
Entry point for the CLI tool. Orchestrates all modules to produce
a full intelligence report about a GitHub repository.

Usage:
    python main.py <github_repo_url> [options]

Examples:
    python main.py https://github.com/user/repo
    python main.py https://github.com/user/repo --output json
    python main.py https://github.com/user/repo --output json --save report.json
    python main.py https://github.com/user/repo --llm --llm-provider openai
"""

import sys
import argparse
import json
import time
from pathlib import Path

from config import Config
from data_collection.github_client import GitHubClient
from data_collection.code_extractor import CodeExtractor
from analysis.feature_engineering import FeatureEngineer
from analysis.code_analyzer import CodeAnalyzer
from analysis.pattern_detector import PatternDetector
from ml_model.classifier import RepositoryClassifier
from reporting.report_builder import ReportBuilder
from reporting.formatter import ReportFormatter


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ai-repo-analyzer",
        description="🤖 AI Repository Intelligence Analyzer — Analyze GitHub repos with ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py https://github.com/user/repo
  python main.py https://github.com/user/repo --output json --save report.json
  python main.py https://github.com/user/repo --llm --llm-provider openai
        """,
    )

    parser.add_argument(
        "repo_url",
        type=str,
        help="Full GitHub repository URL (e.g., https://github.com/user/repo)",
    )
    parser.add_argument(
        "--output",
        choices=["console", "json", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="FILE",
        help="Save JSON report to file (e.g., --save report.json)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help="Enable LLM-powered deep code analysis (requires API key)",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "mock"],
        default="mock",
        help="LLM provider to use (default: mock)",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        metavar="TOKEN",
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=50,
        help="Maximum number of files to analyze (default: 50)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    return parser.parse_args()


def validate_github_url(url: str) -> tuple[str, str]:
    """
    Validate and parse a GitHub URL into (owner, repo) tuple.

    Raises:
        SystemExit: If the URL is not a valid GitHub repo URL.
    """
    import re

    pattern = r"(?:https?://)?github\.com/([^/]+)/([^/\s]+?)(?:\.git)?$"
    match = re.match(pattern, url.strip())
    if not match:
        print(f"❌ Invalid GitHub URL: {url}")
        print("   Expected format: https://github.com/owner/repo")
        sys.exit(1)

    owner, repo = match.group(1), match.group(2)
    return owner, repo


def print_banner() -> None:
    """Print the CLI banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          🤖  AI Repository Intelligence Analyzer             ║
║              Powered by ML + Code Analysis                   ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_progress(step: int, total: int, label: str) -> None:
    """Print a simple progress indicator."""
    bar_len = 30
    filled = int(bar_len * step / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = int(100 * step / total)
    print(f"\r  [{bar}] {pct:3d}%  {label:<40}", end="", flush=True)
    if step == total:
        print()


def run_analysis(args: argparse.Namespace) -> dict:
    """
    Main orchestration pipeline.

    Steps:
        1. Collect data from GitHub API
        2. Extract and parse source files
        3. Run static + heuristic analysis
        4. Engineer features
        5. Run ML classifier
        6. (Optional) LLM deep analysis
        7. Build and return report
    """
    total_steps = 7 if args.llm else 6
    step = 0

    owner, repo_name = validate_github_url(args.repo_url)
    config = Config(
        github_token=args.github_token,
        max_files=args.max_files,
        verbose=args.verbose,
        llm_provider=args.llm_provider,
    )

    # ── Step 1: GitHub Data Collection ──────────────────────────────────────
    step += 1
    print_progress(step, total_steps, "Fetching repository metadata...")

    github_client = GitHubClient(config)
    repo_data = github_client.fetch_repository(owner, repo_name)

    # ── Step 2: Source Code Extraction ──────────────────────────────────────
    step += 1
    print_progress(step, total_steps, "Extracting source files...")

    extractor = CodeExtractor(config)
    source_files = extractor.extract(github_client, owner, repo_name, repo_data)

    # ── Step 3: Code Analysis (static + heuristic) ──────────────────────────
    step += 1
    print_progress(step, total_steps, "Running code analysis...")

    analyzer = CodeAnalyzer(config)
    code_metrics = analyzer.analyze(source_files)

    # ── Step 4: Pattern Detection ────────────────────────────────────────────
    step += 1
    print_progress(step, total_steps, "Detecting design patterns & code smells...")

    detector = PatternDetector(config)
    patterns = detector.detect(source_files, code_metrics)

    # ── Step 5: Feature Engineering ──────────────────────────────────────────
    step += 1
    print_progress(step, total_steps, "Engineering ML features...")

    engineer = FeatureEngineer(config)
    feature_vector = engineer.build(repo_data, source_files, code_metrics, patterns)

    # ── Step 6: ML Classification ─────────────────────────────────────────────
    step += 1
    print_progress(step, total_steps, "Running ML classifier...")

    classifier = RepositoryClassifier(config)
    classification = classifier.predict(feature_vector)

    # ── Step 7 (Optional): LLM Deep Analysis ──────────────────────────────────
    llm_insights = {}
    if args.llm:
        step += 1
        print_progress(step, total_steps, "Running LLM deep analysis...")
        from llm.llm_analyzer import LLMAnalyzer

        llm = LLMAnalyzer(config)
        llm_insights = llm.analyze(source_files, code_metrics, patterns, classification)

    print_progress(total_steps, total_steps, "Generating report...")

    # ── Build Final Report ────────────────────────────────────────────────────
    builder = ReportBuilder(config)
    report = builder.build(
        repo_data=repo_data,
        source_files=source_files,
        code_metrics=code_metrics,
        patterns=patterns,
        feature_vector=feature_vector,
        classification=classification,
        llm_insights=llm_insights,
    )

    return report


def main() -> None:
    args = parse_arguments()
    print_banner()

    print(f"  📦 Repository : {args.repo_url}")
    print(f"  🔬 LLM Analysis: {'Enabled (' + args.llm_provider + ')' if args.llm else 'Disabled'}")
    print(f"  📄 Output      : {args.output}")
    print()

    start_time = time.time()

    try:
        report = run_analysis(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n\n❌ Analysis failed: {exc}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n  ✅ Analysis completed in {elapsed:.1f}s\n")

    # ── Output ────────────────────────────────────────────────────────────────
    formatter = ReportFormatter()

    if args.output in ("console", "both"):
        formatter.print_console(report)

    if args.output in ("json", "both"):
        json_output = json.dumps(report, indent=2, ensure_ascii=False)
        if args.save:
            path = Path(args.save)
            path.write_text(json_output, encoding="utf-8")
            print(f"\n  💾 Report saved to: {path.resolve()}")
        else:
            print("\n── JSON Report ──────────────────────────────────────────────")
            print(json_output)


if __name__ == "__main__":
    main()
