"""
llm/llm_analyzer.py
=====================
Optional LLM-powered deep code analysis.

Providers:
    openai     → GPT-4o via OpenAI Python SDK
    anthropic  → Claude 3.5 Sonnet via Anthropic Python SDK
    mock       → Deterministic fake responses for CI / offline use

The LLM is given:
  - A representative code sample (up to llm_sample_lines lines)
  - Detected patterns and metrics summary
  - A structured prompt requesting a JSON response

Returns LLMInsights with text feedback and ranked improvement suggestions.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field

from analysis.code_analyzer import CodeMetrics
from analysis.pattern_detector import PatternReport
from config import Config
from data_collection.code_extractor import SourceFile
from ml_model.classifier import ClassificationResult

# ── Result Model ──────────────────────────────────────────────────────────────


@dataclass
class LLMInsights:
    """Structured output from the LLM code review."""

    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    architectural_observations: list[str] = field(default_factory=list)
    security_concerns: list[str] = field(default_factory=list)
    scalability_notes: str = ""
    raw_response: str = ""
    provider: str = "none"
    error: str | None = None


# ── Analyzer ──────────────────────────────────────────────────────────────────


class LLMAnalyzer:
    """Routes to the configured LLM provider and parses the response."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def analyze(
        self,
        source_files: list[SourceFile],
        metrics: CodeMetrics,
        patterns: PatternReport,
        classification: ClassificationResult,
    ) -> LLMInsights:
        """Run LLM analysis and return structured insights."""
        code_sample = self._build_code_sample(source_files)
        context_summary = self._build_context_summary(metrics, patterns, classification)
        prompt = self._build_prompt(code_sample, context_summary)

        provider = self.config.llm_provider

        if provider == "openai":
            return self._call_openai(prompt)
        elif provider == "anthropic":
            return self._call_anthropic(prompt)
        else:
            return self._mock_response(metrics, patterns, classification)

    # ── Prompt Construction ────────────────────────────────────────────────────

    def _build_prompt(self, code_sample: str, context_summary: str) -> str:
        return textwrap.dedent(f"""
        You are a senior software engineer performing a code review of a GitHub repository.

        ## Context
        {context_summary}

        ## Code Sample
        ```
        {code_sample}
        ```

        ## Task
        Analyze this repository's code quality, architecture, and engineering practices.
        Respond ONLY with a valid JSON object matching this exact schema:

        {{
          "summary": "2-3 sentence overall assessment",
          "strengths": ["strength 1", "strength 2", "..."],
          "improvements": ["improvement 1", "improvement 2", "..."],
          "architectural_observations": ["observation 1", "..."],
          "security_concerns": ["concern 1 or 'none identified'"],
          "scalability_notes": "1-2 sentences on scalability"
        }}

        Be specific, actionable, and reference actual patterns or code you observed.
        Limit each list to 5 items maximum.
        """).strip()

    def _build_code_sample(self, source_files: list[SourceFile]) -> str:
        """Select most representative file(s) for the LLM."""
        if not source_files:
            return "# No source files available"

        # Pick the largest file (most representative)
        sorted_files = sorted(source_files, key=lambda sf: sf.code_lines, reverse=True)
        sample_lines: list[str] = []
        budget = self.config.llm_sample_lines

        for sf in sorted_files[:3]:
            header = f"# File: {sf.path}"
            lines = sf.content.splitlines()[: budget // 3]
            sample_lines.append(header)
            sample_lines.extend(lines)
            sample_lines.append("")
            if len(sample_lines) >= budget:
                break

        return "\n".join(sample_lines[:budget])

    def _build_context_summary(
        self,
        metrics: CodeMetrics,
        patterns: PatternReport,
        classification: ClassificationResult,
    ) -> str:
        pattern_names = ", ".join(p.name for p in patterns.design_patterns) or "none"
        practice_names = ", ".join(p.name for p in patterns.best_practices[:6]) or "none"
        smell_names = ", ".join(s.name for s in patterns.code_smells) or "none"

        return textwrap.dedent(f"""
        - Predicted level: {classification.level} (confidence: {classification.confidence:.0%})
        - Quality score: {metrics.quality_score:.0f}/100
        - Total files: {metrics.total_files}, lines: {metrics.total_lines}
        - Comment density: {metrics.comment_density:.1%}
        - Design patterns detected: {pattern_names}
        - Good practices detected: {practice_names}
        - Code smells detected: {smell_names}
        """).strip()

    # ── Provider Calls ─────────────────────────────────────────────────────────

    def _call_openai(self, prompt: str) -> LLMInsights:
        try:
            import openai

            client = openai.OpenAI(api_key=self.config.openai_api_key)
            response = client.chat.completions.create(
                model=self.config.llm_model_openai,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.llm_max_tokens,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            return self._parse_response(raw, provider="openai")
        except ImportError:
            return LLMInsights(error="openai package not installed. Run: pip install openai")
        except Exception as exc:
            return LLMInsights(error=f"OpenAI error: {exc}", provider="openai")

    def _call_anthropic(self, prompt: str) -> LLMInsights:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            response = client.messages.create(
                model=self.config.llm_model_anthropic,
                max_tokens=self.config.llm_max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text if response.content else ""
            return self._parse_response(raw, provider="anthropic")
        except ImportError:
            return LLMInsights(error="anthropic package not installed. Run: pip install anthropic")
        except Exception as exc:
            return LLMInsights(error=f"Anthropic error: {exc}", provider="anthropic")

    def _mock_response(
        self,
        metrics: CodeMetrics,
        patterns: PatternReport,
        classification: ClassificationResult,
    ) -> LLMInsights:
        """Deterministic mock response based on metric values."""
        level = classification.level
        qs = metrics.quality_score

        strengths = []
        improvements = []

        if metrics.docstring_ratio > 0.5:
            strengths.append("Good documentation coverage with docstrings present in most files.")
        if metrics.type_hint_ratio > 0.4:
            strengths.append("Type annotations improve code readability and IDE support.")
        if patterns.design_patterns:
            names = ", ".join(p.name for p in patterns.design_patterns[:2])
            strengths.append(f"Effective use of {names} design pattern(s).")
        if metrics.comment_density > 0.15:
            strengths.append("Healthy comment density aids code comprehension.")
        if not strengths:
            strengths.append("Code structure shows basic organization and separation.")

        if metrics.duplication_score > 20:
            improvements.append("Reduce code duplication — extract shared logic into utilities.")
        if metrics.avg_complexity > 50:
            improvements.append("Simplify complex functions; consider breaking them into smaller units.")
        if metrics.docstring_ratio < 0.3:
            improvements.append("Add docstrings to public functions and classes.")
        if not any(p.name == "Unit Tests" for p in patterns.best_practices):
            improvements.append("Introduce unit tests to improve reliability and catch regressions.")
        if not any(p.name == "Logging" for p in patterns.best_practices):
            improvements.append("Replace print() statements with a structured logging framework.")
        if len(improvements) < 2:
            improvements.append("Consider adding type hints for better static analysis support.")

        return LLMInsights(
            summary=(
                f"This appears to be a {level}-level repository with a quality score of "
                f"{qs:.0f}/100. "
                + (
                    "The codebase demonstrates solid engineering practices."
                    if qs > 65
                    else "There are opportunities to improve code quality and maintainability."
                )
            ),
            strengths=strengths[:5],
            improvements=improvements[:5],
            architectural_observations=[
                f"Project has {metrics.total_files} source files across multiple modules.",
                f"{'Good' if metrics.modularity_score > 60 else 'Limited'} modularization with {metrics.total_classes} classes detected.",
            ],
            security_concerns=["No obvious secrets hardcoded (surface-level scan only)."],
            scalability_notes=(
                "Consider dependency injection and interface-based design for easier scaling."
                if level != "Senior"
                else "Architecture shows good separation of concerns for horizontal scaling."
            ),
            provider="mock",
        )

    # ── Response Parsing ──────────────────────────────────────────────────────

    def _parse_response(self, raw: str, provider: str) -> LLMInsights:
        """Parse JSON response from the LLM into an LLMInsights object."""
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
            data = json.loads(cleaned)
            return LLMInsights(
                summary=data.get("summary", ""),
                strengths=data.get("strengths", []),
                improvements=data.get("improvements", []),
                architectural_observations=data.get("architectural_observations", []),
                security_concerns=data.get("security_concerns", []),
                scalability_notes=data.get("scalability_notes", ""),
                raw_response=raw,
                provider=provider,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            return LLMInsights(
                summary=raw[:500] if raw else "Could not parse LLM response.",
                raw_response=raw,
                provider=provider,
                error=f"JSON parse error: {exc}",
            )
