"""
reporting/report_builder.py
=============================
Assembles all analysis results into a single, schema-compliant
report dictionary (JSON-serializable).

The output schema is documented in the module docstring and
matches what the README promises.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from analysis.code_analyzer import CodeMetrics
from analysis.feature_engineering import FeatureVector
from analysis.pattern_detector import PatternReport
from config import Config
from data_collection.code_extractor import SourceFile
from data_collection.github_client import RepositoryData
from ml_model.classifier import ClassificationResult


class ReportBuilder:
    """
    Constructs the final report dict from all pipeline stages.

    Output schema (top-level keys):
        meta           — tool info, timestamp, version
        repository     — GitHub metadata
        classification — ML prediction result
        code_metrics   — static analysis numbers
        patterns       — design patterns detected
        practices      — best practices detected
        smells         — code smells detected
        llm_insights   — (optional) LLM feedback
        recommendations— actionable improvement suggestions
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def build(
        self,
        repo_data: RepositoryData,
        source_files: list[SourceFile],
        code_metrics: CodeMetrics,
        patterns: PatternReport,
        feature_vector: FeatureVector,
        classification: ClassificationResult,
        llm_insights: dict,
    ) -> dict[str, Any]:
        """Assemble and return the complete report dictionary."""

        # ── Determine quality label ────────────────────────────────────────────
        score = classification.composite_score
        if score >= 80:
            quality_label = "Alta"
        elif score >= 60:
            quality_label = "Media-Alta"
        elif score >= 40:
            quality_label = "Media"
        else:
            quality_label = "Baja"

        # ── Build recommendations ──────────────────────────────────────────────
        recommendations = self._build_recommendations(code_metrics, patterns, classification)

        # ── Build best practices list ──────────────────────────────────────────
        buenas_practicas = [{"nombre": p.name, "descripcion": p.description, "archivos": p.files} for p in patterns.best_practices]

        # ── Build smells list ──────────────────────────────────────────────────
        code_smells = [
            {
                "nombre": s.name,
                "severidad": s.severity,
                "descripcion": s.description,
                "archivos_afectados": s.files,
            }
            for s in patterns.code_smells
        ]

        # ── Build design patterns list ─────────────────────────────────────────
        design_patterns = [
            {
                "patron": p.name,
                "confianza": p.confidence,
                "descripcion": p.description,
                "archivos": p.files,
            }
            for p in patterns.design_patterns
        ]

        # ── Language breakdown ────────────────────────────────────────────────
        total_bytes = sum(repo_data.languages.values()) or 1
        language_pct = {
            lang: round(bytes_ / total_bytes * 100, 1) for lang, bytes_ in sorted(repo_data.languages.items(), key=lambda x: x[1], reverse=True)
        }

        report = {
            # ── Top-level summary (matches README spec) ─────────────────────
            "nivel": classification.level,
            "calidad": quality_label,
            "score": classification.composite_score,
            "buenas_practicas": [p.name for p in patterns.best_practices],
            "mejoras": recommendations["quick_wins"],
            # ── Metadata ─────────────────────────────────────────────────────
            "meta": {
                "herramienta": self.config.tool_name,
                "version": self.config.report_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "repositorio_analizado": repo_data.url,
            },
            # ── Repository ────────────────────────────────────────────────────
            "repositorio": {
                "nombre": repo_data.full_name,
                "descripcion": repo_data.description,
                "url": repo_data.url,
                "estrellas": repo_data.stars,
                "forks": repo_data.forks,
                "lenguaje_principal": repo_data.language,
                "lenguajes": language_pct,
                "topics": repo_data.topics,
                "licencia": repo_data.license_name,
                "creado": repo_data.created_at,
                "ultimo_push": repo_data.pushed_at,
                "commits_totales": repo_data.commits_count,
                "colaboradores": repo_data.contributors_count,
                "branches": repo_data.branches_count,
                "releases": repo_data.releases_count,
                "tiene_ci": repo_data.has_ci,
                "tiene_tests": repo_data.has_tests,
                "tiene_docker": repo_data.has_docker,
                "tiene_readme": repo_data.readme_content is not None,
                "tiene_licencia": repo_data.license_name is not None,
            },
            # ── Classification ────────────────────────────────────────────────
            "clasificacion": {
                "nivel": classification.level,
                "confianza": classification.confidence,
                "probabilidades": classification.probabilities,
                "score_compuesto": classification.composite_score,
                "top_features": dict(list(classification.feature_importances.items())[:5]),
            },
            # ── Code Metrics ──────────────────────────────────────────────────
            "metricas_codigo": {
                "archivos_analizados": code_metrics.total_files,
                "lineas_totales": code_metrics.total_lines,
                "lineas_codigo": code_metrics.total_code_lines,
                "lineas_comentario": code_metrics.total_comment_lines,
                "densidad_comentarios": round(code_metrics.comment_density, 3),
                "funciones_totales": code_metrics.total_functions,
                "clases_totales": code_metrics.total_classes,
                "longitud_promedio_archivo": round(code_metrics.avg_file_length, 1),
                "longitud_promedio_funcion": round(code_metrics.avg_function_length, 1),
                "complejidad_promedio": round(code_metrics.avg_complexity, 1),
                "score_duplicacion": round(code_metrics.duplication_score, 1),
                "score_modularidad": round(code_metrics.modularity_score, 1),
                "score_nombres": round(code_metrics.naming_score, 1),
                "ratio_docstrings": round(code_metrics.docstring_ratio, 3),
                "ratio_type_hints": round(code_metrics.type_hint_ratio, 3),
                "funciones_largas": code_metrics.long_method_count,
                "numeros_magicos": code_metrics.magic_number_count,
                "score_calidad": code_metrics.quality_score,
            },
            # ── Patterns ──────────────────────────────────────────────────────
            "patrones_diseno": design_patterns,
            # ── Practices ─────────────────────────────────────────────────────
            "buenas_practicas_detalle": buenas_practicas,
            # ── Smells ────────────────────────────────────────────────────────
            "code_smells": code_smells,
            # ── LLM Insights ──────────────────────────────────────────────────
            "analisis_llm": self._serialize_llm(llm_insights),
            # ── Recommendations ───────────────────────────────────────────────
            "recomendaciones": recommendations,
        }

        return report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_recommendations(
        self,
        metrics: CodeMetrics,
        patterns: PatternReport,
        classification: ClassificationResult,
    ) -> dict[str, list[str]]:
        """Generate tiered, actionable recommendations."""
        quick_wins: list[str] = []
        medium_term: list[str] = []
        architectural: list[str] = []

        level = classification.level

        # Quick wins
        if metrics.docstring_ratio < 0.3:
            quick_wins.append("Agregar docstrings a funciones y clases públicas.")
        if metrics.comment_density < 0.08:
            quick_wins.append("Incrementar la densidad de comentarios para mejorar la legibilidad.")
        if any(s.name == "print() Debugging" for s in patterns.code_smells):
            quick_wins.append("Reemplazar print() por un logger configurado (logging.getLogger).")
        if any(s.name == "TODO / FIXME" for s in patterns.code_smells):
            quick_wins.append("Resolver o crear issues para los TODO/FIXME pendientes.")
        if any(s.name == "Bare Except" for s in patterns.code_smells):
            quick_wins.append("Cambiar 'except:' por 'except ExceptionType:' específico.")
        if metrics.magic_number_count > 10:
            quick_wins.append("Extraer números mágicos a constantes con nombres descriptivos.")
        if not any(p.name == "Type Annotations" for p in patterns.best_practices):
            quick_wins.append("Añadir type hints (Python) para mejorar el análisis estático.")

        # Medium term
        if not any(p.name == "Unit Tests" for p in patterns.best_practices):
            medium_term.append("Implementar suite de pruebas unitarias con pytest o unittest.")
        if not any(p.name == "Logging" for p in patterns.best_practices):
            medium_term.append("Integrar sistema de logging estructurado (loguru o structlog).")
        if metrics.duplication_score > 20:
            medium_term.append("Refactorizar código duplicado extraiendo funciones/clases reutilizables.")
        if any(s.name == "Long Method" for s in patterns.code_smells):
            medium_term.append("Dividir métodos largos siguiendo el Single Responsibility Principle.")
        if not any(p.name == "Environment Variables" for p in patterns.best_practices):
            medium_term.append("Mover configuración a variables de entorno (.env + python-dotenv).")
        if not any(p.name == "Linting Config" for p in patterns.best_practices):
            medium_term.append("Configurar linter/formatter (ruff + black para Python) y pre-commit hooks.")

        # Architectural
        if level in ("Junior", "Mid-level"):
            if not any(p.name == "Dependency Injection" for p in patterns.design_patterns):
                architectural.append("Adoptar inyección de dependencias para mejorar la testabilidad.")
            architectural.append("Considerar separar capas: dominio, infraestructura y presentación (Clean Architecture).")
        if not any(p.name == "Repository Pattern" for p in patterns.design_patterns):
            architectural.append("Implementar Repository Pattern para abstraer el acceso a datos.")
        if not any(p.name == "Interface / Protocol / ABC" for p in patterns.best_practices):
            architectural.append("Definir interfaces/protocolos para inversión de dependencias (SOLID-D).")
        if classification.composite_score < 60:
            architectural.append("Evaluar adoptar una arquitectura hexagonal (ports & adapters) para mayor mantenibilidad.")

        return {
            "quick_wins": quick_wins[:5],
            "mediano_plazo": medium_term[:5],
            "arquitecturales": architectural[:4],
        }

    def _serialize_llm(self, llm_insights) -> dict:
        """Convert LLMInsights object or empty dict to serializable dict."""
        if not llm_insights:
            return {"disponible": False}

        from llm.llm_analyzer import LLMInsights

        if isinstance(llm_insights, LLMInsights):
            return {
                "disponible": True,
                "proveedor": llm_insights.provider,
                "resumen": llm_insights.summary,
                "fortalezas": llm_insights.strengths,
                "mejoras": llm_insights.improvements,
                "observaciones_arquitectura": llm_insights.architectural_observations,
                "preocupaciones_seguridad": llm_insights.security_concerns,
                "notas_escalabilidad": llm_insights.scalability_notes,
                "error": llm_insights.error,
            }
        return {"disponible": False}
