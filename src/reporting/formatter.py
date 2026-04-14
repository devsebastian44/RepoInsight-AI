"""
reporting/formatter.py
=======================
Rich console output for the analysis report.
Renders a color-coded, well-structured terminal report
using only the Python standard library (no rich/curses dependency).
"""

from __future__ import annotations

from typing import Any


# ANSI escape codes for color output
class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    BG_DARK = "\033[40m"


def _b(text: str) -> str:
    return f"{_C.BOLD}{text}{_C.RESET}"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{_C.RESET}"


def _level_color(level: str) -> str:
    return {
        "Junior": _C.YELLOW,
        "Mid-level": _C.CYAN,
        "Senior": _C.GREEN,
    }.get(level, _C.WHITE)


def _quality_color(quality: str) -> str:
    return {
        "Alta": _C.GREEN,
        "Media-Alta": _C.CYAN,
        "Media": _C.YELLOW,
        "Baja": _C.RED,
    }.get(quality, _C.WHITE)


def _score_bar(score: float, width: int = 30) -> str:
    filled = int(width * score / 100)
    color = _C.GREEN if score >= 70 else (_C.YELLOW if score >= 40 else _C.RED)
    bar = "█" * filled + "░" * (width - filled)
    return f"{color}{bar}{_C.RESET} {score:.1f}/100"


def _severity_icon(severity: str) -> str:
    return {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "⚪")


class ReportFormatter:
    """Renders the report dict to the terminal in a human-friendly format."""

    LINE = "─" * 64

    def print_console(self, report: dict[str, Any]) -> None:
        """Print the full console report."""
        self._print_header(report)
        self._print_classification(report)
        self._print_code_metrics(report)
        self._print_design_patterns(report)
        self._print_best_practices(report)
        self._print_code_smells(report)
        self._print_llm_insights(report)
        self._print_recommendations(report)
        self._print_footer(report)

    # ── Sections ──────────────────────────────────────────────────────────────

    def _print_header(self, r: dict) -> None:
        repo = r.get("repositorio", {})
        print(f"\n{_C.BOLD}{_C.BLUE}{self.LINE}{_C.RESET}")
        print(f"  {_b('📦 Repository:')} {_c(_C.WHITE, repo.get('nombre', 'N/A'))}")
        desc = repo.get("descripcion") or "No description"
        print(f"  {_c(_C.DIM, desc[:80])}")
        print(f"  {_c(_C.DIM, repo.get('url', ''))}")
        print(
            f"  ⭐ {repo.get('estrellas', 0)}  🍴 {repo.get('forks', 0)}  "
            f"🌿 {repo.get('branches', 0)} branches  "
            f"👥 {repo.get('colaboradores', 0)} contributors"
        )
        lang = repo.get("lenguaje_principal") or "Unknown"
        topics = ", ".join(repo.get("topics", [])[:5]) or "none"
        print(f"  🔤 {lang}  🏷️  {topics}")
        print(f"{_C.BOLD}{_C.BLUE}{self.LINE}{_C.RESET}\n")

    def _print_classification(self, r: dict) -> None:
        clf = r.get("clasificacion", {})
        level = r.get("nivel", "?")
        quality = r.get("calidad", "?")
        score = r.get("score", 0)

        lc = _level_color(level)
        qc = _quality_color(quality)

        print(f"  {_b('🤖 ML Classification')}")
        print(f"  {self.LINE}")
        print(f"  Nivel del Proyecto : {_c(lc, _b(level))}  (confianza: {clf.get('confidence', 0):.0%})")
        print(f"  Calidad General    : {_c(qc, _b(quality))}")
        print(f"  Score Compuesto    : {_score_bar(score)}")

        probs = clf.get("probabilidades", {})
        if probs:
            print("\n  Distribución de probabilidades:")
            for lbl, prob in probs.items():
                bar = "▪" * int(prob * 20)
                color = _level_color(lbl)
                print(f"    {lbl:<12} {_c(color, bar):<40} {prob:.1%}")
        print()

    def _print_code_metrics(self, r: dict) -> None:
        m = r.get("metricas_codigo", {})
        print(f"  {_b('📊 Code Metrics')}")
        print(f"  {self.LINE}")

        def row(label: str, value, unit: str = "") -> None:
            print(f"  {label:<30} {_c(_C.CYAN, str(value))}{unit}")

        row("Archivos analizados:", m.get("archivos_analizados", 0))
        row("Líneas totales:", m.get("lineas_totales", 0))
        row("Líneas de código puro:", m.get("lineas_codigo", 0))
        row("Funciones detectadas:", m.get("funciones_totales", 0))
        row("Clases detectadas:", m.get("clases_totales", 0))
        row("Densidad de comentarios:", f"{m.get('densidad_comentarios', 0):.1%}")
        row("Ratio docstrings:", f"{m.get('ratio_docstrings', 0):.1%}")
        row("Ratio type hints:", f"{m.get('ratio_type_hints', 0):.1%}")
        row("Complejidad promedio:", f"{m.get('complejidad_promedio', 0):.1f}/100")
        row("Duplicación estimada:", f"{m.get('score_duplicacion', 0):.1f}/100")
        row("Score nombres:", f"{m.get('score_nombres', 0):.1f}/100")
        row("Score modularidad:", f"{m.get('score_modularidad', 0):.1f}/100")
        row("Score calidad:", f"{m.get('score_calidad', 0):.1f}/100")
        row("Funciones largas:", m.get("funciones_largas", 0))
        row("Números mágicos:", m.get("numeros_magicos", 0))
        print()

    def _print_design_patterns(self, r: dict) -> None:
        patterns = r.get("patrones_diseno", [])
        print(f"  {_b('🏗️  Design Patterns Detected')} ({len(patterns)})")
        print(f"  {self.LINE}")
        if not patterns:
            print(f"  {_c(_C.DIM, 'None detected')}")
        else:
            for p in patterns:
                conf_icon = "●" if p["confianza"] == "high" else ("◑" if p["confianza"] == "medium" else "○")
                print(f"  {_c(_C.GREEN, conf_icon)} {_b(p['patron'])}")
                print(f"    {_c(_C.DIM, p['descripcion'])}")
                if p["archivos"]:
                    files_str = ", ".join(p["archivos"][:3])
                    print(f"    {_c(_C.DIM, f'→ {files_str}')}")
        print()

    def _print_best_practices(self, r: dict) -> None:
        practices = r.get("buenas_practicas_detalle", [])
        print(f"  {_b('✅ Best Practices')} ({len(practices)})")
        print(f"  {self.LINE}")
        if not practices:
            print(f"  {_c(_C.DIM, 'None detected')}")
        else:
            for p in practices:
                print(f"  {_c(_C.GREEN, '✓')} {_b(p['nombre'])}")
                print(f"    {_c(_C.DIM, p['descripcion'])}")
        print()

    def _print_code_smells(self, r: dict) -> None:
        smells = r.get("code_smells", [])
        print(f"  {_b('⚠️  Code Smells')} ({len(smells)})")
        print(f"  {self.LINE}")
        if not smells:
            print(f"  {_c(_C.GREEN, '✓ Sin code smells detectados')}")
        else:
            for s in sorted(smells, key=lambda x: ["critical", "warning", "info"].index(x["severidad"])):
                icon = _severity_icon(s["severidad"])
                sev_color = _C.RED if s["severidad"] == "critical" else (_C.YELLOW if s["severidad"] == "warning" else _C.BLUE)
                print(f"  {icon} {_c(sev_color, _b(s['nombre']))} [{s['severidad'].upper()}]")
                print(f"    {_c(_C.DIM, s['descripcion'])}")
                if s["archivos_afectados"]:
                    files_str = ", ".join(s["archivos_afectados"][:3])
                    print(f"    {_c(_C.DIM, f'→ {files_str}')}")
        print()

    def _print_llm_insights(self, r: dict) -> None:
        llm = r.get("analisis_llm", {})
        if not llm.get("disponible"):
            return

        print(f"  {_b('🧠 LLM Deep Analysis')} ({llm.get('proveedor', 'N/A')})")
        print(f"  {self.LINE}")
        print(f"  {_c(_C.WHITE, llm.get('resumen', ''))}")

        if llm.get("fortalezas"):
            print(f"\n  {_b('Fortalezas:')}")
            for s in llm["fortalezas"]:
                print(f"  {_c(_C.GREEN, '+')} {s}")

        if llm.get("mejoras"):
            print(f"\n  {_b('Mejoras sugeridas:')}")
            for i in llm["mejoras"]:
                print(f"  {_c(_C.YELLOW, '→')} {i}")

        if llm.get("preocupaciones_seguridad"):
            print(f"\n  {_b('Seguridad:')}")
            for sc in llm["preocupaciones_seguridad"]:
                print(f"  {_c(_C.RED, '🔒')} {sc}")

        if llm.get("notas_escalabilidad"):
            print(f"\n  {_b('Escalabilidad:')}")
            print(f"  {_c(_C.CYAN, llm['notas_escalabilidad'])}")

        print()

    def _print_recommendations(self, r: dict) -> None:
        recs = r.get("recomendaciones", {})
        print(f"  {_b('🚀 Recommendations')}")
        print(f"  {self.LINE}")

        sections = [
            ("⚡ Quick Wins", recs.get("quick_wins", []), _C.GREEN),
            ("📅 Mediano Plazo", recs.get("mediano_plazo", []), _C.YELLOW),
            ("🏛️  Arquitectura", recs.get("arquitecturales", []), _C.CYAN),
        ]
        for title, items, color in sections:
            if items:
                print(f"\n  {_b(title)}:")
                for idx, item in enumerate(items, 1):
                    print(f"  {_c(color, str(idx) + '.')} {item}")
        print()

    def _print_footer(self, r: dict) -> None:
        meta = r.get("meta", {})
        print(f"{_C.BOLD}{_C.BLUE}{self.LINE}{_C.RESET}")
        print(f"  {_c(_C.DIM, meta.get('herramienta', ''))} v{meta.get('version', '')}  |  {meta.get('timestamp', '')[:19]}")
        print(f"{_C.BOLD}{_C.BLUE}{self.LINE}{_C.RESET}\n")
