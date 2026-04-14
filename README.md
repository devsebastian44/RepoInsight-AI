# 🤖 Analizador de Inteligencia de Repositorios AI

> **Analiza cualquier repositorio de GitHub y obtén un informe de inteligencia potenciado por ML en segundos.**  
> Clasificación de nivel · Puntuación de calidad · Detección de patrones · Revisión de código con LLM

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Clean%20Modular-purple)]()

---

## 📋 Tabla de Contenidos

- [¿Qué hace?](#-qué-hace)
- [Ejemplo de Salida](#-ejemplo-de-salida)
- [Arquitectura](#️-arquitectura)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Referencia de Módulos](#-referencia-de-módulos)
- [Diseño del Modelo ML](#-diseño-del-modelo-ml)
- [Extensión y Escalado](#-extensión-y-escalado)
- [Referencia de Configuración](#️-referencia-de-configuración)

---

## 🎯 ¿Qué hace?

Esta herramienta **evalúa automáticamente repositorios de GitHub** en tres dimensiones:

| Dimensión | ¿Qué se mide? |
|-----------|----------------|
| **Nivel** | Junior / Nivel medio / Senior (clasificador RandomForest) |
| **Calidad** | Densidad de comentarios, complejidad, duplicación, nomenclatura, modularidad |
| **Patrones** | Patrones de diseño, mejores prácticas, code smells |

### Capacidades de detección

**12 Patrones de Diseño**: Singleton, Factory, Observer, Strategy, Decorator,
Repository, Command, Builder, Adapter, MVC/MVP/MVVM, Dependency Injection, Context Manager

**12 Mejores Prácticas**: Anotaciones de tipo, docstrings, manejo de excepciones, logging,
variables de entorno, constantes/enums, dataclasses, pruebas unitarias, configuración de linting,
separación de preocupaciones, programación asíncrona, interfaces/protocolos

**10 Code Smells**: God Class, Método Largo, Anidamiento Profundo, Números Mágicos,
Lista Larga de Parámetros, Código Comentado, depuración con print(), TODO/FIXME,
Código Duplicado, Except Vacío

---

## 📊 Ejemplo de Salida

```json
{
  "nivel": "Senior",
  "calidad": "Alta",
  "score": 84.3,
  "buenas_practicas": [
    "Type Annotations",
    "Docstrings / JSDoc",
    "Exception Handling",
    "Logging",
    "Unit Tests"
  ],
  "mejoras": [
    "Configurar linter/formatter (ruff + black para Python).",
    "Mover configuración a variables de entorno (.env + python-dotenv)."
  ],
  "clasificacion": {
    "nivel": "Senior",
    "confianza": 0.89,
    "probabilidades": {
      "Junior": 0.03,
      "Mid-level": 0.08,
      "Senior": 0.89
    },
    "score_compuesto": 84.3
  },
  "metricas_codigo": {
    "archivos_analizados": 38,
    "lineas_totales": 4812,
    "funciones_totales": 142,
    "clases_totales": 31,
    "densidad_comentarios": 0.18,
    "ratio_docstrings": 0.76,
    "ratio_type_hints": 0.82,
    "complejidad_promedio": 28.4,
    "score_calidad": 81.0
  }
}
```

---

## 🏗️ Arquitectura

```
RepoInsight-AI/
│
├── src/                         ← Código fuente principal
│   ├── main.py                  ← Punto de entrada CLI y orquestador
│   ├── config.py                ← Configuración centralizada
│   ├── train_model.py           ← Entrenamiento de modelo ML
│   ├── data_collection/         ← Capa 1: Adquisición de Datos
│   ├── analysis/                ← Capa 2: Análisis Estático
│   ├── ml_model/                ← Capa 3: Machine Learning (Privado)
│   ├── llm/                     ← Capa 4: Análisis LLM (Privado)
│   └── reporting/               ← Capa 5: Generación de reportes
│
├── scripts/
│   └── publish_public.ps1       ← Sanitización automatizada GitLab -> GitHub
│
├── tests/                       ← Suite de pruebas unitarias y de integración
├── configs/                     ← Plantillas y configuración (ej. .env.example)
├── data/                        ← Modelos entrenados e insumos (ej. bbdd/pkl)
├── docs/                        ← Documentación del proyecto
└── diagrams/                    ← Diagramas arquitectónicos y modelado

### DevSecOps Workflow (Privado vs Público)

Este proyecto emplea un modelo de seguridad por diseño, dividiendo el trabajo en dos entornos estrictos:
- **GitLab (Laboratorio Privado)**: Fuente de verdad centralizada. Contiene código crítico (pipelines de CI/CD, tests integrales, módulos internos `llm/` y `ml_model/`).
- **GitHub (Portafolio Público)**: Versión de solo lectura sanitizada. 

El script `scripts/publish_public.ps1` se ejecuta tras completar validaciones continuas (CI) para empaquetar una **versión sanitizada** desprovista de automatizaciones privadas hacia el portafolio público de demostración.
```

### Flujo de datos

```
GitHub URL
    │
    ▼
GitHubClient ──► RepositoryData (metadata, commits, tree)
    │
    ▼
CodeExtractor ──► [SourceFile, ...] (parsed source with functions, classes, imports)
    │
    ├──► CodeAnalyzer ──────► CodeMetrics (complexity, quality, duplication)
    │
    ├──► PatternDetector ───► PatternReport (patterns, practices, smells)
    │
    ├──► FeatureEngineer ───► FeatureVector (47-dim numpy array)
    │
    ├──► RepositoryClassifier ─► ClassificationResult (level, confidence, score)
    │
    └──► LLMAnalyzer (opt.) ──► LLMInsights (summary, strengths, improvements)
                │
                ▼
          ReportBuilder ──► report dict (JSON-serializable)
                │
                ▼
          ReportFormatter ──► Terminal / JSON file
```

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/devsebastian44/RepoInsight-AI.git
cd RepoInsight-AI
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt

# Opcional: Proveedores LLM
pip install openai          # Para análisis GPT-4o
pip install anthropic       # Para análisis Claude
```

### 4. Configurar variables de entorno (opcional pero recomendado)

```bash
# Archivo .env o exportar directamente:
export GITHUB_TOKEN=ghp_your_token_here      # Incrementa límite 60→5000 req/h
export OPENAI_API_KEY=sk-...                 # Para análisis LLM
export ANTHROPIC_API_KEY=sk-ant-...          # Para análisis Claude
```

### 5. Pre-entrenar el modelo (opcional — auto-entrena en primer uso)

```bash
python src/train_model.py --show-importances
```

---

## 💻 Uso

### Análisis básico

```bash
python src/main.py https://github.com/tiangolo/fastapi
```

### Guardar reporte en JSON

```bash
python src/main.py https://github.com/psf/requests --output json --save report.json
```

### Con análisis profundo LLM (OpenAI)

```bash
python src/main.py https://github.com/pallets/flask \
  --llm \
  --llm-provider openai \
  --output both
```

### Con Anthropic Claude

```bash
python src/main.py https://github.com/encode/httpx \
  --llm \
  --llm-provider anthropic
```

### Todas las opciones

```
usage: ai-repo-analyzer [-h] [--output {console,json,both}]
                        [--save FILE] [--llm]
                        [--llm-provider {openai,anthropic,mock}]
                        [--github-token TOKEN]
                        [--max-files MAX_FILES]
                        [--verbose]
                        repo_url

positional arguments:
  repo_url              URL completa del repositorio GitHub

options:
  --output              Formato de salida: console | json | both (default: both)
  --save FILE           Guardar reporte JSON en archivo
  --llm                 Habilitar análisis profundo LLM
  --llm-provider        Proveedor LLM: openai | anthropic | mock
  --github-token TOKEN  Token GitHub PAT (sobrescribe variable GITHUB_TOKEN)
  --max-files MAX_FILES Máx archivos fuente a analizar (default: 50)
  --verbose             Habilitar logging de depuración
```

### API Python (uso programático)

```python
from src.config import Config
from src.data_collection.github_client import GitHubClient
from src.data_collection.code_extractor import CodeExtractor
from src.analysis.code_analyzer import CodeAnalyzer
from src.analysis.pattern_detector import PatternDetector
from src.analysis.feature_engineering import FeatureEngineer
from src.ml_model.classifier import RepositoryClassifier
from src.reporting.report_builder import ReportBuilder

config = Config(github_token="ghp_your_token")

# Recolectar
client = GitHubClient(config)
repo_data = client.fetch_repository("tiangolo", "fastapi")

extractor = CodeExtractor(config)
source_files = extractor.extract(client, "tiangolo", "fastapi", repo_data)

# Analizar
metrics = CodeAnalyzer(config).analyze(source_files)
patterns = PatternDetector(config).detect(source_files, metrics)
features = FeatureEngineer(config).build(repo_data, source_files, metrics, patterns)

# Clasificar
result = RepositoryClassifier(config).predict(features)
print(f"Nivel: {result.level} ({result.confidence:.0%} confianza)")
print(f"Score: {result.composite_score}/100")

# Reporte
report = ReportBuilder(config).build(
    repo_data=repo_data, source_files=source_files,
    code_metrics=metrics, patterns=patterns,
    feature_vector=features, classification=result, llm_insights={}
)
```

---

## 📦 Referencia de Módulos

### `config.py` — `Config`

Dataclass de configuración central. Todos los módulos aceptan una instancia `Config`.

```python
Config(
    github_token = None,          # str | None
    max_files = 50,               # int — máx archivos fuente a descargar
    llm_provider = "mock",        # "openai" | "anthropic" | "mock"
    model_n_estimators = 200,     # int — árboles RF
    verbose = False,              # bool — salida de depuración
)
```

---

### `data_collection/github_client.py` — `GitHubClient`

Envuelve GitHub REST API v3. Maneja límites de velocidad, paginación y reintentos.

```python
client = GitHubClient(config)
repo_data: RepositoryData = client.fetch_repository("owner", "repo")
content: str = client.get_file_content("owner", "repo", "path/to/file.py")
```

**`RepositoryData`** fields: `stars`, `forks`, `commits_count`, `languages`,
`has_ci`, `has_tests`, `has_docker`, `commit_history`, `tree_entries`, `readme_content`, …

---

### `data_collection/code_extractor.py` — `CodeExtractor`

Descarga y parsea archivos fuente en objetos estructurados `SourceFile`.

```python
files: list[SourceFile] = CodeExtractor(config).extract(client, owner, repo, repo_data)
```

**`SourceFile`** fields: `path`, `extension`, `content`, `functions`, `classes`,
`imports`, `line_count`, `code_lines`, `comment_lines`, `depth`, …

---

### `analysis/code_analyzer.py` — `CodeAnalyzer`

Calcula métricas de calidad estática. Todo el análisis es basado en regex (sin ejecución de código).

```python
metrics: CodeMetrics = CodeAnalyzer(config).analyze(source_files)
# metrics.quality_score  → 0-100
# metrics.duplication_score
# metrics.naming_score
```

---

### `analysis/pattern_detector.py` — `PatternDetector`

Detecta patrones de diseño, mejores prácticas y code smells.

```python
report: PatternReport = PatternDetector(config).detect(source_files, metrics)
# report.design_patterns  → list[Finding]
# report.best_practices   → list[Finding]
# report.code_smells      → list[Finding]
```

---

### `ml_model/classifier.py` — `RepositoryClassifier`

Clasificador de nivel basado en RandomForest. Auto-entrena en primer uso.

```python
result: ClassificationResult = RepositoryClassifier(config).predict(feature_vector)
# result.level          → "Junior" | "Mid-level" | "Senior"
# result.confidence     → 0.0–1.0
# result.composite_score→ 0–100
# result.probabilities  → {"Junior": 0.1, "Mid-level": 0.2, "Senior": 0.7}
```

---

### `llm/llm_analyzer.py` — `LLMAnalyzer`

Análisis profundo opcional potenciado por LLM. Usa prompts JSON estructurados.

```python
insights: LLMInsights = LLMAnalyzer(config).analyze(
    source_files, metrics, patterns, classification
)
# insights.summary
# insights.strengths
# insights.improvements
# insights.security_concerns
```

---

## 🧠 Diseño del Modelo ML

### Vector de Características (47 características)

| Grupo | Características | Cantidad |
|-------|----------------|----------|
| Metadatos del repositorio | stars_log, forks_log, repo_age_days, commits_log, … | 12 |
| Volumen de código | file_count, total_lines_log, functions_log, classes_log, … | 7 |
| Métricas de calidad | comment_density, complexity, duplication, naming, … | 8 |
| Estructura del proyecto | unique_dirs, max_depth, language_count, … | 4 |
| Prácticas de desarrollo | has_ci, has_tests, has_docker, has_requirements | 4 |
| Señales de patrones | design_pattern_count, best_practice_count, smell_count, … | 5 |
| Actividad de commits | commit_frequency, commit_message_quality | 2 |

### Datos de entrenamiento

El modelo se entrena con **1,200 muestras sintéticas etiquetadas** (400 por clase) generadas
con distribuciones calibradas por nivel:

| Característica | Junior | Nivel medio | Senior |
|---------|--------|-----------|--------|
| Score de calidad | ~30 | ~55 | ~80 |
| Ratio de docstrings | ~5% | ~35% | ~75% |
| Ratio de type hints | ~0% | ~20% | ~70% |
| Patrones de diseño | ~0.3 | ~2 | ~5 |
| Mejores prácticas | ~1.5 | ~5 | ~9 |
| Code smells | ~5 | ~3 | ~1 |
| Tiene CI | 15% | 60% | 95% |
| Tiene Tests | 25% | 65% | 95% |

### Fórmula de score compuesto

```
Score = ML_score × 0.40  +  Quality_score × 0.35  +  Practice_score × 0.15  +  Activity × 0.10
```

### Características principales por importancia (ejecución típica)

1. `quality_score`
2. `best_practice_count`
3. `naming_score`
4. `docstring_ratio`
5. `type_hint_ratio`
6. `design_pattern_count`
7. `commits_log`
8. `has_tests`
9. `duplication_score`
10. `modularity_score`

---

## 📈 Extensión y Escalado

### SaaS / REST API

```python
# Envolver pipeline main.py en endpoint FastAPI
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI()

class AnalysisRequest(BaseModel):
    repo_url: str
    enable_llm: bool = False

@app.post("/analyze")
async def analyze(req: AnalysisRequest, bg: BackgroundTasks):
    bg.add_task(run_analysis_and_store, req.repo_url, req.enable_llm)
    return {"status": "queued", "repo": req.repo_url}
```

### Web Dashboard

- **Backend**: FastAPI + Celery + Redis (para cola de trabajos asíncronos)
- **Frontend**: React + Recharts (gráfico radar para dimensiones)
- **Storage**: PostgreSQL para historial de reportes + análisis de tendencias

### Integración CI/CD

```yaml
# .github/workflows/quality-check.yml
- name: Analyze PR
  run: |
    python main.py ${{ github.event.repository.html_url }} \
      --output json --save pr_report.json
    python -c "
    import json, sys
    r = json.load(open('pr_report.json'))
    if r['score'] < 50:
        print('Quality gate FAILED:', r['score'])
        sys.exit(1)
    "
```

### Datos de Entrenamiento Reales

Reemplazar datos sintéticos con repos etiquetados reales:

```python
# Rastrear GitHub con heurísticas de etiquetas:
# Senior → stars > 1000, contributors > 20, age > 2yr, has_ci=True
# Junior → stars < 10, contributors = 1, age < 3mo
```

### Soporte Multi-lenguaje

Extender `CodeExtractor._PATTERNS` con regex específicos de lenguaje para:
Go, Rust, Kotlin, Swift, C++ — siguiendo el patrón existente `.py`/`.js`.

### Análisis por Lotes

```bash
# Analizar múltiples repos desde un archivo de lista
while IFS= read -r url; do
    python main.py "$url" --output json --save "reports/$(basename $url).json"
done < repo_list.txt
```

---

## ⚙️ Referencia de Configuración

| Variable | Variable de entorno | Default | Descripción |
|----------|-------------------|---------|-------------|
| `github_token` | `GITHUB_TOKEN` | `None` | Token GitHub PAT (recomendado) |
| `openai_api_key` | `OPENAI_API_KEY` | `None` | Para análisis LLM OpenAI |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | `None` | Para análisis LLM Claude |
| `max_files` | — | `50` | Máx archivos fuente a analizar |
| `model_n_estimators` | — | `200` | Cantidad de árboles Random Forest |
| `llm_sample_lines` | — | `150` | Líneas de código enviadas al LLM |
| `max_file_size_kb` | — | `500` | Omitir archivos más grandes que esto |

---

## 🧪 Ejecutando Pruebas

```bash
# Con pytest
pytest tests/test_suite.py -v --cov=. --cov-report=term-missing

# Sin pytest (Python puro)
python tests/test_suite.py
```

---

## 📄 Licencia

MIT License — ver [LICENSE](LICENSE)

---

## 🤝 Contribuyendo

1. Fork del repositorio
2. Crear rama de feature (`git checkout -b feature/add-go-support`)
3. Agregar pruebas para nueva funcionalidad
4. Enviar pull request

---

*Construido con Python · scikit-learn · GitHub API · Clean Architecture*
