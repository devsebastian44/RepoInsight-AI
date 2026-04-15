# RepoInsight AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat&logo=scikitlearn)
![LLM](https://img.shields.io/badge/LLM-OpenAI_%7C_Anthropic-412991?style=flat&logo=openai)
![License](https://img.shields.io/badge/License-MIT-green?style=flat&logo=opensourceinitiative)
![Ruff](https://img.shields.io/badge/Linter-Ruff_%2B_Black-orange?style=flat&logo=python)
![GitHub API](https://img.shields.io/badge/GitHub_API-REST_v3-181717?style=flat&logo=github)

---

## 🧠 Overview

RepoInsight AI es un analizador de inteligencia de repositorios de GitHub potenciado por
**Machine Learning** y **LLMs**. Dado únicamente la URL de un repositorio público, la
herramienta descarga su código fuente, extrae un vector de 47 características mediante
análisis estático, y aplica un clasificador **RandomForest** para determinar el nivel de
madurez del código (Junior / Mid-level / Senior), junto con una puntuación de calidad
compuesta de 0 a 100.

El sistema opera en cinco capas claramente desacopladas: adquisición de datos vía GitHub
REST API v3, extracción y parsing de código fuente, análisis estático multidimensional
(métricas, patrones de diseño, code smells), clasificación ML y generación de reportes en
consola o JSON. Opcionalmente integra proveedores LLM (OpenAI GPT-4o o Anthropic Claude)
para análisis semántico profundo del código.

La arquitectura está completamente escrita en Python puro con dependencias mínimas —
`scikit-learn`, `pandas`, `numpy` y `requests` — sin frameworks web ni bases de datos,
lo que la hace ejecutable desde cualquier entorno con Python 3.10+.

---

## ⚙️ Features

- **Clasificación de nivel por ML**: RandomForest entrenado con 1.200 muestras sintéticas
  etiquetadas (400 por clase) clasifica repositorios como Junior, Mid-level o Senior con
  probabilidades y score de confianza.
- **Score compuesto 0–100**: combina score ML (40%), calidad estática (35%), buenas prácticas
  (15%) y actividad de commits (10%).
- **47 características extraídas automáticamente**: metadatos del repo, volumen de código,
  métricas de calidad, estructura del proyecto, prácticas de desarrollo, señales de patrones
  y actividad de commits.
- **Detección de 12 patrones de diseño**: Singleton, Factory, Observer, Strategy, Decorator,
  Repository, Command, Builder, Adapter, MVC/MVP/MVVM, Dependency Injection, Context Manager.
- **Detección de 12 buenas prácticas**: type annotations, docstrings, manejo de excepciones,
  logging, variables de entorno, constantes/enums, dataclasses, tests unitarios, linting,
  separación de responsabilidades, async/await, interfaces/protocolos.
- **Detección de 10 code smells**: God Class, Long Method, Deep Nesting, Magic Numbers,
  Long Parameter List, Commented-Out Code, debug prints, TODO/FIXME, duplicación, Bare Except.
- **Análisis LLM opcional**: envía código representativo a OpenAI o Anthropic para obtener
  resumen de fortalezas, áreas de mejora y alertas de seguridad en lenguaje natural.
- **Salida dual**: reporte legible en consola con colores y reporte JSON serializable para
  integración en pipelines CI/CD o dashboards.
- **Auto-entrenamiento**: el modelo RandomForest se entrena automáticamente en el primer uso
  si no existe un `.pkl` en `data/`; también re-entrenable manualmente con `train_model.py`.
- **CLI completo** con argparse, flags de verbosidad, selección de proveedor LLM, límite de
  archivos y guardado de reportes.
- **Pre-commit hooks**: `black` (formato), `ruff` (linting + autofix), y verificadores
  generales (trailing whitespace, YAML, archivos grandes) en cada commit.
- **Uso programático**: todos los módulos son importables e instanciables directamente como
  librería Python, con dataclasses tipadas para cada artefacto del pipeline.

---

## 🛠️ Tech Stack

| Categoría | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| Motor ML | scikit-learn ≥ 1.4 (RandomForestClassifier) |
| Datos / Vectores | pandas ≥ 2.1, numpy ≥ 1.26 |
| GitHub API | requests ≥ 2.31 (REST v3, paginación, rate-limit) |
| LLM opcional | openai ≥ 1.30 / anthropic ≥ 0.25 |
| Linting | ruff ≥ 0.4 + black ≥ 24.0 |
| Testing | pytest ≥ 8.0 + pytest-cov ≥ 5.0 |
| Pre-commit | pre-commit (black, ruff, trailing-whitespace, check-yaml) |
| Empaquetado | setuptools + pyproject.toml |
| Entorno | python-dotenv (dev) |

---

## 📦 Installation

### Prerrequisitos

- Python `>=3.10`
- pip
- Token de GitHub personal (opcional, aumenta rate-limit de 60 a 5.000 req/h)

### Instalación estándar

```bash
# 1. Clonar el repositorio
git clone https://github.com/devsebastian44/RepoInsight-AI.git
cd RepoInsight-AI

# 2. Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows

# 3. Instalar dependencias de producción
pip install -r requirements.txt

# 4. (Opcional) Instalar extras de desarrollo
pip install -e ".[dev]"
pre-commit install

# 5. (Opcional) Instalar proveedores LLM
pip install openai              # Para análisis con GPT-4o
pip install anthropic           # Para análisis con Claude
```

### Variables de entorno (recomendado)

```bash
# Exportar directamente o mediante archivo .env (requiere python-dotenv en dev)
export GITHUB_TOKEN=ghp_your_token_here      # Rate-limit 60 → 5.000 req/h
export OPENAI_API_KEY=sk-...                 # Para LLM OpenAI
export ANTHROPIC_API_KEY=sk-ant-...          # Para LLM Anthropic / Claude
```

### Pre-entrenar el modelo (opcional)

```bash
# El modelo se auto-entrena en el primer análisis.
# Para pre-entrenarlo manualmente y ver importancias:
python src/train_model.py --show-importances
```

---

## ▶️ Usage

### Análisis básico en consola

```bash
python src/main.py https://github.com/tiangolo/fastapi
```

### Guardar reporte JSON

```bash
python src/main.py https://github.com/psf/requests \
  --output json \
  --save report.json
```

### Con análisis LLM profundo (OpenAI)

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
  --llm-provider anthropic \
  --output both
```

### Referencia completa de flags

```
usage: repoinsight [-h] [--output {console,json,both}]
                   [--save FILE] [--llm]
                   [--llm-provider {openai,anthropic,mock}]
                   [--github-token TOKEN]
                   [--max-files MAX_FILES]
                   [--verbose]
                   repo_url

positional arguments:
  repo_url                    URL completa del repositorio GitHub a analizar

options:
  --output {console,json,both}   Formato de salida (default: both)
  --save FILE                    Guardar reporte JSON en archivo
  --llm                          Habilitar análisis profundo LLM
  --llm-provider                 Proveedor LLM: openai | anthropic | mock
  --github-token TOKEN           Token GitHub PAT (sobrescribe GITHUB_TOKEN)
  --max-files MAX_FILES          Máximo de archivos fuente a analizar (default: 50)
  --verbose                      Habilitar logging de depuración detallado
```

### Uso como librería Python

```python
from src.config import Config
from src.data_collection.github_client import GitHubClient
from src.data_collection.code_extractor import CodeExtractor
from src.analysis.code_analyzer import CodeAnalyzer
from src.analysis.pattern_detector import PatternDetector
from src.analysis.feature_engineering import FeatureEngineer
from src.ml_model.classifier import RepositoryClassifier
from src.reporting.report_builder import ReportBuilder

config = Config(github_token="ghp_your_token", max_files=50)

client  = GitHubClient(config)
repo    = client.fetch_repository("tiangolo", "fastapi")
files   = CodeExtractor(config).extract(client, "tiangolo", "fastapi", repo)
metrics = CodeAnalyzer(config).analyze(files)
patterns= PatternDetector(config).detect(files, metrics)
features= FeatureEngineer(config).build(repo, files, metrics, patterns)
result  = RepositoryClassifier(config).predict(features)

print(f"Nivel: {result.level} — Confianza: {result.confidence:.0%}")
print(f"Score: {result.composite_score}/100")
```

### Ejecutar tests

```bash
# Suite completa con cobertura
pytest tests/ -v --cov=. --cov-report=term-missing

# Ejecución directa sin pytest
python tests/test_suite.py
```

---

## 📁 Project Structure

```
RepoInsight-AI/
│
├── src/                              # Código fuente principal
│   ├── main.py                       # Entry point CLI y orquestador del pipeline
│   ├── config.py                     # Dataclass de configuración centralizada
│   ├── train_model.py                # Script de entrenamiento y re-entrenamiento ML
│   │
│   ├── data_collection/              # Capa 1 — Adquisición de datos
│   │   ├── github_client.py          # Cliente GitHub REST v3: fetch, rate-limit, paginado
│   │   └── code_extractor.py         # Descarga y parsing de archivos fuente → SourceFile[]
│   │
│   ├── analysis/                     # Capa 2 — Análisis estático
│   │   ├── code_analyzer.py          # Métricas: densidad, complejidad, duplicación, naming
│   │   ├── pattern_detector.py       # 12 patrones, 12 buenas prácticas, 10 code smells
│   │   └── feature_engineering.py   # Construcción del vector de 47 características (numpy)
│   │
│   ├── ml_model/                     # Capa 3 — Motor ML (parcialmente privado en GitLab)
│   │   └── classifier.py             # RandomForestClassifier: predict(), auto-train, score
│   │
│   ├── llm/                          # Capa 4 — Análisis LLM (privado en GitLab)
│   │   └── llm_analyzer.py          # Adaptador OpenAI / Anthropic con prompts JSON
│   │
│   └── reporting/                    # Capa 5 — Generación de reportes
│       └── report_builder.py         # Ensambla ReportDict serializable (consola + JSON)
│
├── tests/                            # Suite de tests (unitarios e integración)
│   └── test_suite.py                 # Tests completos con pytest + cobertura
│
├── data/                             # Artefactos de datos y modelos serializados
│   └── *.pkl                         # Modelo RandomForest serializado (auto-generado)
│
├── configs/                          # Plantillas de configuración (ej. .env.example)
├── docs/                             # Documentación técnica del proyecto
├── diagrams/                         # Diagramas de arquitectura C4 / flujo de datos
│
├── scripts/
│   └── publish_public.ps1            # Sanitización automatizada GitLab → GitHub
│
├── .pre-commit-config.yaml           # Hooks: black, ruff, trailing-whitespace, check-yaml
├── pyproject.toml                    # Metadatos, dependencias, ruff, black, pytest
├── requirements.txt                  # Dependencias de producción fijadas
└── LICENSE                           # Licencia MIT
```

---

## 🔐 Security

- **Sin ejecución de código analizado**: todo el análisis es basado puramente en regex y
  parsing de texto. Nunca se ejecuta el código fuente descargado de los repositorios.
- **Token con permisos mínimos**: solo se requiere acceso de lectura a repositorios públicos.
  El token nunca se persiste ni se loguea; se lee desde variable de entorno.
- **Sin persistencia de datos de terceros**: únicamente se serializa el modelo `.pkl`
  propio en `data/`; los archivos fuente descargados se procesan en memoria sin escritura
  en disco.
- **Arquitectura de repositorios segregada**: los módulos `ml_model/` y `llm/` con su
  implementación completa, datasets de entrenamiento, pipelines CI/CD y configuraciones
  sensibles residen exclusivamente en el repositorio privado de GitLab. GitHub expone
  únicamente el código core sanitizado.
- **Pre-commit hooks de seguridad**: `check-added-large-files` previene commits
  accidentales de archivos pesados (modelos, datasets); `check-yaml` valida la integridad
  de la configuración antes de cada commit.
- **Dependencias auditadas**: el extras group `dev` incluye `python-dotenv` para gestión
  segura de variables de entorno; en GitLab se ejecutan auditorías de dependencias (SCA)
  en el pipeline CI/CD.

---

## 🌐 Repository Architecture

Este proyecto sigue una arquitectura distribuida de repositorios con separación estricta
de entornos por diseño:

- **GitLab** — Laboratorio de datos e IA (fuente de verdad): contiene la implementación
  completa de `ml_model/` y `llm/`, datasets de entrenamiento reales, pipelines CI/CD,
  notebooks de experimentación, benchmarks y configuraciones de infraestructura.
- **GitHub** — Portafolio público sanitizado: expone el código core del pipeline, la
  arquitectura del sistema, documentación técnica y diagramas. El script
  `scripts/publish_public.ps1` automatiza la sanitización y sincronización entre entornos.

### 🔗 Full Source Code

👉 Código completo disponible en GitLab:
[https://gitlab.com/group-data-ia-lab/RepoInsight-AI](https://gitlab.com/group-data-ia-lab/RepoInsight-AI)

---

## 🚀 Roadmap

Basado en el análisis de la arquitectura actual detectada en el código fuente:

- [ ] **API REST con FastAPI**: exponer el pipeline completo como microservicio con endpoint
  `POST /analyze` y soporte para colas de trabajos asíncronos (Celery + Redis).
- [ ] **Dashboard web interactivo**: gráfico radar multidimensional (React + Recharts) para
  visualizar las cinco dimensiones del score en tiempo real.
- [ ] **Entrenamiento con datos reales etiquetados**: reemplazar las 1.200 muestras
  sintéticas con repositorios reales etiquetados mediante heurísticas (stars, contributors,
  antigüedad, CI, tests).
- [ ] **Soporte multi-lenguaje extendido**: añadir patrones regex para Go, Rust, Kotlin,
  Swift, C++ y Java siguiendo la arquitectura `_PATTERNS` existente.
- [ ] **Análisis de historial de commits**: extraer métricas temporales de frecuencia,
  calidad de mensajes y evolución de la deuda técnica por periodo.
- [ ] **Persistencia histórica**: base de datos SQLite/PostgreSQL para almacenar análisis
  anteriores y generar comparativas de evolución del proyecto.
- [ ] **GitHub Action oficial**: workflow reutilizable para integrar RepoInsight AI como
  quality gate en pipelines CI/CD de terceros.
- [ ] **Cache de resultados**: evitar re-análisis de repositorios no modificados usando
  el hash del último commit como clave de caché.
- [ ] **Exportación a formatos estándar**: SARIF (GitHub Code Scanning), JUnit XML
  (para integración con pipelines CI) y HTML con estilos para reportes ejecutivos.
- [ ] **Modo batch desde lista de URLs**: soporte nativo para analizar múltiples
  repositorios en paralelo desde un archivo de texto o CSV.
- [ ] **Versiones del modelo por lenguaje**: modelos RandomForest especializados por
  ecosistema (Python, JavaScript, Java) con vectores de características calibrados.

---

## 📄 License

Este proyecto está bajo la licencia **MIT**.

> Licencia detectada directamente desde el archivo `LICENSE` en la raíz del repositorio
> y confirmada en `pyproject.toml` bajo el clasificador
> `License :: OSI Approved :: MIT License`.

---

## 👨‍💻 Author

**Sebastian Zhunaula** — [@devsebastian44](https://github.com/devsebastian44)

Desarrollador full-stack e ingeniero de datos con enfoque en sistemas de análisis
inteligente, arquitecturas ML reproducibles y herramientas de productividad para
equipos de ingeniería de software.