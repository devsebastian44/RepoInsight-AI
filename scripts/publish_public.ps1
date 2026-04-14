# =============================================================================
# scripts/publish_public.ps1 - V2 (DevSecOps Architecture)
# Sincronización Segura: GitLab (Laboratorio) -> GitHub (Portafolio Público)
# =============================================================================

Write-Host "[*] Iniciando sincronización profesional de RepoInsight-AI..." -ForegroundColor Cyan

# 1. Validaciones Iniciales
$currentBranch = git rev-parse --abbrev-ref HEAD
if ($currentBranch -ne "main") {
    Write-Host "[!] Error: Debes estar en 'main' para publicar." -ForegroundColor Red
    exit
}

if (git status --porcelain) {
    Write-Host "[!] Tienes cambios sin guardar. Haz commit antes de publicar." -ForegroundColor Yellow
    exit
}

# 2. Limpieza Local Previa
Write-Host "[*] Limpiando archivos temporales y logs..." -ForegroundColor Yellow
Remove-Item -Path "*.log", "report.json", "reports*.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "__pycache__", ".pytest_cache" -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path "src" -Include "__pycache__", "*.pyc" -Recurse | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue

# 3. Sincronización con Laboratorio Privado (Única fuente de verdad)
Write-Host "[*] Asegurando estado en la fuente de verdad (GitLab)..."
git pull origin-gitlab main --rebase
git push origin-gitlab main

# 4. Estrategia de Rama Pública (Aislamiento de Seguridad)
Write-Host "[*] Creando release sanitizado en rama 'public'..."
git checkout -B public main

# 5. Filtrado de Archivos (Lo que NO va a GitHub)
Write-Host "[*] Aplicando filtros de seguridad DevSecOps para publicación..." -ForegroundColor Cyan
git rm -r --cached tests/ -f 2>$null
git rm -r --cached configs/ -f 2>$null
git rm -r --cached scripts/ -f 2>$null
git rm -r --cached data/ -f 2>$null
git rm --cached .gitlab-ci.yml -f 2>$null
git rm -r --cached src/llm/ -f 2>$null
git rm -r --cached src/ml_model/ -f 2>$null

# 6. Commit de Lanzamiento y Push a GitHub
git commit -m "docs: release update to public portfolio (sanitized)" --allow-empty
Write-Host "[*] Subiendo a GitHub (origin)..." -ForegroundColor Green
git push origin public:main --force

# 7. Retorno Seguro al Entorno de Trabajo
Write-Host "[*] Volviendo al Laboratorio (main)..."
git checkout main -f
git clean -fd 2>$null

Write-Host "[*] Portafolio público actualizado en GitHub. Laboratorio protegido en GitLab." -ForegroundColor Green