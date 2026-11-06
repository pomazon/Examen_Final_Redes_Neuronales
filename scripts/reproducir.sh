#!/usr/bin/env bash
# =============================================================================
# scripts/reproducir.sh
# Regenera tablas y figuras del artículo (Examen Final MIA-203).
#
# Uso:
#   bash scripts/reproducir.sh all              # pipeline completo
#   bash scripts/reproducir.sh datos            # descarga + preparación + particiones
#   bash scripts/reproducir.sh linea_base       # línea base
#   bash scripts/reproducir.sh entrenar         # 6 configs x 3 semillas
#   bash scripts/reproducir.sh metricas         # métricas + resumen
#   bash scripts/reproducir.sh errores          # matrices de confusión + errores
#   bash scripts/reproducir.sh gradcam          # Grad-CAM
#   bash scripts/reproducir.sh costo            # costo computacional
#   bash scripts/reproducir.sh aumentos         # máscaras / aumentos
#   bash scripts/reproducir.sh fondo            # prueba de fondo
#   bash scripts/reproducir.sh figuras          # figuras del artículo
#   bash scripts/reproducir.sh limpiar          # borra results/ (¡cuidado!)
#
# Variables de entorno opcionales:
#   PYTHON=python3        intérprete a usar
#   SEEDS="0 1 2"         semillas a ejecutar
#   SKIP_EXISTING=1       no reentrena si ya existe metricas.json
#   LOG_DIR=logs          carpeta de logs
# =============================================================================

set -euo pipefail

# ----------------------------- Configuración ---------------------------------
PYTHON="${PYTHON:-python3}"
SEEDS="${SEEDS:-0 1 2}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
LOG_DIR="${LOG_DIR:-logs}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIGS=(
  "resnet50_basico"
  "resnet50_fondo"
  "resnet50_fuerte"
  "vit_b_16_basico"
  "vit_b_16_fondo"
  "vit_b_16_fuerte"
)

# Carpeta de datos de ejemplos (para demo/Grad-CAM)
EJEMPLOS_DIR="data/ejemplos"

# ----------------------------- Utilidades ------------------------------------
log()  { printf "\n[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }
info() { printf "  -> %s\n" "$*"; }
die()  { printf "\n[ERROR] %s\n" "$*" >&2; exit 1; }

need() {
  command -v "$1" >/dev/null 2>&1 || die "No se encontró '$1'. Instálalo o ajusta PATH."
}

ensure_dirs() {
  mkdir -p "$LOG_DIR"
  mkdir -p results/{linea_base,entrenamiento,errores,gradcam,costo,inventario,particiones,aumentos,duplicados,prueba_fondo,articulo}
}

run_step() {
  # run_step <nombre_log> <comando...>
  local name="$1"; shift
  local logfile="$LOG_DIR/${name}.log"
  log "Ejecutando: $name"
  info "Comando: $*"
  info "Log: $logfile"
  if ! "$@" >"$logfile" 2>&1; then
    printf "\n[FALLO] Ver %s\n" "$logfile" >&2
    tail -n 30 "$logfile" >&2 || true
    exit 1
  fi
}

# Ejecuta entrenamiento sólo si no existe salida (si SKIP_EXISTING=1)
run_entrenamiento() {
  local cfg="$1" seed="$2"
  local out="results/entrenamiento/${cfg}/semilla_${seed}/metricas.json"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out" ]]; then
    info "SKIP ${cfg} seed=${seed} (ya existe $out)"
    return 0
  fi
  run_step "entrenar_${cfg}_semilla_${seed}" \
    "$PYTHON" src/entrenar.py --config "configs/${cfg}.yaml" --semilla "$seed"
}

# ----------------------------- Etapas ----------------------------------------

stage_datos() {
  log "Etapa 1/10: datos y particiones"
  need "$PYTHON"
  # 1. Descarga (si aplica). Ajusta args según src/descargar_datos.py
  run_step "descargar_datos" "$PYTHON" src/descargar_datos.py

  # 2. Preparación de imágenes (resize, normalización, fondo, etc.)
  run_step "preparar_imagenes" "$PYTHON" src/preparar_imagenes.py

  # 3. Inventario y muestras
  run_step "inventario" "$PYTHON" src/inventario.py \
    --salida results/inventario/inventario.csv \
    --figura results/inventario/muestras.png

  # 4. Duplicados / casi-duplicados
  run_step "duplicados" "$PYTHON" src/duplicados.py \
    --salida results/duplicados/pares.csv

  # 5. Particiones train/val/test (usa data/splits.csv)
  run_step "particiones" "$PYTHON" src/particiones.py \
    --salida results/particiones/conteo.csv \
    --excluidas results/particiones/excluidas.csv
}

stage_linea_base() {
  log "Etapa 2/10: línea base"
  run_step "linea_base" "$PYTHON" src/linea_base.py \
    --salida results/linea_base/metricas.csv \
    --por_semilla results/linea_base/metricas_por_semilla.csv
}

stage_entrenar() {
  log "Etapa 3/10: entrenamiento (${#CONFIGS[@]} configs x ${SEEDS})"
  for cfg in "${CONFIGS[@]}"; do
    for seed in $SEEDS; do
      run_entrenamiento "$cfg" "$seed"
    done
  done
}

stage_metricas() {
  log "Etapa 4/10: métricas y resumen"
  run_step "metricas" "$PYTHON" src/metricas.py
  run_step "resumen"  "$PYTHON" src/resumen.py \
    --salida results/resumen.csv
}

stage_errores() {
  log "Etapa 5/10: matrices de confusión y análisis de errores"
  run_step "errores" "$PYTHON" src/errores.py \
    --salida results/errores \
    --figura results/errores/confusion.png
}

stage_gradcam() {
  log "Etapa 6/10: Grad-CAM"
  run_step "gradcam" "$PYTHON" src/gradcam.py \
    --config configs/vit_b_16_fondo.yaml \
    --salida results/gradcam/gradcam.png \
    --csv results/gradcam/cam_en_hoja.csv
}

stage_costo() {
  log "Etapa 7/10: costo computacional"
  run_step "costo" "$PYTHON" src/costo.py \
    --salida results/costo.csv
}

stage_aumentos() {
  log "Etapa 8/10: máscaras / aumentos"
  run_step "aumentos" "$PYTHON" src/aumentos.py \
    --salida results/aumentos/mascaras.csv \
    --figura results/aumentos/mascaras.png
}

stage_fondo() {
  log "Etapa 9/10: prueba de fondo (background)"
  run_step "prueba_fondo" "$PYTHON" src/prueba_fondo.py \
    --salida results/prueba_fondo/prueba_fondo.csv \
    --figura results/prueba_fondo/versiones.png
}

stage_figuras() {
  log "Etapa 10/10: figuras del artículo"
  # Figuras sueltas de ejemplos (rocole/saposoa/uganda)
  run_step "ejemplos" "$PYTHON" src/ejemplos.py \
    --dir "$EJEMPLOS_DIR" \
    --salida results/articulo/ejemplos.pdf

  # Empaquetado final: curvas, duplicados, errores, fondo, gradcam, arquitectura
  run_step "figuras_articulo" "$PYTHON" src/figuras_articulo.py \
    --results results \
    --salida results/articulo
}

stage_limpiar() {
  log "Borrando results/ (¡esto elimina TODO el progreso!)"
  read -r -p "¿Continuar? [y/N] " ans
  [[ "$ans" =~ ^[yY]$ ]] || { info "Cancelado."; exit 0; }
  rm -rf results
  info "results/ eliminado."
}

usage() {
  cat <<'EOF'
Uso: bash scripts/reproducir.sh <etapa>

Etapas:
  all          pipeline completo (datos -> ... -> figuras)
  datos        descarga, preparación, inventario, duplicados, particiones
  linea_base   entrena y evalúa línea base
  entrenar     entrena 6 configs x semillas (SEEDS="0 1 2")
  metricas     calcula métricas y resumen
  errores      matrices de confusión y análisis de errores
  gradcam      Grad-CAM sobre una config
  costo        costo computacional
  aumentos     máscaras / aumentos
  fondo        prueba de fondo
  figuras      figuras del artículo
  limpiar      borrar results/ (pide confirmación)

Variables:
  PYTHON=python3   SEEDS="0 1 2"   SKIP_EXISTING=1   LOG_DIR=logs
EOF
}

# ----------------------------- Main -----------------------------------------
ensure_dirs
case "${1:-}" in
  all)
    stage_datos
    stage_linea_base
    stage_entrenar
    stage_metricas
    stage_errores
    stage_gradcam
    stage_costo
    stage_aumentos
    stage_fondo
    stage_figuras
    log "Listo. Revisa results/ y logs/."
    ;;
  datos)      stage_datos ;;
  linea_base) stage_linea_base ;;
  entrenar)   stage_entrenar ;;
  metricas)   stage_metricas ;;
  errores)    stage_errores ;;
  gradcam)    stage_gradcam ;;
  costo)      stage_costo ;;
  aumentos)   stage_aumentos ;;
  fondo)      stage_fondo ;;
  figuras)    stage_figuras ;;
  limpiar)    stage_limpiar ;;
  ""|-h|--help|help) usage ;;
  *) die "Etapa desconocida: $1 (usa --help)" ;;
esac