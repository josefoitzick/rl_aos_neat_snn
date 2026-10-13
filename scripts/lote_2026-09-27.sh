#!/bin/bash
# Lote nocturno 2026-09-27 (se lanza en tmux para que sobreviva al cierre de sesión):
#   1. Líneas base preliminares: estático vs AP, 11 réplicas, Pendulum y Acrobot ±1
#   2. Análisis estadístico (§3.4) de ambos entornos
#   3. Estudio del crédito/β de Adaptive Pursuit (Pendulum, 5 semillas)
#   4. Horizonte de convergencia del estático (Pendulum 400 gen, Acrobot 300 gen)
#   5. Resúmenes en ~/tesis/runs/lote_2026-09-27_resumen.txt
# Uso: tmux new-session -d -s tesis_lote "bash scripts/lote_2026-09-27.sh"
set -u
cd "$(dirname "$0")/.."
R="$HOME/tesis/runs"
APY="$HOME/miniconda3/envs/tesis-analisis/bin/python"
LOG="$R/lote_2026-09-27.log"
SUM="$R/lote_2026-09-27_resumen.txt"
GRID="python3 scripts/run_grid.py --jobs 11 --threads 4"
exec > >(tee -a "$LOG") 2>&1

echo "== inicio $(date)"
$GRID configs/lb_preliminar_pendulum.json --seeds 1-11
$GRID configs/lb_preliminar_acrobot.json --seeds 1-11
echo "== análisis $(date)"
$APY analysis/compare.py "$R/lb_preliminar_pendulum" --adaptive ap --baselines static \
     --threshold-from "$R/calib_replace" --threshold-frac 0.95 > /dev/null
$APY analysis/compare.py "$R/lb_preliminar_acrobot" --adaptive ap --baselines static \
     --threshold-from "$R/calib_acrobot" --threshold-frac 0.95 > /dev/null
$GRID configs/ap_credito_pendulum.json --seeds 1-5
$GRID configs/horizonte_static.json --seeds 1-5

{
  echo "Lote 2026-09-27 — terminado $(date)"
  for e in lb_preliminar_pendulum lb_preliminar_acrobot ap_credito_pendulum horizonte_static; do
    echo; echo "### $e"; python3 analysis/summarize.py "$R/$e"
  done
  echo; echo "Reportes estadísticos: $R/lb_preliminar_*/analysis/report.md"
} > "$SUM"
echo "== LOTE-TERMINADO $(date)"
