#!/bin/bash
# Espera a que terminen ambos lotes de la reunión 2 y genera figuras + estadística con las 3 variantes.
set -u
cd "$(dirname "$0")/.."
R="$HOME/tesis/runs"; APY="$HOME/miniconda3/envs/tesis-analisis/bin/python"
until grep -q LOTE-TERMINADO "$R/reunion2.log" 2>/dev/null && grep -q LOTE-TERMINADO "$R/reunion2_snap.log" 2>/dev/null; do sleep 60; done
exec > >(tee -a "$R/reunion2_final.log") 2>&1
$APY analysis/plot_curvas.py "$R/reunion2_pendulum" "$R/reunion2_acrobot" --variants estatico ap ap_snap --out "$R/reunion2_figuras"
$APY analysis/compare.py "$R/reunion2_pendulum" --adaptive ap ap_snap --baselines estatico --threshold-from "$R/calib_replace" > /dev/null
$APY analysis/compare.py "$R/reunion2_acrobot" --adaptive ap ap_snap --baselines estatico --threshold-from "$R/calib_acrobot" > /dev/null
for e in reunion2_pendulum reunion2_acrobot; do echo "### $e"; python3 analysis/summarize.py "$R/$e"; done > "$R/reunion2_resumen.txt"
echo "== FINALIZADO $(date)"
