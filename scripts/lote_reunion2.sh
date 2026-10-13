#!/bin/bash
# Experimento para la 2.ª reunión: estático vs AP, 11 réplicas, Pendulum (400 gen) y Acrobot ±1 (200 gen).
set -u
cd "$(dirname "$0")/.."
R="$HOME/tesis/runs"; APY="$HOME/miniconda3/envs/tesis-analisis/bin/python"
exec > >(tee -a "$R/reunion2.log") 2>&1
echo "== inicio $(date)"
python3 scripts/run_grid.py configs/reunion2_pendulum.json --seeds 1-11 --jobs 11 --threads 4
python3 scripts/run_grid.py configs/reunion2_acrobot.json --seeds 1-11 --jobs 11 --threads 4
$APY analysis/compare.py "$R/reunion2_pendulum" --adaptive ap --baselines estatico --threshold-from "$R/calib_replace" > /dev/null
$APY analysis/compare.py "$R/reunion2_acrobot" --adaptive ap --baselines estatico --threshold-from "$R/calib_acrobot" > /dev/null
echo "== LOTE-TERMINADO $(date)"
