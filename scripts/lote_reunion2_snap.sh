#!/bin/bash
# Tercera curva (AP estilo SNAP-NEAT) para la reunión 2; binario en build-snap/.
set -u
cd "$(dirname "$0")/.."
exec > >(tee -a "$HOME/tesis/runs/reunion2_snap.log") 2>&1
B="--binary build-snap/bin/neat_control --seeds 1-11 --jobs 11 --threads 4"
python3 scripts/run_grid.py configs/reunion2_snap_pendulum.json $B
python3 scripts/run_grid.py configs/reunion2_snap_acrobot.json $B
echo "== LOTE-TERMINADO $(date)"
