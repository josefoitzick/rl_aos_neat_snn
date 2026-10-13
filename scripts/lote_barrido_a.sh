#!/usr/bin/env bash
# Barrido de la etapa A de la selección de entornos (docs/entornos_rltools.md, configs/barrido_etapa_a.json):
# 27 variantes × 5 semillas × 500 generaciones = 135 corridas de NEAT estático.
#
# Reparto en mullo (288 núcleos físicos, 2 NUMA), medido en ~/tesis/runs/bench_hilos: la eficiencia por
# corrida cae sobre 8 hilos (150 genomas por generación), así que conviene muchas corridas con pocos hilos.
#   - ant (lo más caro):  5 procesos × 16 hilos = 80
#   - resto:             20 procesos ×  4 hilos = 80
# Total 160 hilos, con nice 10. Deja 128 núcleos físicos libres para otro usuario (acordado 2026-10-03).
#
# Protecciones contra resultados corruptos:
#   - candado: un segundo lanzamiento del barrido no arranca mientras éste corre;
#   - run_grid.py marca cada corrida con .running (atómico) y no la ejecuta dos veces;
#   - binario congelado en ~/tesis/runs/barrido_etapa_a_bin (sha256 en el log);
#   - reanudable: run_grid.py omite las corridas con summary.txt.
#
# Uso: tmux new -d -s barrido_a 'bash scripts/lote_barrido_a.sh'
set -u
cd "$(dirname "$0")/.."
LOG=~/tesis/runs/barrido_etapa_a_logs
mkdir -p "$LOG"
CFG=configs/barrido_etapa_a.json

exec 9> ~/tesis/runs/barrido_etapa_a.lock
if ! flock -n 9; then
    echo "Ya hay un barrido_etapa_a en curso (candado tomado). No se lanza otro." | tee -a "$LOG/lanzador.log"
    exit 1
fi
# Binario congelado: todas las corridas usan el mismo ejecutable aunque build/ se recompile durante el barrido
BIN=~/tesis/runs/barrido_etapa_a_bin/neat_control
if [ ! -x "$BIN" ]; then
    mkdir -p "$(dirname "$BIN")"
    cp build/bin/neat_control "$BIN"
fi
echo "inicio: $(date)  commit: $(git rev-parse --short HEAD 2>/dev/null)  rl-tools: $(git -C external/rl-tools rev-parse --short HEAD)" \
     "binario sha256: $(sha256sum "$BIN" | cut -c1-16)" | tee -a "$LOG/lanzador.log"

python3 scripts/run_grid.py "$CFG" --seeds 1-5 --only ant --jobs 5 --threads 16 --nice 10 --binary "$BIN" \
    > "$LOG/ant.log" 2>&1 &
python3 scripts/run_grid.py "$CFG" --seeds 1-5 --exclude ant --jobs 20 --threads 4 --nice 10 --binary "$BIN" \
    > "$LOG/resto.log" 2>&1 &
wait
echo "barrido_etapa_a terminado: $(date)" | tee -a "$LOG/lanzador.log" > "$LOG/FIN"
