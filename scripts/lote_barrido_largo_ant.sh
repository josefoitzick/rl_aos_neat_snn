#!/usr/bin/env bash
# ant del barrido largo, aparte y con 1000 generaciones (configs/barrido_largo_ant.json; docs/diseno_neat_ap.md
# §15.4.1): a 2000 generaciones tardaba 4–5 días con el nodo lleno y debe estar antes de la reunión del 2026-10-06.
#   20 procesos (2 protocolos × 10 semillas) × 10 hilos = 200, nice 10.
# Mismas protecciones que lote_barrido_largo.sh: candado propio, .running, binario congelado
# (~/tesis/runs/barrido_largo_bin, el mismo del barrido largo), reanudable.
#
# Uso: tmux new -d -s barrido_largo_ant 'bash scripts/lote_barrido_largo_ant.sh'
set -u
cd "$(dirname "$0")/.."
LOG=~/tesis/runs/barrido_largo_ant_logs
mkdir -p "$LOG"
CFG=configs/barrido_largo_ant.json
BIN=~/tesis/runs/barrido_largo_bin/neat_control

exec 9> ~/tesis/runs/barrido_largo_ant.lock
if ! flock -n 9; then
    echo "Ya hay un barrido_largo_ant en curso (candado tomado). No se lanza otro." | tee -a "$LOG/lanzador.log"
    exit 1
fi
echo "inicio: $(date)  commit: $(git rev-parse --short HEAD 2>/dev/null)  rl-tools: $(git -C external/rl-tools rev-parse --short HEAD)" \
     "binario sha256: $(sha256sum "$BIN" | cut -c1-16)" | tee -a "$LOG/lanzador.log"
python3 scripts/run_grid.py "$CFG" --seeds 1-10 --jobs 20 --threads 10 --nice 10 --binary "$BIN" > "$LOG/ant.log" 2>&1
echo "barrido_largo_ant terminado: $(date)" | tee -a "$LOG/lanzador.log" > "$LOG/FIN"
