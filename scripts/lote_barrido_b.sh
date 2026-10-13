#!/usr/bin/env bash
# Barrido de la etapa A, extensión B (configs/barrido_etapa_b.json): entornos visuales con escenas 3D
# (l2f-visual, hyperdrone-target-frame, hyperdrone-moving-gate), con memoria y reactivos.
# 6 variantes × 5 semillas × 500 generaciones = 30 corridas. Mismo protocolo que A; equivalencia de
# binarios certificada (docs/entornos_rltools.md §6.2).
#
# Presupuesto: el mismo de A (160 hilos; 128 núcleos quedan para otro usuario). Por eso este lanzador
# ESPERA a que termine A: se bloquea en el candado de A y arranca apenas A lo libera.
#   30 procesos × 5 hilos = 150 hilos, nice 10.
#
# Protecciones (las mismas de A): candado propio, marca .running por corrida, binario congelado
# (~/tesis/runs/barrido_etapa_b_bin, sha256 1a5d7d99…, el verificado en B5), reanudable.
#
# Uso: tmux new -d -s barrido_b 'bash scripts/lote_barrido_b.sh'
set -u
cd "$(dirname "$0")/.."
LOG=~/tesis/runs/barrido_etapa_b_logs
mkdir -p "$LOG"
CFG=configs/barrido_etapa_b.json
BIN=~/tesis/runs/barrido_etapa_b_bin/neat_control

exec 8> ~/tesis/runs/barrido_etapa_b.lock
if ! flock -n 8; then
    echo "Ya hay un barrido_etapa_b en curso (candado tomado). No se lanza otro." | tee -a "$LOG/lanzador.log"
    exit 1
fi

echo "esperando a que termine barrido_etapa_a: $(date)" | tee -a "$LOG/lanzador.log"
exec 9> ~/tesis/runs/barrido_etapa_a.lock
flock 9          # bloquea hasta que el lanzador de A suelte su candado
flock -u 9

echo "inicio: $(date)  commit: $(git rev-parse --short HEAD 2>/dev/null)  rl-tools: $(git -C external/rl-tools rev-parse --short HEAD)" \
     "binario sha256: $(sha256sum "$BIN" | cut -c1-16)" | tee -a "$LOG/lanzador.log"
python3 scripts/run_grid.py "$CFG" --seeds 1-5 --jobs 30 --threads 5 --nice 10 --binary "$BIN" > "$LOG/b.log" 2>&1
echo "barrido_etapa_b terminado: $(date)" | tee -a "$LOG/lanzador.log" > "$LOG/FIN"
