#!/usr/bin/env bash
# Sonda de P fija (docs/diseno_neat_ap.md §16, configs/sonda_p.json): en bottleneck, acrobot-swingup y l2f,
# 10 vectores P (5 fijos, 2 que cambian en la generación 250, 3 ablaciones) × fijo/rotación × 10 semillas ×
# 500 generaciones = 600 corridas. Pregunta si la elección de operadores importa antes de entrenar un AOS.
#
# Recursos (2026-10-07): nodo casi libre y el usuario autorizó usarlo al máximo sin saturarlo:
# JOBS × 4 hilos con nice 10 (2026-10-07 22:15: 75 × 4 = 300, otro usuario necesita 257).
#
# Protecciones contra resultados corruptos (las de los barridos anteriores):
#   - candado: un segundo lanzamiento no arranca mientras éste corre;
#   - run_grid.py marca cada corrida con .running y no la ejecuta dos veces; omite las que tienen summary.txt;
#   - binario congelado en ~/tesis/runs/sonda_p_bin (sha256 en el log), verificado con
#     ~/tesis/runs/sonda_p_verif/verificar.sh: sin --p es idéntico al del barrido largo, --p con el P de Stanley
#     es idéntico a no pasarlo, el cambio de P ocurre en la generación indicada y la ablación no usa el operador.
#   - control posterior: stanley + fijo debe reproducir las primeras 500 generaciones del barrido largo.
#
# Uso: tmux new -d -s sonda_p 'bash scripts/lote_sonda_p.sh [SEMILLAS=1-10] [JOBS=75]'
set -u
cd "$(dirname "$0")/.."
SEMILLAS=${1:-1-10}
JOBS=${2:-75}
LOG=~/tesis/runs/sonda_p_logs
mkdir -p "$LOG"
CFG=configs/sonda_p.json
BIN=~/tesis/runs/sonda_p_bin/neat_control

exec 9> ~/tesis/runs/sonda_p.lock
if ! flock -n 9; then
    echo "Ya hay una sonda_p en curso (candado tomado). No se lanza otra." | tee -a "$LOG/lanzador.log"
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "Falta el binario congelado $BIN (copiarlo tras la verificación)." | tee -a "$LOG/lanzador.log"
    exit 1
fi
echo "inicio: $(date)  semillas: $SEMILLAS  jobs: $JOBS × 4 hilos  rl-tools: $(git -C external/rl-tools rev-parse --short HEAD)" \
     " binario sha256: $(sha256sum "$BIN" | cut -c1-16)" | tee -a "$LOG/lanzador.log"
python3 scripts/run_grid.py "$CFG" --seeds "$SEMILLAS" --jobs "$JOBS" --threads 4 --nice 10 --binary "$BIN" \
    > "$LOG/sonda_s$SEMILLAS.log" 2>&1
echo "sonda_p (semillas $SEMILLAS) terminada: $(date)" | tee -a "$LOG/lanzador.log" > "$LOG/FIN_s$SEMILLAS"
