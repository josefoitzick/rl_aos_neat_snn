#!/usr/bin/env bash
# Encadena las semillas 6–10 del barrido largo tras las 1–5 (docs/diseno_neat_ap.md §15.4.1; tope de 256 hilos):
#   - ant 6–10 arranca cuando lote_barrido_largo_ant.sh (semillas 1–5) suelta su candado; ese mismo lanzador
#     omite las corridas con summary.txt y corre solo las pendientes (10 × 10 hilos);
#   - el resto 6–10 arranca cuando lote_barrido_largo.sh 1-5 suelta el suyo (36 × 4 hilos).
# Cada grupo ocupa los hilos que deja su par, así que el total no sube.
#
# Uso: tmux new -d -s encadenar 'bash scripts/encadenar_semillas_6_10.sh'
set -u
cd "$(dirname "$0")/.."
LOG=~/tesis/runs/barrido_largo_logs/encadenar.log

(
    exec 8> ~/tesis/runs/barrido_largo_ant.lock
    echo "ant 6–10: esperando el candado de ant 1–5: $(date)" >> "$LOG"
    flock 8; flock -u 8
    echo "ant 6–10: lanzado: $(date)" >> "$LOG"
    bash scripts/lote_barrido_largo_ant.sh
) &
(
    exec 9> ~/tesis/runs/barrido_largo.lock
    echo "resto 6–10: esperando el candado del resto 1–5: $(date)" >> "$LOG"
    flock 9; flock -u 9
    echo "resto 6–10: lanzado: $(date)" >> "$LOG"
    bash scripts/lote_barrido_largo.sh 6-10 36
) &
wait
echo "encadenamiento terminado: $(date)" >> "$LOG"
