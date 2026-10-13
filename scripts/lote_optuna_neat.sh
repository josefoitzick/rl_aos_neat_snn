#!/usr/bin/env bash
# Estudio de hiperparámetros de NEAT con Optuna (docs/diseno_neat_ap.md §17, scripts/optuna_neat.py) para un entorno:
# W trabajadores comparten el estudio; cada trial usa 3 semillas × 4 hilos = 12 hilos.
# Reparto acordado (2026-10-09, tope 300 hilos): bottleneck 12 trabajadores (144), acrobot-swingup 8 (96), l2f 5 (60).
#
# Protecciones:
#   - candado por entorno: un segundo lanzamiento del mismo entorno no arranca;
#   - binario congelado en ~/tesis/runs/optuna_neat_bin (sha256 en el log), verificado con
#     ~/tesis/runs/optuna_verif/verificar.sh (sin opciones nuevas = binario de la sonda; valores por defecto
#     explícitos = sin opciones; cada opción nueva cambia el resultado);
#   - el estudio vive en un JournalStorage en archivo: si se corta, se relanza y continúa (los trials en curso
#     quedan como RUNNING y no cuentan como completos).
#
# Uso: tmux new -d -s optuna_<entorno> 'bash scripts/lote_optuna_neat.sh <entorno> <trabajadores> [TRIALS=100]'
set -u
cd "$(dirname "$0")/.."
ENV=$1
W=$2
TRIALS=${3:-100}
PY=~/miniconda3/envs/tdi/bin/python
BIN=~/tesis/runs/optuna_neat_bin/neat_control
LOG=~/tesis/runs/optuna_neat/logs
mkdir -p "$LOG"

exec 9> ~/tesis/runs/optuna_neat_"$ENV".lock
if ! flock -n 9; then
    echo "Ya hay un estudio de $ENV en curso (candado tomado). No se lanza otro." | tee -a "$LOG/lanzador.log"
    exit 1
fi
[ -x "$BIN" ] || { echo "Falta el binario congelado $BIN" | tee -a "$LOG/lanzador.log"; exit 1; }
echo "inicio $ENV: $(date)  trabajadores: $W × 12 hilos  trials: $TRIALS  binario sha256: $(sha256sum "$BIN" | cut -c1-16)" \
     | tee -a "$LOG/lanzador.log"
for w in $(seq 1 "$W"); do
    "$PY" scripts/optuna_neat.py --env "$ENV" --trials "$TRIALS" --binary "$BIN" > "$LOG/${ENV}_w$w.log" 2>&1 &
    sleep 5   # el primer trabajador crea el estudio y encola la configuración por defecto
done
wait
echo "estudio $ENV terminado: $(date)" | tee -a "$LOG/lanzador.log" > "$LOG/FIN_$ENV"
